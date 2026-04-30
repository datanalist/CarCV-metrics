# GPU Batch Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement asynchronous batch processing on GPU to achieve 6-8x evaluation speedup with 80% VRAM utilization

**Architecture:** Background loader thread (Python threading) feeds batches to main inference loop via thread-safe queue. Batch preprocessing on CPU, batch inference on GPU, batch postprocessing on CPU. Results accumulated for metrics without per-image timing.

**Tech Stack:** Python threading, queue.Queue, ONNX Runtime (existing), NumPy

---

## File Structure

```
utils/
  batch_data_loader.py      (NEW) — BDD100KBatchLoader class
  batch_inference.py        (NEW) — BatchInferenceEngine class
  model_loader.py           (REUSE) — TrafficCamNetLoader
  data_loader.py            (REUSE) — ImagePreprocessor, BDD100KLoader
  postprocess.py            (UPDATE) — decode_detections for batch mode
  metrics.py                (REUSE) — COCOMetricsComputer

scripts/
  eval_trafficcamnet_gpu.py (NEW) — Main orchestration script
  eval_trafficcamnet.py     (ARCHIVE) — Legacy single-image version

configs/
  experiment/
    trafficcamnet_eval.yaml (UPDATE) — Add batch_size, num_loader_threads
```

---

### Task 1: Create Batch Data Loader (BDD100KBatchLoader)

**Files:**
- Create: `utils/batch_data_loader.py`

- [ ] **Step 1: Write test for BDD100KBatchLoader initialization**

```python
# tests/test_batch_data_loader.py
import pytest
from pathlib import Path
from utils.batch_data_loader import BDD100KBatchLoader

def test_batch_loader_init():
    """Test BDD100KBatchLoader can be initialized with config."""
    loader = BDD100KBatchLoader(
        ann_json_path="/home/mk/Загрузки/DATASETS/bdd100k/labels/bdd100k_labels_images_val.json",
        images_dir="/home/mk/Загрузки/DATASETS/bdd100k/images/100k/val/",
        category_map={"car": "car"},
        batch_size=4,
        num_workers=1,
        max_images=20
    )
    assert loader.batch_size == 4
    assert loader.total_images == 20
    assert loader.current_batch_idx == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/mk/CarCV-metrics
python -m pytest tests/test_batch_data_loader.py::test_batch_loader_init -v
```

Expected: `ModuleNotFoundError: No module named 'utils.batch_data_loader'`

- [ ] **Step 3: Create batch_data_loader.py skeleton**

```python
# utils/batch_data_loader.py
import json
import threading
import queue
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BDD100KBatchLoader:
    """Load and preprocess BDD100K images in batches on background thread."""
    
    def __init__(self, ann_json_path: str, images_dir: str, category_map: Dict[str, str],
                 batch_size: int = 512, num_workers: int = 2, max_images: Optional[int] = None):
        """
        Args:
            ann_json_path: Path to COCO-format annotations JSON
            images_dir: Path to images directory
            category_map: Dict mapping BDD100K categories to target classes
            batch_size: Number of images per batch
            num_workers: Number of background threads (not used in single-thread version, reserved for future)
            max_images: Limit total images (None = all)
        """
        self.batch_size = batch_size
        self.total_images = max_images
        self.current_batch_idx = 0
        
        # Load annotations
        with open(ann_json_path, 'r') as f:
            data = json.load(f)
        
        self.images = data.get('images', [])
        if max_images:
            self.images = self.images[:max_images]
        
        self.annotations = data.get('annotations', [])
        self.categories = {c['id']: c['name'] for c in data.get('categories', [])}
        self.category_map = category_map
        self.images_dir = Path(images_dir)
        
        logger.info(f"BDD100KBatchLoader: {len(self.images)} images, batch_size={batch_size}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_batch_data_loader.py::test_batch_loader_init -v
```

Expected: `PASSED`

- [ ] **Step 5: Write test for get_batch() method**

```python
# Add to tests/test_batch_data_loader.py
def test_batch_loader_get_batch():
    """Test get_batch returns correct shapes and metadata."""
    from utils.data_loader import ImagePreprocessor
    
    loader = BDD100KBatchLoader(
        ann_json_path="/home/mk/Загрузки/DATASETS/bdd100k/labels/bdd100k_labels_images_val.json",
        images_dir="/home/mk/Загрузки/DATASETS/bdd100k/images/100k/val/",
        category_map={"car": "car"},
        batch_size=4,
        max_images=10
    )
    
    batch = loader.get_batch()
    assert batch is not None
    images, metadata, gt = batch
    
    assert images.shape[0] <= 4  # batch_size
    assert images.shape == (len(metadata), 3, 544, 960)
    assert len(gt) == len(metadata)
    assert all('image_id' in m for m in metadata)
```

- [ ] **Step 6: Implement get_batch() method**

```python
# Add to utils/batch_data_loader.py

from utils.data_loader import ImagePreprocessor

class BDD100KBatchLoader:
    def __init__(self, ...):
        # ... existing code ...
        self.preprocessor = ImagePreprocessor(
            input_w=960, input_h=544,
            mean_b=103.939, mean_g=116.779, mean_r=123.68,
            net_scale_factor=0.0039215697906911373
        )
    
    def get_batch(self) -> Optional[Tuple[np.ndarray, List[Dict], List[Dict]]]:
        """Get next batch of images.
        
        Returns:
            (image_batch, metadata_list, gt_annotations_list) or None if done
        """
        if self.current_batch_idx >= len(self.images):
            return None
        
        batch_start = self.current_batch_idx
        batch_end = min(batch_start + self.batch_size, len(self.images))
        batch_images_info = self.images[batch_start:batch_end]
        actual_batch_size = len(batch_images_info)
        
        # Pre-allocate batch tensor
        batch_tensor = np.zeros((self.batch_size, 3, 544, 960), dtype=np.float32)
        metadata = []
        gt_annotations = []
        
        for i, img_info in enumerate(batch_images_info):
            image_id = img_info['id']
            img_path = self.images_dir / img_info['file_name']
            
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Failed to load image: {img_path}")
                    continue
                
                # Preprocess
                tensor, _, _ = self.preprocessor.preprocess(img)
                batch_tensor[i] = tensor[0]  # Remove batch dim from single image
                
                # Metadata
                metadata.append({
                    'image_id': image_id,
                    'filename': img_info['file_name'],
                    'orig_shape': img.shape[:2]
                })
                
                # Ground truth
                gt_boxes = []
                for ann in self.annotations:
                    if ann['image_id'] != image_id:
                        continue
                    cat_name = self.categories.get(ann['category_id'], '')
                    if cat_name not in self.category_map:
                        continue
                    gt_boxes.append(ann['bbox'])  # [x, y, w, h]
                
                gt_annotations.append(gt_boxes)
            
            except Exception as e:
                logger.error(f"Error processing image {image_id}: {e}")
                continue
        
        # Trim batch to actual size
        batch_tensor = batch_tensor[:len(metadata)]
        
        self.current_batch_idx = batch_end
        
        return batch_tensor, metadata, gt_annotations
    
    def reset(self) -> None:
        """Reset loader to beginning."""
        self.current_batch_idx = 0
    
    def is_done(self) -> bool:
        """Check if all batches have been loaded."""
        return self.current_batch_idx >= len(self.images)
```

- [ ] **Step 7: Run test for get_batch()**

```bash
python -m pytest tests/test_batch_data_loader.py::test_batch_loader_get_batch -v
```

Expected: `PASSED`

- [ ] **Step 8: Commit Task 1**

```bash
git add utils/batch_data_loader.py tests/test_batch_data_loader.py
git commit -m "feat(task-1): implement BDD100KBatchLoader for batch image loading"
```

---

### Task 2: Create Batch Inference Engine (BatchInferenceEngine)

**Files:**
- Create: `utils/batch_inference.py`
- Modify: `utils/postprocess.py`

- [ ] **Step 1: Write test for BatchInferenceEngine initialization**

```python
# tests/test_batch_inference.py
import pytest
import numpy as np
from utils.batch_inference import BatchInferenceEngine

def test_batch_inference_engine_init():
    """Test BatchInferenceEngine initialization."""
    engine = BatchInferenceEngine(
        model_path="/home/mk/CarCV-metrics/models/baseline/resnet18_trafficcamnet.onnx"
    )
    assert engine.model is not None
    assert hasattr(engine, 'infer_batch')
    assert hasattr(engine, 'postprocess_batch')
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_batch_inference.py::test_batch_inference_engine_init -v
```

Expected: `ModuleNotFoundError: No module named 'utils.batch_inference'`

- [ ] **Step 3: Create batch_inference.py skeleton**

```python
# utils/batch_inference.py
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from utils.model_loader import TrafficCamNetLoader
from utils.postprocess import decode_detections, apply_nms, Detection

logger = logging.getLogger(__name__)

class BatchInferenceEngine:
    """Manage GPU inference and postprocessing for image batches."""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to ONNX model
        """
        self.model = TrafficCamNetLoader(model_path)
        logger.info(f"BatchInferenceEngine initialized with model: {model_path}")
    
    def infer_batch(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        """Run ONNX inference on batch.
        
        Args:
            batch: (B, 3, 544, 960) preprocessed batch
        
        Returns:
            Dict mapping output names to arrays
        """
        outputs = self.model.session.run(
            self.model.output_names,
            {self.model.input_name: batch.astype(np.float32)}
        )
        return {name: out for name, out in zip(self.model.output_names, outputs)}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_batch_inference.py::test_batch_inference_engine_init -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit skeleton**

```bash
git add utils/batch_inference.py tests/test_batch_inference.py
git commit -m "feat(task-2): add BatchInferenceEngine skeleton"
```

- [ ] **Step 6: Implement postprocess_batch() - full version with all steps**

See implementation plan for complete code structure.

---

### Task 3: Update Configuration

**Files:**
- Modify: `configs/experiment/trafficcamnet_eval.yaml`

- [ ] **Step 1: Add batch configuration**

Add to config file:
```yaml
batch:
  batch_size: 512
  num_loader_threads: 2
  queue_max_size: 3
```

- [ ] **Step 2: Verify and commit**

---

### Task 4: Create Main Evaluation Script (eval_trafficcamnet_gpu.py)

**Files:**
- Create: `scripts/eval_trafficcamnet_gpu.py`

See plan for full implementation.

---

### Task 5: Test with Full Dataset

- [ ] **Step 1: Run full evaluation**
- [ ] **Step 2: Verify output files**
- [ ] **Step 3: Commit results**

---

### Task 6: Update Summary Document

- [ ] **Step 1: Add GPU batch mode note**
- [ ] **Step 2: Commit**

---

### Task 7: Archive Old Script and Final Commit

- [ ] **Step 1: Rename to legacy**
- [ ] **Step 2: Final commit**
