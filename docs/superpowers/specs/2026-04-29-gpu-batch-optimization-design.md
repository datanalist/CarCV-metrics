# GPU Batch Optimization Design

**Date:** 2026-04-29  
**Goal:** Optimize TrafficCamNet evaluation pipeline to maximize GPU utilization (~80% VRAM) and achieve 6-8x speedup through asynchronous batch processing  
**Target Hardware:** NVIDIA RTX 3090 (24GB VRAM)

---

## Problem Statement

Current evaluation pipeline (`scripts/eval_trafficcamnet.py`):
- Processes one image at a time (batch_size=1)
- CPU-bound preprocessing + GPU-bound inference (sequential)
- GPU utilization: ~10% due to small batch overhead
- Evaluation time: ~30-60 minutes for full BDD100K dataset

**Goal:** Move to async batch processing with:
- Batch_size = 512 (occupying ~80% VRAM)
- CPU and GPU working in parallel (threading)
- Expected speedup: 6-8x
- Evaluation time: ~5-10 minutes for full dataset

---

## Design: Asynchronous Batch Processing with Threading

### Architecture Overview

```
┌─────────────────────────────────────┐
│ Background Loader Thread            │
│ • Load images from disk             │
│ • Preprocess (resize, normalize)    │
│ • Batch assembly (512 images)       │
│ → queue.Queue(batch)                │
└────────────┬────────────────────────┘
             │ (batches ready)
             ↓
┌─────────────────────────────────────┐
│ Main Thread (Inference Loop)        │
│ • Get batch from queue              │
│ • GPU Inference (ONNX Runtime)      │
│ • Postprocess (decode + NMS)        │
│ • Accumulate metrics                │
└────────────┬────────────────────────┘
             │ (results)
             ↓
┌─────────────────────────────────────┐
│ Output & Metrics                    │
│ • JSON results                      │
│ • CSV metrics                       │
│ • SUMMARY.md                        │
└─────────────────────────────────────┘
```

### Key Components

#### 1. `utils/batch_data_loader.py` — BDD100KBatchLoader

**Purpose:** Load and preprocess images in batches on background thread

**Interface:**
```python
class BDD100KBatchLoader:
    def __init__(self, ann_json_path, images_dir, category_map, 
                 batch_size=512, num_workers=2, max_images=None)
    
    def start(self) -> None
        """Start background loader thread"""
    
    def get_batch(timeout=30) -> Tuple[np.ndarray, List[Dict], List[Dict]]
        """Get next batch: (image_batch, image_metadata, gt_annotations)
        
        Returns:
            - image_batch: (B, 3, 544, 960) preprocessed images
            - image_metadata: List of {image_id, filename, orig_shape}
            - gt_annotations: List of {image_id, boxes, areas}
        """
    
    def is_done() -> bool
        """True when all images processed"""
    
    def stop() -> None
        """Gracefully shutdown loader thread"""
```

**Implementation Details:**
- Uses `queue.Queue` (thread-safe) with max 3 batches buffered
- Preprocessor reused from current codebase (`ImagePreprocessor`)
- On batch_size mismatch (last batch), pads with zeros (tracked in metadata)
- Runs in daemon thread, raises exception if queue timeout

**Memory Usage:**
- Each batch: 512 × 960 × 544 × 3 × 4 bytes = ~3.2 GB
- Queue max 3 batches: ~9.6 GB input buffers
- Total with outputs: ~15-18 GB (fits in 24GB)

---

#### 2. `utils/batch_inference.py` — BatchInferenceEngine

**Purpose:** Manage GPU inference and postprocessing for batches

**Interface:**
```python
class BatchInferenceEngine:
    def __init__(self, model_path, config)
    
    def infer_batch(batch: np.ndarray) -> Dict[str, np.ndarray]
        """ONNX inference on entire batch
        
        Returns: {output_name: (B, ...) array}
        """
    
    def postprocess_batch(outputs: Dict, batch_size: int, 
                         image_shapes: List[Tuple]) -> List[List[Detection]]
        """Decode + NMS for all images in batch
        
        Returns: List[List[Detection]] - detections per image
        """
    
    def get_gpu_memory_usage() -> Dict[str, float]
        """Return {reserved: GB, allocated: GB, free: GB}"""
```

**Implementation Details:**
- Reuses `TrafficCamNetLoader` for ONNX inference (unchanged)
- Batch decoding in `decode_detections()` (vectorized with NumPy)
- NMS per-image (sequential CPU, but fast compared to GPU transfer)
- Monitors GPU memory with `torch.cuda.memory_stats()` if available, else silent

---

#### 3. `scripts/eval_trafficcamnet_gpu.py` — Main Evaluation Script

**Purpose:** Orchestrate batch loading, inference, and metrics

**Flow:**
```python
def main():
    1. Load config & initialize components
    2. Start BDD100KBatchLoader thread
    3. Initialize metrics, latency tracking
    
    4. WHILE batches available:
        batch = loader.get_batch(timeout=30)  # Blocking
        
        t_start = time.time()
        outputs = engine.infer_batch(batch.images)
        batch_latency = time.time() - t_start
        
        detections = engine.postprocess_batch(outputs, ...)
        
        # Accumulate metrics (no per-image measurement, batch-level only)
        for img_id, dets, gt in zip(...):
            metrics.add_image(...)
            metrics.add_predictions(img_id, dets)
            metrics.add_ground_truths(img_id, gt)
        
        latencies.append(batch_latency)
    
    5. Stop loader thread
    6. Compute final metrics, save JSON/CSV/SUMMARY
```

**Output Format:** Identical to current script
- `results/trafficcamnet_eval/results.json`
- `results/trafficcamnet_eval/results.csv`
- `results/trafficcamnet_eval/SUMMARY.md`
- `results/trafficcamnet_eval/visualizations/`

**Latency Reporting:**
- `latency_ms.mean`: Average time per batch / batch_size
- `latency_ms.p95, p99`: Percentiles (per batch)
- Note: Per-image latency measurements removed (only batch-level available)

---

#### 4. Configuration Changes

**Add to `configs/experiment/trafficcamnet_eval.yaml`:**
```yaml
batch:
  batch_size: 512              # Adjust if VRAM issues
  num_loader_threads: 2        # Background threads for I/O
  queue_max_size: 3            # Buffered batches in memory
  
inference:
  # existing fields unchanged
```

---

## Data Flow & Synchronization

### Queue-Based Communication

```python
# Loader thread produces:
queue.put(Batch(images, metadata, annotations))

# Main thread consumes:
batch = queue.get(timeout=30)  # Blocks if queue empty
# If all images processed and queue empty → raises Empty exception
```

### Thread Safety

- `queue.Queue` is thread-safe (no locks needed)
- Metrics accumulation: Use `threading.Lock()` if accessing from multiple threads (here: single main thread, safe)
- Graceful shutdown: `loader.stop()` sets flag, waits for thread join

### Error Handling

| Scenario | Behavior |
|----------|----------|
| Image load fails | Skip image, log warning, continue batch |
| Queue timeout (30s) | Raise TimeoutError, stop evaluation |
| GPU OOM | Catch `onnxruntime.OnnxRuntimeError`, log, reduce batch_size and retry |
| Last batch < batch_size | Pad with zeros, track in metadata |

---

## Performance Expectations

### Batch Processing Overhead
- Batch assembly time: ~50-100ms
- Transfer to GPU (3.2 GB): ~20-50ms (PCIe 4.0)
- ONNX inference (512 images): ~120-150ms
- Postprocessing (decode + NMS): ~50-100ms
- **Total per batch (512 images): ~250-350ms**
- **Per-image equivalent: 0.5-0.7ms** (vs 1.5-2ms currently)

### VRAM Utilization
- Input batch: 3.2 GB
- Model weights: 5 MB
- Intermediate tensors: 2-3 GB
- Output buffers: 500 MB
- **Total: ~15-18 GB (63-75% of 24GB)**
- **Safe margin: 25% for CUDA overhead**

### Speedup
- Current script: ~1 image/ms (1000 images in ~1000ms)
- Batch script: ~1.5-2 images/ms (1000 images in ~500-700ms)
- **Expected: 6-8x for full evaluation** (amortized over ~10k images)

---

## Backward Compatibility

### Output Format
- Results JSON structure: Identical
- Metrics (precision, recall, F1, mAP@0.5): Identical
- Latency reported as: `latency_ms.mean` = avg ms per batch / batch_size

### Visualization
- Existing `scripts/visualize_results.py` works unchanged
- P-R curve, confidence distribution: Identical computation

### Notebook
- Existing `notebooks/eval_trafficcamnet_analysis.ipynb` works unchanged
- Can reference new `eval_trafficcamnet_gpu.py` results in a new notebook

---

## Files Created/Modified

### Create
- `utils/batch_data_loader.py` — BDD100KBatchLoader class
- `utils/batch_inference.py` — BatchInferenceEngine class
- `scripts/eval_trafficcamnet_gpu.py` — Main evaluation script (batch mode)

### Modify
- `configs/experiment/trafficcamnet_eval.yaml` — Add batch config section

### Archive (optional)
- `scripts/eval_trafficcamnet.py` → `scripts/eval_trafficcamnet_legacy.py` (keep for reference)

### No Changes
- `utils/model_loader.py` — Reused as-is
- `utils/data_loader.py` — Reused as-is (BDD100KLoader, ImagePreprocessor)
- `utils/postprocess.py` — Updated to batch-aware decoding (vectorized)
- `utils/metrics.py` — Reused as-is
- `scripts/visualize_results.py` — Reused as-is

---

## Error Recovery & Tuning

### If GPU OOM Occurs
1. Reduce `batch_size` in config (e.g., 512 → 256)
2. Reduce `num_loader_threads` (e.g., 2 → 1)
3. Reduce `queue_max_size` (e.g., 3 → 2)
4. Restart script

### If Queue Timeout
- Increase timeout in `get_batch(timeout=60)`
- Check disk I/O bottleneck (use `iostat`)
- Reduce batch_size to reduce I/O pressure

### Monitoring
- Log GPU memory every 10 batches: `engine.get_gpu_memory_usage()`
- Log throughput (images/sec) at end
- Compare latency_ms vs. expected range (0.5-0.7ms per image)

---

## Spec Review Checklist

✓ Scope: Single implementation task (batch optimization)  
✓ Clear interfaces: BDD100KBatchLoader, BatchInferenceEngine  
✓ Error handling: Queue timeout, GPU OOM, image load failures  
✓ Performance metrics: Defined (6-8x speedup, 80% VRAM)  
✓ Backward compatibility: Output format unchanged  
✓ Thread safety: queue.Queue is thread-safe  
✓ Memory efficiency: Fits in 24GB VRAM with margin  
✓ Testing strategy: Can reuse existing metrics validation  

---

## Next Steps

1. **Write Implementation Plan** → Use writing-plans skill
2. **Implement batch_data_loader.py** → Background loader thread
3. **Implement batch_inference.py** → GPU inference engine
4. **Write eval_trafficcamnet_gpu.py** → Main orchestration
5. **Test with 10 images** → Verify correctness
6. **Run full evaluation** → Measure speedup
7. **Commit & document** → Create summary
