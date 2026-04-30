"""
Tests for GPU VRAM optimization: maximize usage without system hangs.

Tests verify:
1. Increased batch sizes work
2. GPU memory is actually used
3. Memory stays within safe limits
4. Pipelining works (loader → infer → postprocess)
"""

import pytest
import numpy as np
import psutil
import logging

from utils.batch_inference import BatchInferenceEngine
from utils.batch_data_loader import BDD100KBatchLoader
from utils.gpu_memory import GPUMemoryMonitor, GPUMemoryPool

logger = logging.getLogger(__name__)


class TestGPUMemoryMonitor:
    """Test GPU memory monitoring for safety."""

    def test_monitor_detects_gpu_availability(self):
        """Monitor can detect if GPU is available."""
        monitor = GPUMemoryMonitor()
        # Should not raise regardless of GPU availability
        is_available = monitor.is_gpu_available()
        assert isinstance(is_available, bool)

    def test_monitor_reports_available_memory(self):
        """Monitor reports GPU memory (0 if no GPU)."""
        monitor = GPUMemoryMonitor()
        available_mb = monitor.get_available_gpu_memory_mb()
        assert isinstance(available_mb, (int, float))
        assert available_mb >= 0

    def test_monitor_tracks_peak_usage(self):
        """Monitor tracks peak memory during inference."""
        monitor = GPUMemoryMonitor()

        # Create dummy batch (will be on CPU if no GPU)
        batch = np.random.randn(4, 3, 544, 960).astype(np.float32)

        with monitor.track():
            # Simulate some GPU operation
            _ = batch  # Use it

        peak_mb = monitor.get_peak_memory_mb()
        assert isinstance(peak_mb, (int, float))


class TestGPUMemoryPool:
    """Test GPU memory pooling for reuse."""

    def test_pool_allocates_tensor(self):
        """Pool can allocate tensor on GPU."""
        pool = GPUMemoryPool(max_pool_size_mb=100)

        # Allocate 1 image tensor (3, 544, 960)
        tensor = pool.allocate((3, 544, 960), dtype=np.float32)

        assert tensor is not None
        assert tensor.shape == (3, 544, 960)
        assert tensor.dtype == np.float32

    def test_pool_reuses_tensors(self):
        """Pool reuses tensors instead of reallocating."""
        pool = GPUMemoryPool(max_pool_size_mb=100)

        shape = (3, 544, 960)
        tensor1 = pool.allocate(shape, dtype=np.float32)
        tensor1_id = id(tensor1)

        # Deallocate
        pool.deallocate(tensor1)

        # Allocate again - should reuse
        tensor2 = pool.allocate(shape, dtype=np.float32)
        tensor2_id = id(tensor2)

        assert tensor1_id == tensor2_id, "Pool should reuse tensors"

    def test_pool_respects_max_size(self):
        """Pool prevents allocation if it would exceed max size."""
        pool = GPUMemoryPool(max_pool_size_mb=1)  # 1 MB max (too small)

        # Try to allocate large tensor (3, 544, 960 ≈ 6 MB)
        with pytest.raises(MemoryError):
            pool.allocate((3, 544, 960), dtype=np.float32)


class TestBatchInferenceEngineWithLargeBatches:
    """Test batch inference with larger batches for VRAM optimization."""

    def test_infer_batch_1024_images(self):
        """Inference works with batch_size=1024."""
        engine = BatchInferenceEngine(
            model_path="/home/mk/CarCV-metrics/models/baseline/resnet18_trafficcamnet.onnx"
        )

        # Create dummy batch: 1024 images
        batch = np.random.randn(1024, 3, 544, 960).astype(np.float32) * 0.5

        outputs = engine.infer_batch(batch)

        assert outputs is not None
        assert isinstance(outputs, dict)
        # Should have cov and bbox outputs
        assert any('cov' in name.lower() for name in outputs.keys())
        assert any('bbox' in name.lower() for name in outputs.keys())

    def test_infer_batch_respects_memory_limit(self):
        """Inference respects GPU memory limit during batching."""
        engine = BatchInferenceEngine(
            model_path="/home/mk/CarCV-metrics/models/baseline/resnet18_trafficcamnet.onnx",
            gpu_memory_limit_mb=None  # Auto-detect safe limit
        )

        # Create a very large batch that might exceed limits
        large_batch = np.random.randn(4096, 3, 544, 960).astype(np.float32) * 0.5

        # Should either:
        # 1. Process successfully if GPU has enough memory
        # 2. Split into smaller chunks internally
        # 3. Fall back to CPU
        # But NOT hang or crash
        try:
            outputs = engine.infer_batch(large_batch)
            assert outputs is not None
        except MemoryError:
            # Acceptable - memory limit hit, raised error instead of hanging
            pass


class TestGPUPipelining:
    """Test pipelined loading + inference + postprocessing."""

    def test_pipeline_loader_inference_overlap(self):
        """Loading and inference can overlap without hanging."""
        from utils.batch_data_loader import BDD100KBatchLoaderWithPipeline

        loader = BDD100KBatchLoaderWithPipeline(
            ann_json_path="/home/mk/Загрузки/DATASETS/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
            images_dir="/home/mk/Загрузки/DATASETS/bdd100k/bdd100k/bdd100k/images/100k/val",
            category_map={"car": "car"},
            batch_size=512,
            max_images=10,
            prefetch_batches=2  # Keep 2 batches in GPU queue
        )

        engine = BatchInferenceEngine(
            model_path="/home/mk/CarCV-metrics/models/baseline/resnet18_trafficcamnet.onnx"
        )

        batch_count = 0
        while not loader.is_done():
            batch_result = loader.get_batch()
            if batch_result is None:
                break

            image_batch, metadata_list, gt_annotations_list = batch_result

            # Inference happens while next batch is being loaded
            outputs = engine.infer_batch(image_batch)
            assert outputs is not None

            batch_count += 1

        assert batch_count > 0, "Should process at least one batch"


class TestGPUMemorySafety:
    """Test that system won't hang due to memory exhaustion."""

    def test_memory_exhaustion_raises_error_not_hang(self):
        """System raises MemoryError instead of hanging."""
        monitor = GPUMemoryMonitor()
        available_mb = monitor.get_available_gpu_memory_mb()

        # Try to allocate more than available
        if available_mb > 0:
            pool = GPUMemoryPool(max_pool_size_mb=available_mb * 2)

            with pytest.raises(MemoryError):
                # Try to allocate 10x available memory
                pool.allocate(
                    (10 * available_mb // 4, 1),
                    dtype=np.float32
                )

    def test_memory_release_on_batch_complete(self):
        """Memory is released after each batch for reuse."""
        monitor = GPUMemoryMonitor()
        engine = BatchInferenceEngine(
            model_path="/home/mk/CarCV-metrics/models/baseline/resnet18_trafficcamnet.onnx"
        )

        initial_memory_mb = monitor.get_available_gpu_memory_mb()

        # Process a batch
        batch = np.random.randn(64, 3, 544, 960).astype(np.float32) * 0.5
        outputs = engine.infer_batch(batch)
        del outputs  # Release

        # Clean up
        del batch

        final_memory_mb = monitor.get_available_gpu_memory_mb()

        # Memory should be mostly recovered (allow 5% variance)
        recovered = final_memory_mb >= initial_memory_mb * 0.95
        assert recovered, f"Memory not released: {initial_memory_mb} → {final_memory_mb}"
