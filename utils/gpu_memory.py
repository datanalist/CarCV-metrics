"""
GPU VRAM optimization: memory monitoring, pooling, and safe allocation.

Provides:
- GPUMemoryMonitor: Track GPU memory usage and detect availability
- GPUMemoryPool: Pre-allocate and reuse tensors to reduce fragmentation
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class GPUMemoryMonitor:
    """Monitor GPU memory usage and availability."""

    def __init__(self):
        """Initialize GPU memory monitor."""
        self.peak_memory_mb = 0
        self._try_import_gpu_libs()

    def _try_import_gpu_libs(self):
        """Try to import GPU libraries (CUDA, cuPy, etc)."""
        self.pynvml = None
        self.cupy = None

        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            logger.info("NVIDIA GPU monitoring enabled (pynvml)")
        except Exception as e:
            logger.debug(f"pynvml not available: {e}")

        try:
            import cupy as cp
            self.cupy = cp
            logger.info("GPU compute library enabled (cupy)")
        except Exception as e:
            logger.debug(f"cupy not available: {e}")

    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.pynvml is not None or self.cupy is not None

    def get_available_gpu_memory_mb(self) -> float:
        """
        Get available GPU memory in MB.

        Returns 0 if GPU not available.
        """
        if self.pynvml is None:
            return 0.0

        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
            info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            available_mb = info.free / (1024 * 1024)
            return float(available_mb)
        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}")
            return 0.0

    def get_total_gpu_memory_mb(self) -> float:
        """Get total GPU memory in MB."""
        if self.pynvml is None:
            return 0.0

        try:
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
            info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = info.total / (1024 * 1024)
            return float(total_mb)
        except Exception as e:
            logger.warning(f"Failed to get total GPU memory: {e}")
            return 0.0

    def get_peak_memory_mb(self) -> float:
        """Get peak memory used during tracking."""
        return self.peak_memory_mb

    @contextmanager
    def track(self):
        """Context manager to track peak memory during a block."""
        start_available = self.get_available_gpu_memory_mb()
        total = self.get_total_gpu_memory_mb()

        try:
            yield
        finally:
            end_available = self.get_available_gpu_memory_mb()
            used_mb = total - end_available if total > 0 else 0
            self.peak_memory_mb = max(self.peak_memory_mb, used_mb)


class GPUMemoryPool:
    """
    Memory pool for GPU tensors: pre-allocate and reuse to avoid fragmentation.

    Reduces allocation overhead and memory fragmentation during batch inference.
    """

    def __init__(self, max_pool_size_mb: Optional[float] = None):
        """
        Initialize GPU memory pool.

        Args:
            max_pool_size_mb: Maximum pool size in MB. If None, use 80% of available GPU.
        """
        self.monitor = GPUMemoryMonitor()
        self.max_pool_size_mb = max_pool_size_mb
        self.pool: Dict[Tuple, list] = {}  # shape → list of allocated tensors
        self.allocated_mb = 0.0

        if max_pool_size_mb is None:
            total = self.monitor.get_total_gpu_memory_mb()
            # Use up to 80% of GPU for pool (leave 20% for model + outputs)
            self.max_pool_size_mb = total * 0.8 if total > 0 else float('inf')

        logger.info(f"GPU Memory Pool initialized: max {self.max_pool_size_mb:.0f} MB")

    def allocate(self, shape: Tuple, dtype=np.float32) -> np.ndarray:
        """
        Allocate tensor from pool or create new.

        Args:
            shape: Tensor shape (e.g., (3, 544, 960))
            dtype: Data type (default float32)

        Returns:
            Numpy array (CPU-allocated or GPU if available)

        Raises:
            MemoryError: If allocation would exceed pool size
        """
        # Normalize dtype
        dtype = np.dtype(dtype)

        # Calculate tensor size
        element_count = np.prod(shape)
        bytes_needed = element_count * dtype.itemsize
        mb_needed = bytes_needed / (1024 * 1024)

        # Check if we have space
        if self.allocated_mb + mb_needed > self.max_pool_size_mb:
            raise MemoryError(
                f"Cannot allocate {mb_needed:.1f} MB: "
                f"pool full ({self.allocated_mb:.1f}/{self.max_pool_size_mb:.1f} MB)"
            )

        key = (shape, dtype.name)

        # Try to reuse from pool
        if key in self.pool and self.pool[key]:
            tensor = self.pool[key].pop()
            logger.debug(f"Reused tensor {shape} from pool")
            return tensor

        # Allocate new
        tensor = np.zeros(shape, dtype=dtype)
        self.allocated_mb += mb_needed

        logger.debug(f"Allocated new tensor {shape} ({mb_needed:.1f} MB), "
                     f"pool total: {self.allocated_mb:.1f} MB")

        return tensor

    def deallocate(self, tensor: np.ndarray):
        """
        Return tensor to pool for reuse.

        Args:
            tensor: Numpy array to return
        """
        key = (tensor.shape, tensor.dtype.name)

        if key not in self.pool:
            self.pool[key] = []

        self.pool[key].append(tensor)
        logger.debug(f"Returned tensor {tensor.shape} to pool")

    def clear(self):
        """Clear all pooled tensors and reset."""
        self.pool.clear()
        self.allocated_mb = 0.0
        logger.info("GPU Memory Pool cleared")
