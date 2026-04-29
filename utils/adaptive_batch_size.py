"""
Adaptive batch size calculator: maximize GPU utilization without hanging.

Automatically adjusts batch size based on:
1. Available GPU memory
2. Model weights size
3. Input/output tensor sizes
4. Safety margin (avoid max memory to prevent hangs)
"""

import numpy as np
import logging
from typing import Tuple, Optional

from utils.gpu_memory import GPUMemoryMonitor

logger = logging.getLogger(__name__)


class AdaptiveBatchSize:
    """Calculate optimal batch size for GPU inference."""

    # Model constants (measured from actual TrafficCamNet)
    MODEL_WEIGHTS_MB = 50  # ResNet18 ONNX 5.2 MB file, ~50 MB loaded with buffers
    ONNXRUNTIME_OVERHEAD_MB = 300  # ORT CUDA runtime + scratch allocations

    # Per-image tensor sizes (for 960x544 input)
    INPUT_IMAGE_SIZE = (3, 544, 960)
    BYTES_PER_IMAGE = np.prod(INPUT_IMAGE_SIZE) * 4  # float32
    MB_PER_IMAGE = BYTES_PER_IMAGE / (1024 * 1024)

    # Output sizes per image
    OUTPUT_COV_PER_IMAGE_MB = (1 * 34 * 60 * 4) / (1024 * 1024)  # confidence map
    OUTPUT_BBOX_PER_IMAGE_MB = (4 * 34 * 60 * 4) / (1024 * 1024)  # bbox deltas
    OUTPUT_PER_IMAGE_MB = OUTPUT_COV_PER_IMAGE_MB + OUTPUT_BBOX_PER_IMAGE_MB

    def __init__(self, safety_margin_percent: float = 20.0):
        """
        Initialize adaptive batch size calculator.

        Args:
            safety_margin_percent: Percentage of GPU memory to reserve (default 20% to avoid hangs)
        """
        self.safety_margin_percent = safety_margin_percent
        self.monitor = GPUMemoryMonitor()

    def calculate(self, min_batch_size: int = 64, max_batch_size: int = 4096) -> int:
        """
        Calculate optimal batch size for current GPU.

        Algorithm:
        1. Get available GPU memory
        2. Subtract fixed overhead (model weights + runtime)
        3. Calculate remaining for batch data (inputs + outputs)
        4. Find largest batch that fits with safety margin
        5. Clamp to [min_batch_size, max_batch_size]

        Args:
            min_batch_size: Minimum allowed batch size (default 64)
            max_batch_size: Maximum allowed batch size (default 4096)

        Returns:
            Recommended batch size
        """
        total_gpu_mb = self.monitor.get_total_gpu_memory_mb()

        if total_gpu_mb == 0:
            logger.warning("GPU not available, using conservative batch size")
            return min_batch_size

        # Calculate memory available for batch data
        fixed_overhead = self.MODEL_WEIGHTS_MB + self.ONNXRUNTIME_OVERHEAD_MB
        usable_mb = total_gpu_mb - fixed_overhead

        # Apply safety margin to prevent hangs
        safety_reserve_mb = total_gpu_mb * (self.safety_margin_percent / 100.0)
        available_for_batch = usable_mb - safety_reserve_mb

        if available_for_batch < self.MB_PER_IMAGE:
            logger.warning("GPU memory very tight, using minimum batch size")
            return min_batch_size

        # Memory per image in batch (input + output)
        memory_per_image = self.MB_PER_IMAGE + self.OUTPUT_PER_IMAGE_MB

        # Calculate batch size
        batch_size = int(available_for_batch / memory_per_image)
        batch_size = max(min_batch_size, min(batch_size, max_batch_size))

        logger.info(f"GPU Memory: {total_gpu_mb:.0f} MB total, "
                    f"{available_for_batch:.0f} MB available for batch, "
                    f"suggested batch_size: {batch_size}")

        return batch_size

    def get_memory_breakdown(self, batch_size: int) -> dict:
        """
        Get memory breakdown for a given batch size.

        Returns:
            Dict with memory estimates
        """
        input_mb = batch_size * self.MB_PER_IMAGE
        output_mb = batch_size * self.OUTPUT_PER_IMAGE_MB
        total_data_mb = input_mb + output_mb

        return {
            'model_weights_mb': self.MODEL_WEIGHTS_MB,
            'runtime_overhead_mb': self.ONNXRUNTIME_OVERHEAD_MB,
            'input_batch_mb': input_mb,
            'output_batch_mb': output_mb,
            'total_data_mb': total_data_mb,
            'total_estimated_mb': self.MODEL_WEIGHTS_MB + self.ONNXRUNTIME_OVERHEAD_MB + total_data_mb
        }

    def is_safe(self, batch_size: int) -> Tuple[bool, str]:
        """
        Check if batch size is safe for current GPU.

        Returns:
            (is_safe, reason)
        """
        total_gpu_mb = self.monitor.get_total_gpu_memory_mb()
        if total_gpu_mb == 0:
            return False, "GPU not available"

        memory_needed = self.get_memory_breakdown(batch_size)['total_estimated_mb']
        safety_reserve = total_gpu_mb * (self.safety_margin_percent / 100.0)

        if memory_needed + safety_reserve > total_gpu_mb:
            return False, f"Batch {batch_size} needs {memory_needed:.0f} MB, " \
                         f"only {total_gpu_mb - safety_reserve:.0f} MB available"

        return True, f"Batch {batch_size} safe ({memory_needed:.0f}/{total_gpu_mb:.0f} MB)"
