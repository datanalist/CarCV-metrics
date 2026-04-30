import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import threading
import queue
from utils.data_loader import BDD100KLoader, ImagePreprocessor

logger = logging.getLogger(__name__)


class BDD100KBatchLoader:
    """Load and preprocess BDD100K images in batches for efficient inference."""

    def __init__(self, ann_json_path: str, images_dir: str, category_map: Dict[str, str],
                 batch_size: int = 512, num_workers: int = 2, max_images: Optional[int] = None):
        """
        Initialize BDD100KBatchLoader.

        Args:
            ann_json_path: Path to BDD100K annotations JSON
            images_dir: Path to images directory
            category_map: Dict mapping BDD100K categories to target classes
            batch_size: Number of images per batch (default 512)
            num_workers: Number of workers for data loading (default 2)
            max_images: Limit number of images (None = all)
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load annotations using BDD100KLoader
        self.loader = BDD100KLoader(ann_json_path, images_dir, category_map, max_images)

        # Initialize image preprocessor
        self.preprocessor = ImagePreprocessor()

        # Track current batch index
        self.current_batch_idx = 0

        # Get total images from loader
        self.total_images = len(self.loader.data)

        logger.info(f"BDD100KBatchLoader initialized: {self.total_images} images, "
                    f"batch_size={batch_size}, num_workers={num_workers}")

    def get_batch(self) -> Optional[Tuple[np.ndarray, List[Dict], List[List[Dict]]]]:
        """
        Get next batch of preprocessed images, metadata, and ground truth annotations.

        Returns:
            Tuple of (image_batch, metadata_list, gt_annotations_list) or None if done.
            - image_batch: numpy array shape (batch_size, 3, 544, 960) dtype float32
            - metadata_list: list of dicts with image_id, filename, orig_shape
            - gt_annotations_list: list of lists containing ground truth boxes for each image
        """
        if self.current_batch_idx >= self.total_images:
            return None

        # Calculate batch boundaries
        start_idx = self.current_batch_idx
        end_idx = min(start_idx + self.batch_size, self.total_images)
        batch_size_actual = end_idx - start_idx

        # Collect images, metadata, and ground truth
        image_batch_list = []
        metadata_list = []
        gt_annotations_list = []

        for idx in range(start_idx, end_idx):
            try:
                # Load image
                img, filename = self.loader.get_image_by_id(idx)
                orig_h, orig_w = img.shape[:2]

                # Preprocess image
                preprocessed_tensor, scale_x, scale_y = self.preprocessor.preprocess(img)

                # Remove batch dimension from preprocessor (it returns (1, 3, H, W))
                preprocessed_img = preprocessed_tensor[0]  # (3, H, W)

                image_batch_list.append(preprocessed_img)

                # Create metadata
                metadata = {
                    'image_id': idx,
                    'filename': filename,
                    'orig_shape': (orig_h, orig_w),
                    'scale_x': scale_x,
                    'scale_y': scale_y
                }
                metadata_list.append(metadata)

                # Load ground truth annotations
                gt_boxes = self.loader.get_annotations_for_image(idx)
                gt_annotations_list.append(gt_boxes)

            except Exception as e:
                logger.warning(f"Failed to load image at index {idx}: {e}")
                # Skip this image
                continue

        # If no images loaded successfully, return None
        if not image_batch_list:
            return None

        # Stack images into batch tensor (batch_size, 3, 544, 960)
        image_batch = np.stack(image_batch_list, axis=0).astype(np.float32)

        # Update batch index
        self.current_batch_idx = end_idx

        return (image_batch, metadata_list, gt_annotations_list)

    def reset(self) -> None:
        """Reset loader to initial state."""
        self.current_batch_idx = 0
        logger.info("BDD100KBatchLoader reset to initial state")

    def is_done(self) -> bool:
        """Check if all batches have been consumed."""
        return self.current_batch_idx >= self.total_images


class BDD100KBatchLoaderWithPipeline(BDD100KBatchLoader):
    """
    Pipelined batch loader: background thread loads next batch while main thread processes current.

    Reduces GPU stalls waiting for I/O by overlapping data loading with inference.
    """

    def __init__(self, ann_json_path: str, images_dir: str, category_map: Dict[str, str],
                 batch_size: int = 512, num_workers: int = 2, max_images: Optional[int] = None,
                 prefetch_batches: int = 2):
        """
        Initialize pipelined batch loader.

        Args:
            ann_json_path: Path to BDD100K annotations JSON
            images_dir: Path to images directory
            category_map: Dict mapping BDD100K categories to target classes
            batch_size: Number of images per batch (default 512)
            num_workers: Number of workers for data loading (default 2)
            max_images: Limit number of images (None = all)
            prefetch_batches: Number of batches to prefetch (default 2)
        """
        super().__init__(ann_json_path, images_dir, category_map, batch_size, num_workers, max_images)

        self.prefetch_batches = prefetch_batches
        self.batch_queue: queue.Queue = queue.Queue(maxsize=prefetch_batches)
        self.stop_loader = False
        self.loader_exception = None

        # Start background loader thread
        self.loader_thread = threading.Thread(target=self._loader_worker, daemon=True)
        self.loader_thread.start()

        logger.info(f"Started pipelined loader with {prefetch_batches} prefetch batches")

    def _loader_worker(self):
        """Background thread: continuously load batches into queue."""
        try:
            while not self.stop_loader and not self.is_done():
                batch = super().get_batch()

                if batch is None:
                    break

                self.batch_queue.put(batch, timeout=10)  # 10s timeout to avoid deadlock

            self.batch_queue.put(None)  # Signal end of stream
        except Exception as e:
            self.loader_exception = e
            logger.error(f"Loader worker error: {e}")
            self.batch_queue.put(None)

    def get_batch(self) -> Optional[Tuple[np.ndarray, List[Dict], List[List[Dict]]]]:
        """
        Get next batch from prefetch queue (non-blocking from main thread perspective).

        Returns:
            Tuple or None if all batches consumed
        """
        try:
            batch = self.batch_queue.get(timeout=60)  # 60s timeout per batch
            return batch
        except queue.Empty:
            logger.warning("Batch queue timeout - loader stalled?")
            return None
        except Exception as e:
            if self.loader_exception:
                raise self.loader_exception
            raise

    def stop(self):
        """Stop the loader thread gracefully."""
        self.stop_loader = True
        self.loader_thread.join(timeout=5)
        logger.info("Pipelined loader stopped")
