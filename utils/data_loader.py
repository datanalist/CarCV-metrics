import json
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BDD100KLoader:
    """Load BDD100K annotation JSON and corresponding images (native BDD100K format)."""

    def __init__(self, ann_json_path: str, images_dir: str,
                 category_map: Dict[str, str], max_images: Optional[int] = None):
        """
        Args:
            ann_json_path: Path to BDD100K annotations JSON (native format, list of images)
            images_dir: Path to images directory
            category_map: Dict mapping BDD100K categories to target classes (e.g., {"car": "car"})
            max_images: Limit number of images (None = all)
        """
        self.images_dir = Path(images_dir)
        self.category_map = category_map

        with open(ann_json_path, 'r') as f:
            data = json.load(f)

        # BDD100K format: list of image records
        self.images = data if isinstance(data, list) else data.get('images', [])

        if max_images:
            self.images = self.images[:max_images]

        logger.info(f"Loaded {len(self.images)} images with annotations")

    def get_image_by_id(self, image_index: int) -> Tuple[np.ndarray, str]:
        """Load image as BGR numpy array by index."""
        if image_index >= len(self.images):
            raise ValueError(f"Image index {image_index} out of range")

        img_info = self.images[image_index]
        img_path = self.images_dir / img_info['name']
        img = cv2.imread(str(img_path))
        if img is None:
            raise IOError(f"Failed to load image: {img_path}")

        return img, img_info['name']

    def get_annotations_for_image(self, image_index: int) -> List[Dict]:
        """Get ground truth boxes for image by index, filtered by category_map."""
        if image_index >= len(self.images):
            raise ValueError(f"Image index {image_index} out of range")

        img_info = self.images[image_index]
        boxes = []

        for label in img_info.get('labels', []):
            cat_name = label.get('category', '')

            # Filter by category map
            if cat_name not in self.category_map:
                continue

            target_class = self.category_map[cat_name]

            # BDD100K format: box2d has x1, y1, x2, y2
            if 'box2d' not in label:
                continue

            box2d = label['box2d']
            x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']

            # Convert to [x, y, w, h] format
            w = x2 - x1
            h = y2 - y1

            boxes.append({
                'bbox': [x1, y1, w, h],  # [x, y, w, h]
                'class': target_class,
                'area': w * h,
                'iscrowd': 0
            })

        return boxes

class ImagePreprocessor:
    """Preprocess images for TrafficCamNet (960×544 resize, normalization)."""

    def __init__(self, input_w: int = 960, input_h: int = 544,
                 mean_b: float = 103.939, mean_g: float = 116.779, mean_r: float = 123.68,
                 net_scale_factor: float = 0.0039215697906911373):
        """
        Args:
            input_w, input_h: Target input dimensions
            mean_r, mean_g, mean_b: Mean values for BGR normalization (DeepStream style)
            net_scale_factor: Scaling factor (1/255 = 0.00392...)
        """
        self.input_w = input_w
        self.input_h = input_h
        self.mean_b = mean_b
        self.mean_g = mean_g
        self.mean_r = mean_r
        self.net_scale_factor = net_scale_factor

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocess single image to model input format.

        Returns:
            (preprocessed_array (1, 3, 544, 960), scale_x, scale_y)
        """
        orig_h, orig_w = img.shape[:2]

        # Resize preserving aspect ratio (letterbox)
        scale = min(self.input_w / orig_w, self.input_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create canvas with black padding
        canvas = np.zeros((self.input_h, self.input_w, 3), dtype=np.float32)
        pad_top = (self.input_h - new_h) // 2
        pad_left = (self.input_w - new_w) // 2
        canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

        # BGR mean subtraction (DeepStream style)
        canvas[:, :, 0] -= self.mean_b  # B
        canvas[:, :, 1] -= self.mean_g  # G
        canvas[:, :, 2] -= self.mean_r  # R

        # Scale
        canvas *= self.net_scale_factor

        # NCHW format for ONNX
        tensor = np.transpose(canvas, (2, 0, 1)).astype(np.float32)
        tensor = np.expand_dims(tensor, axis=0)  # Add batch dim: (1, 3, 544, 960)

        return tensor, scale, 1.0  # scale_x = scale_y = scale
