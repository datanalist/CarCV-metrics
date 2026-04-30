import json
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BDD100KLoader:
    """Load BDD100K annotation JSON and corresponding images."""

    def __init__(self, ann_json_path: str, images_dir: str,
                 category_map: Dict[str, str], max_images: Optional[int] = None):
        """
        Args:
            ann_json_path: Path to BDD100K annotations JSON
            images_dir: Path to images directory
            category_map: Dict mapping BDD100K categories to target classes
            max_images: Limit number of images (None = all)
        """
        self.images_dir = Path(images_dir)
        self.category_map = category_map

        with open(ann_json_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            self.data = data
        else:
            self.data = [data]

        if max_images:
            self.data = self.data[:max_images]

        logger.info(f"Loaded {len(self.data)} images")

    def get_image_by_id(self, idx: int) -> Tuple[np.ndarray, str]:
        """Load image as BGR numpy array."""
        if idx >= len(self.data):
            raise ValueError(f"Image index {idx} out of range")

        img_info = self.data[idx]
        filename = img_info['name']
        img_path = self.images_dir / filename
        img = cv2.imread(str(img_path))
        if img is None:
            raise IOError(f"Failed to load image: {img_path}")

        return img, filename

    def get_annotations_for_image(self, idx: int) -> List[Dict]:
        """Get ground truth boxes for image, filtered by category_map."""
        if idx >= len(self.data):
            raise ValueError(f"Image index {idx} out of range")

        img_info = self.data[idx]
        boxes = []

        for label in img_info.get('labels', []):
            cat_name = label.get('category', '')

            if cat_name not in self.category_map:
                continue

            if 'box2d' not in label:
                continue

            target_class = self.category_map[cat_name]
            box2d = label['box2d']

            x1 = box2d['x1']
            y1 = box2d['y1']
            x2 = box2d['x2']
            y2 = box2d['y2']

            w = x2 - x1
            h = y2 - y1

            if w > 0 and h > 0:
                boxes.append({
                    'bbox': [x1, y1, w, h],
                    'class': target_class,
                    'area': w * h,
                    'iscrowd': 0
                })

        return boxes

    @property
    def images(self):
        """Return list of image records for iteration."""
        return [{'id': i, 'name': self.data[i]['name']} for i in range(len(self.data))]


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

        Direct resize to (input_w, input_h) — no letterbox, no mean subtraction.
        Normalization: pixel / 255.0 (net_scale_factor = 1/255).

        Returns:
            (preprocessed_array (1, 3, input_h, input_w), scale_x, scale_y)
        """
        orig_h, orig_w = img.shape[:2]
        scale_x = self.input_w / orig_w
        scale_y = self.input_h / orig_h

        resized = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) * self.net_scale_factor

        tensor = np.transpose(arr, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, scale_x, scale_y
