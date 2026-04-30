import json
import tempfile
import os
import logging
from typing import List, Tuple, Dict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = logging.getLogger(__name__)


class SimpleMetricsComputer:
    """
    Precision / Recall / F1 / mAP@0.5 for single-class detection.

    GT and prediction boxes must be in [x1, y1, x2, y2] pixel coordinates.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self._gt: Dict[int, List[List[float]]] = {}   # image_id → [[x1,y1,x2,y2], ...]
        self._dets: List[Dict] = []                   # [{image_id, bbox, confidence}, ...]

    def add_image_results(
        self,
        image_id: int,
        gt_boxes: List[List[float]],
        pred_boxes: List[Tuple[List[float], float]],
    ) -> None:
        """
        gt_boxes:  list of [x1, y1, x2, y2] pixel coords
        pred_boxes: list of ([x1, y1, x2, y2], confidence)
        """
        self._gt[image_id] = gt_boxes
        for bbox, conf in pred_boxes:
            self._dets.append({"image_id": image_id, "bbox": bbox, "confidence": float(conf)})

    def compute(self) -> Dict[str, float]:
        total_gt = sum(len(v) for v in self._gt.values())
        if total_gt == 0 or not self._dets:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mAP_50": 0.0}

        # ── AP (area under P-R curve) ──────────────────────────────────────
        dets_sorted = sorted(self._dets, key=lambda d: d["confidence"], reverse=True)
        matched: Dict[int, List[bool]] = {k: [False] * len(v) for k, v in self._gt.items()}
        tp_list: List[int] = []
        fp_list: List[int] = []

        for det in dets_sorted:
            img_id = det["image_id"]
            gt_boxes = self._gt.get(img_id, [])
            if not gt_boxes:
                tp_list.append(0); fp_list.append(1)
                continue

            ious = _batch_iou(det["bbox"], gt_boxes)
            best = int(np.argmax(ious))
            if float(ious[best]) >= self.iou_threshold and not matched[img_id][best]:
                tp_list.append(1); fp_list.append(0)
                matched[img_id][best] = True
            else:
                tp_list.append(0); fp_list.append(1)

        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        recall_curve = tp_cum / total_gt
        prec_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        mrec = np.concatenate(([0.0], recall_curve, [1.0]))
        mpre = np.concatenate(([0.0], prec_curve, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

        # ── Fixed-threshold P / R / F1 (last curve point ≥ threshold) ─────
        total_pred = len(self._dets)
        tp_total = int(tp_cum[-1]) if len(tp_cum) else 0
        fp_total = total_pred - tp_total
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / total_gt if total_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mAP_50": float(ap),
        }


def _batch_iou(det_box: List[float], gt_boxes: List[List[float]]) -> np.ndarray:
    """Vectorised IoU: one detection vs all GT boxes, [x1,y1,x2,y2] format."""
    gt = np.asarray(gt_boxes, dtype=np.float32)
    det = np.asarray(det_box, dtype=np.float32)
    ix1 = np.maximum(det[0], gt[:, 0])
    iy1 = np.maximum(det[1], gt[:, 1])
    ix2 = np.minimum(det[2], gt[:, 2])
    iy2 = np.minimum(det[3], gt[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_det = max(0.0, float((det[2] - det[0]) * (det[3] - det[1])))
    area_gt = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(0.0, gt[:, 3] - gt[:, 1])
    union = area_det + area_gt - inter
    return np.where(union > 0, inter / union, 0.0)

class COCOMetricsComputer:
    """Compute COCO metrics (Precision, Recall, mAP) for single class."""

    def __init__(self, class_name: str = "car", iou_threshold: float = 0.5):
        self.class_name = class_name
        self.iou_threshold = iou_threshold
        self.predictions = []
        self.ground_truths = []
        self.image_ids = set()
        self.next_image_id = 1
        self.next_annotation_id = 1

    def add_image(self, height: int, width: int) -> int:
        """Register image and return image_id."""
        img_id = self.next_image_id
        self.next_image_id += 1
        self.image_ids.add(img_id)
        return img_id

    def add_ground_truths(self, image_id: int, boxes: List[List[float]]) -> None:
        """
        Add ground truth boxes for image.

        Args:
            image_id: Image ID (from add_image)
            boxes: List of [x, y, w, h] in pixel coordinates
        """
        for bbox in boxes:
            self.ground_truths.append({
                'id': self.next_annotation_id,
                'image_id': image_id,
                'category_id': 1,  # Single class
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            })
            self.next_annotation_id += 1

    def add_predictions(self, image_id: int, boxes: List[Tuple[List[float], float]]) -> None:
        """
        Add predicted boxes for image.

        Args:
            image_id: Image ID
            boxes: List of (bbox, confidence) where bbox=[x, y, w, h]
        """
        for bbox, conf in boxes:
            self.predictions.append({
                'image_id': image_id,
                'category_id': 1,
                'bbox': bbox,
                'score': float(conf)
            })

    def compute(self) -> Dict[str, float]:
        """Compute COCO metrics."""
        if not self.ground_truths or not self.predictions:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mAP_50': 0.0}

        # Create temporary COCO annotation format
        coco_gt_dict = {
            'images': [{'id': img_id} for img_id in self.image_ids],
            'annotations': self.ground_truths,
            'categories': [{'id': 1, 'name': self.class_name}]
        }

        # Write to temp file and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_gt_dict, f)
            gt_file = f.name

        try:
            coco_gt = COCO(gt_file)
            coco_dt = coco_gt.loadRes(self.predictions)

            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Extract metrics at IoU=0.5
            stats = coco_eval.stats  # [AP, AP50, AP75, APsmall, APmedium, APlarge, AR1, AR10, AR100, ARsmall, ARmedium, ARlarge]
            ap50 = stats[1]  # AP@0.5
            ar = stats[8]     # AR@100 (recall)

            # Compute precision as secondary metric
            tp = sum(1 for pred in self.predictions if self._match_prediction(pred))
            precision = tp / len(self.predictions) if self.predictions else 0.0
            recall = ar
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'mAP_50': float(ap50)
            }
        finally:
            os.unlink(gt_file)

    def _match_prediction(self, pred: Dict) -> bool:
        """Simple match: check if prediction has IoU > threshold with any GT."""
        from utils.postprocess import compute_iou
        pred_box = pred['bbox']

        for gt in self.ground_truths:
            if gt['image_id'] != pred['image_id']:
                continue

            iou = compute_iou(pred_box, gt['bbox'])
            if iou > self.iou_threshold:
                return True

        return False

class ConfidenceStats:
    """Track confidence score distribution."""

    def __init__(self):
        self.scores = []
        self.tp_scores = []
        self.fp_scores = []

    def add_prediction(self, confidence: float, is_tp: bool = False) -> None:
        self.scores.append(confidence)
        if is_tp:
            self.tp_scores.append(confidence)
        else:
            self.fp_scores.append(confidence)

    def get_stats(self) -> Dict[str, float]:
        return {
            'mean': float(np.mean(self.scores)) if self.scores else 0.0,
            'std': float(np.std(self.scores)) if self.scores else 0.0,
            'min': float(np.min(self.scores)) if self.scores else 0.0,
            'max': float(np.max(self.scores)) if self.scores else 0.0,
            'median': float(np.median(self.scores)) if self.scores else 0.0,
        }
