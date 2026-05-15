"""Metrics for CARS model evaluation."""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class DetectionMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ap: float = 0.0        # Average Precision @ IoU 0.5
    map50: float = 0.0
    num_gt: int = 0
    num_pred: int = 0
    num_tp: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class ClassificationMetrics:
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    num_samples: int = 0
    per_class_accuracy: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class OCRMetrics:
    char_accuracy: float = 0.0
    full_plate_accuracy: float = 0.0
    char_error_rate: float = 0.0
    num_samples: int = 0

    def to_dict(self):
        return asdict(self)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_detection_metrics(
    predictions: List[dict],
    ground_truths: List[dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.35,
) -> DetectionMetrics:
    """
    Args:
        predictions: list of {"image_id": str, "boxes": [[x1,y1,x2,y2,conf]], ...}
        ground_truths: list of {"image_id": str, "boxes": [[x1,y1,x2,y2]], ...}
        iou_threshold: IoU threshold for TP
        conf_threshold: minimum confidence to consider a prediction
    """
    gt_by_id = {g["image_id"]: g["boxes"] for g in ground_truths}
    tp_list, fp_list, confs = [], [], []
    num_gt = sum(len(g["boxes"]) for g in ground_truths)

    for pred in predictions:
        img_id = pred["image_id"]
        gt_boxes = np.array(gt_by_id.get(img_id, []))
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        for box in sorted(pred["boxes"], key=lambda b: -b[4]):
            conf = box[4]
            if conf < conf_threshold:
                continue
            confs.append(conf)
            pred_box = np.array(box[:4])

            if len(gt_boxes) == 0:
                tp_list.append(0)
                fp_list.append(1)
                continue

            ious = np.array([iou(pred_box, g) for g in gt_boxes])
            best_idx = ious.argmax()
            if ious[best_idx] >= iou_threshold and not gt_matched[best_idx]:
                gt_matched[best_idx] = True
                tp_list.append(1)
                fp_list.append(0)
            else:
                tp_list.append(0)
                fp_list.append(1)

    if not tp_list:
        return DetectionMetrics(num_gt=num_gt)

    order = np.argsort(-np.array(confs))
    tp_cum = np.cumsum(np.array(tp_list)[order])
    fp_cum = np.cumsum(np.array(fp_list)[order])
    recalls = tp_cum / (num_gt + 1e-9)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    # AP via 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t].max() if any(recalls >= t) else 0.0
        ap += p / 11

    num_tp = int(tp_cum[-1])
    num_pred = len(tp_list)
    precision = num_tp / num_pred if num_pred > 0 else 0.0
    recall = num_tp / num_gt if num_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return DetectionMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        ap=round(ap, 4),
        map50=round(ap, 4),
        num_gt=num_gt,
        num_pred=num_pred,
        num_tp=num_tp,
    )


def compute_classification_metrics(
    predictions: List[dict],
    ground_truths: List[dict],
) -> ClassificationMetrics:
    """
    Args:
        predictions: [{"image_id": str, "top_k": ["label1", ...]}]
        ground_truths: [{"image_id": str, "label": "label1"}]
    """
    gt_by_id = {g["image_id"]: g["label"] for g in ground_truths}
    top1_correct = top3_correct = 0
    per_class_total = {}
    per_class_correct = {}

    for pred in predictions:
        img_id = pred["image_id"]
        gt = gt_by_id.get(img_id)
        if gt is None:
            continue
        topk = pred["top_k"]
        per_class_total[gt] = per_class_total.get(gt, 0) + 1

        if topk[0] == gt:
            top1_correct += 1
            per_class_correct[gt] = per_class_correct.get(gt, 0) + 1
        if gt in topk[:3]:
            top3_correct += 1

    n = len(predictions)
    per_class_acc = {
        cls: round(per_class_correct.get(cls, 0) / per_class_total[cls], 4)
        for cls in per_class_total
    }

    return ClassificationMetrics(
        top1_accuracy=round(top1_correct / n, 4) if n > 0 else 0.0,
        top3_accuracy=round(top3_correct / n, 4) if n > 0 else 0.0,
        num_samples=n,
        per_class_accuracy=per_class_acc,
    )


def compute_ocr_metrics(
    predictions: List[dict],
    ground_truths: List[dict],
) -> OCRMetrics:
    """
    Args:
        predictions: [{"image_id": str, "text": "A123BC77"}]
        ground_truths: [{"image_id": str, "text": "A123BC77"}]
    """
    gt_by_id = {g["image_id"]: g["text"] for g in ground_truths}
    total_chars = correct_chars = 0
    full_correct = 0
    n = 0

    for pred in predictions:
        img_id = pred["image_id"]
        gt_text = gt_by_id.get(img_id, "")
        pred_text = pred.get("text", "")
        if not gt_text:
            continue
        n += 1
        if pred_text == gt_text:
            full_correct += 1

        # Character accuracy via edit distance
        max_len = max(len(gt_text), len(pred_text), 1)
        matches = sum(a == b for a, b in zip(gt_text, pred_text))
        correct_chars += matches
        total_chars += max_len

    cer = 1.0 - (correct_chars / total_chars) if total_chars > 0 else 1.0

    return OCRMetrics(
        char_accuracy=round(correct_chars / total_chars, 4) if total_chars > 0 else 0.0,
        full_plate_accuracy=round(full_correct / n, 4) if n > 0 else 0.0,
        char_error_rate=round(cer, 4),
        num_samples=n,
    )


def check_thresholds(metrics: dict, thresholds: dict) -> dict:
    """Return pass/fail status for each metric threshold."""
    results = {}
    for key, min_val in thresholds.items():
        actual = metrics.get(key, 0.0)
        results[key] = {
            "value": actual,
            "threshold": min_val,
            "status": "PASS" if actual >= min_val else "FAIL",
        }
    return results
