import numpy as np
from typing import List
from collections import namedtuple

Detection = namedtuple('Detection', ['bbox', 'confidence', 'class_id'])

# DetectNet v2 decode constants (matches TrafficCamNet training configuration)
BBOX_SCALE = 35.0
STRIDE = 16
MIN_BBOX_SIZE = 0.02


def decode_detections(cov_output: np.ndarray, bbox_output: np.ndarray,
                      confidence_threshold: float = 0.3,
                      input_w: int = 960, input_h: int = 544) -> List[Detection]:
    """
    Decode DetectNet v2 coverage and bbox outputs to Detection objects.

    Bbox format: [x1, y1, x2, y2] normalized to [0, 1].
    class_id stores batch_idx for grouping in postprocess_batch.

    Decode formula (from DetectNet v2 spec):
        x1 = (xs * STRIDE - dx1 * BBOX_SCALE) / input_w
        y1 = (ys * STRIDE - dy1 * BBOX_SCALE) / input_h
        x2 = ((xs+1) * STRIDE + dx2 * BBOX_SCALE) / input_w
        y2 = ((ys+1) * STRIDE + dy2 * BBOX_SCALE) / input_h
    """
    batch_size = cov_output.shape[0]
    num_classes = cov_output.shape[1]
    detections = []

    for b in range(batch_size):
        for c in range(num_classes):
            cov = cov_output[b, c]
            ys, xs = np.where(cov >= confidence_threshold)
            if len(ys) == 0:
                continue

            confidences = cov[ys, xs]
            base = c * 4
            dx1 = bbox_output[b, base + 0, ys, xs]
            dy1 = bbox_output[b, base + 1, ys, xs]
            dx2 = bbox_output[b, base + 2, ys, xs]
            dy2 = bbox_output[b, base + 3, ys, xs]

            x1 = np.clip((xs * STRIDE - dx1 * BBOX_SCALE) / input_w, 0.0, 1.0)
            y1 = np.clip((ys * STRIDE - dy1 * BBOX_SCALE) / input_h, 0.0, 1.0)
            x2 = np.clip(((xs + 1) * STRIDE + dx2 * BBOX_SCALE) / input_w, 0.0, 1.0)
            y2 = np.clip(((ys + 1) * STRIDE + dy2 * BBOX_SCALE) / input_h, 0.0, 1.0)

            valid = (x2 - x1 > MIN_BBOX_SIZE) & (y2 - y1 > MIN_BBOX_SIZE)
            indices = np.where(valid)[0]

            for i in indices:
                detections.append(Detection(
                    bbox=[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                    confidence=float(confidences[i]),
                    class_id=b,
                ))

    return detections


def apply_nms(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """Apply Non-Maximum Suppression to detections with [x1, y1, x2, y2] bbox format."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept = []
    suppressed = set()

    for i, det_i in enumerate(sorted_dets):
        if i in suppressed:
            continue
        kept.append(det_i)
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue
            if compute_iou(det_i.bbox, sorted_dets[j].bbox) > iou_threshold:
                suppressed.add(j)

    return kept


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """IoU for [x1, y1, x2, y2] format."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0
