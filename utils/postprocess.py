import numpy as np
from typing import List, Tuple
from collections import namedtuple

Detection = namedtuple('Detection', ['bbox', 'confidence', 'class_id'])


def decode_detections(cov_output: np.ndarray, bbox_output: np.ndarray,
                      confidence_threshold: float = 0.3,
                      input_w: int = 960, input_h: int = 544) -> List[Detection]:
    """
    Decode TrafficCamNet coverage and bbox outputs to Detection objects.

    For batch processing, temporarily stores batch index in class_id field.
    Caller should extract and use batch index for grouping.

    Args:
        cov_output: (B, num_classes, H, W) confidence maps
        bbox_output: (B, 4*num_classes, H, W) bbox deltas
        confidence_threshold: Minimum confidence threshold
        input_w: Input width (960)
        input_h: Input height (544)

    Returns:
        List of Detection objects with bbox in normalized coords [0, 1]
        Note: class_id field temporarily stores batch_idx for batch processing
    """
    batch_size = cov_output.shape[0]
    num_classes = cov_output.shape[1]
    grid_h, grid_w = cov_output.shape[2], cov_output.shape[3]

    detections = []
    for b in range(batch_size):
        for c in range(num_classes):
            cov = cov_output[b, c]
            mask = cov >= confidence_threshold
            grid_y, grid_x = np.where(mask)

            for gy, gx in zip(grid_y, grid_x):
                conf = float(cov[gy, gx])
                bbox_offset = c * 4
                dy = float(bbox_output[b, bbox_offset + 0, gy, gx])
                dx = float(bbox_output[b, bbox_offset + 1, gy, gx])
                dh = float(bbox_output[b, bbox_offset + 2, gy, gx])
                dw = float(bbox_output[b, bbox_offset + 3, gy, gx])

                cx = (gx + 0.5) / grid_w
                cy = (gy + 0.5) / grid_h
                bx = cx + dx / input_w
                by = cy + dy / input_h
                bw = dw / input_w
                bh = dh / input_h

                x1 = max(0, bx - bw / 2)
                y1 = max(0, by - bh / 2)
                x2 = min(1, bx + bw / 2)
                y2 = min(1, by + bh / 2)

                w = max(0, x2 - x1)
                h = max(0, y2 - y1)

                if w > 0 and h > 0:
                    # Store batch_idx in class_id field for grouping in batch postprocessing
                    detections.append(Detection(bbox=[x1, y1, w, h], confidence=conf, class_id=b))

    return detections


def apply_nms(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """Apply Non-Maximum Suppression to detections."""
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
            det_j = sorted_dets[j]
            iou = compute_iou(det_i.bbox, det_j.bbox)
            if iou > iou_threshold:
                suppressed.add(j)

    return kept


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes in [x, y, w, h] format (normalized)."""
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1

    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
