# Metrics & Visualization Reference

## Table of Contents

1. [Metrics by Task Type](#metrics-by-task-type)
2. [Target Thresholds](#target-thresholds)
3. [Visualization Patterns](#visualization-patterns)

---

## Metrics by Task Type

### Detection (TrafficCamNet, LPDNet, FaceDetect)

Compute:
- **mAP@0.5**, **mAP@0.5:0.95** — primary quality metric
- **Precision**, **Recall**, **F1** at best confidence threshold
- **Mean IoU** for matched detections
- **Inference time** (ms), **FPS**

IoU matching: prediction matched to GT if IoU ≥ threshold. One GT box matches at most one prediction (highest IoU first).

```python
from collections import defaultdict
import numpy as np

def compute_ap(recalls, precisions):
    """11-point interpolated AP."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        precs_above = [p for r, p in zip(recalls, precisions) if r >= t]
        ap += max(precs_above) / 11 if precs_above else 0
    return ap

def match_detections(preds, gts, iou_thresh=0.5):
    """Match predictions to ground truth boxes. Returns TP/FP labels."""
    matched_gt = set()
    tp = np.zeros(len(preds))
    for i, pred in enumerate(preds):
        best_iou, best_j = 0, -1
        for j, gt in enumerate(gts):
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou and j not in matched_gt:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            tp[i] = 1
            matched_gt.add(best_j)
    return tp
```

### Classification (VehicleMakeNet, VehicleTypeNet)

Compute:
- **Top-1 Accuracy**, **Top-3 Accuracy**
- **Per-class Precision, Recall, F1** (macro + weighted)
- **Confusion Matrix** (raw + normalized)
- **Inference time** (ms)

```python
from sklearn.metrics import (
    classification_report, confusion_matrix, top_k_accuracy_score
)

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
top3 = top_k_accuracy_score(y_true, y_probs, k=3)
```

### OCR (LPR_STN_PRE_POST)

Compute:
- **Character Accuracy** — % correctly recognized characters
- **Full Plate Accuracy** — % fully correct plates
- **Character Error Rate (CER)** — edit_distance / gt_length
- **Per-character error analysis** — confusion between specific chars

```python
def char_accuracy(pred: str, gt: str) -> float:
    correct = sum(p == g for p, g in zip(pred, gt))
    return correct / max(len(gt), 1)

def cer(pred: str, gt: str) -> float:
    import Levenshtein
    return Levenshtein.distance(pred, gt) / max(len(gt), 1)

def full_plate_accuracy(preds: list[str], gts: list[str]) -> float:
    return sum(p == g for p, g in zip(preds, gts)) / len(gts)
```

Russian plate alphabet: `0123456789ABEKMHOPCTYX` (digits + 12 Cyrillic letters that look like Latin).

### Color Recognition (bae_model_f3)

Compute:
- **Overall Accuracy**
- **Per-class Accuracy**
- **Confusion Matrix**
- **Challenging classes analysis** (beige/tan/gold/silver often confused)

---

## Target Thresholds

| Task | Metric | Threshold | Status format |
|------|--------|-----------|---------------|
| Detection | Precision | >0.90 | PASS / FAIL |
| Detection | Recall | >0.85 | PASS / FAIL |
| Detection | F1 | >0.87 | PASS / FAIL |
| Classification (make) | Top-1 | >0.70 | PASS / FAIL |
| Classification (make) | Top-3 | >0.85 | PASS / FAIL |
| OCR | Char accuracy | >0.90 | PASS / FAIL |
| OCR | Full plate | >0.85 | PASS / FAIL |
| Color | Overall accuracy | >0.75 | PASS / FAIL |

Always compare results against these thresholds and display PASS/FAIL in metrics tables.

---

## Visualization Patterns

All figures: `matplotlib` + `seaborn`, label all axes and titles, `plt.tight_layout()`, save at **150 DPI** to `notebooks/{experiment-name}/`.

### 1. Confusion Matrix

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title("Confusion Matrix (counts)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title("Confusion Matrix (normalized)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

plt.tight_layout()
fig.savefig(f"notebooks/{EXP_NAME}/confusion_matrix.png", dpi=150)
```

### 2. Per-class Metrics (horizontal bars)

```python
fig, ax = plt.subplots(figsize=(10, max(6, len(class_names) * 0.4)))
y = range(len(class_names))
ax.barh(y, f1_scores, color="steelblue")
ax.set_yticks(y); ax.set_yticklabels(class_names)
ax.set_xlabel("F1-Score"); ax.set_title("Per-class F1")
ax.axvline(x=target_threshold, color="red", linestyle="--", label="Target")
ax.legend()
plt.tight_layout()
fig.savefig(f"notebooks/{EXP_NAME}/per_class_f1.png", dpi=150)
```

### 3. Confidence Distribution

```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(conf_correct, bins=50, alpha=0.7, label="Correct", color="green")
ax.hist(conf_incorrect, bins=50, alpha=0.7, label="Incorrect", color="red")
ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
ax.set_title("Confidence Distribution")
ax.legend()
plt.tight_layout()
fig.savefig(f"notebooks/{EXP_NAME}/confidence_dist.png", dpi=150)
```

### 4. Error Examples Grid

```python
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for ax, sample in zip(axes.flat, worst_samples[:12]):
    ax.imshow(sample["image"])
    ax.set_title(f"GT: {sample['gt']}\nPred: {sample['pred']}", fontsize=9, color="red")
    ax.axis("off")
fig.suptitle("Worst Predictions", fontsize=14)
plt.tight_layout()
fig.savefig(f"notebooks/{EXP_NAME}/error_examples.png", dpi=150)
```

### 5. Metrics Summary Bar Chart

```python
metrics = {"Precision": 0.93, "Recall": 0.88, "F1": 0.90}
targets = {"Precision": 0.90, "Recall": 0.85, "F1": 0.87}

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(metrics))
colors = ["green" if v >= targets[k] else "red" for k, v in metrics.items()]
ax.bar(x, metrics.values(), color=colors, alpha=0.8)
for i, (k, t) in enumerate(targets.items()):
    ax.hlines(t, i - 0.4, i + 0.4, colors="black", linestyles="--")
ax.set_xticks(x); ax.set_xticklabels(metrics.keys())
ax.set_ylim(0, 1.05); ax.set_title("Metrics vs Targets")
plt.tight_layout()
fig.savefig(f"notebooks/{EXP_NAME}/metrics_summary.png", dpi=150)
```

### Detection-specific Visualizations

Additionally for detection tasks:
- **AP by IoU threshold curve**: plot AP values for IoU thresholds 0.5 to 0.95
- **TP/FP/FN distribution**: stacked bar chart
- **Predicted vs GT bboxes**: overlay on sample images (green=GT, red=pred)

```python
# Predicted vs GT bboxes
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for ax, sample in zip(axes.flat, samples[:6]):
    ax.imshow(sample["image"])
    for gt in sample["gt_boxes"]:
        rect = plt.Rectangle((gt[0], gt[1]), gt[2]-gt[0], gt[3]-gt[1],
                              fill=False, edgecolor="green", linewidth=2)
        ax.add_patch(rect)
    for pred in sample["pred_boxes"]:
        rect = plt.Rectangle((pred[0], pred[1]), pred[2]-pred[0], pred[3]-pred[1],
                              fill=False, edgecolor="red", linewidth=2, linestyle="--")
        ax.add_patch(rect)
    ax.axis("off")
fig.suptitle("Green=GT, Red=Predicted", fontsize=14)
plt.tight_layout()
fig.savefig(f"notebooks/{EXP_NAME}/detection_samples.png", dpi=150)
```
