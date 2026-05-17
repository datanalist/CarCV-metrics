"""
CARS Model Evaluation — main entry point.
Evaluates ONNX models against validation datasets.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from tqdm import tqdm

from metrics import (
    DetectionMetrics, ClassificationMetrics, OCRMetrics,
    compute_detection_metrics, compute_classification_metrics, compute_ocr_metrics,
    check_thresholds,
)
from visualize import (
    plot_confusion_matrix, plot_per_class_accuracy, generate_summary_md,
)

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"eval_{os.uname().nodename}.log"),
    ],
)
log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_ort_session(model_path: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    log.info(f"Loaded {Path(model_path).name} on {sess.get_providers()[0]}")
    return sess


def preprocess_classification(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    img = cv2.resize(img_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[None]  # NCHW


# TAO classification preprocessing (DeepStream nvinfer convention):
# net-scale-factor=1.0, BGR input, mean offsets per channel
def preprocess_tao_bgr(img_bgr: np.ndarray, size: int = 224,
                      offsets=(103.939, 116.779, 123.68)) -> np.ndarray:
    """TAO TF1 classification preprocessing: BGR, no scaling, mean subtract.
    Default offsets are ImageNet BGR means (used when image_mean not set in TAO).
    VehicleMakeNet/TypeNet use offsets=(104, 117, 124) (B;G;R).
    """
    img = cv2.resize(img_bgr, (size, size)).astype(np.float32)
    # cv2 reads as BGR; subtract per-channel offsets (B, G, R)
    img[..., 0] -= offsets[0]
    img[..., 1] -= offsets[1]
    img[..., 2] -= offsets[2]
    return img.transpose(2, 0, 1)[None]  # NCHW


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load_labels(path) -> list:
    """Parse labels file. Supports both newline- and semicolon-separated formats."""
    text = open(path).read().strip()
    if ";" in text and len(text.splitlines()) <= 2:
        return [l.strip() for l in text.split(";") if l.strip()]
    return [l.strip() for l in text.splitlines() if l.strip()]


def nms_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def nms(boxes, iou_thr=0.5):
    """boxes: list of [x1,y1,x2,y2,conf]"""
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if nms_iou(best[:4], b[:4]) < iou_thr]
    return keep


def detectnet_v2_decode(cov, bbox, target_cls, conf_thr=0.4,
                        stride=16, bbox_norm=35.0,
                        img_w=960, img_h=544,
                        scale_w=1.0, scale_h=1.0):
    """Decode DetectNet_v2 output (TrafficCamNet/LPDNet).

    cov: [num_classes, gh, gw] coverage (sigmoid)
    bbox: [num_classes*4, gh, gw] bbox offsets (x1,y1,x2,y2 per class)

    Returns list of [x1, y1, x2, y2, conf] in original image coords.
    """
    num_classes = cov.shape[0]
    gh, gw = cov.shape[1], cov.shape[2]
    boxes = []
    cov_cls = cov[target_cls]  # [gh, gw]
    bbox_cls = bbox[target_cls * 4:(target_cls + 1) * 4]  # [4, gh, gw]

    yy, xx = np.where(cov_cls > conf_thr)
    for i, j in zip(yy, xx):
        # Cell center in input image coords
        cx = (j + 0.5) * stride
        cy = (i + 0.5) * stride
        # bbox values are offsets from cell center, normalized
        dx1, dy1, dx2, dy2 = bbox_cls[:, i, j] * bbox_norm
        x1 = cx - dx1
        y1 = cy - dy1
        x2 = cx + dx2
        y2 = cy + dy2
        # Scale to original image coords
        x1 *= scale_w; x2 *= scale_w
        y1 *= scale_h; y2 *= scale_h
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2, float(cov_cls[i, j])])

    return nms(boxes, iou_thr=0.5)


# ─── TrafficCamNet ────────────────────────────────────────────────────────────

# Goal 4: TrafficCamNet — 4-class native labels (was: car-only on COCO).
# Maps TrafficCamNet label names → set of dataset categories that count as TP.
# BDD100K and COCO/BDD-compatible labels use different vocabularies; merge both.
TRAFFICCAMNET_GT_VOCAB = {
    "car": {"car"},
    "person": {"person", "pedestrian"},
    "bicycle": {"bike", "bicycle", "rider"},
    "road_sign": {"traffic sign", "sign", "traffic_sign", "trafficsign"},
}


def eval_trafficcamnet(cfg: dict) -> dict:
    model_path = ROOT / cfg["model_path"]
    data_dir = ROOT / cfg["data_dir"]
    labels_path = ROOT / cfg["labels_path"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.error(f"TrafficCamNet model not found: {model_path}")
        return {"error": "model not found"}

    labels = load_labels(labels_path)
    labels_lower = [l.strip().lower() for l in labels]
    # conf_thr поднимается из cfg: для cross-domain (COCO street-level) — 0.2;
    # для in-domain (BDD100K traffic-cam, ближе к training-условиям) — 0.4.
    conf_thr = float(cfg.get("conf_thr", 0.2))
    # Class subset (по умолчанию все 4 TrafficCamNet класса; можно ограничить через cfg)
    eval_classes_cfg = cfg.get("eval_classes")
    eval_classes = (
        [c.lower() for c in eval_classes_cfg]
        if eval_classes_cfg
        else [l for l in labels_lower if l in TRAFFICCAMNET_GT_VOCAB]
    )

    sess = get_ort_session(str(model_path))
    input_name = sess.get_inputs()[0].name
    H, W = 544, 960

    meta_path = data_dir / "labels.json"
    if not meta_path.exists():
        log.error(f"detection labels not found: {meta_path}")
        return {"error": "dataset not found"}

    with open(meta_path) as f:
        all_meta = json.load(f)

    # Multi-class collection — отдельные predictions/ground_truths на каждый класс
    per_class_pred = {c: [] for c in eval_classes}
    per_class_gt = {c: [] for c in eval_classes}
    images_dir = data_dir / "images"

    for item in tqdm(all_meta, desc="TrafficCamNet", unit="img"):
        img_path = images_dir / item["file_name"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        inp = cv2.resize(img, (W, H)).astype(np.float32)
        inp = inp[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0  # BGR→RGB NCHW

        outputs = sess.run(None, {input_name: inp})
        # DetectNet_v2: outputs are [cov: B,C,gh,gw] and [bbox: B,4C,gh,gw]
        cov = outputs[0][0]   # [C, gh, gw]
        bbox = outputs[1][0]  # [4C, gh, gw]

        for cls_name in eval_classes:
            cls_idx = labels_lower.index(cls_name)
            boxes_pred = detectnet_v2_decode(
                cov, bbox, target_cls=cls_idx, conf_thr=conf_thr,
                stride=16, bbox_norm=35.0, img_w=W, img_h=H,
                scale_w=orig_w / W, scale_h=orig_h / H,
            )
            per_class_pred[cls_name].append(
                {"image_id": item["image_id"], "boxes": boxes_pred}
            )

            gt_vocab = TRAFFICCAMNET_GT_VOCAB.get(cls_name, {cls_name})
            gt_boxes = []
            for det in item.get("detections", []):
                if det.get("category", "").lower() not in gt_vocab:
                    continue
                b = det.get("bbox2d", det.get("box2d", {}))
                if b:
                    gt_boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
            per_class_gt[cls_name].append(
                {"image_id": item["image_id"], "boxes": gt_boxes}
            )

    per_class_metrics = {}
    for cls_name in eval_classes:
        m_cls = compute_detection_metrics(
            per_class_pred[cls_name], per_class_gt[cls_name],
            conf_threshold=conf_thr,
        )
        per_class_metrics[cls_name] = m_cls.to_dict()

    # Aggregate (macro) — по классам с num_gt > 0
    eligible = [v for v in per_class_metrics.values() if v["num_gt"] > 0]
    if eligible:
        macro = {
            "precision": sum(v["precision"] for v in eligible) / len(eligible),
            "recall": sum(v["recall"] for v in eligible) / len(eligible),
            "f1": sum(v["f1"] for v in eligible) / len(eligible),
            "ap": sum(v["ap"] for v in eligible) / len(eligible),
            "num_classes_evaluated": len(eligible),
        }
    else:
        macro = {"precision": 0, "recall": 0, "f1": 0, "ap": 0,
                 "num_classes_evaluated": 0}

    # Car metrics остаются как primary (для backward-compat sparkline в SUMMARY)
    car_m = per_class_metrics.get("car", {})
    primary = {
        "precision": car_m.get("precision", 0.0),
        "recall": car_m.get("recall", 0.0),
        "f1": car_m.get("f1", 0.0),
        "ap": car_m.get("ap", 0.0),
        "map50": car_m.get("map50", 0.0),
        "num_gt": car_m.get("num_gt", 0),
        "num_pred": car_m.get("num_pred", 0),
        "num_tp": car_m.get("num_tp", 0),
    }
    primary["macro"] = macro
    primary["per_class"] = per_class_metrics
    primary["conf_thr"] = conf_thr

    thresholds = {"precision": 0.90, "recall": 0.85, "f1": 0.87}
    status = check_thresholds(primary, thresholds)

    result = {"metrics": primary, "thresholds": status}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info(
        f"TrafficCamNet (car): P={primary['precision']:.3f} R={primary['recall']:.3f} "
        f"F1={primary['f1']:.3f} AP={primary['ap']:.3f} | macro F1={macro['f1']:.3f} "
        f"over {macro['num_classes_evaluated']} classes (conf_thr={conf_thr})"
    )
    return result


# ─── VehicleMakeNet ───────────────────────────────────────────────────────────

# Mapping: NGC 20 makes → normalized names matching mad-cars brands
NGC_MAKES = [
    "Acura", "Audi", "BMW", "Chevrolet", "Chrysler", "Dodge", "Ford", "GMC",
    "Honda", "Hyundai", "Infiniti", "Jeep", "Kia", "Lexus", "Mazda",
    "Mercedes", "Nissan", "Subaru", "Toyota", "Volkswagen",
]
NGC_MAKES_LOWER = [m.lower() for m in NGC_MAKES]


def normalize_brand(brand: str) -> str:
    # Substring match against NGC 20 makes. Empty/short inputs must NOT match —
    # previously `"" in "acura"` silently routed missing brands to "acura",
    # collapsing the entire eval to one class (see Goal 4 deferred-work.md).
    b = brand.lower().strip()
    if len(b) < 3:
        return b
    for ngc in NGC_MAKES_LOWER:
        if ngc == b:
            return ngc
    for ngc in NGC_MAKES_LOWER:
        if ngc in b or b in ngc:
            return ngc
    return b


def eval_vehiclemakenet(cfg: dict) -> dict:
    model_path = ROOT / cfg["model_path"]
    labels_path = ROOT / cfg["labels_path"]
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.error(f"VehicleMakeNet model not found: {model_path}")
        return {"error": "model not found"}

    labels = load_labels(labels_path)
    labels_norm = [l.strip().lower() for l in labels]

    meta_path = data_dir / "sample_5k.json"
    if not meta_path.exists():
        log.error(f"mad-cars metadata not found: {meta_path}")
        return {"error": "dataset not found"}

    with open(meta_path) as f:
        sample = json.load(f)

    sess = get_ort_session(str(model_path))
    input_name = sess.get_inputs()[0].name
    images_dir = data_dir / "images"

    predictions, ground_truths = [], []
    skipped_oo_dist = 0
    for item in tqdm(sample, desc="VehicleMakeNet", unit="img"):
        img_path = images_dir / item["file_name"]
        if not img_path.exists():
            continue
        gt_label = normalize_brand(item["brand"])
        # Skip out-of-distribution samples (brand not in NGC's 20 makes)
        if gt_label not in NGC_MAKES_LOWER:
            skipped_oo_dist += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # VehicleMakeNet: BGR + offsets=(104, 117, 124)
        inp = preprocess_tao_bgr(img, size=224, offsets=(104.0, 117.0, 124.0))
        out = sess.run(None, {input_name: inp})[0][0]
        probs = softmax(out)
        top_k = [labels[i] for i in probs.argsort()[::-1][:3]]

        predictions.append({"image_id": item["file_name"], "top_k": [t.lower() for t in top_k]})
        ground_truths.append({"image_id": item["file_name"], "label": gt_label})

    log.info(f"VehicleMakeNet: skipped {skipped_oo_dist} out-of-distribution samples (not in 20 NGC makes)")

    m = compute_classification_metrics(predictions, ground_truths)
    thresholds = {"top1_accuracy": 0.70, "top3_accuracy": 0.85}
    status = check_thresholds(m.to_dict(), thresholds)

    result = {"metrics": m.to_dict(), "thresholds": status}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    import csv
    with open(results_dir / "per_class_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "accuracy"])
        for cls, acc in sorted(m.per_class_accuracy.items()):
            w.writerow([cls, acc])

    if m.per_class_accuracy:
        plot_per_class_accuracy(
            m.per_class_accuracy, "VehicleMakeNet", 0.70,
            str(ROOT / "plots" / "vehiclemakenet_per_class.png"),
        )

    log.info(f"VehicleMakeNet: Top1={m.top1_accuracy:.3f} Top3={m.top3_accuracy:.3f}")
    return result


# ─── VehicleTypeNet ───────────────────────────────────────────────────────────

# Goal 4.3: BIT-Vehicle (Kaggle-blocked) → Stanford Cars 8K test + class-name →
# VehicleTypeNet 6 classes mapping. Stanford classes are formatted as
# "<Make> <Model> <BodyType> <Year>" — body type ищется по ключевым словам.
# §7.5 research-datasets-validation.md описывает требование mapping table.

# Keyword priority (longer/more specific first to avoid e.g. "Cab" matching inside other words)
STANFORD_BODY_KEYWORDS = [
    ("convertible", "coupe"),
    ("hatchback", "sedan"),
    ("minivan", "van"),
    ("crew cab", "truck"),
    ("extended cab", "truck"),
    ("regular cab", "truck"),
    ("cargo van", "van"),
    ("supercab", "truck"),
    ("wagon", "sedan"),
    ("sedan", "sedan"),
    ("coupe", "coupe"),
    ("suv", "suv"),
    ("hummer", "suv"),       # AM General Hummer SUV
    ("van", "van"),
    ("cab", "truck"),
]


def derive_typenet_label(stanford_class: str) -> str:
    """Stanford "Acura RL Sedan 2012" → "sedan". Returns "" if no keyword matches."""
    s = stanford_class.lower()
    for kw, typenet in STANFORD_BODY_KEYWORDS:
        if kw in s:
            return typenet
    return ""


def eval_vehicletypenet(cfg: dict) -> dict:
    model_path = ROOT / cfg["model_path"]
    labels_path = ROOT / cfg["labels_path"]
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.error(f"VehicleTypeNet model not found: {model_path}")
        return {"error": "model not found"}

    labels = load_labels(labels_path)
    labels_norm = [l.strip().lower() for l in labels]

    sess = get_ort_session(str(model_path))
    input_name = sess.get_inputs()[0].name

    # Stanford Cars (primary, Goal 4.3)
    sc_meta = data_dir / "test.json"
    sc_images = data_dir / "images"
    bit_legacy = sc_images.exists() is False and data_dir.exists()

    images: list[tuple[Path, str]] = []
    mapping_audit: list[dict] = []
    skipped_no_mapping = 0

    if sc_meta.exists() and sc_images.exists():
        log.info(f"VehicleTypeNet: using Stanford Cars test from {data_dir}")
        with open(sc_meta) as f:
            records = json.load(f)
        for rec in records:
            label_str = (rec.get("label") or "").strip()
            typenet_label = derive_typenet_label(label_str)
            if not typenet_label:
                skipped_no_mapping += 1
                continue
            mapping_audit.append({"stanford_class": label_str, "typenet": typenet_label})
            img_path = sc_images / rec["file_name"]
            images.append((img_path, typenet_label))
    elif bit_legacy:
        # Legacy BIT-Vehicle path: directory-per-class
        log.warning("VehicleTypeNet: Stanford Cars not found, falling back to BIT-Vehicle layout")
        BIT_TO_TYPENET = {
            "bus": "largevehicle", "microbus": "largevehicle",
            "minivan": "van", "van": "van",
            "sedan": "sedan", "suv": "suv", "truck": "truck",
        }
        for type_dir in data_dir.iterdir():
            if not type_dir.is_dir():
                continue
            bit_label = type_dir.name.lower()
            typenet_label = BIT_TO_TYPENET.get(bit_label, bit_label)
            for img_path in type_dir.glob("*.jpg"):
                images.append((img_path, typenet_label))
    else:
        log.error(f"VehicleTypeNet: neither Stanford Cars {sc_meta} nor BIT-Vehicle layout found in {data_dir}")
        return {"error": "dataset not found"}

    if not images:
        log.error(f"VehicleTypeNet: no images discovered in {data_dir}")
        return {"error": "no images"}

    log.info(
        f"VehicleTypeNet: {len(images)} images; skipped {skipped_no_mapping} "
        f"(no body keyword in Stanford class name)"
    )

    # Сохраним audit-таблицу для воспроизводимости §7.5
    if mapping_audit:
        audit_path = data_dir / "type_mapping.csv"
        import csv
        seen = {}
        for row in mapping_audit:
            seen[row["stanford_class"]] = row["typenet"]
        with open(audit_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stanford_class", "typenet"])
            for cls, t in sorted(seen.items()):
                w.writerow([cls, t])
        log.info(f"VehicleTypeNet: wrote mapping audit ({len(seen)} unique classes) → {audit_path}")

    predictions, ground_truths = [], []
    confusion = np.zeros((len(labels_norm), len(labels_norm)), dtype=int)

    for img_path, gt_label in tqdm(images, desc="VehicleTypeNet", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # VehicleTypeNet: BGR + offsets=(104, 117, 124)
        inp = preprocess_tao_bgr(img, size=224, offsets=(104.0, 117.0, 124.0))
        out = sess.run(None, {input_name: inp})[0][0]
        probs = softmax(out)
        top_k = [labels_norm[i] for i in probs.argsort()[::-1][:3]]
        predictions.append({"image_id": img_path.name, "top_k": top_k})
        ground_truths.append({"image_id": img_path.name, "label": gt_label})

        if gt_label in labels_norm and top_k[0] in labels_norm:
            gi = labels_norm.index(gt_label)
            pi = labels_norm.index(top_k[0])
            confusion[gi, pi] += 1

    m = compute_classification_metrics(predictions, ground_truths)
    thresholds = {"top1_accuracy": 0.85}
    status = check_thresholds({"top1_accuracy": m.top1_accuracy}, thresholds)

    result = {"metrics": m.to_dict(), "thresholds": status}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    if confusion.sum() > 0:
        plot_confusion_matrix(
            confusion, labels_norm, "VehicleTypeNet",
            str(ROOT / "plots" / "vehicletypenet_confusion.png"),
        )

    log.info(f"VehicleTypeNet: Top1={m.top1_accuracy:.3f}")
    return result


# ─── LPDNet ──────────────────────────────────────────────────────────────────

def eval_lpdnet(cfg: dict) -> dict:
    model_path = ROOT / cfg["model_path"]
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.error(f"LPDNet model not found: {model_path}")
        return {"error": "model not found"}

    sess = get_ort_session(str(model_path))
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    H = input_shape[2] if len(input_shape) == 4 else 480
    W = input_shape[3] if len(input_shape) == 4 else 640

    # Load nomeroff VIA annotations from val/via_region_data.json
    ann_file = next(data_dir.glob("**/val/via_region_data.json"), None)
    if ann_file is None:
        log.error(f"No val/via_region_data.json found in {data_dir}")
        return {"error": "dataset not found"}

    with open(ann_file) as f:
        ann_data = json.load(f)

    # VIA format: images under _via_img_metadata
    img_metadata = ann_data.get("_via_img_metadata", ann_data)

    predictions, ground_truths = [], []
    img_root = ann_file.parent

    for img_key, img_info in tqdm(img_metadata.items(), desc="LPDNet"):
        img_name = img_info.get("filename", img_key)
        img_path = img_root / img_name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        inp = cv2.resize(img, (W, H)).astype(np.float32)
        inp = inp[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0

        # LPDNet is also DetectNet_v2; outputs: [cov: 1,C,gh,gw] [bbox: 1,4C,gh,gw]
        outputs = sess.run(None, {input_name: inp})
        cov = outputs[0][0]
        bbox = outputs[1][0]
        boxes_pred = detectnet_v2_decode(
            cov, bbox, target_cls=0, conf_thr=0.3,
            stride=16, bbox_norm=35.0, img_w=W, img_h=H,
            scale_w=orig_w / W, scale_h=orig_h / H,
        )

        predictions.append({"image_id": img_name, "boxes": boxes_pred})

        gt_boxes = []
        regions = img_info.get("regions", [])
        # VIA: regions can be list (newer format) or dict (older)
        if isinstance(regions, dict):
            regions = list(regions.values())
        for reg in regions:
            shape = reg.get("shape_attributes", {})
            if shape.get("name") == "rect":
                x, y = shape["x"], shape["y"]
                gt_boxes.append([x, y, x + shape["width"], y + shape["height"]])
            elif shape.get("name") == "polygon":
                xs, ys = shape["all_points_x"], shape["all_points_y"]
                gt_boxes.append([min(xs), min(ys), max(xs), max(ys)])
        ground_truths.append({"image_id": img_name, "boxes": gt_boxes})

    m = compute_detection_metrics(predictions, ground_truths, conf_threshold=0.3)
    thresholds = {"recall": 0.80, "precision": 0.70}
    status = check_thresholds(m.to_dict(), thresholds)

    result = {"metrics": m.to_dict(), "thresholds": status}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info(f"LPDNet: P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")
    return result


# ─── Nomeroff LPD (RU plates) ────────────────────────────────────────────────

def eval_nomeroff_lpd(cfg: dict) -> dict:
    """Детекция номеров через nomeroff-net на VIA-датасете nomeroff_lp.
    Замена US-обученного LPDNet для RU-сегмента (см. eval_lpdnet выше).

    Использует pipeline("number_plate_localization"), который возвращает
    2 слота через unzip: (images_bboxs, images). Каждый bbox — список
    [x1, y1, x2, y2, conf, cls_int, kps_normalized] на детекцию.
    """
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        from nomeroff_net import pipeline
        from nomeroff_net.tools import unzip as nomeroff_unzip
    except ImportError as e:
        log.error(f"nomeroff-net not installed on this host: {e}")
        return {"error": "nomeroff-net not installed"}

    ann_file = next(data_dir.glob("**/val/via_region_data.json"), None)
    if ann_file is None:
        log.error(f"No val/via_region_data.json found in {data_dir}")
        return {"error": "dataset not found"}

    with open(ann_file) as f:
        ann_data = json.load(f)
    img_metadata = ann_data.get("_via_img_metadata", ann_data)
    img_root = ann_file.parent

    detector = pipeline("number_plate_localization", image_loader="opencv")

    predictions, ground_truths = [], []
    skipped = 0

    for img_key, img_info in tqdm(img_metadata.items(), desc="Nomeroff LPD"):
        img_name = img_info.get("filename", img_key)
        img_path = img_root / img_name
        if not img_path.exists():
            skipped += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        try:
            # number_plate_localization returns unzip([model_outputs, images]):
            # a list of (bboxs_per_img, img_array) tuples — 2 slots after unzip.
            # Each bbox element: [x1, y1, x2, y2, conf, cls_int, kps_normalized]
            raw = detector([str(img_path)])
            images_bboxs, _images = nomeroff_unzip(raw)
            bboxs = images_bboxs  # list indexed by image; bboxs[0] for first image
            preds_per_img = [[float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                              float(b[4]) if len(b) > 4 else 1.0]
                             for b in ((bboxs[0] or []) if bboxs else [])
                             if b is not None]
        except Exception as e:
            log.warning(f"nomeroff LPD failed on {img_name}: {e}")
            skipped += 1
            continue

        predictions.append({"image_id": img_name, "boxes": preds_per_img})

        gt_boxes = []
        regions = img_info.get("regions", [])
        if isinstance(regions, dict):
            regions = list(regions.values())
        for reg in regions:
            shape = reg.get("shape_attributes", {})
            if shape.get("name") == "rect":
                x, y = shape["x"], shape["y"]
                gt_boxes.append([x, y, x + shape["width"], y + shape["height"]])
            elif shape.get("name") == "polygon":
                xs, ys = shape["all_points_x"], shape["all_points_y"]
                gt_boxes.append([min(xs), min(ys), max(xs), max(ys)])
        ground_truths.append({"image_id": img_name, "boxes": gt_boxes})

    log.info(f"Nomeroff LPD: processed {len(predictions)} images, skipped {skipped}")

    m = compute_detection_metrics(predictions, ground_truths, conf_threshold=0.3)
    thresholds = {"recall": 0.80, "precision": 0.70}
    status = check_thresholds(m.to_dict(), thresholds)

    result = {"metrics": m.to_dict(), "thresholds": status,
              "skipped": skipped, "model": "nomeroff-net localization (RU)"}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info(f"Nomeroff LPD: P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")
    return result


# ─── LPRNet ──────────────────────────────────────────────────────────────────

# US LPRNet character set
US_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Russian → Latin transliteration for shared-shape characters in license plates.
# Russian plates use only these letters: А В Е К М Н О Р С Т У Х (visually similar to Latin).
RU_TO_LATIN = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
}


def normalize_ru_plate(text: str) -> str:
    """Transliterate Russian license plate to Latin for comparison with US LPRNet output."""
    return "".join(RU_TO_LATIN.get(c, c) for c in text.upper())


def ctc_decode(logits: np.ndarray, chars: str) -> str:
    indices = logits.argmax(axis=-1)
    blank = len(chars)
    result = []
    prev = blank
    for idx in indices:
        if idx != blank and idx != prev:
            if idx < len(chars):
                result.append(chars[idx])
        prev = idx
    return "".join(result)


def eval_lprnet(cfg: dict) -> dict:
    model_path = ROOT / cfg["model_path"]
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.error(f"LPRNet model not found: {model_path}")
        return {"error": "model not found"}

    sess = get_ort_session(str(model_path))
    input_name = sess.get_inputs()[0].name
    W, H = 96, 48  # NVIDIA LPRNet input

    # Nomeroff OCR RU structure: val/img/*.png + val/ann/*.json
    val_img_dir = next(data_dir.glob("**/val/img"), None)
    val_ann_dir = next(data_dir.glob("**/val/ann"), None)
    if not val_img_dir or not val_ann_dir:
        log.error(f"Nomeroff OCR RU val/img + val/ann not found in {data_dir}")
        return {"error": "dataset not found"}

    items = []
    for ann_file in val_ann_dir.glob("*.json"):
        with open(ann_file) as f:
            data = json.load(f)
        # Find matching image (png or jpg)
        stem = ann_file.stem
        img_path = next((p for p in val_img_dir.glob(f"{stem}.*")), None)
        if img_path is None:
            continue
        raw_gt = (data.get("description") or data.get("name") or "").upper().strip()
        gt_text = normalize_ru_plate(raw_gt)  # Cyrillic → Latin for comparison
        if gt_text:
            items.append({"img_path": img_path, "text": gt_text})

    log.info(f"LPRNet: loaded {len(items)} samples from nomeroff OCR RU")

    predictions, ground_truths = [], []

    for item in tqdm(items, desc="LPRNet"):
        img_path = item["img_path"]
        gt_text = item["text"]
        if not img_path.exists() or not gt_text:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        inp = cv2.resize(img, (W, H)).astype(np.float32)
        inp = inp[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0

        # NVIDIA LPRNet outputs: [ArgMax indices: N,T] [Max confs: N,T]
        outputs = sess.run(None, {input_name: inp})
        indices = outputs[0][0]  # [T]
        # NVIDIA US LPRNet uses 'Z' (index 35) as CTC blank token
        BLANK_IDX = US_CHARS.index("Z")  # 35
        result = []
        prev = -1
        for idx in indices:
            idx = int(idx)
            if idx != BLANK_IDX and idx != prev and 0 <= idx < len(US_CHARS):
                result.append(US_CHARS[idx])
            prev = idx
        text = "".join(result)

        predictions.append({"image_id": str(img_path.name), "text": text})
        ground_truths.append({"image_id": str(img_path.name), "text": gt_text})

    m = compute_ocr_metrics(predictions, ground_truths)
    thresholds = {"char_accuracy": 0.90, "full_plate_accuracy": 0.80}
    status = check_thresholds(m.to_dict(), thresholds)

    result = {"metrics": m.to_dict(), "thresholds": status,
              "note": "US Latin model evaluated on RU Cyrillic plates — domain gap expected"}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info(f"LPRNet: CharAcc={m.char_accuracy:.3f} PlateAcc={m.full_plate_accuracy:.3f}")
    return result


# ─── Nomeroff OCR (RU) ───────────────────────────────────────────────────────

def eval_nomeroff_ocr(cfg: dict) -> dict:
    """RU OCR через nomeroff-net на готовых кропах из data/nomeroff_ocr_ru.
    Замена US-обученного LPRNet для RU-сегмента (см. eval_lprnet выше).

    Использует NumberPlateTextReading с DumpyImageLoader напрямую — без YOLO
    детекции. number_plate_detection_and_reading запускал бы YOLO на уже
    вырезанных кропах и давал неправильные результаты (несколько псевдо-bbox
    вместо одного полного текста). NumberPlateTextReading принимает кроп как
    numpy-массив и возвращает через unzip 2 слота: (texts, images).

    PyTorch >= 2.6 изменил дефолт weights_only=True в torch.load, что ломает
    загрузку чекпойнтов nomeroff-net (содержат StrLabelConverter). Временная
    обёртка torch.load с weights_only=False применяется только при инициализации.
    """
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        from nomeroff_net.tools import unzip as nomeroff_unzip
        from nomeroff_net.pipelines.number_plate_text_reading import NumberPlateTextReading
        from nomeroff_net.image_loaders import DumpyImageLoader
    except ImportError as e:
        log.error(f"nomeroff-net not installed on this host: {e}")
        return {"error": "nomeroff-net not installed"}

    val_img_dir = next(data_dir.glob("**/val/img"), None)
    val_ann_dir = next(data_dir.glob("**/val/ann"), None)
    if not val_img_dir or not val_ann_dir:
        log.error(f"Nomeroff OCR RU val/img + val/ann not found in {data_dir}")
        return {"error": "dataset not found"}

    items = []
    for ann_file in val_ann_dir.glob("*.json"):
        with open(ann_file) as f:
            data = json.load(f)
        stem = ann_file.stem
        img_path = next((p for p in val_img_dir.glob(f"{stem}.*")), None)
        if img_path is None:
            continue
        gt_text = (data.get("description") or data.get("name") or "").upper().strip()
        if gt_text:
            items.append({"img_path": img_path, "text": gt_text})

    log.info(f"Nomeroff OCR: loaded {len(items)} samples from {data_dir.name}")

    # PyTorch >= 2.6 changed torch.load default to weights_only=True, which breaks
    # nomeroff-net's old-format checkpoint loading (contains StrLabelConverter).
    # Monkey-patch torch.load to pass weights_only=False during pipeline init only.
    # NOTE: not thread-safe — eval_nomeroff_ocr must not run concurrently with
    # other code that calls torch.load.
    import torch as _torch
    _orig_torch_load = _torch.load

    def _torch_load_unsafe(f, *args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(f, *args, **kwargs)

    _torch.load = _torch_load_unsafe
    try:
        # Use NumberPlateTextReading directly to avoid running YOLO on already-cropped
        # plate images. DumpyImageLoader passes numpy arrays through unchanged.
        text_reader = NumberPlateTextReading(
            task="number_plate_text_reading",
            image_loader="opencv",
            presets={"ru": {"for_regions": ["ru"], "for_count_lines": [1],
                            "model_path": "latest"}},
            default_label="ru",
            off_number_plate_classification=True,
        )
        text_reader.image_loader = DumpyImageLoader()
    finally:
        _torch.load = _orig_torch_load

    predictions, ground_truths = [], []
    skipped = 0

    for item in tqdm(items, desc="Nomeroff OCR"):
        img_path = item["img_path"]
        gt_text = item["text"]
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        try:
            # NumberPlateTextReading input: list of (zone_array, region_label,
            # count_lines, preprocessed_np). Returns list of (text, zone_array).
            # After unzip: (texts_tuple, images_tuple), each with one element
            # per input.
            raw = text_reader([(img, "ru", 1, None)])
            texts, _images = nomeroff_unzip(raw)
            pred_text = (texts[0] if texts else "").upper().strip()
        except Exception as e:
            log.warning(f"nomeroff OCR failed on {img_path.name}: {e}")
            skipped += 1
            continue

        predictions.append({"image_id": img_path.name, "text": pred_text})
        ground_truths.append({"image_id": img_path.name, "text": gt_text})

    log.info(f"Nomeroff OCR: processed {len(predictions)} samples, skipped {skipped}")

    m = compute_ocr_metrics(predictions, ground_truths)
    thresholds = {"char_accuracy": 0.90, "full_plate_accuracy": 0.80}
    status = check_thresholds(m.to_dict(), thresholds)

    result = {"metrics": m.to_dict(), "thresholds": status,
              "skipped": skipped, "model": "nomeroff-net NumberPlateTextReading direct (RU)"}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info(f"Nomeroff OCR: CharAcc={m.char_accuracy:.3f} PlateAcc={m.full_plate_accuracy:.3f}")
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

EVAL_CONFIGS = {
    "trafficcamnet": {
        "model_path": "models/trafficcamnet/resnet18_trafficcamnet_pruned.onnx",
        "labels_path": "models/trafficcamnet/labels.txt",
        "data_dir": "data/bdd100k",
        "results_dir": "results/trafficcamnet",
        "eval_fn": eval_trafficcamnet,
        # Goal 4: multi-class evaluation + lower conf_thr под cross-domain (COCO street-level).
        # Когда BDD100K val 10K окажется доступен на ssh9 — можно поднять до 0.4 (in-domain).
        "conf_thr": 0.2,
        "eval_classes": ["car", "person", "bicycle", "road_sign"],
    },
    "vehiclemakenet": {
        "model_path": "models/vehiclemakenet/resnet18_pruned.onnx",
        "labels_path": "models/vehiclemakenet/labels.txt",
        "data_dir": "data/mad_cars",
        "results_dir": "results/vehiclemakenet",
        "eval_fn": eval_vehiclemakenet,
    },
    "vehicletypenet": {
        "model_path": "models/vehicletypenet/resnet18_pruned.onnx",
        "labels_path": "models/vehicletypenet/labels.txt",
        # Goal 4.3: switched from BIT-Vehicle (Kaggle-blocked) to Stanford Cars test.
        # Body type derived from class name via STANFORD_BODY_KEYWORDS (§7.5 mapping).
        "data_dir": "data/stanford_cars",
        "results_dir": "results/vehicletypenet",
        "eval_fn": eval_vehicletypenet,
    },
    "lpdnet": {
        "model_path": "models/lpdnet/LPDNet_usa_pruned_tao5.onnx",
        "data_dir": "data/nomeroff_lp",
        "results_dir": "results/lpdnet",
        "eval_fn": eval_lpdnet,
    },
    "nomeroff_lpd": {
        "data_dir": "data/nomeroff_lp",
        "results_dir": "results/nomeroff_lpd",
        "eval_fn": eval_nomeroff_lpd,
    },
    "lprnet": {
        "model_path": "models/lprnet/us_lprnet_baseline18_deployable.onnx",
        "data_dir": "data/nomeroff_ocr_ru",
        "results_dir": "results/lprnet",
        "eval_fn": eval_lprnet,
    },
    "nomeroff_ocr": {
        "data_dir": "data/nomeroff_ocr_ru",
        "results_dir": "results/nomeroff_ocr",
        "eval_fn": eval_nomeroff_ocr,
    },
}


def check_all():
    ok = True
    for name, cfg in EVAL_CONFIGS.items():
        model = ROOT / cfg["model_path"]
        data = ROOT / cfg["data_dir"]
        m_ok = "✅" if model.exists() else "❌"
        d_ok = "✅" if data.exists() else "❌"
        print(f"{name:20s}  model:{m_ok}  data:{d_ok}")
        if not model.exists() or not data.exists():
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description="CARS Model Evaluation")
    parser.add_argument("--models", nargs="+", choices=list(EVAL_CONFIGS.keys()) + ["all"],
                        default=["all"], help="Models to evaluate")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--plots", default="plots/")
    parser.add_argument("--check", action="store_true", help="Check readiness only")
    parser.add_argument("--summary", action="store_true", help="Generate summary from existing results")
    args = parser.parse_args()

    (ROOT / args.plots).mkdir(parents=True, exist_ok=True)

    if args.check:
        sys.exit(0 if check_all() else 1)

    if args.summary:
        all_results = {}
        for name in EVAL_CONFIGS:
            p = ROOT / "results" / name / "metrics.json"
            if p.exists():
                with open(p) as f:
                    all_results[name] = json.load(f)
        generate_summary_md(all_results, str(ROOT / "results/SUMMARY.md"),
                            os.uname().nodename)
        return

    models = list(EVAL_CONFIGS.keys()) if "all" in args.models else args.models
    all_results = {}

    for name in models:
        cfg = EVAL_CONFIGS[name]
        log.info(f"─── Evaluating {name} ───")
        t0 = time.time()
        try:
            result = cfg["eval_fn"](cfg)
            all_results[name] = result
            log.info(f"{name} done in {time.time()-t0:.1f}s")
        except Exception as e:
            log.exception(f"{name} failed: {e}")
            all_results[name] = {"error": str(e)}

    generate_summary_md(all_results, str(ROOT / "results/SUMMARY.md"),
                        os.uname().nodename)
    log.info("Evaluation complete. Results in results/ and plots/")


if __name__ == "__main__":
    main()
