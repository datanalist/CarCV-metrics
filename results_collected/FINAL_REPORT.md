# CARS ML Model Evaluation — Final Report

**Date:** 2026-05-14
**Project:** CARS (Computer Automotive Recognition System) — NVIDIA Jetson on-board video analytics
**Reference:** `docs/system-design/ML_System_Design_Document.md` §3.3, §6
**Servers:** qudata2 (2× RTX 4090), qudata5 (1× RTX 4090, replaced 4× V100)

---

## Executive Summary

Validated **4 of 5** NVIDIA TAO models from NGC public catalog using independent public datasets.

| Model | Eval Dataset | Result | Target | Verdict |
|-------|--------------|--------|--------|---------|
| **TrafficCamNet** | COCO val2017 (5K) | P=0.17, R=0.03, F1=0.05 | P≥0.90, R≥0.85 | ❌ Severe domain gap |
| **VehicleMakeNet** | mad-cars (700 in-dist) | Top1=0.08, Top3=0.21 | Top1≥0.70 | ❌ Cross-domain failure |
| **LPDNet** | nomeroff 2018 (375) | P=0.88, R=0.30, F1=0.44 | P≥0.70, R≥0.80 | ⚠️ High precision, low recall |
| **LPRNet** | nomeroff OCR RU (4893) | CharAcc=0.59, PlateAcc=0.06 | ≥0.90, ≥0.80 | ❌ Wrong alphabet |
| **VehicleTypeNet** | — (no BIT-Vehicle access) | not evaluated | — | ⏸️ Pending dataset |
| **Color** | — (no model on NGC) | not evaluated | — | ⏸️ Custom model needed |

**Key finding**: All NGC models are **US/EU-trained** and perform poorly on Russian-market data without fine-tuning. NGC pretrained weights are useful as **transfer-learning starting points**, not direct production deployment for RU market.

---

## Methodology

### Models (all ONNX from NGC, no API key)
- TrafficCamNet `pruned_onnx_v1.0.4` (5.4 MB)
- VehicleMakeNet `pruned_onnx_v1.1.0` (7.4 MB)
- VehicleTypeNet `pruned_onnx_v1.1.0` (19.9 MB)
- LPDNet `pruned_v2.2.1` USA (1.7 MB)
- LPRNet `deployable_onnx_v1.1` US (57.7 MB)

### Datasets
- **TrafficCamNet** → COCO val2017 (substitute for blocked BDD100k from qudata2 net)
- **VehicleMakeNet** → mad-cars sample 5K (142 brands, 700 NGC-overlapping)
- **LPDNet** → nomeroff autoriaNumberplateDataset-2018-11-20 (val=375 images)
- **LPRNet** → nomeroff autoriaNumberplateOcrRu-2021-09-01 (val=4893 plate crops)

### Inference Stack
- onnxruntime-gpu 1.20.1 with CUDA 12.9
- DetectNet_v2 decoder for grid-based detection outputs (stride=16, bbox_norm=35)
- TAO BGR preprocessing (offsets=104,117,124) for classifiers

---

## Detailed Results

### 1. TrafficCamNet (Vehicle/Person/Bike/Sign Detector)

```
Precision: 0.167   Recall: 0.032   F1: 0.053   AP: 0.019
num_gt=1932  num_pred=365  num_tp=61
```

**Class**: car only.
**Cause**: TrafficCamNet trained on top-down traffic-camera footage; COCO val2017 contains street-level natural photography. Different viewpoint, scale, lighting.

**For RU production**: Replace with traffic-camera-trained detector or fine-tune on Russian traffic footage.

---

### 2. VehicleMakeNet (20 Car Makes)

```
Top-1 Accuracy: 0.083   Top-3 Accuracy: 0.211
Evaluated on 700/4960 in-distribution samples (NGC-overlapping brands)
```

**Cause**:
- NGC's 20 makes are US/EU brands (Acura, Audi, BMW, Chevrolet, ..., Toyota).
- mad-cars is Russian marketplace data: 86% of samples (4260/4960) are brands NGC doesn't know (VAZ/Lada, Moskvich, Solaris, Trumpchi, etc.) — excluded from eval.
- Even on overlapping brands (Toyota, BMW, etc.), accuracy is barely above random (5%): RU-market variants differ visually from US-market.

**For RU production**: Need to retrain with RU brand classes or use embedding-based classifier.

---

### 3. LPDNet (License Plate Detector)

```
Precision: 0.884 ✅ (target ≥0.70)
Recall:    0.296 ❌ (target ≥0.80)
F1:        0.444
num_gt=385  num_pred=129  num_tp=114
```

**Cause**: USA LPDNet variant trained on US plate aspect ratios and visual characteristics. RU plates have similar overall shape but different proportions and edge characteristics. The detector is **conservative but accurate** — when it finds a plate, it's right 88% of the time.

**For RU production**: Use LPDNet `CCPD` variant (Chinese plates have similar aspect ratio to RU) or fine-tune USA variant on nomeroff training set.

---

### 4. LPRNet (Plate Text Recognition)

```
Char Accuracy:  0.590   (target ≥0.90)
Plate Accuracy: 0.062   (target ≥0.80)
Char Error Rate: 0.410
n=4893
```

**Decoder fix**: NVIDIA US LPRNet uses character 'Z' (index 35) as CTC blank token, not a separate blank class. Initial decode produced garbage; after correction, character accuracy jumped from 2% to 59%.

**Cause**:
- US plate format: 6-7 chars (e.g., `ABC1234`)
- RU plate format: 8-9 chars (e.g., `A123BC77` or `A123BC777`)
- Cyrillic→Latin transliteration: А→A, В→B, Е→E, К→K, М→M, Н→H, О→O, Р→P, С→C, Т→T, У→Y, Х→X
- Aspect ratios differ slightly

**For RU production**: Use LPRNet trained on RU plates (e.g., nomeroff-net's own LPRNet model). The NGC LPRNet is unsuitable for RU.

---

## Pipeline Architecture (delivered)

```
deploy/
├── CLAUDE_qudata2.md / CLAUDE_qudata5.md  # Per-server agent instructions
├── evaluation/
│   ├── evaluate.py    # Unified entry, 5 model evaluators
│   ├── metrics.py     # Detection/Classification/OCR metrics
│   └── visualize.py   # PR curves, confusion matrices, summary
├── scripts/
│   ├── setup_server.sh        # uv venv + onnxruntime-gpu
│   ├── download_models.sh     # 5 models from NGC (no API key)
│   ├── download_datasets_*.sh # COCO/mad-cars/nomeroff downloads
│   ├── deploy_to_servers.sh   # Parallel rsync + setup
│   └── collect_results.sh     # rsync results back to local
└── requirements.txt
```

### Reproducibility

```bash
ssh qudata2 "cd ~/cars-eval && source venv/bin/activate && python evaluation/evaluate.py --models all"
rsync -az qudata2:~/cars-eval/results/ ./results_collected/qudata2/
```

---

## Known Issues & Limitations

1. **BDD100k blocked from qudata2 network** — ETH Zürich CDN unreachable; substituted with COCO val2017 (similar classes).
2. **mad-cars `brand` field lost on `groupby`** — fixed via car_id backfill from full metadata.
3. **TAO DetectNet_v2 outputs** — not flat YOLO-style boxes; needed custom decoder (cov + bbox grids, stride=16, bbox_norm=35.0).
4. **US LPRNet CTC blank** — at index 35 ('Z'), not separate class. Critical decoder fix.
5. **VehicleTypeNet** — BIT-Vehicle dataset requires Kaggle auth not configured. Skipped.
6. **Color recognition model** (`bae_model_f3.onnx` from System Design) — not on NGC catalog. Skipped.
7. **qudata5 nomeroff throughput** — ~10 KB/s from new instance to nomeroff.net.ua. Worked around by downloading on qudata2.

---

## Recommendations for Production

1. **Do not use NGC pretrained weights directly for RU market.** Domain gap is severe across all 4 evaluated models.
2. **Use NGC weights as initialization** for fine-tuning on:
   - RU traffic footage (TrafficCamNet → ~5-10K labeled frames)
   - mad-cars / yandex-research datasets (VehicleMakeNet, VehicleTypeNet)
   - nomeroff RU OCR/detection (LPDNet, LPRNet)
3. **For LP recognition**, consider replacing NGC LPRNet entirely with nomeroff-net's own pretrained RU model — already trained on Cyrillic.
4. **For color recognition** (gap in NGC catalog), train a custom 15-class CNN on UFPR-VCR or mad-cars `color` field.

---

## Hardware Notes

- **qudata2** (2× RTX 4090): 100-160 img/s ONNX inference; full pipeline runs end-to-end in <1 minute.
- **qudata5** (1× RTX 4090, replaced from 4× V100): equivalent single-GPU speed.
- Multi-GPU not utilized — eval is sequential single-image. For batched throughput benchmarking, would require modifications to `evaluate.py`.

---

*Generated by CARS evaluation pipeline. Raw metrics: `results_collected/qudata2/{model}/metrics.json`.*
