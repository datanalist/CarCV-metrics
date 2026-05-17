# CARS Model Evaluation — Aggregated Summary

_Generated 2026-05-17 by `deploy/evaluation/aggregate_summary.py` from `results_collected/`._

**Runs aggregated:** 13 across 3 host(s): `qudata2`, `ssh1.qudata.ai`, `ssh9.qudata.ai`.
**Families:** classification, detection, ocr.

## Overall Results

| Model | Family | Host | Metric | Value | Threshold | Status |
|---|---|---|---|---:|---|---|
| `lpdnet` | detection | `qudata2` | precision | 0.8837 | ≥0.7 | ✅ PASS |
| `lpdnet` | detection | `qudata2` | recall | 0.2961 | ≥0.8 | ❌ FAIL |
| `lpdnet` | detection | `qudata2` | f1 | 0.4436 | — | — |
| `lpdnet` | detection | `qudata2` | map50 | 0.2617 | — | — |
| `lprnet` | ocr | `qudata2` | char_accuracy | 0.5903 | ≥0.9 | ❌ FAIL |
| `lprnet` | ocr | `qudata2` | full_plate_accuracy | 0.0621 | ≥0.8 | ❌ FAIL |
| `trafficcamnet` | detection | `qudata2` | precision | 0.1671 | ≥0.9 | ❌ FAIL |
| `trafficcamnet` | detection | `qudata2` | recall | 0.0316 | ≥0.85 | ❌ FAIL |
| `trafficcamnet` | detection | `qudata2` | f1 | 0.0531 | ≥0.87 | ❌ FAIL |
| `trafficcamnet` | detection | `qudata2` | map50 | 0.0192 | — | — |
| `vehiclemakenet` | classification | `qudata2` | top1_accuracy | 0.0829 | ≥0.7 | ❌ FAIL |
| `vehiclemakenet` | classification | `qudata2` | top3_accuracy | 0.2114 | ≥0.85 | ❌ FAIL |
| `trafficcamnet` | detection | `ssh1.qudata.ai` | precision | 0.1671 | ≥0.9 | ❌ FAIL |
| `trafficcamnet` | detection | `ssh1.qudata.ai` | recall | 0.0316 | ≥0.85 | ❌ FAIL |
| `trafficcamnet` | detection | `ssh1.qudata.ai` | f1 | 0.0531 | ≥0.87 | ❌ FAIL |
| `trafficcamnet` | detection | `ssh1.qudata.ai` | map50 | 0.0192 | — | — |
| `vehiclemakenet` | classification | `ssh1.qudata.ai` | top1_accuracy | 0.0020 | ≥0.7 | ❌ FAIL |
| `vehiclemakenet` | classification | `ssh1.qudata.ai` | top3_accuracy | 0.0147 | ≥0.85 | ❌ FAIL |
| `lpdnet` | detection | `ssh9.qudata.ai` | precision | 0.8837 | ≥0.7 | ✅ PASS |
| `lpdnet` | detection | `ssh9.qudata.ai` | recall | 0.2961 | ≥0.8 | ❌ FAIL |
| `lpdnet` | detection | `ssh9.qudata.ai` | f1 | 0.4436 | — | — |
| `lpdnet` | detection | `ssh9.qudata.ai` | map50 | 0.2616 | — | — |
| `lprnet` | ocr | `ssh9.qudata.ai` | char_accuracy | 0.5904 | ≥0.9 | ❌ FAIL |
| `lprnet` | ocr | `ssh9.qudata.ai` | full_plate_accuracy | 0.0621 | ≥0.8 | ❌ FAIL |
| `nomeroff_lpd` | detection | `ssh9.qudata.ai` | precision | 0.9056 | ≥0.7 | ✅ PASS |
| `nomeroff_lpd` | detection | `ssh9.qudata.ai` | recall | 0.9221 | ≥0.8 | ✅ PASS |
| `nomeroff_lpd` | detection | `ssh9.qudata.ai` | f1 | 0.9138 | — | — |
| `nomeroff_lpd` | detection | `ssh9.qudata.ai` | map50 | 0.8806 | — | — |
| `nomeroff_ocr` | ocr | `ssh9.qudata.ai` | char_accuracy | 0.9995 | ≥0.9 | ✅ PASS |
| `nomeroff_ocr` | ocr | `ssh9.qudata.ai` | full_plate_accuracy | 0.9978 | ≥0.8 | ✅ PASS |
| `trafficcamnet` | detection | `ssh9.qudata.ai` | precision | 0.0819 | ≥0.9 | ❌ FAIL |
| `trafficcamnet` | detection | `ssh9.qudata.ai` | recall | 0.0528 | ≥0.85 | ❌ FAIL |
| `trafficcamnet` | detection | `ssh9.qudata.ai` | f1 | 0.0642 | ≥0.87 | ❌ FAIL |
| `trafficcamnet` | detection | `ssh9.qudata.ai` | map50 | 0.0192 | — | — |
| `vehiclemakenet` | classification | `ssh9.qudata.ai` | top1_accuracy | 0.0829 | ≥0.7 | ❌ FAIL |
| `vehiclemakenet` | classification | `ssh9.qudata.ai` | top3_accuracy | 0.2114 | ≥0.85 | ❌ FAIL |
| `vehicletypenet` | classification | `ssh9.qudata.ai` | top1_accuracy | 0.3575 | ≥0.85 | ❌ FAIL |
| `vehicletypenet` | classification | `ssh9.qudata.ai` | top3_accuracy | 0.7009 | — | — |

## US baseline vs Nomeroff RU (where both ran)

| Pipeline | Metric | US baseline | Nomeroff RU | Δ |
|---|---|---:|---:|---:|
| LPD | precision | 0.8837 | **0.9056** | +0.0219 |
| LPD | recall | 0.2961 | **0.9221** | +0.6260 |
| LPD | f1 | 0.4436 | **0.9138** | +0.4702 |
| LPD | map50 | 0.2617 | **0.8806** | +0.6189 |
| OCR | char_accuracy | 0.5903 | **0.9995** | +0.4092 |
| OCR | full_plate_accuracy | 0.0621 | **0.9978** | +0.9357 |

## Detailed Results

### `vehiclemakenet` @ `qudata2`  (classification)

- **top1_accuracy**: 0.0829
- **top3_accuracy**: 0.2114
- **num_samples**: 700
- **per_class_accuracy**: `acura`=0.0000, `audi`=0.0000, `bmw`=0.0571, `chevrolet`=0.1143, `chrysler`=0.0000, `dodge`=0.0286, `ford`=0.1429, `gmc`=0.1143, `honda`=0.1143, `hyundai`=0.1714, `infiniti`=0.0000, `jeep`=0.1714, `kia`=0.0286, `lexus`=0.0571, `mazda`=0.1143, `mercedes`=0.0286, `nissan`=0.0857, `subaru`=0.0000, `toyota`=0.2857, `volkswagen`=0.1429
- _source_: `results_collected/qudata2/vehiclemakenet/metrics.json`

### `lpdnet` @ `qudata2`  (detection)

- **precision**: 0.8837
- **recall**: 0.2961
- **f1**: 0.4436
- **ap**: 0.2617
- **map50**: 0.2617
- **num_gt**: 385
- **num_pred**: 129
- **num_tp**: 114
- _source_: `results_collected/qudata2/lpdnet/metrics.json`

### `trafficcamnet` @ `qudata2`  (detection)

- **precision**: 0.1671
- **recall**: 0.0316
- **f1**: 0.0531
- **ap**: 0.0192
- **map50**: 0.0192
- **num_gt**: 1932
- **num_pred**: 365
- **num_tp**: 61
- _source_: `results_collected/qudata2/trafficcamnet/metrics.json`

### `lprnet` @ `qudata2`  (ocr)

- **char_accuracy**: 0.5903
- **full_plate_accuracy**: 0.0621
- **char_error_rate**: 0.4097
- **num_samples**: 4893
- _note_: US Latin model evaluated on RU Cyrillic plates — domain gap expected
- _source_: `results_collected/qudata2/lprnet/metrics.json`

### `vehiclemakenet` @ `ssh1.qudata.ai`  (classification)

- **top1_accuracy**: 0.0020
- **top3_accuracy**: 0.0147
- **num_samples**: 4960
- **per_class_accuracy**: `acura`=0.0020
- _source_: `results_collected/ssh1.qudata.ai/results/vehiclemakenet/metrics.json`

### `trafficcamnet` @ `ssh1.qudata.ai`  (detection)

- **precision**: 0.1671
- **recall**: 0.0316
- **f1**: 0.0531
- **ap**: 0.0192
- **map50**: 0.0192
- **num_gt**: 1932
- **num_pred**: 365
- **num_tp**: 61
- _source_: `results_collected/ssh1.qudata.ai/results/trafficcamnet/metrics.json`

### `vehiclemakenet` @ `ssh9.qudata.ai`  (classification)

- **top1_accuracy**: 0.0829
- **top3_accuracy**: 0.2114
- **num_samples**: 700
- **per_class_accuracy**: `acura`=0.0000, `audi`=0.0000, `bmw`=0.0571, `chevrolet`=0.1143, `chrysler`=0.0000, `dodge`=0.0286, `ford`=0.1429, `gmc`=0.1143, `honda`=0.1143, `hyundai`=0.1714, `infiniti`=0.0000, `jeep`=0.1714, `kia`=0.0286, `lexus`=0.0571, `mazda`=0.1143, `mercedes`=0.0286, `nissan`=0.0857, `subaru`=0.0000, `toyota`=0.2857, `volkswagen`=0.1429
- _source_: `results_collected/ssh9.qudata.ai/results/vehiclemakenet/metrics.json`

### `vehicletypenet` @ `ssh9.qudata.ai`  (classification)

- **top1_accuracy**: 0.3575
- **top3_accuracy**: 0.7009
- **num_samples**: 7483
- **per_class_accuracy**: `coupe`=0.3160, `sedan`=0.2894, `suv`=0.3928, `truck`=0.6361, `van`=0.3815
- _source_: `results_collected/ssh9.qudata.ai/results/vehicletypenet/metrics.json`

### `lpdnet` @ `ssh9.qudata.ai`  (detection)

- **precision**: 0.8837
- **recall**: 0.2961
- **f1**: 0.4436
- **ap**: 0.2616
- **map50**: 0.2616
- **num_gt**: 385
- **num_pred**: 129
- **num_tp**: 114
- _source_: `results_collected/ssh9.qudata.ai/results/lpdnet/metrics.json`

### `nomeroff_lpd` @ `ssh9.qudata.ai`  (detection)

- **precision**: 0.9056
- **recall**: 0.9221
- **f1**: 0.9138
- **ap**: 0.8806
- **map50**: 0.8806
- **num_gt**: 385
- **num_pred**: 392
- **num_tp**: 355
- _note_: nomeroff-net localization (RU)
- _source_: `results_collected/ssh9.qudata.ai/results/nomeroff_lpd/metrics.json`

### `trafficcamnet` @ `ssh9.qudata.ai`  (detection)

- **precision**: 0.0819
- **recall**: 0.0528
- **f1**: 0.0642
- **ap**: 0.0192
- **map50**: 0.0192
- **num_gt**: 1932
- **num_pred**: 1245
- **num_tp**: 102
- **macro**: `ap`=0.0585, `f1`=0.0987, `num_classes_evaluated`=4, `precision`=0.1270, `recall`=0.0831
- **per_class**: `bicycle`=`{'precision': 0.0429, 'recall': 0.0095, 'f1': 0.0155, 'ap': 0.0152, 'map50': 0.0152, 'num_gt': 316, 'num_pred': 70, 'num_tp': 3}`, `car`=`{'precision': 0.0819, 'recall': 0.0528, 'f1': 0.0642, 'ap': 0.0192, 'map50': 0.0192, 'num_gt': 1932, 'num_pred': 1245, 'num_tp': 102}`, `person`=`{'precision': 0.36, 'recall': 0.2435, 'f1': 0.2905, 'ap': 0.1947, 'map50': 0.1947, 'num_gt': 11004, 'num_pred': 7441, 'num_tp': 2679}`, `road_sign`=`{'precision': 0.0233, 'recall': 0.0267, 'f1': 0.0248, 'ap': 0.0048, 'map50': 0.0048, 'num_gt': 75, 'num_pred': 86, 'num_tp': 2}`
- **conf_thr**: 0.2000
- _source_: `results_collected/ssh9.qudata.ai/results/trafficcamnet/metrics.json`

### `lprnet` @ `ssh9.qudata.ai`  (ocr)

- **char_accuracy**: 0.5904
- **full_plate_accuracy**: 0.0621
- **char_error_rate**: 0.4096
- **num_samples**: 4893
- _note_: US Latin model evaluated on RU Cyrillic plates — domain gap expected
- _source_: `results_collected/ssh9.qudata.ai/results/lprnet/metrics.json`

### `nomeroff_ocr` @ `ssh9.qudata.ai`  (ocr)

- **char_accuracy**: 0.9995
- **full_plate_accuracy**: 0.9978
- **char_error_rate**: 0.0005
- **num_samples**: 4893
- _note_: nomeroff-net NumberPlateTextReading direct (RU)
- _source_: `results_collected/ssh9.qudata.ai/results/nomeroff_ocr/metrics.json`

