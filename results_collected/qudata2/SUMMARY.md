# CARS Model Evaluation — 280e00cd198b

## Summary

| Model | Key Metric | Value | Target | Status |
|-------|-----------|-------|--------|--------|
| trafficcamnet | precision | 0.1671 | ≥0.9 | ❌ FAIL |
| trafficcamnet | recall | 0.0316 | ≥0.85 | ❌ FAIL |
| trafficcamnet | f1 | 0.0531 | ≥0.87 | ❌ FAIL |
| vehiclemakenet | top1_accuracy | 0.0829 | ≥0.7 | ❌ FAIL |
| vehiclemakenet | top3_accuracy | 0.2114 | ≥0.85 | ❌ FAIL |
| lpdnet | recall | 0.2961 | ≥0.8 | ❌ FAIL |
| lpdnet | precision | 0.8837 | ≥0.7 | ✅ PASS |
| lprnet | char_accuracy | 0.5903 | ≥0.9 | ❌ FAIL |
| lprnet | full_plate_accuracy | 0.0621 | ≥0.8 | ❌ FAIL |

## Detailed Results

### trafficcamnet
- **precision**: 0.1671
- **recall**: 0.0316
- **f1**: 0.0531
- **ap**: 0.0192
- **map50**: 0.0192
- **num_gt**: 1932.0000
- **num_pred**: 365.0000
- **num_tp**: 61.0000

### vehiclemakenet
- **top1_accuracy**: 0.0829
- **top3_accuracy**: 0.2114
- **num_samples**: 700.0000

### lpdnet
- **precision**: 0.8837
- **recall**: 0.2961
- **f1**: 0.4436
- **ap**: 0.2617
- **map50**: 0.2617
- **num_gt**: 385.0000
- **num_pred**: 129.0000
- **num_tp**: 114.0000

### lprnet
- **char_accuracy**: 0.5903
- **full_plate_accuracy**: 0.0621
- **char_error_rate**: 0.4097
- **num_samples**: 4893.0000

## Known Limitations

- LPDNet trained on US plates, evaluated on RU dataset → domain gap expected
- LPRNet US model (Latin alphabet) evaluated on Russian Cyrillic plates → limited applicability
- VehicleMakeNet evaluated on mad-cars (RU/EU brands); NGC model trained on US brands

*Generated automatically by CARS evaluation pipeline on 280e00cd198b*