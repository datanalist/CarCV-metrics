# CARS Model Evaluation — d072e99b6844

## Summary

| Model | Key Metric | Value | Target | Status |
|-------|-----------|-------|--------|--------|
| trafficcamnet | precision | 0.0819 | ≥0.9 | ❌ FAIL |
| trafficcamnet | recall | 0.0528 | ≥0.85 | ❌ FAIL |
| trafficcamnet | f1 | 0.0642 | ≥0.87 | ❌ FAIL |

## Detailed Results

### trafficcamnet
- **precision**: 0.0819
- **recall**: 0.0528
- **f1**: 0.0642
- **ap**: 0.0192
- **map50**: 0.0192
- **num_gt**: 1932.0000
- **num_pred**: 1245.0000
- **num_tp**: 102.0000
- **conf_thr**: 0.2000

## Known Limitations

- LPDNet trained on US plates, evaluated on RU dataset → domain gap expected
- LPRNet US model (Latin alphabet) evaluated on Russian Cyrillic plates → limited applicability
- VehicleMakeNet evaluated on mad-cars (RU/EU brands); NGC model trained on US brands

*Generated automatically by CARS evaluation pipeline on d072e99b6844*