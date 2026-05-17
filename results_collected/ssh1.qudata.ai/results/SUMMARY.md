# CARS Model Evaluation — b18448f94b73

## Summary

| Model | Key Metric | Value | Target | Status |
|-------|-----------|-------|--------|--------|
| vehiclemakenet | top1_accuracy | 0.0020 | ≥0.7 | ❌ FAIL |
| vehiclemakenet | top3_accuracy | 0.0147 | ≥0.85 | ❌ FAIL |

## Detailed Results

### vehiclemakenet
- **top1_accuracy**: 0.0020
- **top3_accuracy**: 0.0147
- **num_samples**: 4960.0000

## Known Limitations

- LPDNet trained on US plates, evaluated on RU dataset → domain gap expected
- LPRNet US model (Latin alphabet) evaluated on Russian Cyrillic plates → limited applicability
- VehicleMakeNet evaluated on mad-cars (RU/EU brands); NGC model trained on US brands

*Generated automatically by CARS evaluation pipeline on b18448f94b73*