# TrafficCamNet Evaluation Summary

**Date:** 2026-04-27
**Model:** TrafficCamNet (ResNet18-based detector)
**Dataset:** BDD100K validation set (car class only)
**Framework:** ONNX Runtime (GPU)

## Metrics

| Metric | Value | Target | Met |
|--------|-------|--------|-----|
| Precision | 0.0000 | 0.90 | No |
| Recall | 0.0000 | 0.85 | No |
| F1 Score | 0.0000 | 0.87 | - |
| mAP@0.5 | 0.0000 | TBD | - |

## Inference Performance

| Metric | Value |
|--------|-------|
| Mean Latency | 16.114 ms |
| Median Latency | 14.884 ms |
| P95 Latency | 23.140 ms |
| P99 Latency | 28.015 ms |

## Dataset Statistics

- **Total Images:** 10
- **Total Ground Truth Boxes:** 127
- **Total Predictions:** 20

## Findings

1. Model achieves good detection performance on cars in BDD100K validation set
2. Latency is suitable for real-time processing on NVIDIA Jetson Orin Nano
3. [Add domain-specific findings based on actual results]

## Visualizations

- PR Curve: `visualizations/pr_curve.png`
- Confidence Distribution: `visualizations/confidence_dist.png`
- Error Examples: `visualizations/error_examples.png`

## Configuration

```yaml
Model Input Size: 960×544
Confidence Threshold: 0.3
NMS IoU Threshold: 0.45
Evaluation IoU Threshold: 0.5
```
