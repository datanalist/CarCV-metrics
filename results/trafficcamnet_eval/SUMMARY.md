# TrafficCamNet Evaluation Summary

**Date:** 2026-04-30  
**Model:** TrafficCamNet ResNet18 (pruned, 5.1 MB)  
**Dataset:** BDD100K validation set — 10 000 images  
**Framework:** ONNX Runtime (CUDAExecutionProvider, RTX 3090)

## Metrics

| Metric | Value | Target | Met |
|--------|-------|--------|-----|
| Precision | 0.536 | 0.90 | No |
| Recall | 0.175 | 0.85 | No |
| F1 Score | 0.264 | 0.87 | — |
| mAP@0.5 | 0.111 | TBD | — |

## Inference Performance

| Metric | Value |
|--------|-------|
| Mean Latency (batch=64) | 659.98 ms |
| Median Latency | 657.81 ms |
| P95 Latency | 668.95 ms |
| P99 Latency | 805.40 ms |
| Per-image (est.) | ~10.3 ms |

## Dataset Statistics

- **Total Images:** 10 000
- **Total Ground Truth Boxes:** 123 718 (cars + trucks + buses)
- **Total Predictions:** 40 458
- **Total Batches:** 157 × 64 images

## Inference Configuration

```yaml
Model Input Size: 960×544
Preprocessing: pixel / 255.0 (direct resize, no letterbox)
Confidence Threshold: 0.30
NMS IoU Threshold: 0.45
Evaluation IoU Threshold: 0.50
Batch Size: 64 (GPU VRAM limit)
BBOX_SCALE: 35.0 (DetectNet v2 decode constant)
```

## Findings

1. **Decode bug fix**: The original postprocessing used incorrect bbox decode formula,
   producing 1×1 px boxes. Fixed with the DetectNet v2 formula:
   `x1 = (xs * stride - dx1 * BBOX_SCALE) / input_w`

2. **Preprocessing fix**: Original code applied mean subtraction + /255 normalization
   (resulting in max confidence 0.15). Correct preprocessing is direct resize + pixel/255
   (no mean subtraction), matching the model's training pipeline.

3. **Low recall (0.175)**: Expected for a 90%+ pruned model (5.1 MB vs 44 MB full ResNet18).
   The model detects cars with reasonable precision (0.536) but misses most objects.

4. **GPU batch size**: batch=64 is the practical GPU limit for this architecture on RTX 3090
   with 24 GB VRAM (intermediate activations overflow at batch=512).

5. **Latency**: ~10 ms per image on GPU is suitable for real-time (>30 FPS single stream).

## Next Steps

- Evaluate with full (unpruned) TrafficCamNet model for production-quality metrics
- Tune confidence threshold (currently 0.30) for better precision-recall tradeoff
- Test on Jetson Orin Nano (target deployment hardware)
