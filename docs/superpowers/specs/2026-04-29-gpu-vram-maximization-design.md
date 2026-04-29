# GPU VRAM Maximization Design

**Date:** 2026-04-29  
**Goal:** Fix broken GPU execution and maximize VRAM utilization during TrafficCamNet evaluation  
**Target Hardware:** NVIDIA RTX 3090 (24 GB VRAM)  
**Current state:** Inference runs on CPU (ORT CUDA provider silently fails), VRAM ~600 MB used

---

## Problem Statement

`BatchInferenceEngine` uses ONNX Runtime with `CUDAExecutionProvider`, but at runtime ORT silently
falls back to CPU:

```
Failed to load libonnxruntime_providers_cuda.so:
  libcublasLt.so.12: cannot open shared object file: No such file or directory
```

Root cause: ORT 1.24.4 requires CUDA 12 (`libcublasLt.so.12`), but the project environment
installs CUDA 13 packages (`nvidia/cu13/lib/libcublasLt.so.13`) for PyTorch cu130.

Secondary problem: `AdaptiveBatchSize.MODEL_WEIGHTS_MB = 2800` is wrong by 2 orders of magnitude
(actual model: 5.2 MB ONNX file, ~50 MB in GPU memory with activation buffers). This causes the
batch size calculator to waste ~3.2 GB of "phantom" overhead and underutilize VRAM.

**Result today:** ~28 sec/batch on CPU, ~600 MB VRAM, batch_size hardcoded to 512 regardless of
actual VRAM availability.

**Goal:** GPU inference active, batch size ~3000–3200 images, VRAM ~19–20 GB (80%), no system hang.

---

## Architecture

No new files. Three targeted changes to existing files + one dependency addition.

```
pyproject.toml
  └─ add nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12
       └─ ORT scans site-packages/nvidia/*/lib/ → finds libcublasLt.so.12 → CUDA active

utils/model_loader.py  (TrafficCamNetLoader.__init__)
  └─ add assert_gpu_execution() after session creation
       └─ raises RuntimeError immediately if CUDAExecutionProvider not active

utils/adaptive_batch_size.py  (AdaptiveBatchSize constants)
  └─ MODEL_WEIGHTS_MB: 2800 → 50
  └─ ONNXRUNTIME_OVERHEAD_MB: 500 → 300

configs/experiment/trafficcamnet_eval.yaml
  └─ batch_size: 512 → auto  (eval script calls AdaptiveBatchSize().calculate())
```

---

## Component Design

### 1. Dependencies (`pyproject.toml`)

Add to `[project] dependencies`:

```toml
"nvidia-cublas-cu12>=12.0",
"nvidia-cuda-runtime-cu12>=12.0",
"nvidia-cudnn-cu12>=9.0",
```

ORT's dynamic loader scans `site-packages/nvidia/cublas/lib/`, `nvidia/cuda_runtime/lib/`,
`nvidia/cudnn/lib/` at import time. It finds `libcublasLt.so.12` from `nvidia-cublas-cu12` and
activates `CUDAExecutionProvider`. PyTorch continues using cu13 libs — the versioned `.so.12` /
`.so.13` suffixes prevent conflicts.

No `LD_LIBRARY_PATH` changes, no wrapper scripts.

### 2. GPU pre-flight check (`utils/model_loader.py`)

Add to `TrafficCamNetLoader.__init__()` after `InferenceSession` creation:

```python
active_providers = self.session.get_providers()
if "CUDAExecutionProvider" not in active_providers:
    raise RuntimeError(
        f"GPU inference not active. Active providers: {active_providers}\n"
        "Fix: ensure nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, "
        "nvidia-cudnn-cu12 are installed (uv sync)."
    )
logger.info(f"GPU confirmed. Active providers: {active_providers}")
```

Placed in `TrafficCamNetLoader` (not eval script) so every caller gets protection automatically.
Fails before loading 10k images, before starting background threads.

### 3. Fix VRAM calculation (`utils/adaptive_batch_size.py`)

```python
# Before:
MODEL_WEIGHTS_MB = 2800      # WRONG: claimed ResNet18 weighs 2.8 GB
ONNXRUNTIME_OVERHEAD_MB = 500

# After:
MODEL_WEIGHTS_MB = 50        # ResNet18 ONNX 5.2 MB file, ~50 MB loaded with buffers
ONNXRUNTIME_OVERHEAD_MB = 300  # ORT CUDA runtime + scratch allocations
```

Resulting batch size on RTX 3090 with `safety_margin_percent=20`:
- Total VRAM: 24 576 MB
- Safety reserve (20%): 4 915 MB
- Fixed overhead: 350 MB
- Available for batch data: ~19 311 MB
- Per image (input 5.98 MB + output 0.039 MB): ~6.02 MB
- **Calculated batch size: ~3207** (clamped to max_batch_size=4096)

### 4. Adaptive batch size in config

`configs/experiment/trafficcamnet_eval.yaml`: change `batch_size: 512` to `batch_size: auto`.

Eval script reads this value; when `"auto"`, calls `AdaptiveBatchSize().calculate()` at startup
and logs the result. The `min_batch_size=64` / `max_batch_size=4096` bounds remain unchanged.

Required change in `scripts/eval_trafficcamnet_gpu.py` (lines 93, 108, 215):

```python
raw_batch_size = batch_cfg['batch_size']
if raw_batch_size == 'auto':
    from utils.adaptive_batch_size import AdaptiveBatchSize
    batch_size = AdaptiveBatchSize().calculate()
    logger.info(f"Adaptive batch size calculated: {batch_size}")
else:
    batch_size = int(raw_batch_size)
```

Then use `batch_size` variable instead of `batch_cfg['batch_size']` throughout the script.

---

## Safety

- **20% VRAM reserve** (~4.9 GB) covers CUDA context overhead, cuDNN workspace, and OS display
- RTX 3090 desktop GPU shares VRAM with display: current usage is 368 MB (nvidia-smi), leaving
  23 755 MB free — well within margin
- No OOM retry needed at this safety margin; if OOM occurs, the standard ORT exception propagates
  and the user can reduce batch_size in config

---

## Testing

1. After `uv sync`: `python3 -c "import onnxruntime as ort; s = ort.InferenceSession('models/baseline/resnet18_trafficcamnet.onnx', providers=['CUDAExecutionProvider','CPUExecutionProvider']); print(s.get_providers())"` → must show `CUDAExecutionProvider` first
2. Run eval on 100 images: verify VRAM > 10 GB in `nvidia-smi`, latency < 5 sec/batch
3. Run full eval: results JSON metrics unchanged (GPU vs CPU produces identical fp32 outputs for ResNet18)

---

## Files Changed

| File | Change |
|------|--------|
| `pyproject.toml` | Add 3 `nvidia-*-cu12` dependencies |
| `utils/model_loader.py` | Add GPU provider assertion in `__init__` |
| `utils/adaptive_batch_size.py` | Fix `MODEL_WEIGHTS_MB` and `ONNXRUNTIME_OVERHEAD_MB` constants |
| `configs/experiment/trafficcamnet_eval.yaml` | `batch_size: auto` |
| `scripts/eval_trafficcamnet_gpu.py` | Handle `batch_size: auto` → call `AdaptiveBatchSize().calculate()` |

No new files. No changes to `batch_inference.py`, `batch_data_loader.py`, or tests.
