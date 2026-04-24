---
name: ml-experimenter
description: "Conduct reproducible, valid, and visual ML experiments in Computer Vision domain — model evaluation, comparison, ablation studies, fine-tuning. Use proactively when the user asks to evaluate a model, run an experiment, measure metrics, benchmark a neural network, compare models, or perform an ablation study."
---

# ML Experimenter

Conduct reproducible ML experiments with quantitative metrics, mandatory visualizations, error analysis, and professional reports. All results must come from actual model inference — never fabricate data.

## Workflow Overview

1. **Gather requirements** — model, dataset, experiment type
2. **Study context** — read rules, architecture, dataset docs
3. **Plan** — objective, hypothesis, metrics, artifacts
4. **Implement** — notebook + helper scripts
5. **Compute metrics** — minimum 3 quantitative metrics per experiment
6. **Visualize** — confusion matrix, per-class metrics, confidence distribution, error examples
7. **Error analysis** — FP/FN analysis, worst-case predictions, edge cases
8. **Report** — markdown report in `docs/experiments/`
9. **Update docs** — architecture, plan

---

## Phase 1: Gather Requirements

Determine: **model**, **dataset**, **experiment type**. If the user already specified all three — proceed directly.

**Models available:**

| Model | Task | Path |
|-------|------|------|
| TrafficCamNet | Vehicle detection | `models/baseline/resnet18_trafficcamnet_fp16.engine` |
| VehicleMakeNet | Brand classification | `models/baseline/vehiclemakenet.engine` |
| VehicleTypeNet | Type classification | `models/baseline/vehicletypenet.engine` |
| LPDNet | License plate detection | `models/baseline/lpdnet.engine` |
| LPR_STN_PRE_POST | Plate OCR | `models/baseline/LPR_STN_PRE_POST.onnx` |
| bae_model_f3 | Color recognition | `models/baseline/bae_model_f3.onnx` |
| FaceDetect | Face detection | `models/baseline/facedetect.engine` |

**Datasets:**

| Dataset | Path |
|---------|------|
| BDD100K | `/home/mk/Загрузки/DATASETS/bdd100k` |
| VMMRdb | `/home/mk/Загрузки/DATASETS/VMMRdb` |
| YMAD | `data/external/ymad_cars` |
| autoriaNumberplateOcrRu | Check `data/` |

**Experiment types:** Baseline evaluation, Model comparison (A vs B), Ablation study, Fine-tuning.

## Phase 2: Study Context

Read these files before writing code:

1. `.cursor/rules/experiments/dl-experiments.mdc` — experiment structure and conventions. **Follow strictly.**
2. `docs/architecture.md` — model specs, current metrics, pipeline.
3. `docs/about_datasets/{dataset-name}.md` — dataset structure and format. **If missing — create first.**
4. `docs/rules/task.md` — current task requirements.
5. Check `scripts/` for existing reusable inference code.

## Phase 3: Plan

Update `docs/rules/plan.md` with:
- Objective and hypothesis
- Model path and configuration
- Dataset path, split, filtering
- Metrics to compute (≥3)
- Visualization list
- Expected artifacts

## Phase 4: Implement

### Artifact Structure

| Artifact | Path |
|----------|------|
| Notebook | `notebooks/{experiment-name}.ipynb` |
| Figures | `notebooks/{experiment-name}/` |
| Helper scripts | `scripts/{task-name}/` |
| Results | `results/baseline/{model-name}/` or `results/{experiment-name}/` |

### Notebook Structure (mandatory sections)

Build using **Jupyter MCP tools only** (`add_cell`, `edit_cell`, `execute_cell`, `read_output_of_cell`):

1. **Header** (markdown) — title, date, objective, hypothesis, model, dataset
2. **Setup** (code) — imports, paths, device config, `random seeds`
3. **Data Loading** (code + markdown) — load, show statistics, class distribution
4. **Preprocessing / Analysis** (code) — data stats, distributions, sample visualizations
5. **Model / Inference** (code) — load model, configure, run inference
6. **Metrics** (code + markdown) — compute and display metrics in tables and charts
7. **Error Analysis** (code + markdown) — FP/FN, worst-case examples, edge cases
8. **Conclusions** (markdown) — summary, comparison with targets, recommendations

### Heavy Computation

For large-dataset inference — extract to `.py` script:

```
scripts/{task-name}/{script}.py  →  uv run python scripts/{task-name}/{script}.py
```

Save results to `data/processed/` or `results/`, load in notebook for analysis.

Reuse existing code: check `scripts/` before writing new scripts.

### Code Standards

- PEP 8, type hints, descriptive names
- Helper `.py` code: concise, no docstrings (temporary)
- **Always GPU** (CUDA) with ONNX Runtime or PyTorch
- Mixed precision where applicable
- Fix random seeds (`random`, `numpy`, `torch`) for reproducibility
- `num_workers` 4–8 per GPU in DataLoader

## Phase 5: Compute Metrics

Select metrics based on task type — see [references/metrics-and-viz.md](references/metrics-and-viz.md) for detailed metrics per task and code snippets.

**Minimum requirement:** ≥3 quantitative metrics + inference performance per experiment.

Always compare results against target thresholds and mark **PASS / FAIL**.

## Phase 6: Visualize

**Mandatory visualizations** (save to `notebooks/{experiment-name}/`):

1. **Confusion Matrix** — raw counts + normalized (classification / OCR / color)
2. **Per-class Metrics** — horizontal bar chart P/R/F1 per class
3. **Confidence Distribution** — histogram correct vs incorrect
4. **Error Examples** — grid of worst-case predictions with GT and predicted labels
5. **Metrics Summary** — bar chart of aggregate metrics vs targets

For detection tasks additionally: AP by IoU curve, TP/FP/FN distribution, predicted vs GT bboxes overlay.

Code patterns for all visualizations — see [references/metrics-and-viz.md](references/metrics-and-viz.md#visualization-patterns).

Use `matplotlib` + `seaborn`. Label axes and titles. `plt.tight_layout()`. Save at **150 DPI**.

## Phase 7: Error Analysis

- Analyze **false positives** and **false negatives**
- Check edge cases: night (IR), rain/snow, motion blur, unusual angles
- Show worst-case prediction examples in notebook
- Identify per-class weak spots and failure patterns
- For OCR: per-character confusion analysis

## Phase 8: Report

Create `docs/experiments/{experiment-name}.md` — see [references/report-template.md](references/report-template.md) for the exact template.

## Phase 9: Update Documentation

1. Update `docs/architecture.md` if model metrics changed (Current Metrics section).
2. Update `docs/rules/plan.md` — mark experiment as completed.
3. Delete temporary helper scripts if no longer needed.

---

## Rules

- **Reproducibility first** — fix random seeds, log all hyperparameters, save raw predictions.
- **No fabricated data** — all metrics from actual inference, never invented.
- **GPU mandatory** — always CUDA for inference.
- **≥3 metrics** per experiment.
- **Error analysis mandatory** — FP/FN analysis + worst-case examples.
- **Visualizations mandatory** — confusion matrix, per-class, confidence, error examples.
- **Save artifacts** — metrics (JSON), predictions (pickle/JSON), figures (PNG), notebook (ipynb).
- **Professional reports** — every experiment gets `docs/experiments/` report.
- **Reuse code** — check `scripts/` for existing inference code first.
- **Target comparison** — always compare against thresholds from `edge-ai.mdc`.
- **Clean up** — remove temporary scripts after experiment.
- Never modify model files or dataset files.
