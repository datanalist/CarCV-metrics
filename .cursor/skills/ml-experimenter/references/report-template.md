# Experiment Report Template

Save reports to `docs/experiments/{experiment-name}.md`. Use this exact structure:

```markdown
# {Experiment Title}

**Date:** {YYYY-MM-DD}
**Status:** {Completed / In Progress / Failed}

---

## Overview
{1-2 sentences: what was evaluated and why}

---

## Experiment Configuration

### Data and Model
| Parameter | Value |
|-----------|-------|
| **Dataset** | {name + path} |
| **Split** | {train/val/test, count} |
| **Classes** | {list or count} |
| **Images** | {total count} |
| **Model** | {name + path} |
| **Model size** | {MB} |

### Inference
| Parameter | Value |
|-----------|-------|
| **Framework** | {ONNX Runtime / TensorRT / PyTorch} |
| **Device** | {GPU model or CPU} |
| **Precision** | {FP32 / FP16 / INT8} |
| **Confidence threshold** | {value} |
| **Batch size** | {value} |

---

## Methodology
{Numbered steps describing the experiment procedure}

Key scripts:
- Inference: `scripts/{task-name}/{script}.py`
- Evaluation: `notebooks/{experiment-name}.ipynb`

---

## Results

### Quality Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| {metric1} | {value} | {target} | {PASS/FAIL} |
| {metric2} | {value} | {target} | {PASS/FAIL} |
| {metric3} | {value} | {target} | {PASS/FAIL} |

### Performance
| Metric | Value |
|--------|-------|
| Avg inference time | {ms} |
| FPS | {value} |
| GPU memory | {MB} |

---

## Error Analysis
{Key failure modes, edge cases, per-class weak spots}

---

## Artifacts
- Metrics: `results/{...}/metrics.json`
- Predictions: `results/{...}/predictions.pkl`
- Figures: `notebooks/{experiment-name}/`
- Notebook: `notebooks/{experiment-name}.ipynb`

---

## Observations
{Key findings, unexpected behaviors, recommendations for improvement}

---

## How to Reproduce
\```bash
# Inference
uv run python scripts/{task-name}/{script}.py

# Evaluation notebook
jupyter notebook notebooks/{experiment-name}.ipynb
\```

---

**Document version:** 1.0
**Created:** {date}
```
