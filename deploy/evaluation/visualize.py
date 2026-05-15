"""Visualization utilities for CARS evaluation results."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def plot_pr_curve(precisions, recalls, ap, model_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, "b-", linewidth=2, label=f"AP={ap:.3f}")
    ax.axhline(y=0.90, color="r", linestyle="--", alpha=0.7, label="Precision target 0.90")
    ax.axvline(x=0.85, color="g", linestyle="--", alpha=0.7, label="Recall target 0.85")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {model_name}")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, model_name, save_path):
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(per_class: dict, model_name: str, threshold: float, save_path):
    classes = sorted(per_class.keys())
    values = [per_class[c] for c in classes]
    colors = ["green" if v >= threshold else "salmon" for v in values]

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.6), 5))
    bars = ax.bar(classes, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold {threshold}")
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Class Accuracy — {model_name}")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_summary_md(all_results: dict, save_path: str, server_name: str):
    lines = [
        f"# CARS Model Evaluation — {server_name}",
        "",
        "## Summary",
        "",
        "| Model | Key Metric | Value | Target | Status |",
        "|-------|-----------|-------|--------|--------|",
    ]

    for model, result in all_results.items():
        thresholds = result.get("thresholds", {})
        for metric, info in thresholds.items():
            status = "✅ PASS" if info["status"] == "PASS" else "❌ FAIL"
            lines.append(
                f"| {model} | {metric} | {info['value']:.4f} "
                f"| ≥{info['threshold']} | {status} |"
            )

    lines += ["", "## Detailed Results", ""]
    for model, result in all_results.items():
        lines.append(f"### {model}")
        metrics = result.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                lines.append(f"- **{k}**: {v:.4f}")
            elif isinstance(v, dict):
                pass  # skip per_class in summary
        lines.append("")

    lines += [
        "## Known Limitations",
        "",
        "- LPDNet trained on US plates, evaluated on RU dataset → domain gap expected",
        "- LPRNet US model (Latin alphabet) evaluated on Russian Cyrillic plates → limited applicability",
        "- VehicleMakeNet evaluated on mad-cars (RU/EU brands); NGC model trained on US brands",
        "",
        f"*Generated automatically by CARS evaluation pipeline on {server_name}*",
    ]

    Path(save_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {save_path}")
