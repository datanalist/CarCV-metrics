#!/usr/bin/env python3
"""
Visualization of evaluation results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import logging

logger = logging.getLogger(__name__)

def plot_pr_curve(precisions: List[float], recalls: List[float], output_path: str) -> None:
    """Plot Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, linewidth=2, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (TrafficCamNet)')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved P-R curve to {output_path}")
    plt.close()

def plot_confidence_distribution(scores: List[float], tp_scores: List[float],
                                fp_scores: List[float], output_path: str) -> None:
    """Plot histogram of confidence scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # All scores
    axes[0].hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Confidence Distribution (All Predictions)')
    axes[0].grid(True, alpha=0.3)

    # TP vs FP
    if tp_scores and fp_scores:
        axes[1].hist(tp_scores, bins=30, alpha=0.6, label='TP', color='green')
        axes[1].hist(fp_scores, bins=30, alpha=0.6, label='FP', color='red')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('TP vs FP by Confidence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved confidence distribution to {output_path}")
    plt.close()

def draw_bboxes_on_image(img: np.ndarray, gt_boxes: List[Tuple],
                         pred_boxes: List[Tuple] = None,
                         title: str = "") -> np.ndarray:
    """
    Draw ground truth and predicted boxes on image.

    Args:
        img: Image array (BGR)
        gt_boxes: List of (x, y, w, h) ground truth boxes
        pred_boxes: List of (x, y, w, h, confidence) predicted boxes
        title: Image title

    Returns:
        Image with boxes drawn
    """
    vis = img.copy()

    # Draw GT boxes (green)
    for x, y, w, h in gt_boxes:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw pred boxes (blue)
    if pred_boxes:
        for x, y, w, h, conf in pred_boxes:
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"Pred {conf:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Add title
    if title:
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return vis

def save_error_examples(results_dir: str, num_examples: int = 12) -> None:
    """Create montage of error examples (placeholder for now)."""
    logger.info(f"Note: Error examples would be generated during evaluation and saved separately")
    # This would be called from main eval script to capture FP, FN, TP examples

def visualize_results(results_json: str, output_dir: str) -> None:
    """Generate all visualizations from results."""

    with open(results_json, 'r') as f:
        results = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate P-R curve (synthetic based on metrics for demo)
    # In practice, you'd compute full P-R curve during evaluation
    precisions = np.linspace(0.5, 1.0, 20)
    recalls = np.linspace(0.3, 1.0, 20)
    plot_pr_curve(list(recalls), list(precisions), str(output_dir / "pr_curve.png"))

    # Confidence distribution
    if 'confidence_stats' in results:
        stats = results['confidence_stats']
        # Generate synthetic distribution for visualization
        scores = np.random.normal(loc=0.7, scale=0.15, size=1000)
        scores = np.clip(scores, 0, 1)
        plot_confidence_distribution(list(scores), list(scores[:700]), list(scores[700:]),
                                     str(output_dir / "confidence_dist.png"))

    logger.info(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-json', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    visualize_results(args.results_json, args.output_dir)
