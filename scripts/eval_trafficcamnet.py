#!/usr/bin/env python3
"""
Evaluation pipeline for TrafficCamNet on BDD100K validation set.

Usage:
    python scripts/eval_trafficcamnet.py
    python scripts/eval_trafficcamnet.py --config configs/experiment/trafficcamnet_eval.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import json
import csv
import logging
from typing import Dict, List, Tuple
import time
import numpy as np
import cv2
from tqdm import tqdm

from utils.model_loader import TrafficCamNetLoader
from utils.data_loader import BDD100KLoader, ImagePreprocessor
from utils.postprocess import decode_detections, apply_nms
from utils.metrics import COCOMetricsComputer, ConfidenceStats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/experiment/trafficcamnet_eval.yaml") -> Dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(config: Dict) -> Dict:
    """Run evaluation pipeline."""

    logger.info("=" * 60)
    logger.info("TrafficCamNet Evaluation Pipeline")
    logger.info("=" * 60)

    # Parse config
    model_cfg = config['model']
    data_cfg = config['data']
    eval_cfg = config['evaluation']
    out_cfg = config['artifacts']

    # Create output directory
    output_dir = Path(out_cfg['local_output_root'])
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Load model
    logger.info(f"Loading model: {model_cfg['local_path']}")
    model = TrafficCamNetLoader(model_cfg['local_path'])

    # Load data
    logger.info(f"Loading dataset: {data_cfg['local_ann_json']}")
    loader = BDD100KLoader(
        ann_json_path=data_cfg['local_ann_json'],
        images_dir=data_cfg['local_images_dir'],
        category_map=data_cfg['category_map'],
        max_images=data_cfg['max_images']
    )

    preprocessor = ImagePreprocessor(
        input_w=model_cfg['input_w'],
        input_h=model_cfg['input_h'],
        mean_b=model_cfg['mean_b'],
        mean_g=model_cfg['mean_g'],
        mean_r=model_cfg['mean_r'],
        net_scale_factor=model_cfg['net_scale_factor']
    )

    # Initialize metrics
    metrics_comp = COCOMetricsComputer(class_name='car', iou_threshold=eval_cfg['iou_threshold'])
    conf_stats = ConfidenceStats()

    latencies = []
    total_gt_boxes = 0
    total_predictions = 0
    error_examples = {'fp': [], 'fn': [], 'tp': []}

    logger.info(f"Running inference on {len(loader.images)} images...")

    # Main inference loop
    for img_info in tqdm(loader.images, desc="Evaluating"):
        image_id = img_info['id']

        # Load image
        try:
            img, filename = loader.get_image_by_id(image_id)
        except Exception as e:
            logger.warning(f"Failed to load image {image_id}: {e}")
            continue

        # Get ground truth
        gt_boxes = loader.get_annotations_for_image(image_id)
        gt_boxes_pixel = [b['bbox'] for b in gt_boxes]  # [x, y, w, h]
        total_gt_boxes += len(gt_boxes)

        # Preprocess
        tensor, scale_x, scale_y = preprocessor.preprocess(img)

        # Inference with timing
        t_start = time.time()
        outputs = model.infer(tensor)
        latency_ms = (time.time() - t_start) * 1000
        latencies.append(latency_ms)

        # Postprocess
        cov_name = model_cfg['output_cov_name']
        bbox_name = model_cfg['output_bbox_name']
        # Try with and without :0 suffix for ONNX runtime compatibility
        if cov_name not in outputs:
            cov_name = cov_name + ':0'
        if bbox_name not in outputs:
            bbox_name = bbox_name + ':0'

        cov_output = outputs[cov_name]
        bbox_output = outputs[bbox_name]

        detections = decode_detections(
            cov_output, bbox_output,
            confidence_threshold=model_cfg['confidence_threshold'],
            input_w=model_cfg['input_w'],
            input_h=model_cfg['input_h']
        )
        detections = apply_nms(detections, iou_threshold=model_cfg['nms_iou_threshold'])

        # Convert normalized to pixel coordinates
        h, w = img.shape[:2]
        pred_boxes_pixel = []
        for det in detections:
            x_norm, y_norm, w_norm, h_norm = det.bbox
            x_pix = x_norm * w
            y_pix = y_norm * h
            w_pix = w_norm * w
            h_pix = h_norm * h
            pred_boxes_pixel.append(([x_pix, y_pix, w_pix, h_pix], det.confidence))
            conf_stats.add_prediction(det.confidence)
            total_predictions += 1

        # Register with metrics
        img_id_metric = metrics_comp.add_image(h, w)
        metrics_comp.add_ground_truths(img_id_metric, gt_boxes_pixel)
        metrics_comp.add_predictions(img_id_metric, pred_boxes_pixel)

    logger.info("Computing metrics...")
    metrics = metrics_comp.compute()

    # Compute latency statistics
    latencies_np = np.array(latencies)
    latency_stats = {
        'mean': float(np.mean(latencies_np)),
        'median': float(np.median(latencies_np)),
        'p95': float(np.percentile(latencies_np, 95)),
        'p99': float(np.percentile(latencies_np, 99)),
        'min': float(np.min(latencies_np)),
        'max': float(np.max(latencies_np))
    }

    # Check targets
    target_met = {
        'precision': metrics['precision'] >= eval_cfg['target_precision'],
        'recall': metrics['recall'] >= eval_cfg['target_recall'],
        'mAP_50': metrics['mAP_50'] >= eval_cfg.get('target_f1', 0.5)
    }

    # Prepare results
    results = {
        'model': 'TrafficCamNet',
        'config': {
            'input_size': f"{model_cfg['input_w']}x{model_cfg['input_h']}",
            'confidence_threshold': model_cfg['confidence_threshold'],
            'nms_iou_threshold': model_cfg['nms_iou_threshold'],
            'evaluation_iou_threshold': eval_cfg['iou_threshold']
        },
        'metrics': metrics,
        'latency_ms': latency_stats,
        'dataset': {
            'total_images': len(loader.images),
            'total_gt_boxes': total_gt_boxes,
            'total_predictions': total_predictions
        },
        'target_met': target_met,
        'confidence_stats': conf_stats.get_stats()
    }

    # Save JSON results
    results_json = output_dir / "results.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_json}")

    # Save CSV results
    results_csv = output_dir / "results.csv"
    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
        writer.writeheader()
        for key, val in results['metrics'].items():
            writer.writerow({'metric': key, 'value': val})
        for key, val in results['latency_ms'].items():
            writer.writerow({'metric': f'latency_{key}', 'value': val})
    logger.info(f"Saved CSV to {results_csv}")

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"mAP@0.5: {metrics['mAP_50']:.4f}")
    logger.info(f"Latency (mean): {latency_stats['mean']:.3f} ms")
    logger.info(f"Targets met: {target_met}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrafficCamNet evaluation")
    parser.add_argument('--config', type=str, default='configs/experiment/trafficcamnet_eval.yaml',
                        help='Path to config YAML')
    args = parser.parse_args()

    config = load_config(args.config)
    results = evaluate(config)
