#!/usr/bin/env python3
"""
Batch evaluation pipeline for TrafficCamNet on BDD100K using GPU-optimized inference.

This script orchestrates batch inference with GPU acceleration, using:
- BDD100KBatchLoader for efficient batch data loading
- BatchInferenceEngine for GPU batch inference
- COCOMetricsComputer for metrics computation

Usage:
    python scripts/eval_trafficcamnet_gpu.py
    python scripts/eval_trafficcamnet_gpu.py --config configs/experiment/trafficcamnet_eval.yaml
    python scripts/eval_trafficcamnet_gpu.py --config configs/experiment/trafficcamnet_eval_test.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import json
import csv
import logging
import time
import os
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

from utils.batch_data_loader import BDD100KBatchLoader
from utils.batch_inference import BatchInferenceEngine
from utils.metrics import COCOMetricsComputer, ConfidenceStats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(config_path: str) -> Dict:
    """
    Run batch GPU evaluation pipeline.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing results
    """
    logger.info("=" * 60)
    logger.info("TrafficCamNet Batch Evaluation Pipeline (GPU)")
    logger.info("=" * 60)

    config = load_config(config_path)

    # Parse config sections
    model_cfg = config['model']
    data_cfg = config['data']
    eval_cfg = config['evaluation']
    batch_cfg = config['batch']
    out_cfg = config['artifacts']

    # Validate file existence before processing
    assert Path(model_cfg['local_path']).exists(), f"Model not found: {model_cfg['local_path']}"
    assert Path(data_cfg['local_ann_json']).exists(), f"Annotations not found: {data_cfg['local_ann_json']}"
    assert Path(data_cfg['local_images_dir']).exists(), f"Images directory not found: {data_cfg['local_images_dir']}"
    logger.info("File existence checks passed")

    # Create output directory with write validation
    output_dir = Path(out_cfg['local_output_root'])
    output_dir.mkdir(parents=True, exist_ok=True)
    assert output_dir.exists() and os.access(output_dir, os.W_OK), f"Output directory not writable: {output_dir}"
    logger.info(f"Output directory: {output_dir}")

    # Initialize BatchInferenceEngine with model
    logger.info(f"Loading model: {model_cfg['local_path']}")
    inference_engine = BatchInferenceEngine(model_cfg['local_path'])

    # Initialize BDD100KBatchLoader with batch config
    logger.info(f"Loading dataset: {data_cfg['local_ann_json']}")
    batch_loader = BDD100KBatchLoader(
        ann_json_path=data_cfg['local_ann_json'],
        images_dir=data_cfg['local_images_dir'],
        category_map=data_cfg['category_map'],
        batch_size=batch_cfg['batch_size'],
        num_workers=batch_cfg['num_loader_threads'],
        max_images=data_cfg['max_images']
    )

    # Initialize metrics collectors
    metrics_comp = COCOMetricsComputer(class_name='car', iou_threshold=eval_cfg['iou_threshold'])
    conf_stats = ConfidenceStats()

    # Tracking variables
    latencies = []
    total_gt_boxes = 0
    total_predictions = 0
    total_batches = 0

    logger.info(f"Starting batch evaluation (batch_size={batch_cfg['batch_size']})...")

    # Main evaluation loop - process batches until loader is done
    progress_bar = tqdm(desc="Evaluating batches", unit="batch")

    while not batch_loader.is_done():
        # Get next batch
        batch_result = batch_loader.get_batch()

        if batch_result is None:
            logger.warning("Received None batch, skipping")
            continue

        image_batch, metadata_list, gt_annotations_list = batch_result
        batch_size_actual = len(metadata_list)

        logger.debug(f"Processing batch {total_batches + 1} with {batch_size_actual} images")

        # Measure batch-level inference latency (inference only, excludes I/O and postprocessing)
        t_start = time.time()
        outputs = inference_engine.infer_batch(image_batch)
        batch_latency_ms = (time.time() - t_start) * 1000
        latencies.append(batch_latency_ms)

        # Extract original image shapes for postprocessing
        image_shapes = [
            metadata['orig_shape'] for metadata in metadata_list
        ]

        # Postprocess batch
        try:
            detections_per_image = inference_engine.postprocess_batch(
                outputs=outputs,
                image_shapes=image_shapes,
                confidence_threshold=model_cfg['confidence_threshold'],
                iou_threshold=model_cfg['nms_iou_threshold'],
                input_w=model_cfg['input_w'],
                input_h=model_cfg['input_h']
            )
        except Exception as e:
            logger.error(f"Error postprocessing batch {total_batches}: {e}")
            total_batches += 1
            progress_bar.update(1)
            continue

        # Accumulate metrics per image in batch
        # Use zip to properly align metadata, detections, and ground truth
        for metadata, detections, gt_boxes in zip(metadata_list, detections_per_image, gt_annotations_list):
            image_id = metadata['image_id']
            orig_h, orig_w = metadata['orig_shape']

            # Ground truth boxes for this image
            gt_boxes_pixel = [b['bbox'] for b in gt_boxes]
            total_gt_boxes += len(gt_boxes)

            # Register image with metrics
            metric_img_id = metrics_comp.add_image(orig_h, orig_w)
            metrics_comp.add_ground_truths(metric_img_id, gt_boxes_pixel)

            # Predicted boxes for this image
            pred_boxes_pixel = [
                (det.bbox, det.confidence) for det in detections
            ]
            total_predictions += len(detections)

            # Add predictions and confidence stats
            metrics_comp.add_predictions(metric_img_id, pred_boxes_pixel)
            for det in detections:
                conf_stats.add_prediction(det.confidence)

        total_batches += 1
        progress_bar.update(1)

    progress_bar.close()

    logger.info(f"Processed {metrics_comp.next_annotation_id - 1} predictions from {total_batches} batches")

    # Compute final metrics
    logger.info("Computing COCO metrics...")
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

    # Check if targets are met
    target_met = {
        'precision': metrics['precision'] >= eval_cfg.get('target_precision', 0.9),
        'recall': metrics['recall'] >= eval_cfg.get('target_recall', 0.85),
        'mAP_50': metrics['mAP_50'] >= eval_cfg.get('target_mAP_50', 0.5)
    }

    # Prepare results dictionary
    results = {
        'model': 'TrafficCamNet',
        'config': {
            'input_size': f"{model_cfg['input_w']}x{model_cfg['input_h']}",
            'confidence_threshold': model_cfg['confidence_threshold'],
            'nms_iou_threshold': model_cfg['nms_iou_threshold'],
            'evaluation_iou_threshold': eval_cfg['iou_threshold'],
            'batch_size': batch_cfg['batch_size'],
            'num_loader_threads': batch_cfg['num_loader_threads']
        },
        'metrics': metrics,
        'latency_ms': latency_stats,
        'dataset': {
            'total_images': batch_loader.total_images,
            'total_batches': total_batches,
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
    logger.info(f"Saved JSON results to {results_json}")

    # Save CSV results
    results_csv = output_dir / "results.csv"
    with open(results_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
        writer.writeheader()

        # Write metrics
        for key, val in results['metrics'].items():
            writer.writerow({'metric': key, 'value': val})

        # Write latency stats
        for key, val in results['latency_ms'].items():
            writer.writerow({'metric': f'latency_{key}', 'value': val})

        # Write dataset stats
        for key, val in results['dataset'].items():
            writer.writerow({'metric': f'dataset_{key}', 'value': val})

        # Write confidence stats
        for key, val in results['confidence_stats'].items():
            writer.writerow({'metric': f'confidence_{key}', 'value': val})

    logger.info(f"Saved CSV results to {results_csv}")

    # Log results summary
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"mAP@0.5: {metrics['mAP_50']:.4f}")
    logger.info(f"Latency (mean): {latency_stats['mean']:.3f} ms")
    logger.info(f"Latency (median): {latency_stats['median']:.3f} ms")
    logger.info(f"Latency (p95): {latency_stats['p95']:.3f} ms")
    logger.info(f"Latency (p99): {latency_stats['p99']:.3f} ms")
    logger.info(f"Total images: {batch_loader.total_images}")
    logger.info(f"Total batches: {total_batches}")
    logger.info(f"Ground truth boxes: {total_gt_boxes}")
    logger.info(f"Predictions: {total_predictions}")
    logger.info(f"Targets met: {target_met}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TrafficCamNet batch GPU evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/eval_trafficcamnet_gpu.py
    python scripts/eval_trafficcamnet_gpu.py --config configs/experiment/trafficcamnet_eval.yaml
    python scripts/eval_trafficcamnet_gpu.py --config configs/experiment/trafficcamnet_eval_test.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment/trafficcamnet_eval.yaml',
        help='Path to config YAML (default: configs/experiment/trafficcamnet_eval.yaml)'
    )
    args = parser.parse_args()

    results = evaluate(args.config)
