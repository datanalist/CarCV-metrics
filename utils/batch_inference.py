import numpy as np
from typing import Dict, List, Tuple
import logging

from utils.model_loader import TrafficCamNetLoader
from utils.postprocess import decode_detections, apply_nms, Detection

logger = logging.getLogger(__name__)


class BatchInferenceEngine:
    """
    GPU batch inference engine for TrafficCamNet model.

    Handles:
    - Loading ONNX model via TrafficCamNetLoader
    - Running batch inference on (B, 3, 544, 960) images
    - Postprocessing outputs (decode + NMS)
    - Converting normalized coords to pixel coords
    """

    def __init__(self, model_path: str):
        """
        Initialize BatchInferenceEngine.

        Args:
            model_path: Path to ONNX model file
        """
        self.model = TrafficCamNetLoader(model_path)
        logger.info(f"BatchInferenceEngine initialized with model: {model_path}")

    def infer_batch(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run ONNX inference on image batch.

        Args:
            batch: Numpy array of shape (B, 3, 544, 960), dtype float32

        Returns:
            Dictionary mapping output names to arrays
            - 'output_cov': (B, 1, 34, 60) confidence maps
            - 'output_bbox': (B, 4, 34, 60) bbox deltas
        """
        if not isinstance(batch, np.ndarray):
            raise TypeError(f"batch must be numpy array, got {type(batch)}")
        if batch.dtype != np.float32:
            raise TypeError(f"batch must be float32, got {batch.dtype}")
        if len(batch.shape) != 4:
            raise ValueError(f"batch must be 4D (B,C,H,W), got shape {batch.shape}")

        outputs = self.model.infer(batch)
        return outputs

    def postprocess_batch(
        self,
        outputs: Dict[str, np.ndarray],
        image_shapes: List[Tuple[int, int]],
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.45,
        input_w: int = 960,
        input_h: int = 544
    ) -> List[List[Detection]]:
        """
        Postprocess batch outputs: decode + NMS per image + convert to pixel coords.

        Args:
            outputs: Dict with 'output_cov' and 'output_bbox' arrays
                    - output_cov: (B, 1, 34, 60) confidence maps
                    - output_bbox: (B, 4, 34, 60) bbox deltas
            image_shapes: List of (H, W) tuples for each image in batch
            confidence_threshold: Threshold for detection confidence
            iou_threshold: NMS IoU threshold
            input_w: Input width (960)
            input_h: Input height (544)

        Returns:
            List[List[Detection]] - detections grouped by image
        """
        # Extract output arrays, handling optional :0 suffix in names
        cov_output = None
        bbox_output = None

        for name, array in outputs.items():
            if 'cov' in name.lower():
                cov_output = array
            elif 'bbox' in name.lower():
                bbox_output = array

        if cov_output is None or bbox_output is None:
            raise ValueError(f"outputs must contain 'cov' and 'bbox' arrays, got keys: {outputs.keys()}")

        batch_size = cov_output.shape[0]

        if len(image_shapes) != batch_size:
            raise ValueError(
                f"image_shapes length ({len(image_shapes)}) must match "
                f"batch size ({batch_size})"
            )

        # Decode detections from entire batch at once
        # This uses the updated decode_detections that marks batch_idx in class_id
        detections = decode_detections(
            cov_output,
            bbox_output,
            confidence_threshold=confidence_threshold,
            input_w=input_w,
            input_h=input_h
        )

        # Group detections by batch image index
        # We use the class_id field to store batch_idx temporarily
        detections_per_image = [[] for _ in range(batch_size)]
        for det in detections:
            batch_idx = int(det.class_id)
            # Create new Detection with batch_idx temporarily stored
            detections_per_image[batch_idx].append(det)

        # Apply NMS per image
        nms_results = []
        for b, dets in enumerate(detections_per_image):
            kept_dets = apply_nms(dets, iou_threshold=iou_threshold)

            # Convert from normalized to pixel coordinates
            img_h, img_w = image_shapes[b]
            pixel_dets = []
            for det in kept_dets:
                x_norm, y_norm, w_norm, h_norm = det.bbox
                x_pix = x_norm * img_w
                y_pix = y_norm * img_h
                w_pix = w_norm * img_w
                h_pix = h_norm * img_h

                # Create new Detection with pixel coords and reset class_id to 0
                pixel_det = Detection(
                    bbox=[x_pix, y_pix, w_pix, h_pix],
                    confidence=det.confidence,
                    class_id=0  # Reset to 0 after processing
                )
                pixel_dets.append(pixel_det)

            nms_results.append(pixel_dets)

        return nms_results
