# utils/model_loader.py
import onnxruntime as ort
import numpy as np
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class TrafficCamNetLoader:
    """Load and manage ONNX TrafficCamNet model inference."""

    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Args:
            model_path: Path to ONNX model file
            providers: onnxruntime execution providers.
                      Default: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        """
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        logger.info(f"Model loaded: {model_path}")
        logger.info(f"Input: {self.input_name}, shape: {self.input_shape}")
        logger.info(f"Outputs: {self.output_names}")

    def infer(self, image_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on preprocessed image batch.

        Args:
            image_data: Numpy array, shape (N, 1, H, W) or (N, 3, H, W)

        Returns:
            Dictionary mapping output names to arrays
        """
        outputs = self.session.run(self.output_names, {self.input_name: image_data})
        return {name: out for name, out in zip(self.output_names, outputs)}
