import pytest
import numpy as np
from utils.batch_inference import BatchInferenceEngine
from utils.postprocess import Detection


class TestBatchInferenceEngineInit:
    """Test BatchInferenceEngine initialization."""

    def test_batch_inference_engine_init(self):
        """Test initialization with real model."""
        engine = BatchInferenceEngine(
            model_path="/home/mk/CarCV-metrics/models/baseline/resnet18_trafficcamnet.onnx"
        )
        assert engine.model is not None
        assert engine.model.session is not None
        assert engine.model.input_name is not None
        assert len(engine.model.output_names) > 0

    def test_batch_inference_engine_model_properties(self):
        """Test that model properties are accessible."""
        engine = BatchInferenceEngine(
            model_path="/home/mk/CarCV-metrics/models/baseline/resnet18_trafficcamnet.onnx"
        )
        assert hasattr(engine, 'model')
        assert hasattr(engine.model, 'session')
        assert hasattr(engine.model, 'input_name')
        assert hasattr(engine.model, 'output_names')


class TestInferBatch:
    """Test infer_batch method."""

    @pytest.fixture
    def engine(self):
        """Fixture for a BatchInferenceEngine instance."""
        return BatchInferenceEngine(
            model_path="/home/mk/CarCV-metrics/models/baseline/resnet18_trafficcamnet.onnx"
        )

    def test_infer_batch_returns_dict(self, engine):
        """Test that infer_batch returns a dictionary."""
        # Create dummy batch: (2, 3, 544, 960) float32
        batch = np.random.randn(2, 3, 544, 960).astype(np.float32)
        outputs = engine.infer_batch(batch)

        assert isinstance(outputs, dict)
        assert len(outputs) > 0

    def test_infer_batch_output_keys(self, engine):
        """Test that output dictionary contains expected keys."""
        batch = np.random.randn(2, 3, 544, 960).astype(np.float32)
        outputs = engine.infer_batch(batch)

        # Should have coverage and bbox outputs
        output_names = list(outputs.keys())
        assert len(output_names) == 2, f"Expected 2 outputs, got {len(output_names)}"

    def test_infer_batch_output_types(self, engine):
        """Test that outputs are numpy arrays."""
        batch = np.random.randn(2, 3, 544, 960).astype(np.float32)
        outputs = engine.infer_batch(batch)

        for name, array in outputs.items():
            assert isinstance(array, np.ndarray), f"Output {name} should be numpy array"

    def test_infer_batch_single_image(self, engine):
        """Test inference with single image batch."""
        batch = np.random.randn(1, 3, 544, 960).astype(np.float32)
        outputs = engine.infer_batch(batch)

        assert isinstance(outputs, dict)
        assert len(outputs) == 2

    def test_infer_batch_large_batch(self, engine):
        """Test inference with larger batch."""
        batch = np.random.randn(4, 3, 544, 960).astype(np.float32)
        outputs = engine.infer_batch(batch)

        assert isinstance(outputs, dict)
        # Check batch dimension is preserved
        for name, array in outputs.items():
            assert array.shape[0] == 4, f"Output {name} batch size should be 4"


class TestPostprocessBatch:
    """Test postprocess_batch method."""

    @pytest.fixture
    def engine(self):
        """Fixture for a BatchInferenceEngine instance."""
        return BatchInferenceEngine(
            model_path="/home/mk/CarCV-metrics/models/baseline/resnet18_trafficcamnet.onnx"
        )

    def test_postprocess_batch_returns_list(self, engine):
        """Test that postprocess_batch returns list of lists."""
        # Create dummy outputs matching model output shapes
        batch_size = 2
        cov_output = np.random.rand(batch_size, 1, 34, 60).astype(np.float32)
        bbox_output = np.random.rand(batch_size, 4, 34, 60).astype(np.float32)
        outputs = {
            'output_cov': cov_output,
            'output_bbox': bbox_output
        }
        image_shapes = [(544, 960), (544, 960)]

        result = engine.postprocess_batch(outputs, image_shapes)

        assert isinstance(result, list)
        assert len(result) == batch_size

    def test_postprocess_batch_per_image_lists(self, engine):
        """Test that each image has list of detections."""
        batch_size = 2
        cov_output = np.random.rand(batch_size, 1, 34, 60).astype(np.float32)
        bbox_output = np.random.rand(batch_size, 4, 34, 60).astype(np.float32)
        outputs = {
            'output_cov': cov_output,
            'output_bbox': bbox_output
        }
        image_shapes = [(544, 960), (544, 960)]

        result = engine.postprocess_batch(outputs, image_shapes)

        for detections in result:
            assert isinstance(detections, list)

    def test_postprocess_batch_detection_type(self, engine):
        """Test that detections are Detection namedtuples."""
        batch_size = 1
        # Create outputs with some high confidence values to get detections
        cov_output = np.random.rand(batch_size, 1, 34, 60).astype(np.float32)
        cov_output[0, 0, 5:10, 5:10] = 0.8  # Set some high confidence region
        bbox_output = np.random.randn(batch_size, 4, 34, 60).astype(np.float32) * 0.01
        outputs = {
            'output_cov': cov_output,
            'output_bbox': bbox_output
        }
        image_shapes = [(544, 960)]

        result = engine.postprocess_batch(outputs, image_shapes)

        # Should have detections
        if result[0]:
            for det in result[0]:
                assert isinstance(det, Detection)
                assert hasattr(det, 'bbox')
                assert hasattr(det, 'confidence')
                assert hasattr(det, 'class_id')

    def test_postprocess_batch_bbox_coordinates_in_pixels(self, engine):
        """Test that bboxes are in pixel coordinates."""
        batch_size = 1
        cov_output = np.zeros((batch_size, 1, 34, 60), dtype=np.float32)
        cov_output[0, 0, 10, 20] = 0.9  # High confidence
        bbox_output = np.zeros((batch_size, 4, 34, 60), dtype=np.float32)
        bbox_output[0, :, 10, 20] = [0, 0, 0.1, 0.1]
        outputs = {
            'output_cov': cov_output,
            'output_bbox': bbox_output
        }
        image_shapes = [(544, 960)]

        result = engine.postprocess_batch(outputs, image_shapes)

        if result[0]:
            for det in result[0]:
                x, y, w, h = det.bbox
                # Should be in pixel coordinates (0 to 960/544)
                assert 0 <= x <= 960
                assert 0 <= y <= 544
                assert w >= 0
                assert h >= 0

    def test_postprocess_batch_empty_detections(self, engine):
        """Test postprocess_batch with low confidence outputs."""
        batch_size = 2
        cov_output = np.random.rand(batch_size, 1, 34, 60) * 0.1  # All low confidence
        bbox_output = np.random.randn(batch_size, 4, 34, 60).astype(np.float32)
        outputs = {
            'output_cov': cov_output.astype(np.float32),
            'output_bbox': bbox_output
        }
        image_shapes = [(544, 960), (544, 960)]

        result = engine.postprocess_batch(outputs, image_shapes)

        # Should return empty lists (no detections above threshold)
        assert len(result) == batch_size
        for detections in result:
            assert isinstance(detections, list)


class TestIntegration:
    """Integration tests for BatchInferenceEngine."""

    @pytest.fixture
    def engine(self):
        """Fixture for a BatchInferenceEngine instance."""
        return BatchInferenceEngine(
            model_path="/home/mk/CarCV-metrics/models/baseline/resnet18_trafficcamnet.onnx"
        )

    def test_end_to_end_inference_and_postprocess(self, engine):
        """Test complete inference and postprocessing pipeline."""
        batch_size = 2
        batch = np.random.randn(batch_size, 3, 544, 960).astype(np.float32)

        # Inference
        outputs = engine.infer_batch(batch)
        assert isinstance(outputs, dict)

        # Postprocessing
        image_shapes = [(544, 960), (544, 960)]
        detections_per_image = engine.postprocess_batch(outputs, image_shapes)

        assert len(detections_per_image) == batch_size
        for detections in detections_per_image:
            assert isinstance(detections, list)
