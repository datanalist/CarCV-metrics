import pytest
import numpy as np
from utils.batch_data_loader import BDD100KBatchLoader


class TestBDD100KBatchLoaderInit:
    """Test BDD100KBatchLoader initialization."""

    @pytest.fixture
    def loader_params(self):
        """Fixture for loader parameters."""
        return {
            "ann_json_path": "/home/mk/Загрузки/DATASETS/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
            "images_dir": "/home/mk/Загрузки/DATASETS/bdd100k/bdd100k/bdd100k/images/100k/val/",
            "category_map": {"car": "car"},
            "batch_size": 4,
            "num_workers": 1,
            "max_images": 20,
        }

    def test_batch_loader_init(self, loader_params):
        """Test initialization of BDD100KBatchLoader."""
        loader = BDD100KBatchLoader(**loader_params)
        assert loader.batch_size == 4
        assert loader.total_images == 20
        assert loader.current_batch_idx == 0

    def test_batch_loader_init_with_large_max_images(self, loader_params):
        """Test initialization with max_images larger than dataset."""
        loader_params["max_images"] = 50000
        loader = BDD100KBatchLoader(**loader_params)
        # Should load actual dataset size (max ~10000)
        assert loader.total_images > 0
        assert loader.current_batch_idx == 0


class TestBDD100KBatchLoaderGetBatch:
    """Test BDD100KBatchLoader get_batch functionality."""

    @pytest.fixture
    def loader(self):
        """Fixture for a loader instance."""
        return BDD100KBatchLoader(
            ann_json_path="/home/mk/Загрузки/DATASETS/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
            images_dir="/home/mk/Загрузки/DATASETS/bdd100k/bdd100k/bdd100k/images/100k/val/",
            category_map={"car": "car"},
            batch_size=4,
            num_workers=1,
            max_images=20,
        )

    def test_get_batch_returns_correct_structure(self, loader):
        """Test that get_batch returns (images, metadata, gt) tuple."""
        batch = loader.get_batch()
        assert batch is not None, "First batch should not be None"
        images, metadata, gt = batch
        assert isinstance(images, np.ndarray), "images should be numpy array"
        assert isinstance(metadata, list), "metadata should be list"
        assert isinstance(gt, list), "gt should be list"

    def test_get_batch_image_shape(self, loader):
        """Test that batch images have correct shape and dtype."""
        batch = loader.get_batch()
        assert batch is not None
        images, metadata, gt = batch

        # Shape should be (batch_size, 3, 544, 960)
        assert len(images.shape) == 4, f"Expected 4D tensor, got {images.shape}"
        assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"
        assert images.shape[2] == 544, f"Expected height 544, got {images.shape[2]}"
        assert images.shape[3] == 960, f"Expected width 960, got {images.shape[3]}"

        # Check dtype is float32
        assert images.dtype == np.float32, f"Expected float32, got {images.dtype}"

        # Batch size should be <= 4
        assert images.shape[0] <= 4, f"Batch size should be <= 4, got {images.shape[0]}"

    def test_get_batch_metadata_alignment(self, loader):
        """Test that metadata length matches batch size."""
        batch = loader.get_batch()
        assert batch is not None
        images, metadata, gt = batch

        assert len(metadata) == images.shape[0], "metadata length should match batch size"
        assert all("image_id" in m for m in metadata), "All metadata should have 'image_id'"
        assert all("filename" in m for m in metadata), "All metadata should have 'filename'"
        assert all("orig_shape" in m for m in metadata), "All metadata should have 'orig_shape'"

    def test_get_batch_gt_alignment(self, loader):
        """Test that gt annotations length matches batch size."""
        batch = loader.get_batch()
        assert batch is not None
        images, metadata, gt = batch

        assert len(gt) == len(metadata), "gt length should match metadata/batch length"
        # Each gt can be empty list (no cars) or list of boxes
        assert all(isinstance(g, list) for g in gt), "Each gt should be a list"

    def test_get_batch_updates_idx(self, loader):
        """Test that get_batch updates current_batch_idx."""
        assert loader.current_batch_idx == 0
        batch1 = loader.get_batch()
        assert batch1 is not None
        idx_after_first = loader.current_batch_idx
        assert idx_after_first > 0, "current_batch_idx should increase after get_batch"

        batch2 = loader.get_batch()
        if batch2 is not None:
            idx_after_second = loader.current_batch_idx
            assert idx_after_second > idx_after_first, "current_batch_idx should keep increasing"

    def test_get_batch_returns_none_when_done(self, loader):
        """Test that get_batch returns None when all batches exhausted."""
        batches = []
        while True:
            batch = loader.get_batch()
            if batch is None:
                break
            batches.append(batch)

        # Should have gotten at least one batch (max_images=20, batch_size=4)
        assert len(batches) >= 1, "Should get at least one batch"
        # Total images should match
        total_images_from_batches = sum(b[0].shape[0] for b in batches)
        assert total_images_from_batches <= 20, "Total should not exceed max_images"


class TestBDD100KBatchLoaderReset:
    """Test BDD100KBatchLoader reset functionality."""

    @pytest.fixture
    def loader(self):
        """Fixture for a loader instance."""
        return BDD100KBatchLoader(
            ann_json_path="/home/mk/Загрузки/DATASETS/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
            images_dir="/home/mk/Загрузки/DATASETS/bdd100k/bdd100k/bdd100k/images/100k/val/",
            category_map={"car": "car"},
            batch_size=4,
            num_workers=1,
            max_images=20,
        )

    def test_reset_functionality(self, loader):
        """Test that reset resets the loader state."""
        # Get first batch
        batch1 = loader.get_batch()
        assert batch1 is not None
        assert loader.current_batch_idx > 0

        # Reset
        loader.reset()
        assert loader.current_batch_idx == 0

        # Should get same first batch again
        batch2 = loader.get_batch()
        assert batch2 is not None
        assert batch1[0].shape == batch2[0].shape


class TestBDD100KBatchLoaderIsDone:
    """Test BDD100KBatchLoader is_done functionality."""

    @pytest.fixture
    def loader(self):
        """Fixture for a loader instance."""
        return BDD100KBatchLoader(
            ann_json_path="/home/mk/Загрузки/DATASETS/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
            images_dir="/home/mk/Загрузки/DATASETS/bdd100k/bdd100k/bdd100k/images/100k/val/",
            category_map={"car": "car"},
            batch_size=4,
            num_workers=1,
            max_images=20,
        )

    def test_is_done_initial_state(self, loader):
        """Test that is_done is False initially."""
        assert loader.is_done() is False

    def test_is_done_after_exhausting(self, loader):
        """Test that is_done is True after exhausting all batches."""
        while not loader.is_done():
            batch = loader.get_batch()
            if batch is None:
                break

        assert loader.is_done() is True

    def test_is_done_after_reset(self, loader):
        """Test that is_done resets after reset()."""
        # Exhaust loader
        while not loader.is_done():
            batch = loader.get_batch()
            if batch is None:
                break

        assert loader.is_done() is True

        # Reset
        loader.reset()
        assert loader.is_done() is False
