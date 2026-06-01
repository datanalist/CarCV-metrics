"""Юнит-тесты чистых конвертеров датасетов (stdlib only, без ML-зависимостей)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy" / "evaluation"))

from dataset_prep import bdd100k_to_labels  # noqa: E402


def test_bdd100k_to_labels_basic():
    native = [
        {
            "name": "0000f77c-6257be58.jpg",
            "attributes": {"weather": "clear", "scene": "city street",
                           "timeofday": "daytime"},
            "labels": [
                {"category": "car", "box2d": {"x1": 45.2, "y1": 254.5,
                                              "x2": 357.8, "y2": 487.9}},
                {"category": "traffic sign", "box2d": {"x1": 1.0, "y1": 2.0,
                                                       "x2": 3.0, "y2": 4.0}},
                # лейбл полосы без box2d (только poly2d) — должен быть отброшен
                {"category": "lane", "poly2d": [{"vertices": [[0, 0]]}]},
            ],
        },
        {"name": "b1c66a42-6f7d68ca.jpg", "attributes": {}, "labels": []},
    ]
    out = bdd100k_to_labels(native)
    assert len(out) == 2
    first = out[0]
    assert first["image_id"] == "0000f77c-6257be58"
    assert first["file_name"] == "0000f77c-6257be58.jpg"
    assert first["weather"] == "clear"
    # только два box2d-лейбла, категории в нижнем регистре
    assert len(first["detections"]) == 2
    assert first["detections"][0]["category"] == "car"
    assert first["detections"][1]["category"] == "traffic sign"
    assert first["detections"][0]["box2d"] == {"x1": 45.2, "y1": 254.5,
                                               "x2": 357.8, "y2": 487.9}
    # второе изображение — без детекций, но присутствует
    assert out[1]["detections"] == []
