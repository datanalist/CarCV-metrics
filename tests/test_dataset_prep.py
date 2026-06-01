"""Юнит-тесты чистых конвертеров датасетов (stdlib only, без ML-зависимостей)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy" / "evaluation"))

from dataset_prep import bdd100k_to_labels  # noqa: E402
from dataset_prep import vmmrdb_make, iter_vmmrdb_samples  # noqa: E402


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


def test_vmmrdb_make_first_token_lowercased():
    assert vmmrdb_make("Honda_Accord_2003") == "honda"
    assert vmmrdb_make("BMW_3_Series_2010") == "bmw"
    # многословные марки сворачиваются к первому токену (normalize_brand добьёт)
    assert vmmrdb_make("Mercedes_Benz_C_Class_2008") == "mercedes"


def test_iter_vmmrdb_samples_caps_per_class(tmp_path):
    # каталоги-по-классам с изображениями
    honda = tmp_path / "Honda_Civic_2005"
    honda.mkdir()
    for i in range(5):
        (honda / f"img{i}.jpg").write_bytes(b"x")
    bmw = tmp_path / "BMW_X5_2012"
    bmw.mkdir()
    (bmw / "a.jpg").write_bytes(b"x")

    samples = iter_vmmrdb_samples(tmp_path, per_class_cap=3)
    # Honda обрезана до 3, BMW — 1 → всего 4 пары
    assert len(samples) == 4
    makes = sorted({make for _, make in samples})
    assert makes == ["bmw", "honda"]
    # детерминированный отбор: одни и те же файлы при повторе
    assert iter_vmmrdb_samples(tmp_path, per_class_cap=3) == samples
    # элементы — (Path, make)
    p, m = samples[0]
    assert isinstance(p, Path) and isinstance(m, str)
