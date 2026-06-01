"""Чистые конвертеры аннотаций датасетов в формат пайплайна.

Только stdlib — модуль импортируется и тестами (без ML-зависимостей), и
download-скриптами на удалённом сервере. Не импортировать cv2/numpy/onnx.
"""
from __future__ import annotations

from pathlib import Path


def bdd100k_to_labels(native: list) -> list:
    """Нативные BDD100K val-аннотации → формат labels.json пайплайна.

    Вход: список {name, attributes, labels:[{category, box2d:{x1,y1,x2,y2}}]}.
    Лейблы без box2d (полосы/области — только poly2d) отбрасываются.
    Выход: список {image_id, file_name, detections:[{category, box2d}], ...}.
    """
    items = []
    for ex in native:
        name = ex.get("name") or ""
        if not name:
            continue
        detections = []
        for lab in ex.get("labels", []):
            box = lab.get("box2d")
            cat = (lab.get("category") or "").lower()
            if cat and box:
                detections.append({
                    "category": cat,
                    "box2d": {"x1": box["x1"], "y1": box["y1"],
                              "x2": box["x2"], "y2": box["y2"]},
                })
        attrs = ex.get("attributes", {}) or {}
        items.append({
            "image_id": Path(name).stem,
            "file_name": name,
            "detections": detections,
            "weather": attrs.get("weather", ""),
            "timeofday": attrs.get("timeofday", ""),
            "scene": attrs.get("scene", ""),
        })
    return items
