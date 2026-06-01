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


def vmmrdb_make(dirname: str) -> str:
    """Имя каталога VMMRdb '<make>_<model>_<year>' → марка (первый токен, lower)."""
    return dirname.split("_")[0].lower().strip()


def iter_vmmrdb_samples(root: Path, per_class_cap: int) -> list:
    """Обойти каталоги-по-классам VMMRdb, вернуть детерминированный список
    (image_path, make) с не более чем per_class_cap изображений на каталог.

    Сортировка по имени каталога и по имени файла → воспроизводимый отбор.
    """
    root = Path(root)
    samples = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        make = vmmrdb_make(class_dir.name)
        imgs = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for img in imgs[:per_class_cap]:
            samples.append((img, make))
    return samples


def widerface_faces_to_detections(faces: dict) -> list:
    """WIDER FACE faces-словарь HF → detections в формате пайплайна.

    HF-схема: faces = {"bbox": [[x, y, w, h], ...], ...}. Боксы с нулевой/
    отрицательной шириной или высотой отбрасываются (в WIDER FACE есть
    вырожденные аннотации «invalid»).
    """
    dets = []
    for bbox in faces.get("bbox", []):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        if w <= 0 or h <= 0:
            continue
        dets.append({
            "category": "face",
            "box2d": {"x1": float(x), "y1": float(y),
                      "x2": float(x + w), "y2": float(y + h)},
        })
    return dets
