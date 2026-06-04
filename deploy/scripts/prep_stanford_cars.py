#!/usr/bin/env python3
"""Stanford Cars devkit (.mat) → test.json + симлинк images/.

test.json: [{file_name, label:"<Make> <Model> <BodyType> <Year>"}].
Тип кузова выводит сам eval_vehicletypenet (derive_typenet_label).
По умолчанию используем TRAIN split (с метками) — у официального test нет меток.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.io import loadmat


def parse_class_names(meta_mat: Path) -> list[str]:
    m = loadmat(str(meta_mat))
    raw = m["class_names"][0]
    return [str(x[0]) for x in raw]


def parse_annos(annos_mat: Path) -> list[tuple[str, int]]:
    """Возвращает [(fname, class_1based)] из cars_*_annos.mat."""
    m = loadmat(str(annos_mat))
    ann = m["annotations"][0]
    out = []
    for rec in ann:
        cls = int(rec["class"][0][0]) if "class" in rec.dtype.names else int(rec[-2][0][0])
        fname = str(rec["fname"][0]) if "fname" in rec.dtype.names else str(rec[-1][0])
        out.append((fname, cls))
    return out


def build_stanford_records(class_names: list[str],
                           annos: list[tuple[str, int]]) -> list[dict]:
    return [{"file_name": fn, "label": class_names[cls - 1]} for fn, cls in annos]


def link_images(out_dir: Path, images_dir: Path) -> None:
    link = out_dir / "images"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(images_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-mat", required=True, type=Path)
    ap.add_argument("--annos-mat", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    names = parse_class_names(args.meta_mat)
    annos = parse_annos(args.annos_mat)
    recs = build_stanford_records(names, annos)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "test.json").write_text(json.dumps(recs))
    link_images(args.out_dir, args.images_dir)
    print(f"test.json: {len(recs)} записей · {len(names)} классов · images → {args.images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
