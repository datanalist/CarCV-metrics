#!/usr/bin/env python3
"""BDD100K val JSON → labels.json формата пайплайна + симлинк images/.

Формат на выходе: [{file_name, image_id, detections:[{category, bbox2d:{x1,y1,x2,y2}}]}].
Маппинг BDD→TrafficCamNet делает сам eval_trafficcamnet через TRAFFICCAMNET_GT_VOCAB —
здесь сохраняем сырые BDD-категории, только приводя box2d→bbox2d.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_bdd_to_labels(bdd_records: list) -> list:
    out = []
    for rec in bdd_records:
        dets = []
        for lab in rec.get("labels", []):
            box = lab.get("box2d")
            if not box:
                continue
            dets.append({
                "category": lab.get("category", ""),
                "bbox2d": {"x1": box["x1"], "y1": box["y1"],
                           "x2": box["x2"], "y2": box["y2"]},
            })
        out.append({"file_name": rec["name"], "image_id": rec["name"],
                    "detections": dets})
    return out


def link_images(out_dir: Path, images_dir: Path) -> None:
    link = out_dir / "images"
    if link.is_symlink():          # снять прежний/битый симлинк, чтобы пере-указать
        link.unlink()
    link.symlink_to(images_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd-json", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    records = json.loads(args.bdd_json.read_text())
    labels = convert_bdd_to_labels(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "labels.json").write_text(json.dumps(labels))

    link_images(args.out_dir, args.images_dir)
    print(f"labels.json: {len(labels)} записей · images → {args.images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
