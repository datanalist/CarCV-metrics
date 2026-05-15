#!/bin/bash
# qudata2 datasets v3: COCO val2017 for TrafficCamNet + mad-cars for Make/Color
# Replaces BDD100k (blocked from qudata2) with COCO val2017 (publicly accessible)

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Downloading datasets for qudata2 (v3 — COCO + mad-cars) ==="

# 1. COCO val2017 images (~1 GB, 5K images)
echo "[1/3] COCO val2017 images"
mkdir -p data/coco
if [ ! -d data/coco/val2017 ] || [ "$(ls data/coco/val2017 2>/dev/null | wc -l)" -lt 4000 ]; then
    wget -q --show-progress "http://images.cocodataset.org/zips/val2017.zip" \
        -O data/coco/val2017.zip
    echo "Extracting val2017..."
    cd data/coco && unzip -q val2017.zip && cd ../..
    rm -f data/coco/val2017.zip
fi

# 2. COCO annotations
echo "[2/3] COCO val2017 annotations"
if [ ! -f data/coco/annotations/instances_val2017.json ]; then
    wget -q --show-progress "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
        -O data/coco/annotations.zip
    cd data/coco && unzip -q annotations.zip && cd ../..
    rm -f data/coco/annotations.zip
fi

# Convert COCO to our BDD100k-compatible format pointing to coco data
python - <<'EOF'
import json
from pathlib import Path

root = Path("data/coco")
ann_path = root / "annotations/instances_val2017.json"

with open(ann_path) as f:
    coco = json.load(f)

# COCO category IDs: 1=person, 2=bicycle, 3=car (matches TrafficCamNet classes)
cat_map = {1: "person", 2: "bike", 3: "car"}
imgs = {im["id"]: im for im in coco["images"]}
per_img = {im_id: [] for im_id in imgs}

for ann in coco["annotations"]:
    if ann["category_id"] not in cat_map:
        continue
    x, y, w, h = ann["bbox"]
    per_img[ann["image_id"]].append({
        "category": cat_map[ann["category_id"]],
        "box2d": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
    })

# Save to expected location used by evaluate.py for bdd100k path
out_dir = Path("data/bdd100k")
out_dir.mkdir(exist_ok=True)
images_dir = out_dir / "images"
images_dir.mkdir(exist_ok=True)

# Symlink COCO images into bdd100k/images
src_imgs = root / "val2017"
for img_path in src_imgs.glob("*.jpg"):
    link = images_dir / img_path.name
    if not link.exists():
        try:
            link.symlink_to(img_path.resolve())
        except FileExistsError:
            pass

items = []
for im_id, im in imgs.items():
    items.append({
        "image_id": im["file_name"].replace(".jpg", ""),
        "file_name": im["file_name"],
        "detections": per_img[im_id],
        "weather": "",
        "timeofday": "",
    })

with open(out_dir / "labels.json", "w") as f:
    json.dump(items, f)

print(f"COCO→BDD format: {len(items)} images, "
      f"{sum(len(p) for p in per_img.values())} annotations")
EOF

# 3. mad-cars sample
echo "[3/3] mad-cars sample (5K images)"
python - <<'EOF'
from datasets import load_dataset
import pandas as pd
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import json

save_dir = Path("data/mad_cars")
images_dir = save_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)
sample_path = save_dir / "sample_5k.json"

SAMPLE_SIZE = 5000

if sample_path.exists() and len(list(images_dir.glob("*.jpg"))) >= SAMPLE_SIZE * 0.9:
    print("mad-cars already downloaded, skipping")
else:
    print("Loading mad-cars metadata from HuggingFace...")
    ds = load_dataset("yandex/mad-cars", split="train")
    df = ds.to_pandas()

    n_brands = df['brand'].nunique()
    per_brand = max(1, SAMPLE_SIZE // n_brands)
    sample = df.groupby("brand", group_keys=False).apply(
        lambda x: x.sample(min(len(x), per_brand), random_state=42)
    ).reset_index(drop=True)
    sample = sample.sample(min(SAMPLE_SIZE, len(sample)), random_state=42)

    print(f"Downloading {len(sample)} images...")
    records, failed = [], 0
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        img_name = f"{row['car_id']}_{row['view_id']}.jpg"
        img_path = images_dir / img_name
        if not img_path.exists():
            try:
                resp = requests.get(row['url'], timeout=15)
                img = Image.open(BytesIO(resp.content)).convert('RGB')
                img.save(img_path)
            except Exception:
                failed += 1
                continue
        records.append({
            "file_name": img_name,
            "brand": str(row.get('brand', '')),
            "model": str(row.get('model', '')),
            "color": str(row.get('color', '')),
            "car_id": str(row.get('car_id', '')),
        })

    with open(sample_path, 'w') as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"mad-cars saved: {len(records)} images ({failed} failed)")
EOF

echo ""
echo "=== Dataset download v3 complete ==="
du -sh data/coco data/bdd100k data/mad_cars 2>/dev/null
