#!/bin/bash
# Alternative download for qudata2 — uses ETHZ direct for BDD100k (no HF rate limit)
# and downloads mad-cars sample independently.

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Downloading datasets for qudata2 (v2 — direct sources) ==="

# 1. BDD100k validation images via ETHZ direct (~1.0 GB, ~10K images)
echo "[1/3] BDD100k val images via ETHZ"
mkdir -p data/bdd100k
if [ ! -f data/bdd100k/bdd100k_images_100k_val.zip ] && [ ! -d data/bdd100k/bdd100k/images/100k/val ]; then
    wget --no-check-certificate -q --show-progress \
        "https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip" \
        -O data/bdd100k/bdd100k_images_100k_val.zip
    echo "Extracting BDD100k val images..."
    cd data/bdd100k && unzip -q bdd100k_images_100k_val.zip && cd ../..
fi

# 2. BDD100k detection labels (~50 MB)
echo "[2/3] BDD100k detection labels"
if [ ! -f data/bdd100k/bdd100k_det_20_labels_trainval.json ]; then
    wget --no-check-certificate -q --show-progress \
        "https://dl.cv.ethz.ch/bdd100k/data/bdd100k_det_20_labels_trainval.zip" \
        -O data/bdd100k/labels.zip
    cd data/bdd100k && unzip -q labels.zip && cd ../..
fi

# Normalize layout: copy val labels to expected path
python - <<'EOF'
import json, os
from pathlib import Path

root = Path("data/bdd100k")
labels_path = root / "labels.json"

# Find the detection labels
candidates = list(root.rglob("det_val.json")) + list(root.rglob("bdd100k_det_*_val.json"))
if candidates:
    src = candidates[0]
    with open(src) as f:
        data = json.load(f)
    # Convert to our expected format
    items = []
    for item in data:
        items.append({
            "image_id": item.get("name", "").replace(".jpg", ""),
            "file_name": item.get("name", ""),
            "detections": [
                {
                    "category": l.get("category", ""),
                    "box2d": l.get("box2d", {})
                }
                for l in item.get("labels", []) if "box2d" in l
            ],
            "weather": item.get("attributes", {}).get("weather", ""),
            "timeofday": item.get("attributes", {}).get("timeofday", ""),
        })
    with open(labels_path, "w") as f:
        json.dump(items, f)
    print(f"Saved {len(items)} BDD100k labels")
else:
    print("WARN: no detection labels found")

# Link val images to expected path
val_imgs = list(root.rglob("100k/val/*.jpg"))
if val_imgs:
    target = root / "images"
    target.mkdir(exist_ok=True)
    src_dir = val_imgs[0].parent
    for img in val_imgs[:11000]:
        link = target / img.name
        if not link.exists():
            try:
                link.symlink_to(img.resolve())
            except FileExistsError:
                pass
    print(f"Linked {len(val_imgs)} val images to data/bdd100k/images/")
EOF

# 3. mad-cars sample (for VehicleMakeNet + Color)
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
echo "=== Dataset download v2 complete ==="
du -sh data/bdd100k data/mad_cars
