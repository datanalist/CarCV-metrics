#!/bin/bash
# Download validation datasets for qudata2
# Models: TrafficCamNet, VehicleMakeNet, Color
# Datasets: BDD100k val, mad-cars sample

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Downloading datasets for qudata2 ==="

# 1. BDD100k validation set (for TrafficCamNet)
echo "[1/2] BDD100k validation set (10K images, ~695 MB)"
python - <<'EOF'
from datasets import load_dataset
import os, json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

save_dir = Path("data/bdd100k")
images_dir = save_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)
labels_path = save_dir / "labels.json"

if labels_path.exists() and len(list(images_dir.glob("*.jpg"))) > 9000:
    print("BDD100k already downloaded, skipping")
else:
    print("Loading BDD100k from HuggingFace...")
    ds = load_dataset("dgural/bdd100k", split="validation", trust_remote_code=True)

    labels = []
    for i, sample in enumerate(tqdm(ds, desc="Saving BDD100k")):
        img_path = images_dir / f"{sample['image_id']}.jpg"
        if not img_path.exists():
            sample['image'].save(img_path)
        labels.append({
            "image_id": sample['image_id'],
            "file_name": f"{sample['image_id']}.jpg",
            "detections": sample.get('detections', []),
            "weather": sample.get('weather', ''),
            "timeofday": sample.get('timeofday', ''),
        })

    with open(labels_path, 'w') as f:
        json.dump(labels, f)
    print(f"BDD100k saved: {len(labels)} images")
EOF

# 2. mad-cars sample (for VehicleMakeNet + Color)
echo "[2/2] mad-cars sample (5K images for validation)"
python - <<'EOF'
from datasets import load_dataset
import pandas as pd
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import json, random

save_dir = Path("data/mad_cars")
images_dir = save_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)
meta_path = save_dir / "metadata.parquet"
sample_path = save_dir / "sample_5k.json"

SAMPLE_SIZE = 5000

if sample_path.exists() and len(list(images_dir.glob("*.jpg"))) >= SAMPLE_SIZE * 0.9:
    print("mad-cars already downloaded, skipping")
else:
    print("Loading mad-cars metadata from HuggingFace...")
    ds = load_dataset("yandex/mad-cars", split="train", trust_remote_code=True)
    df = ds.to_pandas()

    # Stratified sample: ~35 images per brand
    sample = df.groupby("brand").apply(
        lambda x: x.sample(min(len(x), max(1, SAMPLE_SIZE // df['brand'].nunique())),
                           random_state=42)
    ).reset_index(drop=True)
    sample = sample.sample(min(SAMPLE_SIZE, len(sample)), random_state=42)

    print(f"Downloading {len(sample)} images...")
    records = []
    failed = 0
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        img_name = f"{row['car_id']}_{row['view_id']}.jpg"
        img_path = images_dir / img_name
        if not img_path.exists():
            try:
                resp = requests.get(row['url'], timeout=10)
                img = Image.open(BytesIO(resp.content)).convert('RGB')
                img.save(img_path)
            except Exception:
                failed += 1
                continue
        records.append({
            "file_name": img_name,
            "brand": row.get('brand', ''),
            "model": row.get('model', ''),
            "color": row.get('color', ''),
            "car_id": str(row.get('car_id', '')),
        })

    with open(sample_path, 'w') as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"mad-cars saved: {len(records)} images ({failed} failed)")
EOF

echo ""
echo "=== Dataset download complete for qudata2 ==="
du -sh data/bdd100k data/mad_cars
