#!/bin/bash
# Goal 4 datasets for ssh9.qudata.ai (после переноса с ssh1):
#   - COCO val2017 (TrafficCamNet, default; BDD100K val 10K — опционально)
#   - mad-cars sample 5K (VehicleMakeNet)
#   - Stanford Cars test (VehicleTypeNet, через HF mirror)
#
# Запускается на удалённом сервере после `run_remote.py deploy`:
#   bash deploy/scripts/download_datasets_ssh9_goal4.sh

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Goal 4 datasets for ssh9 ==="

# ── 1. TrafficCamNet dataset ─────────────────────────────────────────────────
# Primary: COCO val2017 (open, 5K images, ~1 GB) — переиспользуем существующий формат.
# Опционально: если переменная BDD100K_HF_REPO задана, попытаемся скачать BDD100K val 10K
# из HuggingFace и заменим labels.json.

if [ -n "${BDD100K_HF_REPO:-}" ]; then
    echo "[1/3] BDD100K val 10K (HF: ${BDD100K_HF_REPO}) — TrafficCamNet primary"
    mkdir -p data/bdd100k
    python - <<EOF || echo "BDD100K HF download FAILED — will fall back to COCO"
from datasets import load_dataset
from pathlib import Path
import json

repo = "${BDD100K_HF_REPO}"
print(f"Loading BDD100K val from {repo}...")
ds = load_dataset(repo, split="val")

out_dir = Path("data/bdd100k")
img_dir = out_dir / "images"
img_dir.mkdir(parents=True, exist_ok=True)

items = []
for ex in ds:
    name = ex.get("name") or ex.get("file_name") or ex.get("image_id", "")
    if not name:
        continue
    # Save image if it's a PIL.Image
    if "image" in ex and hasattr(ex["image"], "save"):
        img_path = img_dir / name
        if not img_path.exists():
            ex["image"].save(img_path)
    # Normalize labels into our format
    detections = []
    for lab in ex.get("labels", []):
        cat = (lab.get("category") or "").lower()
        box = lab.get("box2d", {})
        if cat and box:
            detections.append({"category": cat, "box2d": box})
    items.append({
        "image_id": Path(name).stem,
        "file_name": name,
        "detections": detections,
        "weather": ex.get("attributes", {}).get("weather", ""),
        "timeofday": ex.get("attributes", {}).get("timeofday", ""),
        "scene": ex.get("attributes", {}).get("scene", ""),
    })

(out_dir / "labels.json").write_text(json.dumps(items, ensure_ascii=False))
print(f"BDD100K val: {len(items)} images, {sum(len(it['detections']) for it in items)} dets")
EOF
fi

# COCO fallback (always run if BDD100K labels.json is absent)
if [ ! -f data/bdd100k/labels.json ]; then
    echo "[1/3] COCO val2017 (TrafficCamNet primary, fallback)"
    mkdir -p data/coco data/bdd100k/images
    if [ ! -d data/coco/val2017 ] || [ "$(ls data/coco/val2017 2>/dev/null | wc -l)" -lt 4000 ]; then
        wget -q --show-progress "http://images.cocodataset.org/zips/val2017.zip" \
            -O data/coco/val2017.zip
        echo "Extracting val2017..."
        (cd data/coco && unzip -q val2017.zip)
        rm -f data/coco/val2017.zip
    fi
    if [ ! -f data/coco/annotations/instances_val2017.json ]; then
        wget -q --show-progress \
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
            -O data/coco/annotations.zip
        (cd data/coco && unzip -q annotations.zip)
        rm -f data/coco/annotations.zip
    fi

    python - <<'EOF'
import json
from pathlib import Path

root = Path("data/coco")
with open(root / "annotations/instances_val2017.json") as f:
    coco = json.load(f)

# COCO category IDs — map к TrafficCamNet 4 классам (car/person/bicycle/road_sign).
# COCO IDs: 1=person, 2=bicycle, 3=car, 13=stop sign (proxy для road_sign).
cat_map = {1: "person", 2: "bicycle", 3: "car", 13: "traffic sign"}
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

out_dir = Path("data/bdd100k")
out_dir.mkdir(exist_ok=True)
images_dir = out_dir / "images"
images_dir.mkdir(exist_ok=True)

src_imgs = root / "val2017"
for img_path in src_imgs.glob("*.jpg"):
    link = images_dir / img_path.name
    if not link.exists():
        try:
            link.symlink_to(img_path.resolve())
        except FileExistsError:
            pass

items = [{
    "image_id": im["file_name"].replace(".jpg", ""),
    "file_name": im["file_name"],
    "detections": per_img[im_id],
    "weather": "", "timeofday": "", "scene": "",
} for im_id, im in imgs.items()]

(out_dir / "labels.json").write_text(json.dumps(items))
print(f"COCO→TrafficCamNet 4-class: {len(items)} images, "
      f"{sum(len(p) for p in per_img.values())} annotations")
EOF
fi

# ── 2. mad-cars sample 5K (VehicleMakeNet) ───────────────────────────────────
echo "[2/3] mad-cars sample (5K, VehicleMakeNet)"
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
    # Goal 4.1: validate brand field is non-empty (regression check)
    with open(sample_path) as f:
        records = json.load(f)
    empty_brand = sum(1 for r in records if not r.get("brand", "").strip())
    if empty_brand > len(records) * 0.05:
        print(f"WARN: {empty_brand}/{len(records)} records have empty brand — re-downloading")
        sample_path.unlink()
    else:
        print(f"mad-cars OK: {len(records)} records, {empty_brand} empty-brand")
        raise SystemExit

print("Loading mad-cars metadata from HuggingFace...")
ds = load_dataset("yandex/mad-cars", split="train")
df = ds.to_pandas()
assert "brand" in df.columns, f"brand field missing in mad-cars schema; got {list(df.columns)}"

n_brands = df['brand'].nunique()
per_brand = max(1, SAMPLE_SIZE // n_brands)
# pandas 3.0 breaking change: `groupby(K, group_keys=False).apply(...)` DROPS the
# groupby column from the result (became default in 3.0). Explicit per-group loop
# preserves all columns deterministically.
import pandas as pd
parts = []
for brand_val, g in df.groupby("brand", sort=True):
    n = min(len(g), per_brand)
    parts.append(g.sample(n=n, random_state=42))
sample = pd.concat(parts).reset_index(drop=True)
assert "brand" in sample.columns and sample["brand"].notna().all(), \
    f"brand column lost after groupby+sample; cols={list(sample.columns)}"
sample = sample.sample(min(SAMPLE_SIZE, len(sample)), random_state=42).reset_index(drop=True)

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
    brand = str(row.get('brand', '') or '').strip()
    records.append({
        "file_name": img_name,
        "brand": brand,
        "model": str(row.get('model', '') or ''),
        "color": str(row.get('color', '') or ''),
        "car_id": str(row.get('car_id', '')),
    })

with open(sample_path, 'w') as f:
    json.dump(records, f, ensure_ascii=False)
empty = sum(1 for r in records if not r["brand"])
print(f"mad-cars saved: {len(records)} ({failed} failed, {empty} empty-brand)")
assert empty < len(records) * 0.05, f"Too many empty brand values: {empty}"
EOF

# ── 3. Stanford Cars test (VehicleTypeNet via Make→Type mapping) ──────────────
echo "[3/3] Stanford Cars test (VehicleTypeNet)"
python - <<'EOF'
from pathlib import Path
import json
from tqdm import tqdm

save_dir = Path("data/stanford_cars")
images_dir = save_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)
meta_path = save_dir / "test.json"

if meta_path.exists() and len(list(images_dir.glob("*.jpg"))) >= 7000:
    print(f"Stanford Cars OK: {len(list(images_dir.glob('*.jpg')))} images")
    raise SystemExit

# Try multiple HF mirrors (no Kaggle auth needed):
candidates = [
    ("Multimodal-Fatima/StanfordCars_test", "test"),
    ("tanganke/stanford_cars", "test"),
    ("nateraw/stanford-cars", "test"),
]

from datasets import load_dataset
ds = None
for repo, split in candidates:
    try:
        print(f"trying {repo}…")
        ds = load_dataset(repo, split=split)
        print(f"OK from {repo}: {len(ds)} samples; columns: {ds.column_names}")
        break
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {e}")

if ds is None:
    raise SystemExit("All Stanford Cars HF mirrors failed; need manual download")

# Schema varies between mirrors — try common field names
records = []
for i, ex in enumerate(tqdm(ds)):
    img = ex.get("image")
    if img is None or not hasattr(img, "save"):
        continue
    # label may be int (class id) or str
    label = ex.get("label") if "label" in ex else ex.get("class")
    label_str = ex.get("label_text") or ex.get("class_name") or ""
    if not label_str and isinstance(label, int) and "label" in ds.features:
        try:
            label_str = ds.features["label"].names[label]
        except Exception:
            label_str = str(label)
    fname = f"sc_{i:06d}.jpg"
    img_path = images_dir / fname
    if not img_path.exists():
        img.save(img_path)
    records.append({"file_name": fname, "label": label_str})

with open(meta_path, "w") as f:
    json.dump(records, f, ensure_ascii=False)
print(f"Stanford Cars saved: {len(records)} test samples")
EOF

echo ""
echo "=== Goal 4 dataset download complete ==="
du -sh data/bdd100k data/mad_cars data/stanford_cars 2>/dev/null || true
