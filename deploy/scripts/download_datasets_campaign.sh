#!/bin/bash
# Загрузка и конвертация датасетов валидационной кампании 2026-06:
#   - BDD100K val 10K       → data/bdd100k   (TrafficCamNet)
#   - VMMRdb (NGC-20 марки)  → data/vmmrdb    (VehicleMakeNet)   [Task B*]
#   - WIDER FACE val         → data/widerface (FaceDetect)       [Task C*]
#
# Идемпотентен: повторный запуск пропускает уже готовые датасеты.
# Запуск (на сервере, после `run_remote.py deploy`):
#   cd ~/cars-eval && source venv/bin/activate
#   bash scripts/download_datasets_campaign.sh [bdd100k|vmmrdb|widerface|all]
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

WHICH="${1:-all}"

# Nextcloud публичная шара: download-API возвращает zip по path+files.
NC_BASE="https://next.ogoun.name/index.php/s/6T4zxG4eek3AdGG/download?path=/DATASETS"

# ── BDD100K ──────────────────────────────────────────────────────────────────
if [ "$WHICH" = "bdd100k" ] || [ "$WHICH" = "all" ]; then
  echo "=== [BDD100K] TrafficCamNet ==="
  if [ -f data/bdd100k/labels.json ] && \
     [ "$(ls data/bdd100k/images 2>/dev/null | wc -l)" -ge 9000 ]; then
    echo "  [skip] data/bdd100k уже готов"
  else
    mkdir -p data/bdd100k_raw data/bdd100k/images
    if [ ! -f data/bdd100k_raw/bdd100k.zip ]; then
      echo "  скачиваю bdd100k.zip с Nextcloud…"
      wget -q --show-progress "${NC_BASE}&files=bdd100k.zip" -O data/bdd100k_raw/bdd100k.zip
    fi
    echo "  распаковка…"
    unzip -q -o data/bdd100k_raw/bdd100k.zip -d data/bdd100k_raw
    python - <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "evaluation")
from dataset_prep import bdd100k_to_labels

raw = Path("data/bdd100k_raw")
# Нативные val-аннотации (имя стабильно во всех раскладках BDD100K).
ann = next(raw.glob("**/bdd100k_labels_images_val.json"), None)
assert ann is not None, f"bdd100k_labels_images_val.json не найден в {raw} (см. unzip -l)"
native = json.loads(ann.read_text())
items = bdd100k_to_labels(native)

# Val-изображения: каталог .../images/100k/val
val_img_dir = next((p for p in raw.glob("**/images/100k/val") if p.is_dir()), None)
assert val_img_dir is not None, f"каталог images/100k/val не найден в {raw}"

out_img = Path("data/bdd100k/images")
out_img.mkdir(parents=True, exist_ok=True)
linked = 0
present = []
for it in items:
    src = val_img_dir / it["file_name"]
    if not src.exists():
        continue
    dst = out_img / it["file_name"]
    if not dst.exists():
        dst.symlink_to(src.resolve())
    linked += 1
    present.append(it)

Path("data/bdd100k/labels.json").write_text(json.dumps(present))
print(f"BDD100K val: {len(present)} изображений с аннотациями, "
      f"{sum(len(i['detections']) for i in present)} детекций (слинковано {linked})")
EOF
  fi
  echo "  готово: $(du -sh data/bdd100k 2>/dev/null | cut -f1)"
fi

# ── VMMRdb ───────────────────────────────────────────────────────────────────
if [ "$WHICH" = "vmmrdb" ] || [ "$WHICH" = "all" ]; then
  echo "=== [VMMRdb] VehicleMakeNet ==="
  if [ -f data/vmmrdb/manifest.json ] && \
     [ "$(ls data/vmmrdb/images 2>/dev/null | wc -l)" -ge 1000 ]; then
    echo "  [skip] data/vmmrdb уже готов"
  else
    mkdir -p data/vmmrdb_raw data/vmmrdb/images
    if [ ! -f data/vmmrdb_raw/VMMRdb.zip ]; then
      echo "  скачиваю VMMRdb.zip с Nextcloud…"
      wget -q --show-progress "${NC_BASE}&files=VMMRdb.zip" -O data/vmmrdb_raw/VMMRdb.zip
    fi
    echo "  распаковка…"
    unzip -q -o data/vmmrdb_raw/VMMRdb.zip -d data/vmmrdb_raw
    PER_CLASS_CAP="${VMMRDB_PER_CLASS_CAP:-50}" python - <<'EOF'
import json, os, sys
from pathlib import Path
sys.path.insert(0, "evaluation")
from dataset_prep import iter_vmmrdb_samples

raw = Path("data/vmmrdb_raw")
# Корень с каталогами-по-классам: каталог, где больше всего подкаталогов
# вида make_model_year.
candidates = [raw] + [p for p in raw.iterdir() if p.is_dir()]
def n_subdirs(d):
    return sum(1 for x in d.iterdir() if x.is_dir()) if d.is_dir() else 0
root = max(candidates, key=n_subdirs)
print(f"VMMRdb корень классов: {root}")

cap = int(os.environ["PER_CLASS_CAP"])
samples = iter_vmmrdb_samples(root, per_class_cap=cap)

# Импортируем правила марок из эвалуатора, чтобы заранее отсеять
# out-of-distribution (не из 20 NGC-марок) — экономит копирование. Эвалуатор
# всё равно повторно отфильтрует через normalize_brand.
from evaluate import normalize_brand, NGC_MAKES_LOWER

out_img = Path("data/vmmrdb/images")
out_img.mkdir(parents=True, exist_ok=True)
records, kept, skipped_ood = [], 0, 0
for img_path, make in samples:
    if normalize_brand(make) not in NGC_MAKES_LOWER:
        skipped_ood += 1
        continue
    fname = f"{kept:06d}_{img_path.name}"
    dst = out_img / fname
    if not dst.exists():
        dst.symlink_to(img_path.resolve())
    records.append({"file_name": fname, "brand": make})
    kept += 1

Path("data/vmmrdb/manifest.json").write_text(json.dumps(records, ensure_ascii=False))
print(f"VMMRdb manifest: {kept} изображений в 20 NGC-марках "
      f"({skipped_ood} отфильтровано как out-of-distribution, cap={cap}/класс)")
EOF
  fi
  echo "  готово: $(du -sh data/vmmrdb 2>/dev/null | cut -f1)"
fi
