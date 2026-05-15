#!/bin/bash
# Download validation datasets for qudata5
# Models: VehicleTypeNet, LPDNet, LPRNet
# Datasets: BIT-Vehicle, nomeroff RU OCR, nomeroff detection subset

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=== Downloading datasets for qudata5 ==="

# 1. nomeroff OCR RU — for LPRNet evaluation
echo "[1/3] nomeroff OCR RU (~1.4 GB)"
if [ ! -f "data/nomeroff_ocr_ru/autoriaNumberplateOcrRu-2021-09-01.zip" ]; then
    mkdir -p data/nomeroff_ocr_ru
    wget -q --show-progress \
        "https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2021-09-01.zip" \
        -O "data/nomeroff_ocr_ru/autoriaNumberplateOcrRu-2021-09-01.zip"
    echo "Extracting..."
    unzip -q "data/nomeroff_ocr_ru/autoriaNumberplateOcrRu-2021-09-01.zip" \
        -d "data/nomeroff_ocr_ru/"
    echo "nomeroff OCR RU ready"
else
    echo "nomeroff OCR RU already exists, skipping"
fi

# 2. nomeroff detection subset — for LPDNet evaluation (2GB subset via partial extract)
echo "[2/3] nomeroff LP detection dataset (subset)"
if [ ! -d "data/nomeroff_lp/images" ] || [ "$(find data/nomeroff_lp/images -name '*.jpg' | wc -l)" -lt 1000 ]; then
    mkdir -p data/nomeroff_lp
    # Download a smaller, older version suitable for validation
    wget -q --show-progress \
        "https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2019-02-19.zip" \
        -O "data/nomeroff_lp/autoriaNumberplateDataset-2019.zip" 2>/dev/null || \
    wget -q --show-progress \
        "https://nomeroff.net.ua/datasets/autoriaNumberplateDataset-2018-11-20.zip" \
        -O "data/nomeroff_lp/autoriaNumberplateDataset-2019.zip"
    echo "Extracting..."
    unzip -q "data/nomeroff_lp/autoriaNumberplateDataset-2019.zip" -d "data/nomeroff_lp/"
    echo "nomeroff LP detection ready"
else
    echo "nomeroff LP detection already exists, skipping"
fi

# 3. BIT-Vehicle — for VehicleTypeNet evaluation (6 classes)
echo "[3/3] BIT-Vehicle dataset (~500 MB)"
if [ ! -d "data/bit_vehicle/images" ] || [ "$(find data/bit_vehicle/images -name '*.jpg' | wc -l)" -lt 9000 ]; then
    mkdir -p data/bit_vehicle
    python - <<'EOF'
import kaggle, zipfile, os
from pathlib import Path

dest = Path("data/bit_vehicle")

# Download via kaggle API
try:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "kuanghangdong/bitvehicle",
        path=str(dest),
        unzip=True,
        quiet=False
    )
    print("BIT-Vehicle downloaded via Kaggle API")
except Exception as e:
    print(f"Kaggle API failed: {e}")
    print("Trying alternative download...")
    import subprocess
    result = subprocess.run([
        "wget", "-q", "--show-progress",
        "https://github.com/Curig7/VehicleType/raw/master/BIT-Vehicle.zip",
        "-O", str(dest / "BIT-Vehicle.zip")
    ])
    if result.returncode == 0:
        import zipfile
        with zipfile.ZipFile(dest / "BIT-Vehicle.zip") as z:
            z.extractall(dest)
        print("BIT-Vehicle extracted")
    else:
        print("WARNING: BIT-Vehicle download failed, will use mad-cars fallback for type eval")
EOF
else
    echo "BIT-Vehicle already exists, skipping"
fi

echo ""
echo "=== Dataset download complete for qudata5 ==="
du -sh data/nomeroff_ocr_ru data/nomeroff_lp data/bit_vehicle 2>/dev/null
