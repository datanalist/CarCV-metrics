#!/bin/bash
# Download all CARS evaluation models from NVIDIA NGC (no API key required)
# Models are in ONNX format, compatible with onnxruntime

set -e

BASE_URL="https://api.ngc.nvidia.com/v2/models/nvidia/tao"
MODELS_DIR="$(dirname "$0")/../models"

download_file() {
    local url="$1"
    local dest="$2"
    local name="$3"
    if [ -f "$dest" ]; then
        echo "  [skip] $name already exists"
        return
    fi
    echo "  Downloading $name..."
    wget -q --show-progress -L "$url" -O "$dest"
    echo "  Done: $(du -sh "$dest" | cut -f1)"
}

echo "=== Downloading CARS models from NGC ==="

# 1. TrafficCamNet (vehicle detector, 4 classes)
echo "[1/5] TrafficCamNet pruned_onnx_v1.0.4"
DIR="$MODELS_DIR/trafficcamnet"
V="pruned_onnx_v1.0.4"
download_file "$BASE_URL/trafficcamnet/versions/$V/files/resnet18_trafficcamnet_pruned.onnx" \
    "$DIR/resnet18_trafficcamnet_pruned.onnx" "trafficcamnet.onnx"
download_file "$BASE_URL/trafficcamnet/versions/$V/files/labels.txt" \
    "$DIR/labels.txt" "trafficcamnet labels"

# 2. VehicleMakeNet (20 car makes)
echo "[2/5] VehicleMakeNet pruned_onnx_v1.1.0"
DIR="$MODELS_DIR/vehiclemakenet"
V="pruned_onnx_v1.1.0"
download_file "$BASE_URL/vehiclemakenet/versions/$V/files/resnet18_pruned.onnx" \
    "$DIR/resnet18_pruned.onnx" "vehiclemakenet.onnx"
download_file "$BASE_URL/vehiclemakenet/versions/$V/files/labels.txt" \
    "$DIR/labels.txt" "vehiclemakenet labels"

# 3. VehicleTypeNet (6 body types)
echo "[3/5] VehicleTypeNet pruned_onnx_v1.1.0"
DIR="$MODELS_DIR/vehicletypenet"
V="pruned_onnx_v1.1.0"
download_file "$BASE_URL/vehicletypenet/versions/$V/files/resnet18_pruned.onnx" \
    "$DIR/resnet18_pruned.onnx" "vehicletypenet.onnx"
download_file "$BASE_URL/vehicletypenet/versions/$V/files/labels.txt" \
    "$DIR/labels.txt" "vehicletypenet labels"

# 4. LPDNet (license plate detector, USA model)
echo "[4/5] LPDNet pruned_v2.2.1 (USA)"
DIR="$MODELS_DIR/lpdnet"
V="pruned_v2.2.1"
download_file "$BASE_URL/lpdnet/versions/$V/files/LPDNet_usa_pruned_tao5.onnx" \
    "$DIR/LPDNet_usa_pruned_tao5.onnx" "lpdnet.onnx"

# 5. LPRNet (license plate OCR, US plates)
echo "[5/5] LPRNet deployable_onnx_v1.1 (US)"
DIR="$MODELS_DIR/lprnet"
V="deployable_onnx_v1.1"
download_file "$BASE_URL/lprnet/versions/$V/files/us_lprnet_baseline18_deployable.onnx" \
    "$DIR/us_lprnet_baseline18_deployable.onnx" "lprnet.onnx"

echo ""
echo "=== Models downloaded ==="
du -sh "$MODELS_DIR"/*/
