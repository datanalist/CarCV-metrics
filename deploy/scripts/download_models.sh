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
echo "[1/6] TrafficCamNet pruned_onnx_v1.0.4"
DIR="$MODELS_DIR/trafficcamnet"
V="pruned_onnx_v1.0.4"
download_file "$BASE_URL/trafficcamnet/versions/$V/files/resnet18_trafficcamnet_pruned.onnx" \
    "$DIR/resnet18_trafficcamnet_pruned.onnx" "trafficcamnet.onnx"
download_file "$BASE_URL/trafficcamnet/versions/$V/files/labels.txt" \
    "$DIR/labels.txt" "trafficcamnet labels"

# 2. VehicleMakeNet (20 car makes)
echo "[2/6] VehicleMakeNet pruned_onnx_v1.1.0"
DIR="$MODELS_DIR/vehiclemakenet"
V="pruned_onnx_v1.1.0"
download_file "$BASE_URL/vehiclemakenet/versions/$V/files/resnet18_pruned.onnx" \
    "$DIR/resnet18_pruned.onnx" "vehiclemakenet.onnx"
download_file "$BASE_URL/vehiclemakenet/versions/$V/files/labels.txt" \
    "$DIR/labels.txt" "vehiclemakenet labels"

# 3. VehicleTypeNet (6 body types)
echo "[3/6] VehicleTypeNet pruned_onnx_v1.1.0"
DIR="$MODELS_DIR/vehicletypenet"
V="pruned_onnx_v1.1.0"
download_file "$BASE_URL/vehicletypenet/versions/$V/files/resnet18_pruned.onnx" \
    "$DIR/resnet18_pruned.onnx" "vehicletypenet.onnx"
download_file "$BASE_URL/vehicletypenet/versions/$V/files/labels.txt" \
    "$DIR/labels.txt" "vehicletypenet labels"

# 4. LPDNet (license plate detector, USA model)
echo "[4/6] LPDNet pruned_v2.2.1 (USA)"
DIR="$MODELS_DIR/lpdnet"
V="pruned_v2.2.1"
download_file "$BASE_URL/lpdnet/versions/$V/files/LPDNet_usa_pruned_tao5.onnx" \
    "$DIR/LPDNet_usa_pruned_tao5.onnx" "lpdnet.onnx"

# 5. LPRNet (license plate OCR, US plates)
echo "[5/6] LPRNet deployable_onnx_v1.1 (US)"
DIR="$MODELS_DIR/lprnet"
V="deployable_onnx_v1.1"
download_file "$BASE_URL/lprnet/versions/$V/files/us_lprnet_baseline18_deployable.onnx" \
    "$DIR/us_lprnet_baseline18_deployable.onnx" "lprnet.onnx"

# 6. FaceNet (face detector, DetectNet_v2, 1 класс)
echo "[6/6] FaceNet — поиск ONNX-файла на NGC"
DIR="$MODELS_DIR/facenet"
mkdir -p "$DIR"
if [ -f "$DIR/facenet.onnx" ]; then
    echo "  [skip] facenet.onnx уже есть"
else
    # У FaceNet ONNX может лежать под разными версиями. Перебираем кандидатов,
    # для каждой запрашиваем список файлов и ищем первый .onnx.
    FOUND=""
    for V in deployable_v1.0 pruned_quantized_v2.0.1 pruned_v2.0.1 deployable_onnx_v1.0; do
        FILES_JSON="$(wget -qO- "$BASE_URL/facenet/versions/$V/files" || true)"
        ONNX_NAME="$(echo "$FILES_JSON" | grep -oE '"[^"/]+\.onnx"' | head -1 | tr -d '"')"
        if [ -n "$ONNX_NAME" ]; then
            echo "  нашёл $ONNX_NAME в версии $V"
            download_file "$BASE_URL/facenet/versions/$V/files/$ONNX_NAME" \
                "$DIR/facenet.onnx" "facenet.onnx"
            FOUND="yes"
            break
        fi
    done
    if [ -z "$FOUND" ]; then
        echo "  ОШИБКА: ONNX FaceNet не найден ни в одной из проверенных версий NGC."
        echo "  Выполнить вручную: wget -qO- '$BASE_URL/facenet/versions/<V>/files'"
        echo "  чтобы найти версию с .onnx, и дописать её в список кандидатов."
        exit 1
    fi
fi

echo ""
echo "=== Models downloaded ==="
du -sh "$MODELS_DIR"/*/
