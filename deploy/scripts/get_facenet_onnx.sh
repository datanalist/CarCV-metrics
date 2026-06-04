#!/usr/bin/env bash
# Попытка получить deployable FaceNet ONNX. 3 шага; при неудаче — UNDEF.
set -u
OUT=/home/mk/CarCV-metrics/data/prep/facedetect
mkdir -p "$OUT"
ETLT=/home/mk/Загрузки/facenet_pruned_quantized_v2.0.1/model.etlt

URL="https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/deployable_v1.0/files/model.onnx"
echo "[1/3] NGC deployable ONNX…"
wget -q "$URL" -O "$OUT/facenet.onnx" && [ -s "$OUT/facenet.onnx" ] && { echo "OK NGC"; exit 0; }
rm -f "$OUT/facenet.onnx"

echo "[2/3] etlt→onnx экспорт…"
for KEY in tlt_encode nvidia_tlt; do
  if command -v tao_converter >/dev/null 2>&1; then
    tao_converter -k "$KEY" -e "$OUT/facenet.onnx" "$ETLT" 2>/dev/null && [ -s "$OUT/facenet.onnx" ] && { echo "OK export ($KEY)"; exit 0; }
  fi
done

echo "[3/3] FAILED — FaceNet ONNX недоступен → UNDEF"
exit 1
