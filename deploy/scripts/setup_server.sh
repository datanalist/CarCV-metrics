#!/bin/bash
# Setup script for CARS evaluation server
# Run once on each server after rsync

set -e

echo "=== CARS Evaluation Server Setup ==="
echo "Server: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Install uv
if ! command -v uv &>/dev/null; then
    echo "[1/4] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
else
    echo "[1/4] uv already installed: $(uv --version)"
fi

export PATH="$HOME/.local/bin:$PATH"

# Create virtual environment and install deps (idempotent)
# NOTE: --system-site-packages is required so the venv can see the host's
# pre-installed torch (provided by NVIDIA-driver host setup). Changing this
# flag later requires a manual `rm -rf venv` on each host — the reuse branch
# below never re-creates an existing venv.
echo "[2/4] Creating Python environment..."
cd "$(dirname "$0")/.."
if [ -d venv ]; then
    echo "    venv already exists, reusing"
else
    uv venv --python 3.12 --system-site-packages venv
fi
source venv/bin/activate
uv pip install -r requirements.txt

# nomeroff-net declares modelhub-client as a dependency in its own
# requirements.txt (via git URL) but NOT in its PyPI METADATA, so
# `uv pip install nomeroff-net` alone leaves it missing.  Install separately.
# PyTurboJPEG is likewise imported at module init but not in PyPI metadata.
# Also, nomeroff-net pulls in opencv-python (GUI) which conflicts with the
# headless variant we need on headless servers; force reinstall headless.
echo "    Installing nomeroff-net git dependencies..."
uv pip install "git+https://github.com/ria-com/modelhub-client.git" --quiet \
    || echo "    WARNING: modelhub-client install failed (nomeroff-net may not work)"
uv pip install PyTurboJPEG --quiet \
    || echo "    WARNING: PyTurboJPEG install failed (nomeroff-net may not work)"
uv pip install "opencv-python-headless>=4.8" --reinstall --quiet

# Create data directories
echo "[3/4] Creating directories..."
mkdir -p models/{trafficcamnet,vehiclemakenet,vehicletypenet,lpdnet,lprnet,color}
mkdir -p data/{bdd100k,mad_cars,nomeroff_lp,nomeroff_ocr_ru,bit_vehicle}
mkdir -p results/{trafficcamnet,vehiclemakenet,vehicletypenet,lpdnet,lprnet,color}
mkdir -p plots logs

# Verify GPU access
echo "[4/4] Verifying GPU..."
python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())"

echo ""
echo "=== Setup complete ==="
echo "Next: run download_models.sh then download_datasets.sh"
