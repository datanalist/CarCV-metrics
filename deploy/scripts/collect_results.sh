#!/bin/bash
# Collect results from both servers to local machine
# Run from local machine after evaluation completes

set -e

LOCAL_DIR="$(dirname "$0")/../../results_collected"
mkdir -p "$LOCAL_DIR"

echo "=== Collecting results from qudata2 ==="
rsync -avz --progress qudata2:~/cars-eval/results/ "$LOCAL_DIR/qudata2/"
rsync -avz --progress qudata2:~/cars-eval/plots/ "$LOCAL_DIR/qudata2_plots/"

echo ""
echo "=== Collecting results from qudata5 ==="
rsync -avz --progress qudata5:~/cars-eval/results/ "$LOCAL_DIR/qudata5/"
rsync -avz --progress qudata5:~/cars-eval/plots/ "$LOCAL_DIR/qudata5_plots/"

echo ""
echo "=== Generating combined report ==="
python3 - <<EOF
import json, glob
from pathlib import Path

collected = Path("$LOCAL_DIR")
lines = [
    "# CARS Evaluation — Combined Report",
    "",
    "| Server | Model | Key Metric | Value | Target | Status |",
    "|--------|-------|-----------|-------|--------|--------|",
]

for server in ["qudata2", "qudata5"]:
    for metrics_file in sorted((collected / server).glob("*/metrics.json")):
        model = metrics_file.parent.name
        with open(metrics_file) as f:
            data = json.load(f)
        for metric, info in data.get("thresholds", {}).items():
            status = "✅" if info["status"] == "PASS" else "❌"
            lines.append(
                f"| {server} | {model} | {metric} | {info['value']:.4f} "
                f"| ≥{info['threshold']} | {status} |"
            )

(collected / "COMBINED_REPORT.md").write_text("\\n".join(lines))
print("Combined report saved to", collected / "COMBINED_REPORT.md")
EOF

echo ""
echo "Results collected to: $LOCAL_DIR"
ls -la "$LOCAL_DIR"
