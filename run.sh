#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== AI Decision Intelligence Inventory Experiment ==="
echo "Python: $(python --version)"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "--- GPU check (optional) ---"
nvidia-smi || true

echo "--- Running experiment ---"
python experiment.py --config config.yaml

echo "--- Complete ---"
echo "Open outputs/summary.md first."
