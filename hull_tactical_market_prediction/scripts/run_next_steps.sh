#!/bin/bash
set -euo pipefail

echo "=== Hull Tactical Market Prediction Pipeline ==="

echo "1. Creating validation split..."
PYTHONPATH=. python3 -m competitions.hull_tactical_market_prediction.src.experimental.create_validation_split "$@"

echo "2. Running two-phase Perceiver..."
PYTHONPATH=. python3 -m competitions.hull_tactical_market_prediction.src.experimental.two_phase_perceiver "$@"

echo "3. Staging TSMixer dataset..."
echo "   (Provide --tsmixer-root when rerunning this script to stage files and follow the printed instructions.)"
