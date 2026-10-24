#!/bin/bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'Usage: ./run-sharp-multiview.sh REFERENCE.jpg SECOND_VIEW.jpg' >&2
  exit 2
fi
# Resolve paths before changing working directory.
first="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
second="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
cd "$(dirname "$0")"
mkdir -p output/sharp-multiview
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/sharp_baseline.py "$first" | tee output/sharp-multiview/baseline.log
.venv/bin/python scripts/splat_export.py output/sharp-multiview/baseline.pt output/sharp-multiview/sharp-baseline.splat
.venv/bin/python scripts/multiview_depth.py "$first" "$second" | tee output/sharp-multiview/depth.log
.venv/bin/python scripts/sharp_depth_conditioned.py native | tee output/sharp-multiview/conditioned-native.log
.venv/bin/python scripts/build_sharp_viewer.py
echo 'Comparison: http://127.0.0.1:8873/web/sharp-conditioned-comparison.html'
