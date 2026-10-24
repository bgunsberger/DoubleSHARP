#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# == 0 ]]; then
  set -- examples/tree/2026_10_L1101215.jpg examples/tree/2026_10_L1101219.jpg
fi
exec ./run-sharp-multiview.sh "$@"
