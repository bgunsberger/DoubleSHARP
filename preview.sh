#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
port="${PORT:-8873}"
echo "Viewer: http://127.0.0.1:$port/vendor/free-splatter.cpp/web/index.html?splat=/output/three-views.splat"
exec python3 -m http.server "$port" --bind 127.0.0.1
