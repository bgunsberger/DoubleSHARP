#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python python3.12 .venv --allow-existing
uv pip install --python .venv/bin/python -r requirements-multiview.txt
mkdir -p vendor
if [[ ! -d vendor/ml-sharp/.git ]]; then
  git clone https://github.com/apple/ml-sharp.git vendor/ml-sharp
fi
git -C vendor/ml-sharp checkout aed6527499ef91cba3b54c18d49a870f25947190
if [[ ! -d vendor/Depth-Anything-3/.git ]]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git vendor/Depth-Anything-3
fi
git -C vendor/Depth-Anything-3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
if ! git -C vendor/Depth-Anything-3 apply --reverse --check ../../patches/da3-lazy-export.patch 2>/dev/null; then
  git -C vendor/Depth-Anything-3 apply ../../patches/da3-lazy-export.patch
fi

# Only the upstream WebGL viewer is used; no C++ build is required.
if [[ ! -d vendor/free-splatter.cpp/.git ]]; then
  git clone https://github.com/localai-org/free-splatter.cpp.git vendor/free-splatter.cpp
fi
git -C vendor/free-splatter.cpp checkout bf636c4fc2cf4117649eccab64909d3937bbbcdb
