#!/usr/bin/env bash
# Build the submission image as "mandated base layers + one layer" without pulling 20 GB locally.
# usage: build/assemble.sh <target image ref>        (needs crane and a registry login)
set -euo pipefail
TARGET="$1"
BASE="index.docker.io/rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
L="$(mktemp -d)"
mkdir -p "$L/app/output" "$L/models" "$L/opt/pinpoint/deps"
cp -r "$ROOT/app/." "$L/app/"
find "$L/app" -name __pycache__ -prune -exec rm -rf {} +
# wheels for the base image's interpreter (CPython 3.14, manylinux x86_64); all are pure-python or abi3/cp314
pip install --no-deps --only-binary=:all: --target "$L/opt/pinpoint/deps" \
  --python-version 3.14 --implementation cp --abi cp314 --abi abi3 --abi none \
  --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 --platform any \
  -r "$ROOT/app/requirements.txt"
tar -C "$L" --owner=0 --group=0 --numeric-owner -czf "$L.tar.gz" app models opt
crane append -b "$BASE" -f "$L.tar.gz" -t "$TARGET.tmp"
crane mutate "$TARGET.tmp" \
  --env PYTHONPATH=/opt/pinpoint/deps --env PINPOINT_MODEL_DIR=/models --env HF_HUB_DISABLE_TELEMETRY=1 \
  --workdir /app --cmd python3,/app/app.py,--serve -t "$TARGET"
crane manifest "$TARGET" | python3 -c "import sys,json; m=json.load(sys.stdin); print(len(m['layers']), 'layers', round(sum(l['size'] for l in m['layers'])/2**30,2), 'GiB compressed')"
