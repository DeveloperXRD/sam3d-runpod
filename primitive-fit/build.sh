#!/bin/bash
set -e

echo "=== Primitive Fitting RunPod Pod Build Script ==="

# ── System deps ──
apt-get update -q && apt-get install -y -q --no-install-recommends \
    build-essential libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ──
pip install --no-cache-dir \
    'numpy>=1.24,<2.0' \
    'open3d>=0.18.0' \
    'ifcopenshell>=0.8.0' \
    'cadquery>=2.4' \
    'scipy>=1.11' \
    'trimesh>=4.0' \
    'runpod>=1.6' \
    'Pillow>=10.0'

echo "=== Build complete ==="
echo "Now upload primitive_fit_handler.py to /workspace/handler.py"
echo "Then use the start command from start_cmd.txt"
