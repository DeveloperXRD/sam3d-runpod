#!/bin/bash
set -e

echo "=== Primitive Fitting RunPod Pod Build Script ==="

# ── System deps ──
apt-get update -q && apt-get install -y -q --no-install-recommends \
    build-essential libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ──
# Step 1: Install core deps (already present on pytorch image, but ensure correct versions)
pip install --no-cache-dir --upgrade-strategy only-if-needed \
    'numpy>=1.24,<2.0' \
    'open3d>=0.18.0' \
    'ifcopenshell>=0.8.0' \
    'scipy>=1.11' \
    'trimesh>=4.0' \
    'Pillow>=10.0'

# Step 2: Install cadquery without dependencies first (avoids backtracking)
pip install --no-cache-dir --no-deps cadquery-ocp
pip install --no-cache-dir --no-deps cadquery

# Step 3: Install only the cadquery deps that aren't already there
pip install --no-cache-dir --upgrade-strategy only-if-needed \
    ezdxf multimethod nlopt vtk path

echo "=== Build complete ==="
echo "Now upload primitive_fit_handler.py to /workspace/handler.py"
echo "Then use the start command from start_cmd.txt"
