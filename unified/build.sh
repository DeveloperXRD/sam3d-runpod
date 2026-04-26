#!/bin/bash
set -e

echo "=== Unified RunPod Pod Build Script (Cloud2BIM + Primitive Fitting) ==="

# ── System deps ──
apt-get update -q && apt-get install -y -q --no-install-recommends \
    build-essential libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Core Python deps (shared by both pipelines) ──
pip install --no-cache-dir --upgrade-strategy only-if-needed \
    'numpy>=1.24,<2.0' \
    'open3d>=0.18.0' \
    'ifcopenshell>=0.8.0' \
    'scipy>=1.11' \
    'trimesh>=4.0' \
    'laspy>=2.5' \
    'Pillow>=10.0'

# ── CadQuery (optional, for STEP export in primitive-fit) ──
# Install with --no-deps to avoid pip backtracking through ancient matplotlib versions
echo "--- Installing CadQuery (optional) ---"
pip install --no-cache-dir --no-deps 'cadquery-ocp>=7.8.1,<7.9' || echo "cadquery-ocp install failed (optional)"
pip install --no-cache-dir --no-deps 'cadquery>=2.4,<2.7' || echo "cadquery install failed (optional)"
pip install --no-cache-dir --upgrade-strategy only-if-needed \
    'ezdxf' 'multimethod<2.0' 'nlopt' 'path' || echo "Some cadquery deps failed (STEP export will be disabled)"

echo "=== Build complete ==="
echo "Start the unified server with:"
echo "  python -u /workspace/repo/unified/handler.py"
