#!/bin/bash
set -e

echo "=== Unified RunPod Pod Build Script (Cloud2BIM + Primitive Fitting) ==="

# ── System deps ──
apt-get update -q && apt-get install -y -q --no-install-recommends \
    build-essential libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Core Python deps (shared by both pipelines) ──
# CRITICAL: scipy<1.14 and scikit-learn<1.6 — later versions require numpy 2.x
# which breaks open3d/trimesh compat.
# --ignore-installed blinker: works around Ubuntu's apt-installed blinker 1.4 conflict
pip install --no-cache-dir --upgrade-strategy only-if-needed --ignore-installed blinker \
    'numpy>=1.24,<2.0' \
    'ifcopenshell>=0.8.0' \
    'shapely==2.0.2' \
    'trimesh>=4.0' \
    'laspy>=2.5' \
    'Pillow>=10.0'
# scipy not needed — both handlers use pure-numpy convex hull and connected components
# shapely<2.1 — 2.1+ requires numpy 2.x

# Open3D separately with --no-deps to avoid blinker/flask conflicts; install only essential deps
pip install --no-cache-dir --no-deps 'open3d>=0.18.0'
# Open3D imports dash/flask at module load for its visualization submodule — must be installed
# scikit-learn<1.6 is required (1.6+ needs numpy 2.x binary compat)
pip install --no-cache-dir --upgrade-strategy only-if-needed --ignore-installed blinker \
    'matplotlib>=3.7,<3.10' 'pyyaml' 'tqdm' 'pyquaternion' 'scikit-learn<1.6' 'pandas<2.3' 'addict' 'configargparse' \
    'dash>=2.6.0' 'flask>=3.0.0' 'werkzeug' 'plotly' || echo "Some open3d deps failed"

# ── CadQuery (optional, for STEP export in primitive-fit) ──
# Install with --no-deps to avoid pip pulling in heavy deps (vtk, matplotlib, etc)
# that break numpy/open3d compatibility.
echo "--- Installing CadQuery (optional, with --no-deps to preserve numpy<2) ---"
pip install --no-cache-dir --no-deps 'cadquery-ocp>=7.8.1,<7.9' || echo "cadquery-ocp install failed (optional)"
pip install --no-cache-dir --no-deps 'cadquery>=2.4,<2.7' || echo "cadquery install failed (optional)"
# Install only the minimal python-only deps cadquery needs; skip vtk/matplotlib that force numpy upgrade
# nlopt<2.10 is required — nlopt 2.10 requires numpy 2.x
pip install --no-cache-dir --no-deps 'multimethod<2.0' 'nlopt<2.10' 'path' 'ezdxf' || echo "Some cadquery deps failed (STEP export will be disabled)"

# ── CRITICAL: Force numpy back to <2.0 after cadquery install ──
# Some transitive deps may have pulled numpy 2.x which breaks open3d/trimesh.
echo "--- Enforcing numpy<2.0 for open3d/trimesh compatibility ---"
pip install --no-cache-dir --force-reinstall --no-deps 'numpy>=1.24,<2.0'

# ── Stub out open3d.ml to avoid sklearn auto-import (numpy ABI hell) ──
# We don't use open3d's ML features, so we don't need its heavy deps.
# This avoids the scipy/sklearn numpy 2.x ABI mismatch on import.
echo "--- Stubbing out open3d.ml (we don't use ML features) ---"
OPEN3D_PATH=$(python -c "import open3d, os; print(os.path.dirname(open3d.__file__))" 2>/dev/null || echo "/usr/local/lib/python3.11/dist-packages/open3d")
echo "" > "$OPEN3D_PATH/ml/__init__.py" || true
echo "" > "$OPEN3D_PATH/_ml3d/__init__.py" || true

echo "=== Build complete ==="
echo "Start the unified server with:"
echo "  python -u /workspace/repo/unified/handler.py"
