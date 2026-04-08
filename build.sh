#!/bin/bash
set -e

echo "=== SAM 3D Objects RunPod Build Script ==="

# ── System deps ──
apt-get update -q && apt-get install -y -q --no-install-recommends \
    git build-essential ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Clone SAM 3D Objects ──
git clone https://github.com/facebookresearch/sam-3d-objects.git /workspace/sam-3d-objects
cd /workspace/sam-3d-objects

# ── Install base package (no [dev]) ──
pip install --no-cache-dir -e .

# ── Install PyTorch3D ──
pip install --no-cache-dir -e '.[p3d]'

# ── Install inference deps (Kaolin + gsplat) ──
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install --no-cache-dir -e '.[inference]'

# ── Patch Hydra ──
bash ./patching/hydra || true

# ── Extra deps for handler ──
pip install --no-cache-dir runpod open3d trimesh opencv-python-headless

# ── Download checkpoints from HuggingFace ──
pip install --no-cache-dir 'huggingface-hub[cli]<1.0'
huggingface-cli login --token ${HF_TOKEN}
huggingface-cli download \
    --repo-type model \
    --local-dir /workspace/checkpoints/hf-download \
    --max-workers 1 \
    facebook/sam-3d-objects
mv /workspace/checkpoints/hf-download/checkpoints /workspace/checkpoints/hf
rm -rf /workspace/checkpoints/hf-download

echo "=== Build complete ==="
