# ============================================================
# SAM 3D Objects — RunPod Serverless Worker
# ============================================================
# Build:
#   docker build --build-arg HF_TOKEN=hf_xxx -t sam3d-runpod .
# Push:
#   docker tag sam3d-runpod <your-dockerhub>/sam3d-runpod:latest
#   docker push <your-dockerhub>/sam3d-runpod:latest
# ============================================================

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LIDRA_SKIP_INIT=true
ENV PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5

# ── System deps ──────────────────────────────────────────
RUN apt-get update -q && apt-get install -y -q --no-install-recommends \
    git build-essential ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Clone SAM 3D Objects ──────────────────────────────────
RUN git clone https://github.com/facebookresearch/sam-3d-objects.git /workspace/sam-3d-objects

WORKDIR /workspace/sam-3d-objects

# ── Install base package (NO [dev] — skips bpy, jupyter, wandb, etc.) ──
RUN pip install --no-cache-dir -e .

# ── Install PyTorch3D ──
RUN pip install --no-cache-dir -e '.[p3d]'

# ── Install inference deps (Kaolin + gsplat) ──
# TORCH_CUDA_ARCH_LIST needed because no GPU present during CI build
ENV PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
RUN pip install --no-cache-dir -e '.[inference]'

# ── Patch Hydra ──
RUN bash ./patching/hydra || true

# ── Fix numpy binary compat (kaolin wheels need numpy 2.x) ──
RUN pip install --no-cache-dir 'numpy>=2.0'

# ── Extra deps for handler + inference.py imports ──
RUN pip install --no-cache-dir runpod open3d trimesh opencv-python-headless \
    gradio seaborn matplotlib

# ── HuggingFace hub (checkpoints are downloaded at runtime via handler.py) ──
RUN pip install --no-cache-dir 'huggingface-hub<1.0'

# ── Copy handler ──
WORKDIR /workspace
COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]
