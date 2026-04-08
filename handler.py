"""
RunPod Serverless Handler — SAM 3D Objects
Accepts image + mask → returns GLB mesh (converted from Gaussian splat).
"""
import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
import base64
import tempfile
import traceback

import numpy as np
import torch
from io import BytesIO
from PIL import Image

import runpod

sys.path.insert(0, "/workspace/sam-3d-objects")
sys.path.insert(0, "/workspace/sam-3d-objects/notebook")

# ---------------------------------------------------------------------------
# Download checkpoints on first cold start (avoids baking ~10 GB into image)
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "/workspace/checkpoints/hf"


def ensure_checkpoints():
    """Download SAM 3D Objects checkpoints from HuggingFace if not present."""
    marker = os.path.join(CHECKPOINT_DIR, "pipeline.yaml")
    if os.path.exists(marker):
        print("[handler] Checkpoints already present.", flush=True)
        return

    print("[handler] Downloading checkpoints from HuggingFace …", flush=True)
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("HF_TOKEN env var is required to download checkpoints")

    from huggingface_hub import snapshot_download

    dl_dir = "/workspace/checkpoints/hf-download"
    snapshot_download(
        repo_id="facebook/sam-3d-objects",
        repo_type="model",
        local_dir=dl_dir,
        token=hf_token,
    )
    # Move checkpoints subfolder to expected location
    import shutil
    src = os.path.join(dl_dir, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for item in os.listdir(src):
        shutil.move(os.path.join(src, item), os.path.join(CHECKPOINT_DIR, item))
    shutil.rmtree(dl_dir, ignore_errors=True)
    print("[handler] Checkpoints downloaded!", flush=True)


# ---------------------------------------------------------------------------
# Lazy model loading (loaded once, reused across requests)
# ---------------------------------------------------------------------------
_inference = None


def get_model():
    global _inference
    if _inference is None:
        ensure_checkpoints()
        os.environ.setdefault("CONDA_PREFIX", "/usr/local/cuda")
        from inference import Inference

        config_path = os.path.join(CHECKPOINT_DIR, "pipeline.yaml")
        print("[handler] Loading SAM 3D Objects model …", flush=True)
        _inference = Inference(config_path, compile=False)
        print("[handler] Model loaded!", flush=True)
    return _inference


# ---------------------------------------------------------------------------
# Gaussian-splat → GLB mesh conversion
# ---------------------------------------------------------------------------
def gs_to_glb(gs_obj, out_path: str) -> str:
    """
    Convert a GaussianModel object to a GLB mesh file.
    Uses Open3D Poisson surface reconstruction on the high-opacity splat centres.
    """
    import open3d as o3d
    import trimesh

    # Extract positions & opacities from the Gaussian splat
    xyz = gs_obj.get_xyz.detach().cpu().numpy()           # (N, 3)
    opacity = gs_obj.get_opacity.detach().cpu().squeeze().numpy()  # (N,)

    # Keep only visible splats
    mask = opacity > 0.3
    pts = xyz[mask]

    if len(pts) < 50:
        raise ValueError(f"Too few visible Gaussians ({len(pts)})")

    # Try to extract vertex colours (SH band-0 → RGB)
    colors = None
    try:
        sh0 = gs_obj._features_dc.detach().cpu().numpy()[mask]  # (N, 3)
        C0 = 0.28209479177387814
        colors = np.clip(0.5 + C0 * sh0, 0.0, 1.0)
    except Exception:
        pass

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Poisson surface reconstruction
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    # Trim low-density vertices (remove reconstruction artefacts)
    densities_np = np.asarray(densities)
    threshold = np.quantile(densities_np, 0.05)
    vertices_to_remove = densities_np < threshold
    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    # Convert to trimesh for GLB export
    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    vert_colors = np.asarray(mesh_o3d.vertex_colors) if mesh_o3d.has_vertex_colors() else None

    tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vert_colors)
    tm.export(out_path, file_type="glb")
    return out_path


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(event):
    try:
        inp = event["input"]

        # --- decode image ------------------------------------------------
        image_b64 = inp["image"].split(",")[-1]
        image = np.array(
            Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB"),
            dtype=np.uint8,
        )

        # --- decode mask -------------------------------------------------
        mask_b64 = inp["mask"].split(",")[-1]
        mask_img = Image.open(BytesIO(base64.b64decode(mask_b64))).convert("L")
        mask = np.array(mask_img) > 128  # boolean mask

        seed = int(inp.get("seed", 42))

        # --- run SAM 3D Objects ------------------------------------------
        model = get_model()
        print(f"[handler] Running inference (image {image.shape}, mask {mask.shape}) …", flush=True)
        output = model(image, mask, seed=seed)
        gs = output["gs"]

        # --- export ------------------------------------------------------
        tmp_ply = "/tmp/output.ply"
        tmp_glb = "/tmp/output.glb"
        gs.save_ply(tmp_ply)

        # Convert Gaussian splat → GLB mesh
        try:
            gs_to_glb(gs, tmp_glb)
            with open(tmp_glb, "rb") as f:
                model_b64 = base64.b64encode(f.read()).decode()
            fmt = "glb"
            mime = "model/gltf-binary"
        except Exception as conv_err:
            print(f"[handler] GLB conversion failed ({conv_err}), returning PLY", flush=True)
            with open(tmp_ply, "rb") as f:
                model_b64 = base64.b64encode(f.read()).decode()
            fmt = "ply"
            mime = "application/octet-stream"

        print(f"[handler] Done! format={fmt}, size={len(model_b64)} chars", flush=True)
        return {
            "modelUrl": f"data:{mime};base64,{model_b64}",
            "format": fmt,
        }

    except Exception:
        traceback.print_exc()
        return {"error": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
