# Cloud2BIM + Primitive Fitting — RunPod Pod Deployment

Two pipelines for converting 3D scans/meshes into clean architectural BIM/CAD models.
Uses **RunPod GPU Pods** (not serverless) for stable, always-on processing.

## Pipelines

### 1. Cloud2BIM — Point Cloud → IFC BIM
- **Input**: Point cloud (PLY, PCD, XYZ, LAS) or mesh (OBJ, STL, GLB)
- **Output**: IFC BIM file + Pascal Editor JSON nodes
- **Algorithm**: RANSAC plane fitting → storey segmentation → wall/opening detection

### 2. Primitive Fitting — Mesh → STEP CAD + IFC + Pascal
- **Input**: OBJ, GLB, STL mesh files
- **Output**: STEP (parametric B-Rep CAD) + IFC (BIM) + Pascal Editor JSON
- **Algorithm**: Dense sampling → RANSAC → angle/grid snapping → CadQuery B-Rep export

## Architecture

```
User uploads scan/mesh
  → /api/scan-to-bim    (Cloud2BIM Pod)
  → /api/mesh-to-cad    (Primitive Fitting Pod)
    → RunPod GPU Pod (HTTP server on port 8000)
      → Detect walls, slabs, doors, windows
      → Export IFC / STEP / Pascal nodes
    → Pascal Editor imports nodes directly into scene
```

## Prerequisites

| Requirement | Notes |
|---|---|
| RunPod account | https://runpod.io/console/user/settings |

## Deploy — Cloud2BIM Pod

### Step 1: Create Pod

1. Go to https://www.runpod.io/console/pods
2. Click **+ Deploy**
3. Configure:
   - **Template:** `RunPod Pytorch 2.1` (or any Python 3.10+ template)
   - **GPU:** Any (RTX 3090, A4000, etc.) — used for fast numpy/scipy
   - **Volume:** 20 GB (to persist deps across restarts)
   - **Expose HTTP Ports:** `8000`
4. Click **Deploy**

### Step 2: Build on the Pod

SSH into the pod (or use Web Terminal):

```bash
# Upload handler from your local machine
# OR clone your repo:
git clone https://github.com/YOUR_USER/YOUR_REPO.git /workspace/repo
cp /workspace/repo/runpod/cloud2bim/handler.py /workspace/handler.py

# Run build script (installs all deps):
bash /workspace/repo/runpod/cloud2bim/build.sh
```

### Step 3: Set Start Command

In the RunPod Pod settings, paste this as **Docker Start Command**
(copy from `runpod/cloud2bim/start_cmd.txt`):

```
bash -c 'echo "<base64>" | python -m base64 -d > /workspace/boot.py && python -u /workspace/boot.py'
```

Or for quicker manual start, just SSH in and run:
```bash
python -u /workspace/handler.py
```

### Step 4: Get Pod URL

Your Pod's HTTP endpoint will be:
```
https://<POD_ID>-8000.proxy.runpod.net
```
Find the Pod ID in the RunPod dashboard.

---

## Deploy — Primitive Fitting Pod

Same steps as Cloud2BIM, but use:
- `runpod/primitive-fit/handler.py` → `/workspace/handler.py`
- `runpod/primitive-fit/build.sh` for deps
- `runpod/primitive-fit/start_cmd.txt` for start command

---

## Configure the Editor App

Add to `apps/editor/.env.local`:

```env
# Existing (SAM 3D Objects)
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_sam3d_endpoint_id

# Cloud2BIM Pod
RUNPOD_CLOUD2BIM_URL=https://<CLOUD2BIM_POD_ID>-8000.proxy.runpod.net

# Primitive Fitting Pod
RUNPOD_PRIMITIVE_FIT_URL=https://<PRIMITIVE_FIT_POD_ID>-8000.proxy.runpod.net
```

Restart the dev server.

## Verify Pods Are Running

```bash
# Cloud2BIM health check
curl https://<POD_ID>-8000.proxy.runpod.net/health
# → {"status": "ok", "service": "cloud2bim"}

# Primitive Fitting health check
curl https://<POD_ID>-8000.proxy.runpod.net/health
# → {"status": "ok", "service": "primitive-fit"}

# Or via the editor API:
curl http://localhost:3000/api/scan-to-bim
curl http://localhost:3000/api/mesh-to-cad
```

## API Usage

### Scan-to-BIM (Cloud2BIM)

```bash
curl -X POST "http://localhost:3000/api/scan-to-bim" \
  -H "Content-Type: application/json" \
  -d '{
    "data": "<base64-encoded-point-cloud>",
    "format": "ply",
    "output": "both",
    "params": {
      "voxel_size": 0.05,
      "wall_distance_threshold": 0.05,
      "min_wall_points": 200
    }
  }'

# Response (synchronous):
# {
#   "status": "COMPLETED",
#   "storeys": 2,
#   "summary": [{ "floor_elevation": 0.0, "walls": 12, "openings": 8, "has_slab": true }],
#   "ifc_b64": "...",
#   "pascal_nodes": { "site_abc123": {...}, "wall_def456": {...}, ... }
# }
```

### Mesh-to-CAD (Primitive Fitting)

```bash
curl -X POST "http://localhost:3000/api/mesh-to-cad" \
  -H "Content-Type: application/json" \
  -d '{
    "data": "<base64-encoded-mesh>",
    "format": "obj",
    "output": "all",
    "params": {
      "distance_threshold": 0.03,
      "snap_angle": 5.0,
      "snap_grid": 0.05
    }
  }'

# Response:
# {
#   "status": "COMPLETED",
#   "summary": { "walls": 15, "slabs": 2, "doors": 5, "windows": 8 },
#   "ifc_b64": "...",
#   "step_b64": "...",
#   "pascal_nodes": { ... }
# }
```

## Pascal Node Import

Both pipelines return `pascal_nodes` — a flat dictionary of scene nodes that map directly to `packages/core/schema/` types:

- **SiteNode** → building container
- **BuildingNode** → with position/rotation
- **LevelNode** → per-storey, with children
- **WallNode** → start/end [x,z] points, thickness, height
- **SlabNode** → polygon boundary [[x,z], ...]
- **DoorNode** → position, width, height, frame params
- **WindowNode** → position, width, height, frame params

To import into the editor, iterate `pascal_nodes` and call `createNode(NodeType.parse(node), node.parentId)` for each.

## Pipeline Parameters

### Cloud2BIM

| Parameter | Default | Description |
|---|---|---|
| `voxel_size` | 0.05 | Point cloud downsampling (meters) |
| `wall_distance_threshold` | 0.05 | RANSAC distance threshold for walls |
| `min_wall_points` | 200 | Minimum points to consider a wall |

### Primitive Fitting

| Parameter | Default | Description |
|---|---|---|
| `distance_threshold` | 0.03 | RANSAC plane fitting tolerance |
| `min_wall_points` | 300 | Minimum points per wall |
| `snap_angle` | 5.0 | Snap walls to nearest 45° increment (degrees) |
| `snap_grid` | 0.05 | Snap endpoints to grid (meters, 0 = disable) |

## Cost Estimate (RunPod GPU Pods)

| GPU | Cost/hr | Notes |
|---|---|---|
| RTX 3090 | ~$0.22/hr | Good for both pipelines |
| RTX A4000 | ~$0.20/hr | Budget option |
| A100 40GB | ~$0.79/hr | Overkill but very fast |

Pods are **always on** — stop when not in use to save cost.

## File Structure

```
runpod/
  cloud2bim/
    handler.py        # HTTP server handler (port 8000)
    boot.py           # Auto-installs deps + starts handler
    build.sh          # Manual build script (run on Pod)
    gen_cmd.py        # Generates start_cmd.txt
    start_cmd.txt     # Docker Start Command (paste in RunPod)
  primitive-fit/
    handler.py        # HTTP server handler (port 8000)
    boot.py           # Auto-installs deps + starts handler
    build.sh          # Manual build script (run on Pod)
    gen_cmd.py        # Generates start_cmd.txt
    start_cmd.txt     # Docker Start Command (paste in RunPod)
```

## Comparison

| Feature | Cloud2BIM | Primitive Fitting |
|---|---|---|
| Input | Point cloud + mesh | Mesh only |
| Output | IFC + Pascal | STEP + IFC + Pascal |
| Wall detection | Good | Better (grid/angle snap) |
| Opening detection | Basic (density gaps) | Advanced (adaptive threshold) |
| Parametric CAD (STEP) | No | Yes (CadQuery) |
| Speed | Faster | Slower (denser sampling) |
| Best for | Quick import | Production-quality CAD |
