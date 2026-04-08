# SAM 3D Objects — RunPod Serverless Deployment

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker Desktop | With WSL 2 backend on Windows |
| NVIDIA Container Toolkit | For GPU-accelerated build (optional but faster) |
| HuggingFace account | Must request access to [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) |
| HuggingFace token | Generate at https://huggingface.co/settings/tokens |
| Docker Hub account | To push the built image |
| RunPod account | With API key from https://runpod.io/console/user/settings |

## 1. Build the Docker Image

> **Note:** The build downloads ~10 GB of model weights and compiles CUDA
> extensions. It takes **30–60 minutes** depending on your machine. Building on
> a RunPod GPU Pod is faster (see "Build on RunPod Pod" below).

```bash
cd runpod

# Replace hf_xxx with your HuggingFace token
docker build --build-arg HF_TOKEN=hf_xxx -t sam3d-runpod .
```

### Build on a RunPod GPU Pod (recommended)

1. Create an A100 40 GB Pod on RunPod
2. SSH in, clone this repo, `cd runpod`
3. Build + push from the Pod:

```bash
docker build --build-arg HF_TOKEN=hf_xxx -t sam3d-runpod .
docker tag sam3d-runpod YOUR_DOCKERHUB_USER/sam3d-runpod:latest
docker login
docker push YOUR_DOCKERHUB_USER/sam3d-runpod:latest
```

4. Destroy the Pod after pushing.

## 2. Push to Docker Hub

```bash
docker tag sam3d-runpod YOUR_DOCKERHUB_USER/sam3d-runpod:latest
docker push YOUR_DOCKERHUB_USER/sam3d-runpod:latest
```

## 3. Create RunPod Serverless Endpoint

1. Go to https://console.runpod.io/serverless
2. Click **+ New endpoint** → **Custom deployment**
3. Configure:
   - **Docker Image:** `YOUR_DOCKERHUB_USER/sam3d-runpod:latest`
   - **GPU:** A100 40 GB (or A100 80 GB for faster inference)
   - **Active Workers:** 0 (scales to zero when idle)
   - **Max Workers:** 1–3
   - **Idle Timeout:** 5 seconds
   - **Execution Timeout:** 300 seconds
4. Click **Create**
5. Copy the **Endpoint ID** from the dashboard

## 4. Configure the Editor App

Add to `apps/editor/.env.local`:

```
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
```

Remove (or leave empty):
```
COLAB_3D_URL=
```

Restart the dev server.

## 5. Test the Endpoint

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "data:image/png;base64,iVBOR...",
      "mask": "data:image/png;base64,iVBOR..."
    }
  }'
```

Response:
```json
{
  "id": "...",
  "status": "COMPLETED",
  "output": {
    "modelUrl": "data:model/gltf-binary;base64,...",
    "format": "glb"
  }
}
```

## Architecture

```
User click → /api/segment (DETR, free HF Inference API)
           → masks displayed in editor

User clicks "Generate 3D"
           → /api/generate-3d
           → RunPod Serverless (SAM 3D Objects on A100)
           → GLB mesh returned to viewer
```

## Cost Estimate

- **A100 40 GB Serverless:** ~$0.00076/sec ($2.74/hr) — pay only when running
- **Cold start:** ~30–60s (first request after idle)
- **Inference time:** ~15–30s per object
- **Idle:** $0 (scales to zero)
