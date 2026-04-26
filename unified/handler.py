"""
Unified RunPod Pod Handler — Cloud2BIM + Primitive Fitting
Routes:
  POST /cloud2bim/run     → Cloud2BIM pipeline (point cloud → IFC + Pascal)
  POST /primitive-fit/run → Primitive Fitting pipeline (mesh → STEP + IFC + Pascal)
  GET  /health            → service status
  GET  /                  → service info

Imports the two pipeline modules from their respective folders and routes
incoming HTTP requests to the right one.
"""
import os
import sys
import json
import base64
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

# Make both pipeline modules importable
REPO_ROOT = os.environ.get("REPO_ROOT", "/workspace/repo")
sys.path.insert(0, os.path.join(REPO_ROOT, "cloud2bim"))
sys.path.insert(0, os.path.join(REPO_ROOT, "primitive-fit"))

PORT = int(os.environ.get("PORT", "8000"))

# ── Import pipelines lazily so a failure in one doesn't kill the other ──
_cloud2bim = None
_primitive_fit = None

def get_cloud2bim():
    global _cloud2bim
    if _cloud2bim is None:
        try:
            sys.path.insert(0, os.path.join(REPO_ROOT, "cloud2bim"))
            import handler as c2b
            _cloud2bim = c2b
            print("[unified] Cloud2BIM pipeline loaded", flush=True)
        except Exception as e:
            print(f"[unified] Cloud2BIM load failed: {e}", flush=True)
            _cloud2bim = False
    return _cloud2bim if _cloud2bim is not False else None

def get_primitive_fit():
    global _primitive_fit
    if _primitive_fit is None:
        try:
            # Re-import with a different module name to avoid clash with cloud2bim's "handler"
            import importlib.util
            pf_path = os.path.join(REPO_ROOT, "primitive-fit", "handler.py")
            spec = importlib.util.spec_from_file_location("primitive_fit_handler", pf_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _primitive_fit = module
            print("[unified] Primitive Fitting pipeline loaded", flush=True)
        except Exception as e:
            print(f"[unified] Primitive Fitting load failed: {e}", flush=True)
            _primitive_fit = False
    return _primitive_fit if _primitive_fit is not False else None


# ─────────────────────────────────────────────────────────────────────
# Pipeline runners
# ─────────────────────────────────────────────────────────────────────

def run_cloud2bim(inp):
    c2b = get_cloud2bim()
    if c2b is None:
        return {"error": "Cloud2BIM pipeline not available"}, 503

    data_b64 = inp.get("data", "")
    if "," in data_b64:
        data_b64 = data_b64.split(",")[-1]
    data_bytes = base64.b64decode(data_b64)
    fmt = inp.get("format", "ply").lower()
    params = inp.get("params", {})
    output_format = inp.get("output", "both")

    print(f"[cloud2bim] Input: {len(data_bytes)} bytes, format={fmt}", flush=True)
    pcd = c2b.load_point_cloud(data_bytes, fmt)
    print(f"[cloud2bim] Loaded {len(pcd.points)} points", flush=True)
    storeys_data = c2b.run_pipeline(pcd, params)

    result = {"storeys": len(storeys_data), "summary": []}
    for sd in storeys_data:
        result["summary"].append({
            "floor_elevation": sd["floor_elevation"],
            "walls": len(sd.get("walls", [])),
            "openings": sum(len(w.get("openings", [])) for w in sd.get("walls", [])),
            "has_slab": "slab_polygon" in sd,
        })

    if output_format in ("ifc", "both"):
        ifc_path = "/tmp/cloud2bim_output.ifc"
        c2b.build_ifc(storeys_data, ifc_path)
        with open(ifc_path, "rb") as f:
            result["ifc_b64"] = base64.b64encode(f.read()).decode()

    if output_format in ("pascal", "both"):
        result["pascal_nodes"] = c2b.generate_pascal_nodes(storeys_data)

    print("[cloud2bim] Done!", flush=True)
    return result, 200


def run_primitive_fit(inp):
    pf = get_primitive_fit()
    if pf is None:
        return {"error": "Primitive Fitting pipeline not available"}, 503

    data_b64 = inp.get("data", "")
    if "," in data_b64:
        data_b64 = data_b64.split(",")[-1]
    data_bytes = base64.b64decode(data_b64)
    fmt = inp.get("format", "obj").lower()
    params = inp.get("params", {})
    output_format = inp.get("output", "all")

    print(f"[primitive-fit] Input: {len(data_bytes)} bytes, format={fmt}", flush=True)
    pcd = pf.load_mesh_to_pointcloud(data_bytes, fmt)
    print(f"[primitive-fit] Sampled {len(pcd.points)} points", flush=True)
    walls, slabs = pf.run_pipeline(pcd, params)

    result = {
        "summary": {
            "walls": len(walls), "slabs": len(slabs),
            "doors": sum(1 for w in walls for o in w.openings if o.type == "door"),
            "windows": sum(1 for w in walls for o in w.openings if o.type == "window"),
            "total_openings": sum(len(w.openings) for w in walls),
        }
    }

    if output_format in ("ifc", "all"):
        ifc_path = "/tmp/primitive_fit_output.ifc"
        pf.export_ifc(walls, slabs, ifc_path)
        with open(ifc_path, "rb") as f:
            result["ifc_b64"] = base64.b64encode(f.read()).decode()

    if output_format in ("step", "all"):
        step_path = "/tmp/primitive_fit_output.step"
        if pf.export_step(walls, slabs, step_path):
            with open(step_path, "rb") as f:
                result["step_b64"] = base64.b64encode(f.read()).decode()

    if output_format in ("pascal", "all"):
        result["pascal_nodes"] = pf.generate_pascal_nodes(walls, slabs)

    print("[primitive-fit] Done!", flush=True)
    return result, 200


# ─────────────────────────────────────────────────────────────────────
# HTTP server
# ─────────────────────────────────────────────────────────────────────

class UnifiedHandler(BaseHTTPRequestHandler):
    def _json(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self):
        if self.path in ("/health", "/cloud2bim/health", "/primitive-fit/health"):
            self._json(200, {
                "status": "ok",
                "service": "unified-bim-cad",
                "pipelines": {
                    "cloud2bim": get_cloud2bim() is not None,
                    "primitive-fit": get_primitive_fit() is not None,
                },
            })
        elif self.path == "/":
            self._json(200, {
                "service": "unified-bim-cad",
                "endpoints": [
                    "POST /cloud2bim/run",
                    "POST /primitive-fit/run",
                    "GET  /health",
                ],
            })
        else:
            self._json(404, {"error": f"Unknown route: {self.path}"})

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            inp = body.get("input", body)

            if self.path == "/cloud2bim/run":
                result, status = run_cloud2bim(inp)
            elif self.path == "/primitive-fit/run":
                result, status = run_primitive_fit(inp)
            else:
                result, status = {"error": f"Unknown route: {self.path}"}, 404

            self._json(status, result)

        except Exception:
            traceback.print_exc()
            self._json(500, {"error": traceback.format_exc()})

    def log_message(self, format, *args):
        print(f"[http] {args[0]}", flush=True)


if __name__ == "__main__":
    # Pre-load pipelines so /health reflects real status
    print("[unified] Loading pipelines...", flush=True)
    get_cloud2bim()
    get_primitive_fit()

    server = HTTPServer(("0.0.0.0", PORT), UnifiedHandler)
    print(f"[unified] HTTP server listening on port {PORT}", flush=True)
    print("[unified] Routes:", flush=True)
    print("  POST /cloud2bim/run", flush=True)
    print("  POST /primitive-fit/run", flush=True)
    print("  GET  /health", flush=True)
    server.serve_forever()
