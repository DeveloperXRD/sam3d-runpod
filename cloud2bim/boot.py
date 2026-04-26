import sys, os, subprocess

print("[boot] Cloud2BIM boot starting...", flush=True)

# Install deps if not already installed
deps = ["open3d", "ifcopenshell", "scipy", "trimesh", "laspy", "Pillow"]
try:
    import open3d
    print("[boot] Dependencies already installed", flush=True)
except ImportError:
    print("[boot] Installing dependencies...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "numpy>=1.24,<2.0", "open3d>=0.18.0", "ifcopenshell>=0.8.0",
        "scipy>=1.11", "trimesh>=4.0", "laspy>=2.5", "Pillow>=10.0"])
    print("[boot] Dependencies installed!", flush=True)

# Run the handler
hp = "/workspace/handler.py"
if os.path.exists(hp):
    print("[boot] Starting Cloud2BIM HTTP server...", flush=True)
    import runpy
    runpy.run_path(hp, run_name="__main__")
else:
    print(f"[boot] ERROR: {hp} not found!", flush=True)
    print("[boot] Upload cloud2bim handler.py to /workspace/handler.py", flush=True)
    sys.exit(1)
