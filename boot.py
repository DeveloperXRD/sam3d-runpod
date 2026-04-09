import sys, types, os

class _D:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return None
    def __getattr__(self, n): return _D()
    def __bool__(self): return False

class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _D()

class _KaolinFinder:
    def find_module(self, fullname, path=None):
        if fullname == "kaolin" or fullname.startswith("kaolin."):
            return self
        return None
    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = _StubModule(fullname)
        mod.__path__ = []
        mod.__loader__ = self
        mod.__package__ = fullname
        mod.__file__ = "<kaolin_stub>"
        sys.modules[fullname] = mod
        return mod

sys.meta_path.insert(0, _KaolinFinder())
print("[boot] kaolin import hook installed", flush=True)

# Patch 1: skip mesh decoder in inference_pipeline.py
pipe = "/workspace/sam-3d-objects/sam3d_objects/pipeline/inference_pipeline.py"
if os.path.exists(pipe):
    code = open(pipe).read()
    old = 'ret["mesh"] = self.models["slat_decoder_mesh"](slat)'
    if old in code:
        code = code.replace(old, "pass  # mesh decoder skipped")
        open(pipe, "w").write(code)
        print("[boot] Patched: mesh decoder skipped", flush=True)

# Patch 2: fix handler.py to handle gs/gaussian key + add logging
hp = "/workspace/handler.py"
if os.path.exists(hp):
    hc = open(hp).read()
    old_gs = 'gs = output["gs"]'
    new_gs = ('print(f"[handler] Pipeline output keys: {list(output.keys())}", flush=True)\n'
              '        gs = output.get("gs") or output.get("gaussian")\n'
              '        if gs is None:\n'
              '            raise ValueError(f"No GS in output. Keys: {list(output.keys())}")')
    if old_gs in hc:
        hc = hc.replace(old_gs, new_gs)
        open(hp, "w").write(hc)
        print("[boot] Patched: handler.py gs key fix", flush=True)

import runpy
runpy.run_path(hp, run_name="__main__")
