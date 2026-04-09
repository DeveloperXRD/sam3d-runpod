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

pipe = "/workspace/sam-3d-objects/sam3d_objects/pipeline/inference_pipeline.py"
if os.path.exists(pipe):
    code = open(pipe).read()
    old = 'ret["mesh"] = self.models["slat_decoder_mesh"](slat)'
    if old in code:
        code = code.replace(old, "pass  # mesh decoder skipped")
        open(pipe, "w").write(code)
        print("[boot] Patched: mesh decoder skipped", flush=True)

import runpy
runpy.run_path("/workspace/handler.py", run_name="__main__")
