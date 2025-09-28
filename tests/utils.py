import contextlib
import importlib
import importlib.util
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def load_test_app(monkeypatch, tmp_path, extra_env=None):
    env = {
        "DATA_BASE_DIR": str(tmp_path),
        "DEDUPE_ON_SAVE": "0",
        "DEDUPE_USE_AI": "0",
        "ADMIN_IP_ENFORCE": "0",
        "CSRF_STRICT": "0",
    }
    if extra_env:
        for key, value in extra_env.items():
            env[key] = None if value is None else str(value)

    base_path = Path(env["DATA_BASE_DIR"])
    for sub in ("pamphlets", "system"):
        (base_path / sub).mkdir(parents=True, exist_ok=True)

    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    if "config" in sys.modules:
        importlib.reload(sys.modules["config"])

    module_name = f"app_for_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    try:
        prepare_once = getattr(module, "_ensure_dirs_once", None)
        if callable(prepare_once):
            prepare_once()
        else:
            prepare = getattr(module, "_prepare_runtime_dirs", None)
            if callable(prepare):
                prepare()
        yield module
    finally:
        sys.modules.pop(module_name, None)
