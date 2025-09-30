import contextlib
import importlib.util
import sys
import types
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

    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    if "dotenv" not in sys.modules:
        sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

    if "openai" not in sys.modules:
        class _StubOpenAI:
            def __init__(self, *args, **kwargs):
                pass

        sys.modules["openai"] = types.SimpleNamespace(OpenAI=_StubOpenAI)

    if "linebot" not in sys.modules:
        linebot_module = types.ModuleType("linebot")

        class _StubApi:
            def __init__(self, *args, **kwargs):
                pass

            def reply_message(self, *args, **kwargs):
                return None

            def push_message(self, *args, **kwargs):
                return None

        class _StubHandler:
            def __init__(self, *args, **kwargs):
                pass

            def add(self, *args, **kwargs):
                def _decorator(func):
                    return func

                return _decorator

            def handle(self, *args, **kwargs):
                return None

        linebot_module.LineBotApi = _StubApi
        linebot_module.WebhookHandler = _StubHandler

        models_module = types.ModuleType("linebot.models")

        class _StubModel:
            def __init__(self, *args, **kwargs):
                pass

        for name in [
            "MessageEvent",
            "TextMessage",
            "TextSendMessage",
            "QuickReply",
            "QuickReplyButton",
            "MessageAction",
            "ImageSendMessage",
            "LocationMessage",
            "FlexSendMessage",
            "LocationAction",
            "FollowEvent",
            "PostbackEvent",
            "PostbackAction",
            "URIAction",
        ]:
            setattr(models_module, name, _StubModel)

        exceptions_module = types.ModuleType("linebot.exceptions")

        class _StubLineBotApiError(Exception):
            pass

        class _StubInvalidSignatureError(Exception):
            pass

        exceptions_module.LineBotApiError = _StubLineBotApiError
        exceptions_module.InvalidSignatureError = _StubInvalidSignatureError

        linebot_module.models = models_module
        linebot_module.exceptions = exceptions_module

        sys.modules["linebot"] = linebot_module
        sys.modules["linebot.models"] = models_module
        sys.modules["linebot.exceptions"] = exceptions_module

    sys.modules.pop("config", None)

    module_name = f"app_for_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
