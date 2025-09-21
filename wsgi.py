import logging

# Codex移行用の間口：create_app() があればそれを使う。ダメなら既存 app にフォールバック。
try:
    from app import create_app as _create_app  # type: ignore
except Exception as e:
    logging.getLogger(__name__).warning("create_app import failed: %s", e)
    _create_app = None  # type: ignore

_app = None
if _create_app:
    try:
        _app = _create_app()
    except Exception as e:
        logging.getLogger(__name__).warning("create_app() raised: %s; falling back to global app", e)
        _app = None

if _app is None:
    from app import app as app  # type: ignore
else:
    app = _app
