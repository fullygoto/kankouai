import logging

# Codex移行用の間口：app.py が持つルーティングを優先的にロードする。
try:
    from app import app as app  # type: ignore
except Exception as e:  # pragma: no cover - defensive fallback
    logging.getLogger(__name__).warning("app import failed: %s; falling back to create_app", e)
    try:
        from kankouai_app import create_app as _create_app  # type: ignore
    except Exception as e2:  # pragma: no cover
        logging.getLogger(__name__).warning("create_app import failed: %s", e2)
        _create_app = None  # type: ignore
    else:
        _create_app = _create_app  # type: ignore

    if _create_app:
        try:
            app = _create_app()
        except Exception as e3:  # pragma: no cover
            logging.getLogger(__name__).warning("create_app() raised: %s", e3)
            raise
