import logging

# Codex移行用の間口：create_app() があればそれを使う。無ければ既存 app を使う。
try:
    from app import create_app as _create_app  # type: ignore
except Exception as e:
    logging.getLogger(__name__).warning("create_app import failed: %s", e)
    _create_app = None  # type: ignore

if _create_app:
    app = _create_app()
else:
    # 既存の app インスタンスにフォールバック
    from app import app as app  # type: ignore
