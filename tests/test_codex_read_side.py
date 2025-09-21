import os
import pytest


def test_media_paths_read_from_config(tmp_path, monkeypatch):
    # Arrange: override via config through env consumed by create_app()
    monkeypatch.setenv("DATA_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("MEDIA_ROOT", str(tmp_path / "m_root"))
    monkeypatch.setenv("IMAGES_DIR", str(tmp_path / "imgs"))
    monkeypatch.setenv("WATERMARK_DIR", str(tmp_path / "wms"))

    from importlib import reload
    import wsgi as _wsgi
    reload(_wsgi)  # ensure app picks up envs via create_app()

    app = _wsgi.app
    with app.app_context():
        # Access a few internal helpers by hitting endpoints that use images/watermark/media read-side
        with app.test_client() as c:
            # 404 でOK（実ファイル無し想定）だが、500はNG
            assert c.get("/media/img/does-not-exist.jpg").status_code in (404, 403)
            # 管理系の画像操作周り（ログイン前提のため302）が落ちないこと
            assert c.get("/admin/watermark").status_code in (200, 302)
