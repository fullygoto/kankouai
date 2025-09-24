import re
from urllib.parse import urlparse

import pytest
from PIL import Image

from tests.utils import load_test_app


def _login_admin(client):
    with client.session_transaction() as sess:
        sess["user_id"] = "admin"
        sess["role"] = "admin"


def _create_watermark_assets(tmp_path):
    media_root = tmp_path / "media"
    wm_root = tmp_path / "wm"
    media_root.mkdir(parents=True, exist_ok=True)
    wm_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (400, 300), "blue").save(media_root / "base.jpg")
    Image.new("RGBA", (200, 80), (255, 255, 255, 180)).save(wm_root / "overlay.png")
    return media_root, wm_root


def test_watermark_pages_render(monkeypatch, tmp_path):
    media_root, wm_root = _create_watermark_assets(tmp_path)
    extra_env = {"MEDIA_ROOT": media_root, "IMAGES_DIR": media_root, "WATERMARK_DIR": wm_root}
    with load_test_app(monkeypatch, tmp_path, extra_env=extra_env) as module:
        app = module.app
        assert (media_root / "base.jpg").exists()
        with app.app_context():
            with app.test_client() as client:
                _login_admin(client)
                rv = client.get("/admin/watermark")
                assert rv.status_code == 200

                # Ensure admin links resolve to real endpoints
                html = rv.data.decode("utf-8")
                paths = re.findall(r"(?:href|action)=\"([^\"]+)\"", html)
                adapter = app.url_map.bind("")
                for raw in paths:
                    parsed = urlparse(raw)
                    if parsed.scheme or parsed.netloc:
                        path = parsed.path
                    else:
                        path = raw.split("?")[0]
                    if not path.startswith("/admin"):
                        continue
                    if path in {"", "#"}:
                        continue
                    try:
                        adapter.match(path, method="GET")
                    except Exception:
                        try:
                            adapter.match(path, method="POST")
                        except Exception:  # pragma: no cover - failure path
                            pytest.fail(f"unresolvable admin link: {raw}")

                # Additional pages render without error
                resp_one = client.get("/admin_watermark_one", query_string={"src": "base.jpg"})
                assert resp_one.status_code == 200, resp_one.location
                assert client.get("/admin_watermark_edit", query_string={"filename": "base.jpg"}).status_code == 200
