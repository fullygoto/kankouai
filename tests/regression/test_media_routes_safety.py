from pathlib import Path

from PIL import Image

from tests.utils import load_test_app


def _login_admin(client):
    with client.session_transaction() as sess:
        sess["user_id"] = "admin"
        sess["role"] = "admin"


def _prepare_dirs(tmp_path: Path) -> tuple[Path, Path]:
    media_root = tmp_path / "media"
    wm_root = tmp_path / "wm"
    media_root.mkdir(parents=True, exist_ok=True)
    wm_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 200), "gray").save(media_root / "safe.jpg")
    Image.new("RGBA", (100, 40), (255, 0, 0, 160)).save(wm_root / "overlay.png")
    return media_root, wm_root


def test_media_routes_reject_traversal(monkeypatch, tmp_path):
    media_root, wm_root = _prepare_dirs(tmp_path)
    extra_env = {"MEDIA_ROOT": media_root, "IMAGES_DIR": media_root, "WATERMARK_DIR": wm_root}
    with load_test_app(monkeypatch, tmp_path, extra_env=extra_env) as module:
        app = module.app
        with app.app_context():
            with app.test_client() as client:
                _login_admin(client)
                resp = client.get("/admin/_media_img/../../etc/passwd")
                assert resp.status_code in {400, 404}

                resp = client.post("/admin/media/delete", data={"filename": "../evil.jpg"})
                assert resp.status_code == 400
