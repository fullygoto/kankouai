import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image
from werkzeug.datastructures import MultiDict

from tests.utils import load_test_app


def _login_admin(client):
    with client.session_transaction() as sess:
        sess["user_id"] = "admin"
        sess["role"] = "admin"


def _prepare_media(tmp_path: Path) -> tuple[Path, Path]:
    media_root = tmp_path / "media"
    wm_root = tmp_path / "wm"
    media_root.mkdir(parents=True, exist_ok=True)
    wm_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (500, 400), "white").save(media_root / "sample.jpg")
    overlay = Image.new("RGBA", (300, 120), (0, 0, 0, 0))
    for x in range(overlay.width):
        for y in range(overlay.height):
            overlay.putpixel((x, y), (255, 0, 0, int(200 * (x / overlay.width))))
    overlay.save(wm_root / "overlay.png")
    return media_root, wm_root


def test_watermark_generation_and_regeneration(monkeypatch, tmp_path):
    media_root, wm_root = _prepare_media(tmp_path)
    extra_env = {"MEDIA_ROOT": media_root, "IMAGES_DIR": media_root, "WATERMARK_DIR": wm_root}
    with load_test_app(monkeypatch, tmp_path, extra_env=extra_env) as module:
        app = module.app
        assert Path(app.config["WATERMARK_DIR"]).resolve() == wm_root.resolve()
        sample_path = media_root / "sample.jpg"
        original_hash = hashlib.sha256(sample_path.read_bytes()).hexdigest()

        with app.app_context():
            with app.test_client() as client:
                _login_admin(client)

                # 1) Bulk generation for existing file
                data = MultiDict(
                    [
                        ("selected_existing", "sample.jpg"),
                        ("watermark_file", "overlay.png"),
                        ("scale", "0.10"),
                        ("opacity", "0.80"),
                    ]
                )
                resp = client.post(
                    "/admin/watermark",
                    data=data,
                )
                assert resp.status_code == 200
                generated = media_root / "sample__wm.jpg"
                assert generated.exists()
                wm_hash_1 = hashlib.sha256(generated.read_bytes()).hexdigest()
                assert wm_hash_1 != original_hash

                # 2) Regenerate with new parameters via edit endpoint
                resp = client.post(
                    "/admin_watermark_edit/apply",
                    data={
                        "filename": "sample.jpg",
                        "watermark_file": "overlay.png",
                        "scale": "0.12",
                        "opacity": "0.40",
                    },
                    follow_redirects=True,
                )
                assert resp.status_code == 200
                wm_hash_2 = hashlib.sha256(generated.read_bytes()).hexdigest()
                assert wm_hash_2 != wm_hash_1
                backups = list(media_root.glob("sample__wm.jpg.bak-*"))
                assert backups, "backup file should be created before overwrite"

                # 3) Upload invalid data -> skipped without touching originals
                bad_file = (BytesIO(b"not an image"), "bad.png")
                resp = client.post(
                    "/admin/watermark",
                    data={
                        "watermark_file": "overlay.png",
                        "scale": "0.10",
                        "opacity": "0.80",
                        "files": bad_file,
                    },
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200
                # original hash unchanged
                assert hashlib.sha256(sample_path.read_bytes()).hexdigest() == original_hash
