import io
from pathlib import Path

import pytest
from PIL import Image
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge

from tests.utils import load_test_app


def _make_image_bytes(min_bytes: int = 0, size: int = 512, quality: int = 85) -> io.BytesIO:
    while True:
        buf = io.BytesIO()
        Image.new("RGB", (size, size), color="white").save(buf, format="JPEG", quality=quality)
        if buf.tell() >= min_bytes:
            buf.seek(0)
            return buf
        size += 256
        quality = min(100, quality + 5)


def test_default_max_upload_is_64mb(monkeypatch, tmp_path):
    with load_test_app(monkeypatch, tmp_path, extra_env={"SECRET_KEY": "test"}) as module:
        assert module.app.config["MAX_CONTENT_LENGTH"] == 64 * 1024 * 1024


def test_save_jpeg_rejects_massive_image(monkeypatch, tmp_path):
    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={"SECRET_KEY": "test"},
    ) as module:
        payload = _make_image_bytes(size=12000, quality=100)
        storage = FileStorage(stream=payload, filename="huge.png")
        with pytest.raises(RequestEntityTooLarge):
            module._save_jpeg_1080_350kb(storage, previous=None, delete=False)


def test_watermark_variants_created(monkeypatch, tmp_path):
    with load_test_app(monkeypatch, tmp_path, extra_env={"SECRET_KEY": "test"}) as module:
        payload = _make_image_bytes(size=640, quality=80)
        storage = FileStorage(stream=payload, filename="sample.png")
        filename = module._save_jpeg_1080_350kb(storage, previous=None, delete=False)
        assert filename
        base_path = Path(module.IMAGES_DIR) / filename
        assert base_path.exists()
        variants = module._ensure_wm_variants(filename)
        for key in ("gotocity", "fullygoto"):
            assert key in variants
            variant_path = Path(module.IMAGES_DIR) / variants[key]
            assert variant_path.exists()
