import io

from flask import session

import admin_pamphlets

from tests.utils import load_test_app


def _make_client(monkeypatch, tmp_path):
    base_dir = tmp_path / "pamphlets"
    extra_env = {
        "PAMPHLET_BASE_DIR": str(base_dir),
        "SECRET_KEY": "test",
        "PAMPHLET_EDIT_MAX_MB": "2",
        "MAX_UPLOAD_MB": "128",
    }
    from services import pamphlet_store

    pamphlet_store.BASE = base_dir
    pamphlet_store.ensure_dirs()

    context = load_test_app(monkeypatch, tmp_path, extra_env=extra_env)
    return context, base_dir


def _login_admin(client):
    with client.session_transaction() as sess:
        sess["role"] = "admin"


def test_upload_accepts_japanese_filename(monkeypatch, tmp_path):
    ctx, base_dir = _make_client(monkeypatch, tmp_path)
    with ctx as module:
        client = module.app.test_client()
        _login_admin(client)

        data = {
            "city": "goto",
            "file": (io.BytesIO("こんにちは".encode("utf-8")), "日本語ガイド.txt"),
        }

        response = client.post(
            "/admin/pamphlets/upload",
            data=data,
            content_type="multipart/form-data",
        )
        assert response.status_code == 302

        saved = base_dir / "goto" / "日本語ガイド.txt"
        assert saved.exists()
        assert saved.read_text(encoding="utf-8") == "こんにちは"


def test_edit_save_within_limit_creates_backup(monkeypatch, tmp_path):
    ctx, base_dir = _make_client(monkeypatch, tmp_path)
    city_dir = base_dir / "goto"
    city_dir.mkdir(parents=True, exist_ok=True)
    target = city_dir / "sample.txt"
    target.write_text("旧コンテンツ", encoding="utf-8")
    expected_mtime = str(target.stat().st_mtime)

    with ctx as module:
        client = module.app.test_client()
        _login_admin(client)

        new_body = "x" * (273 * 1024)
        response = client.post(
            "/admin/pamphlets/save",
            data={
                "city": "goto",
                "name": "sample.txt",
                "expected_mtime": expected_mtime,
                "content": new_body,
            },
        )
        assert response.status_code == 302

        with client.session_transaction() as sess:
            assert ("success", "保存しました。") in sess.get("_flashes", [])

        assert target.read_text(encoding="utf-8") == new_body
        backup = target.with_suffix(target.suffix + ".bak")
        assert backup.exists()
        assert backup.read_text(encoding="utf-8") == "旧コンテンツ"


def test_edit_save_rejects_large_payload(monkeypatch, tmp_path):
    ctx, base_dir = _make_client(monkeypatch, tmp_path)
    city_dir = base_dir / "goto"
    city_dir.mkdir(parents=True, exist_ok=True)
    target = city_dir / "limit.txt"
    target.write_text("base", encoding="utf-8")
    expected_mtime = str(target.stat().st_mtime)

    with ctx as module:
        app = module.app
        app.config["PAMPHLET_EDIT_MAX_BYTES"] = 1024
        oversize = "a" * 1500

        with app.test_request_context(
            "/admin/pamphlets/save",
            method="POST",
            data={
                "city": "goto",
                "name": "limit.txt",
                "expected_mtime": expected_mtime,
                "content": oversize,
            },
        ):
            session["role"] = "admin"
            response = admin_pamphlets.pamphlets_save()
            assert response.status_code == 302

            messages = session.get("_flashes", [])
            assert any(
                "編集内容が大きすぎます" in msg and "64MB" not in msg
                for _cat, msg in messages
            ), messages

        assert target.read_text(encoding="utf-8") == "base"
