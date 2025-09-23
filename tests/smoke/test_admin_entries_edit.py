import pytest
from flask import url_for


def test_entries_edit_page_renders():
    from app import app
    with app.app_context():
        with app.test_client() as c:
            with c.session_transaction() as sess:
                sess["user_id"] = "admin"
                sess["role"] = "admin"
            rv = c.get("/admin/entries_edit")
            assert rv.status_code == 200


def test_entries_edit_endpoints_resolve():
    from app import app
    with app.app_context():
        with app.test_request_context():
            # テンプレ内で使う endpoint が url_for で組み立て可能か（BuildErrorを起こさないか）
            url_for("admin_entry")
            url_for("admin_entries_edit")
            url_for("admin_entries_import_csv")
            url_for("admin_entries_dedupe")
            url_for("admin_snapshot_create")
            url_for("admin_snapshot_restore")
            url_for("admin_snapshot_download", fname="dummy")
