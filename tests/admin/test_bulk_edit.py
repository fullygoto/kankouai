import pytest

from tests.utils import load_test_app


def _login_as_admin(client):
    with client.session_transaction() as sess:
        sess["user_id"] = "admin"
        sess["role"] = "admin"


def _write_entries(module, entries):
    with module.app.app_context():
        module.save_entries(entries)


def _read_entries(module):
    with module.app.app_context():
        return module.load_entries()


def test_bulk_edit_updates_without_data_loss(monkeypatch, tmp_path):
    sample_entries = [
        {"category": "観光", "title": "A", "desc": "説明A", "areas": ["五島市"]},
        {"category": "観光", "title": "B", "desc": "説明B", "areas": ["新上五島町"]},
    ]

    with load_test_app(monkeypatch, tmp_path) as module:
        _write_entries(module, sample_entries)

        client = module.app.test_client()
        _login_as_admin(client)

        payload = {
            "row_id[]": ["0"],
            "category[]": ["観光"],
            "title[]": ["A"],
            "desc[]": ["更新済み"],
            "address[]": [""],
            "map[]": [""],
            "tags[]": [""],
            "areas[]": ["五島市"],
            "tel[]": [""],
            "holiday[]": [""],
            "open_hours[]": [""],
            "parking[]": [""],
            "parking_num[]": [""],
            "payment[]": [""],
            "remark[]": [""],
            "links[]": [""],
            "allow_empty": "",
        }

        response = client.post("/admin/entries_edit", data=payload, follow_redirects=False)
        assert response.status_code == 302

        saved = _read_entries(module)
        assert len(saved) == 2
        assert saved[0]["desc"] == "更新済み"
        assert saved[1]["title"] == "B"


def test_bulk_edit_rejects_missing_required_fields(monkeypatch, tmp_path):
    sample_entries = [
        {"category": "観光", "title": "A", "desc": "説明A", "areas": ["五島市"]},
    ]

    with load_test_app(monkeypatch, tmp_path) as module:
        _write_entries(module, sample_entries)

        client = module.app.test_client()
        _login_as_admin(client)

        payload = {
            "row_id[]": ["0"],
            "category[]": ["観光"],
            "title[]": [""],
            "desc[]": [""],
            "address[]": [""],
            "map[]": [""],
            "tags[]": [""],
            "areas[]": [""],
            "tel[]": [""],
            "holiday[]": [""],
            "open_hours[]": [""],
            "parking[]": [""],
            "parking_num[]": [""],
            "payment[]": [""],
            "remark[]": [""],
            "links[]": [""],
            "allow_empty": "",
        }

        response = client.post("/admin/entries_edit", data=payload, follow_redirects=False)
        assert response.status_code == 422

        saved = _read_entries(module)
        assert len(saved) == 1
        assert saved[0]["title"] == "A"


def test_bulk_edit_rejects_unknown_columns(monkeypatch, tmp_path):
    sample_entries = [
        {"category": "観光", "title": "A", "desc": "説明A", "areas": ["五島市"]},
    ]

    with load_test_app(monkeypatch, tmp_path) as module:
        _write_entries(module, sample_entries)

        client = module.app.test_client()
        _login_as_admin(client)

        payload = {
            "row_id[]": ["0"],
            "category[]": ["観光"],
            "title[]": ["A"],
            "desc[]": ["説明A"],
            "address[]": [""],
            "map[]": [""],
            "tags[]": [""],
            "areas[]": ["五島市"],
            "tel[]": [""],
            "holiday[]": [""],
            "open_hours[]": [""],
            "parking[]": [""],
            "parking_num[]": [""],
            "payment[]": [""],
            "remark[]": [""],
            "links[]": [""],
            "allow_empty": "",
            "unexpected[]": ["boom"],
        }

        response = client.post("/admin/entries_edit", data=payload, follow_redirects=False)
        assert response.status_code == 422

        saved = _read_entries(module)
        assert len(saved) == 1
        assert saved[0]["title"] == "A"


def test_bulk_edit_deletes_only_flagged_rows(monkeypatch, tmp_path):
    sample_entries = [
        {"category": "観光", "title": "A", "desc": "説明A", "areas": ["五島市"]},
        {"category": "観光", "title": "B", "desc": "説明B", "areas": ["新上五島町"]},
    ]

    with load_test_app(monkeypatch, tmp_path) as module:
        _write_entries(module, sample_entries)

        client = module.app.test_client()
        _login_as_admin(client)

        payload = {
            "row_id[]": ["0"],
            "category[]": ["観光"],
            "title[]": ["A"],
            "desc[]": ["説明A"],
            "address[]": [""],
            "map[]": [""],
            "tags[]": [""],
            "areas[]": ["五島市"],
            "tel[]": [""],
            "holiday[]": [""],
            "open_hours[]": [""],
            "parking[]": [""],
            "parking_num[]": [""],
            "payment[]": [""],
            "remark[]": [""],
            "links[]": [""],
            "allow_empty": "",
            "deleted_row_ids[]": ["1"],
        }

        response = client.post("/admin/entries_edit", data=payload, follow_redirects=False)
        assert response.status_code == 302

        saved = _read_entries(module)
        assert len(saved) == 1
        assert saved[0]["title"] == "A"


def test_bulk_edit_rolls_back_on_duplicate_keys(monkeypatch, tmp_path):
    sample_entries = [
        {"category": "観光", "title": "A", "desc": "説明A", "areas": ["五島市"]},
        {"category": "観光", "title": "B", "desc": "説明B", "areas": ["新上五島町"]},
    ]

    with load_test_app(monkeypatch, tmp_path) as module:
        _write_entries(module, sample_entries)

        client = module.app.test_client()
        _login_as_admin(client)

        payload = {
            "row_id[]": ["0", "0"],
            "category[]": ["観光", "観光"],
            "title[]": ["A", "B"],
            "desc[]": ["説明A", "説明B"],
            "address[]": ["", ""],
            "map[]": ["", ""],
            "tags[]": ["", ""],
            "areas[]": ["五島市", "新上五島町"],
            "tel[]": ["", ""],
            "holiday[]": ["", ""],
            "open_hours[]": ["", ""],
            "parking[]": ["", ""],
            "parking_num[]": ["", ""],
            "payment[]": ["", ""],
            "remark[]": ["", ""],
            "links[]": ["", ""],
            "allow_empty": "",
        }

        response = client.post("/admin/entries_edit", data=payload, follow_redirects=False)
        assert response.status_code == 422

        saved = _read_entries(module)
        assert len(saved) == 2
        assert saved[1]["title"] == "B"


@pytest.mark.parametrize("count", [1000])
def test_bulk_edit_handles_large_payload(monkeypatch, tmp_path, count):
    large_entries = [
        {
            "category": "観光",
            "title": f"スポット{i}",
            "desc": f"説明{i}",
            "areas": ["五島市" if i % 2 == 0 else "新上五島町"],
        }
        for i in range(count)
    ]

    with load_test_app(monkeypatch, tmp_path) as module:
        _write_entries(module, large_entries)

        client = module.app.test_client()
        _login_as_admin(client)

        payload = {
            "row_id[]": ["0"],
            "category[]": ["観光"],
            "title[]": ["スポット0"],
            "desc[]": ["更新後の説明"],
            "address[]": [""],
            "map[]": [""],
            "tags[]": [""],
            "areas[]": ["五島市"],
            "tel[]": [""],
            "holiday[]": [""],
            "open_hours[]": [""],
            "parking[]": [""],
            "parking_num[]": [""],
            "payment[]": [""],
            "remark[]": [""],
            "links[]": [""],
            "allow_empty": "",
        }

        response = client.post("/admin/entries_edit", data=payload, follow_redirects=False)
        assert response.status_code == 302

        saved = _read_entries(module)
        assert len(saved) == count
        assert saved[0]["desc"] == "更新後の説明"
