import json
from pathlib import Path

from tests.utils import load_test_app


def test_snapshots_and_dedupe_endpoints(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "snapshots"
    with load_test_app(monkeypatch, tmp_path, extra_env={"SNAPSHOT_DIR": snapshot_dir}) as mod:
        app = mod.app
        data_file = Path(tmp_path) / "entries.json"
        sample_entries = [
            {"category": "観光", "title": "重複スポット", "desc": "説明1", "areas": ["五島市"]},
            {"category": "観光", "title": "重複スポット", "desc": "説明2", "areas": ["五島市"]},
        ]
        data_file.write_text(json.dumps(sample_entries, ensure_ascii=False, indent=2), encoding="utf-8")

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["user_id"] = "admin"
                sess["role"] = "admin"

            resp = client.post("/admin/snapshot/create")
            assert resp.status_code in (302, 303)

            snapshots = sorted(snapshot_dir.glob("snapshot-*.zip"))
            assert snapshots
            fname = snapshots[-1].name

            resp = client.post("/admin/snapshot/restore", data={"fname": fname})
            assert resp.status_code in (302, 303)

            resp = client.get(f"/admin/snapshot/download/{fname}")
            assert resp.status_code == 200
            assert resp.headers.get("Content-Type", "").startswith("application/zip")

            resp = client.post("/admin/entries_dedupe")
            assert resp.status_code == 200
            payload = resp.get_json()
            assert payload and payload.get("ok") is True
