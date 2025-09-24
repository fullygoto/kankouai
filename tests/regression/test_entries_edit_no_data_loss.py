import hashlib
import json
from pathlib import Path

from tests.utils import load_test_app


def test_entries_edit_rejects_invalid_payload(tmp_path, monkeypatch):
    with load_test_app(monkeypatch, tmp_path) as mod:
        app = mod.app
        data_file = Path(tmp_path) / "entries.json"
        sample_entries = [
            {"category": "観光", "title": "朝日", "desc": "朝日スポット", "areas": ["五島市"]},
            {"category": "観光", "title": "夕日", "desc": "夕日スポット", "areas": ["新上五島町"]},
        ]
        data_file.write_text(json.dumps(sample_entries, ensure_ascii=False, indent=2), encoding="utf-8")
        original_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["user_id"] = "admin"
                sess["role"] = "admin"

            resp = client.post("/admin/entries_edit", data={"entries_raw": "", "allow_empty": ""})
            assert resp.status_code == 422
            assert hashlib.sha256(data_file.read_bytes()).hexdigest() == original_hash

            resp = client.post("/admin/entries_edit", data={})
            assert resp.status_code == 422
            assert hashlib.sha256(data_file.read_bytes()).hexdigest() == original_hash

            resp = client.post("/admin/entries_edit", data={"entries_raw": "not json"})
            assert resp.status_code == 422
            assert hashlib.sha256(data_file.read_bytes()).hexdigest() == original_hash

            resp = client.post("/admin/entries_edit", data={"entries_raw": "[]"})
            assert resp.status_code == 422
            assert hashlib.sha256(data_file.read_bytes()).hexdigest() == original_hash

            valid_payload = json.dumps([
                {"category": "観光", "title": "更新後のタイトル", "desc": "更新済み", "areas": ["五島市"], "tags": ["景勝"]},
            ], ensure_ascii=False)
            resp = client.post("/admin/entries_edit", data={"entries_raw": valid_payload})
            assert resp.status_code == 302

            new_bytes = data_file.read_bytes()
            assert hashlib.sha256(new_bytes).hexdigest() != original_hash
            saved_entries = json.loads(new_bytes.decode("utf-8"))
            assert saved_entries and saved_entries[0]["title"] == "更新後のタイトル"

            backups = sorted(data_file.parent.glob("entries.json.bak-*"))
            assert backups
            with backups[-1].open("r", encoding="utf-8") as f:
                backup_entries = json.load(f)
            assert backup_entries == sample_entries
