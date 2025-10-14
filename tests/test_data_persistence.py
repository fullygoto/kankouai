import io
from pathlib import Path

from werkzeug.datastructures import FileStorage

from services import pamphlet_search, pamphlet_store
from tests.utils import load_test_app


def test_entries_and_pamphlets_persist_between_reloads(monkeypatch, tmp_path):
    data_base = tmp_path / "storage"
    pamphlet_dir = data_base / "pamphlets"
    extra_env = {
        "DATA_BASE_DIR": data_base,
        "PAMPHLET_BASE_DIR": pamphlet_dir,
        "SECRET_KEY": "test",
    }

    entry_payload = [
        {
            "title": "高浜海水浴場",
            "desc": "白砂が続く遠浅のビーチです。",
            "areas": ["五島市"],
            "tags": ["海", "ビーチ"],
            "category": "観光",
        }
    ]

    data_base.mkdir()
    with load_test_app(monkeypatch, tmp_path, extra_env=extra_env) as module:
        module.save_entries(entry_payload)
        pamphlet_store.BASE = Path(pamphlet_dir)
        pamphlet_store.ensure_dirs()
        pamphlet_search.configure({"PAMPHLET_BASE_DIR": str(pamphlet_dir)})
        payload = io.BytesIO("高浜海水浴場の見どころを紹介するパンフレット。".encode("utf-8"))
        storage = FileStorage(stream=payload, filename="takahama.txt")
        pamphlet_store.save_file("goto", storage)
        status = pamphlet_search.reindex_all()
        assert status.get("goto", {}).get("state") in {"ready", "empty"}

    with load_test_app(monkeypatch, tmp_path, extra_env=extra_env) as module:
        entries = module.load_entries()
        assert any(entry.get("title") == "高浜海水浴場" for entry in entries)

        pamphlet_store.BASE = Path(pamphlet_dir)
        pamphlet_store.ensure_dirs()
        pamphlet_search.configure({"PAMPHLET_BASE_DIR": str(pamphlet_dir)})
        status = pamphlet_search.reindex_all()
        assert status.get("goto", {}).get("state") in {"ready", "empty"}
        snapshot = pamphlet_search.snapshot("goto")
        assert snapshot.chunks, "再インデックス後にチャンクが生成されていません"

        results = pamphlet_search.search("goto", "高浜", 5)
        if not results:
            assert any("高浜" in chunk.text for chunk in snapshot.chunks)
        else:
            assert any("高浜" in result.chunk.text for result in results)
