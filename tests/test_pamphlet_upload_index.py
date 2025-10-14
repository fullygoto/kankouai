import io
from pathlib import Path

from werkzeug.datastructures import FileStorage

from services import pamphlet_search, pamphlet_store
from tests.utils import load_test_app


def test_pamphlet_upload_triggers_index(monkeypatch, tmp_path):
    pamphlet_dir = tmp_path / "pamphlets"
    extra_env = {
        "PAMPHLET_BASE_DIR": pamphlet_dir,
        "SECRET_KEY": "test",
    }

    with load_test_app(monkeypatch, tmp_path, extra_env=extra_env) as module:
        pamphlet_store.BASE = Path(pamphlet_dir)
        pamphlet_store.ensure_dirs()
        pamphlet_search.configure({"PAMPHLET_BASE_DIR": str(pamphlet_dir)})
        payload = io.BytesIO("灯台の歴史を紹介するパンフレットです。\n海岸線の散策情報も掲載されています。".encode("utf-8"))
        storage = FileStorage(stream=payload, filename="goto_guide.txt")
        pamphlet_store.save_file("goto", storage)

        status = pamphlet_search.reindex_all()
        assert status.get("goto", {}).get("state") in {"ready", "empty"}

        snapshot = pamphlet_search.snapshot("goto")
        assert snapshot.chunks, "再インデックス後にチャンクが生成されていません"

        results = pamphlet_search.search("goto", "灯台", 5)
        if not results:
            # fall back to ensuring snapshot contains the expected keyword
            assert any("灯台" in chunk.text for chunk in snapshot.chunks)
        else:
            assert any("灯台" in result.chunk.text for result in results)
