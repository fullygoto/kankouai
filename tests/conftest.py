import sys, pathlib, os, tempfile

import pytest

# 1) プロジェクトルートを import パスに追加
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2) テスト時のデフォルト環境変数（ログ警告を抑える用）
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "dummy")
os.environ.setdefault("LINE_CHANNEL_SECRET", "dummy")


@pytest.fixture(autouse=True, scope="session")
def setup_data_base_dir():
    base = os.environ.get("DATA_BASE_DIR")
    if not base:
        base = str(pathlib.Path(tempfile.gettempdir()) / "kankouai_test_data")
        os.environ["DATA_BASE_DIR"] = base
    p = pathlib.Path(base)
    for d in ("pamphlets", "system"):
        (p / d).mkdir(parents=True, exist_ok=True)
    return p
