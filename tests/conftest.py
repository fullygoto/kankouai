import os
import pathlib
import sys
import tempfile

# 1) プロジェクトルートを import パスに追加
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2) テスト時のデフォルト環境変数（ログ警告を抑える用）
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "dummy")
os.environ.setdefault("LINE_CHANNEL_SECRET", "dummy")


if "DATA_BASE_DIR" not in os.environ:
    base_dir = pathlib.Path(tempfile.mkdtemp(prefix="kankouai-data-"))
    pamphlets_dir = base_dir / "pamphlets"
    system_dir = base_dir / "system"
    pamphlets_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_BASE_DIR"] = str(base_dir)
