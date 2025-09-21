import sys, pathlib, os

# 1) プロジェクトルートを import パスに追加
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2) テスト時のデフォルト環境変数（ログ警告を抑える用）
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LINE_CHANNEL_ACCESS_TOKEN", "dummy")
os.environ.setdefault("LINE_CHANNEL_SECRET", "dummy")
