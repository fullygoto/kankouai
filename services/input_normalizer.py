import re

TS_NAME_PREFIX = re.compile(
    r"""^\s*(?:\d{1,2}:\d{2})\s+          # 例: 15:19
         (?:[\u4E00-\u9FFF\u3040-\u30FF]+(?:\s+|　)+[\u4E00-\u9FFF\u3040-\u30FF]+)\s+ # 漢字+かなの氏名風
    """,
    re.X,
)

LOG_PREFIX = re.compile(r"^\s*\[\d{4}-\d{2}-\d{2}.*?\]\s*|\s*^user[:：]\s*", re.I)

TRAILING_PLUS = re.compile(r"\s*\++$")


def normalize_user_query(text: str) -> str:
    """Normalize incoming user utterances for consistent downstream handling."""

    t = text or ""
    t = TS_NAME_PREFIX.sub("", t)
    t = LOG_PREFIX.sub("", t)
    t = TRAILING_PLUS.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()
