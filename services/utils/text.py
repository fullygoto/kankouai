"""Pure text utilities extracted from app.py."""

from __future__ import annotations

import os
import re
import unicodedata
from typing import List

# Shared defaults for message-safe lengths
LINE_SAFE_CHARS = int(os.getenv("LINE_SAFE_CHARS", "3000"))

RENTACAR_KEYWORDS = ("レンタカー",)


def is_rentacar_query(text: str) -> bool:
    norm = unicodedata.normalize("NFKC", text or "")
    return any(keyword in norm for keyword in RENTACAR_KEYWORDS)


def _split_lines_commas(val: str) -> List[str]:
    """改行・カンマ両対応で分割 → 余白除去 → 空要素除去"""
    if not val:
        return []
    parts = re.split(r'[\n,]+', str(val))
    return [p.strip() for p in parts if p and p.strip()]


def _split_for_line(text: str,
                    limit: int | None = None,
                    max_len: int | None = None,
                    **_ignored) -> List[str]:
    """
    LINEの1通上限を超える長文を“安全に”分割するユーティリティ。
    - limit 未指定なら env/グローバルの LINE_SAFE_CHARS（既定4800）を採用
    - 段落→文の順に自然に切る。超過時はハードスプリット
    - どんな入力でも最低1要素返す（空配列にしない）
    - len()ではなくUTF-16コードユニット数でカウント（絵文字混在に強い）
    """
    import os, re

    def u16len(s: str) -> int:
        # UTF-16 LE のコードユニット数（LINEの仕様に近い）
        return len(s.encode("utf-16-le")) // 2

    # 実効上限を決定
    if limit is None:
        limit = max_len
    if limit is None:
        # env > グローバル > 既定 の順で採用
        try:
            limit = int(os.getenv("LINE_SAFE_CHARS", str(globals().get("LINE_SAFE_CHARS", 4800))))
        except Exception:
            limit = 4800
    if limit <= 0:
        return ["" if text is None else str(text)]

    s = "" if text is None else str(text)
    if u16len(s) <= limit:
        return [s]

    # 段落単位（空行で分割）
    paragraphs = [p for p in re.split(r"\n\s*\n", s) if p != ""]
    chunks: List[str] = []

    def flush_buf(buf: str) -> None:
        if buf:
            chunks.append(buf)

    def hard_split(token: str) -> None:
        """1トークン自体が長過ぎる場合、UTF-16長を見ながら強制分割"""
        buf = ""
        for ch in token:
            if u16len(buf + ch) > limit:
                flush_buf(buf)
                buf = ch
            else:
                buf += ch
        flush_buf(buf)

    # 1) 段落→2) 文→3) ハードスプリット の順で収める
    buf = ""
    SENT_SPLIT = re.compile(r"(?<=[。．！？!?])")  # 句点等の直後で区切る
    for para in paragraphs:
        para = para.strip("\n")
        # 段落丸ごと入るなら入れる
        if u16len(para) <= limit:
            if u16len(buf + (("\n\n" + para) if buf else para)) <= limit:
                buf = (buf + ("\n\n" if buf else "") + para)
            else:
                flush_buf(buf); buf = para
            continue

        # 文単位で詰める
        sentences = [x for x in SENT_SPLIT.split(para) if x]
        for sent in sentences:
            if u16len(sent) > limit:
                # 文がそもそも長い → いったん今のbufを吐き出してハード分割
                flush_buf(buf); buf = ""
                hard_split(sent)
                continue
            # 既存bufに足せるなら足す
            sep = ("\n" if (buf and not buf.endswith("\n")) else "")
            candidate = buf + (sep + sent if buf else sent)
            if u16len(candidate) <= limit:
                buf = candidate
            else:
                flush_buf(buf); buf = sent

        # 段落の終わりで改行を入れたい場合はここで調整してもOK
    flush_buf(buf)

    # 念のため空にならない保証
    if not chunks:
        chunks = [s[:limit]]

    return chunks


_re2 = re


def _parse_dms_block(s: str):
    """
    DMS（度分秒）1ブロックを小数に変換（例: 35°41'6.6"N / 北緯35度41分6.6秒 / 139°41'30"E）
    戻り値: (value, axis)  axis は 'lat' / 'lng' / None
    """
    s = s.strip()
    hemi = None
    if _re2.search(r'[N北]', s, _re2.I): hemi = 'N'
    if _re2.search(r'[S南]', s, _re2.I): hemi = 'S'
    if _re2.search(r'[E東]', s, _re2.I): hemi = 'E'
    if _re2.search(r'[W西]', s, _re2.I): hemi = 'W'

    m = _re2.search(
        r'(\d+(?:\.\d+)?)\s*[°度]\s*(\d+(?:\.\d+)?)?\s*[\'’′分]?\s*(\d+(?:\.\d+)?)?\s*["”″秒]?',
        s
    )
    if not m:
        return None, None

    deg = float(m.group(1))
    minutes = float(m.group(2) or 0.0)
    seconds = float(m.group(3) or 0.0)
    val = deg + minutes/60.0 + seconds/3600.0
    if hemi in ('S','W'):
        val = -val

    axis = None
    if hemi in ('N','S'):
        axis = 'lat'
    elif hemi in ('E','W'):
        axis = 'lng'
    return val, axis


def parse_latlng_any(text: str):
    """
    Googleマップのコピペ（URL/小数/DMS/日本語表記）を (lat, lng) へ正規化。
    例:
      35.681236, 139.767125
      https://www.google.com/maps?q=35.681236,139.767125
      https://www.google.com/maps/@35.681236,139.767125,17z
      35°41'6.6"N 139°41'30"E
      北緯35度41分6.6秒 東経139度41分30秒
    """
    if not text:
        return None
    s = text.strip().replace('，', ',').replace('、', ',')
    s = _re2.sub(r'\s+', ' ', s)

    # URL ?q=lat,lng / ?query=lat,lng
    m = _re2.search(r'[?&](?:q|query)=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)', s)
    if m:
        return float(m.group(1)), float(m.group(2))

    # URL /@lat,lng,...
    m = _re2.search(r'/@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)(?:[,/?]|$)', s)
    if m:
        return float(m.group(1)), float(m.group(2))

    # 純粋な「lat,lng」
    m = _re2.search(r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)', s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if abs(a) > 90 and abs(b) <= 90:
            a, b = b, a
        return a, b

    # DMS ブロック2つ拾う
    dms_blocks = _re2.findall(
        r'(\d+(?:\.\d+)?\s*[°度]\s*\d*(?:\.\d+)?\s*[\'’′分]?\s*\d*(?:\.\d+)?\s*["”″秒]?\s*[NSEW北南東西]?)',
        s, flags=_re2.I
    )
    if len(dms_blocks) >= 2:
        v1, a1 = _parse_dms_block(dms_blocks[0])
        v2, a2 = _parse_dms_block(dms_blocks[1])
        if (v1 is not None) and (v2 is not None):
            if a1 == 'lat' and a2 == 'lng':
                return v1, v2
            if a1 == 'lng' and a2 == 'lat':
                return v2, v1

    # それでも見つからない場合は None
    return None


def _split_for_messaging(text: str, chunk_size: int = None) -> List[str]:
    """長文を自然な区切り（段落→句点）で2〜複数通に分割。最低1通は返す。"""
    # 上位で定義済みの安全値を使う
    lim = int(chunk_size) if chunk_size else int(globals().get("LINE_SAFE_CHARS", LINE_SAFE_CHARS))
    if not text:
        return [""]
    text = text.strip()
    if len(text) <= lim:
        return [text]

    parts: List[str] = []
    rest = text
    while len(rest) > lim:
        cut = rest.rfind("\n\n", 0, lim)
        if cut < int(lim * 0.5):
            cut = rest.rfind("。", 0, lim)
        if cut < int(lim * 0.5):
            cut = lim
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        parts.append(rest)
    return parts


_unic = unicodedata


def _norm_text_jp(s: str) -> str:
    return _unic.normalize("NFKC", (s or "")).strip().lower()


def _first_int_in_text(text: str) -> int | None:
    import re, unicodedata
    m = re.search(r'([0-9０-９]+)', text or "")
    return int(unicodedata.normalize("NFKC", m.group(1))) if m else None
