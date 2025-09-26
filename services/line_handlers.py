"""Utilities for LINE message handling (control commands, parsing, etc.)."""

from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import unicodedata
from typing import Callable, Optional, Tuple

from services import state as user_state

LOGGER = logging.getLogger(__name__)

CONTROL_CMD_ENABLED = os.getenv("CONTROL_CMD_ENABLED", "true").lower() in {"1", "true", "on", "yes"}
PAUSE_DEFAULT_TTL_SEC = int(os.getenv("PAUSE_DEFAULT_TTL_SEC", "86400"))

_PAUSE_KEYWORDS = ("停止", "一時停止", "ストップ", "黙って", "しずかにして")
_RESUME_KEYWORDS = ("解除", "再開", "やめ", "解除して", "オン", "復帰")
_TIME_PATTERN = re.compile(r"(\d+)\s*(分|m|min|h|時間|d|日)", re.IGNORECASE)
_JST = _dt.timezone(_dt.timedelta(hours=9))


def normalize_control_text(text: str | None) -> str:
    base = unicodedata.normalize("NFKC", text or "")
    base = base.replace("\u3000", " ")
    base = re.sub(r"\s+", " ", base).strip().lower()
    return base


def _extract_ttl(text: str) -> Optional[int]:
    match = _TIME_PATTERN.search(text)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit in {"分", "m", "min"}:
        return value * 60
    if unit in {"h", "時間"}:
        return value * 3600
    if unit in {"d", "日"}:
        return value * 86400
    return None


def parse_control_command(text: str | None) -> Optional[Tuple[str, Optional[int]]]:
    if not CONTROL_CMD_ENABLED:
        return None
    normalized = normalize_control_text(text)
    if not normalized:
        return None

    ttl = _extract_ttl(normalized)
    if ttl is not None and ttl <= 0:
        ttl = None
    for keyword in _PAUSE_KEYWORDS:
        k = normalize_control_text(keyword)
        if normalized == k or normalized.startswith(f"{k} ") or normalized.startswith(k):
            ttl_sec = ttl or PAUSE_DEFAULT_TTL_SEC
            return ("pause", ttl_sec)

    for keyword in _RESUME_KEYWORDS:
        k = normalize_control_text(keyword)
        if normalized == k or normalized.startswith(f"{k} ") or normalized.startswith(k):
            return ("resume", None)

    return None


def format_pause_until(paused_until: int) -> str:
    dt = _dt.datetime.fromtimestamp(paused_until, tz=_dt.timezone.utc).astimezone(_JST)
    return dt.strftime("%Y-%m-%d %H:%M (%Z)")


def process_control_command(
    raw_text: str | None,
    *,
    user_id: str,
    event,
    reply_func: Callable[[object, object], object],
    logger: logging.Logger,
    global_pause: Tuple[bool, Optional[str]] = (False, None),
    now: Optional[int] = None,
) -> Optional[dict]:
    if not CONTROL_CMD_ENABLED:
        return None

    parsed = parse_control_command(raw_text)
    if not parsed:
        return None

    paused_global, paused_by = global_pause
    if paused_global and paused_by == "admin":
        try:
            reply_func(event, "現在、管理者によって応答が停止されています。管理画面から再開されるまでお待ちください。")
        except Exception:
            logger.exception("control command reply failed (admin pause)")
        return {"action": "blocked", "by": "admin"}

    now_epoch = int(now if now is not None else _dt.datetime.now(tz=_dt.timezone.utc).timestamp())
    action, ttl = parsed

    if action == "pause":
        ttl_sec = int(ttl or PAUSE_DEFAULT_TTL_SEC)
        state = user_state.pause(user_id, ttl_sec=ttl_sec, now=now_epoch)
        paused_until = state["paused_until"]
        message = f"{format_pause_until(paused_until)}まで応答を停止します。『解除』で再開できます。"
        try:
            reply_func(event, message)
        except Exception:
            logger.exception("control command reply failed (pause)")
        return {"action": "pause", "paused_until": paused_until, "ttl_sec": ttl_sec}

    # resume command
    current_paused = user_state.is_paused(user_id, now_epoch)
    user_state.resume(user_id, now=now_epoch)
    if current_paused:
        message = "応答を再開しました。"
    else:
        message = "現在、応答は稼働中です。"
    try:
        reply_func(event, message)
    except Exception:
        logger.exception("control command reply failed (resume)")
    return {"action": "resume", "changed": current_paused}


def control_is_paused(user_id: str, now_epoch: Optional[int] = None) -> bool:
    if not CONTROL_CMD_ENABLED:
        return False
    return user_state.is_paused(user_id, now_epoch)
