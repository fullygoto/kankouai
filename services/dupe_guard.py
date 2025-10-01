"""Simple in-process guard to avoid duplicate responses."""
from __future__ import annotations

import hashlib
import os
from time import time
from typing import Dict, Optional

from coreapp.search.query_limits import min_query_chars, normalize_for_length

_CACHE: Dict[str, float] = {}
_RECENT_INPUTS: Dict[str, tuple[str, float]] = {}
_RECENT_REQUESTS: Dict[str, float] = {}
_RECENT_HASHES: Dict[str, float] = {}
_EVENT_KEYS: Dict[str, float] = {}
_EVENT_TEXT_KEYS: Dict[str, float] = {}

WINDOW = float(os.getenv("ANTIFLOOD_TTL_SEC", "120"))
_REPLAY_WINDOW = float(os.getenv("REPLAY_GUARD_SEC", "150"))


def _purge(cache: Dict[str, float], *, now: Optional[float] = None) -> None:
    now = time() if now is None else now
    for key, ts in list(cache.items()):
        if now - ts > WINDOW:
            cache.pop(key, None)


def seen_recent(key: str) -> bool:
    now = time()
    _purge(_CACHE, now=now)
    if key in _CACHE:
        return True
    _CACHE[key] = now
    return False


def _event_key(message_id: Optional[str]) -> Optional[str]:
    if not message_id:
        return None
    return f"event:{message_id}"


def _event_text_key(user_id: str, text: str) -> str:
    cleaned = (text or "").strip()
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
    return f"evt:{user_id}:{digest}"


def should_process_incoming(
    *,
    user_id: str,
    message_id: Optional[str],
    text: str,
    event_ts: Optional[float],
    now: Optional[float] = None,
) -> bool:
    """Return ``True`` when the inbound message should be processed."""

    now_ts = time() if now is None else now

    if event_ts is not None and _REPLAY_WINDOW > 0:
        if now_ts - event_ts > _REPLAY_WINDOW:
            return False

    msg_key = _event_key(message_id)
    if msg_key:
        _purge(_EVENT_KEYS, now=now_ts)
        ts = _EVENT_KEYS.get(msg_key)
        if ts and now_ts - ts <= WINDOW:
            return False
        _EVENT_KEYS[msg_key] = now_ts

    if text:
        _purge(_EVENT_TEXT_KEYS, now=now_ts)
        text_key = _event_text_key(user_id or "anon", text)
        ts = _EVENT_TEXT_KEYS.get(text_key)
        if ts and now_ts - ts <= WINDOW:
            return False
        _EVENT_TEXT_KEYS[text_key] = now_ts

    return True


def evaluate_utterance(user_id: str, text: str, *, min_tokens: int = 2) -> Optional[str]:
    """Return a reason string if the utterance should be ignored."""

    normalized = (text or "").strip()
    if not normalized:
        return "empty"

    compact = normalize_for_length(normalized)
    effective_min = max(min_query_chars(), min_tokens)
    if len(compact) < effective_min:
        return "too_short"

    now = time()
    for key, (prev_text, ts) in list(_RECENT_INPUTS.items()):
        if now - ts > WINDOW:
            _RECENT_INPUTS.pop(key, None)

    prev = _RECENT_INPUTS.get(user_id)
    if prev and prev[0] == normalized and now - prev[1] < WINDOW:
        return "duplicate"

    _RECENT_INPUTS[user_id] = (normalized, now)
    return None


def should_suppress_response(request_id: Optional[str], payload: str) -> bool:
    """Return True when the response should be dropped to avoid rapid duplicates."""

    cleaned = (payload or "").strip()
    if not cleaned:
        return False

    now = time()
    _purge(_RECENT_REQUESTS, now=now)
    _purge(_RECENT_HASHES, now=now)

    hashed = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    if request_id and request_id in _RECENT_REQUESTS:
        return True

    if hashed in _RECENT_HASHES:
        return True

    if request_id:
        _RECENT_REQUESTS[request_id] = now
    _RECENT_HASHES[hashed] = now
    return False


def reset() -> None:
    """Reset guard state (used in tests)."""

    _CACHE.clear()
    _RECENT_INPUTS.clear()
    _RECENT_REQUESTS.clear()
    _RECENT_HASHES.clear()
    _EVENT_KEYS.clear()
    _EVENT_TEXT_KEYS.clear()
