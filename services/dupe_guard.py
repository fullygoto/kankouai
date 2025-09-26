"""Simple in-process guard to avoid duplicate responses."""
from __future__ import annotations

import hashlib
import os
import re
from time import time
from typing import Dict, Optional

_CACHE: Dict[str, float] = {}
_RECENT_INPUTS: Dict[str, tuple[str, float]] = {}
_RECENT_REQUESTS: Dict[str, float] = {}
_RECENT_HASHES: Dict[str, float] = {}

WINDOW = float(os.getenv("ANTIFLOOD_TTL_SEC", "120"))
_TOKEN_RE = re.compile(r"[\w]+|[\u3040-\u30FF]+|[\u4E00-\u9FFF]+")


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


def evaluate_utterance(user_id: str, text: str, *, min_tokens: int = 2) -> Optional[str]:
    """Return a reason string if the utterance should be ignored."""

    normalized = (text or "").strip()
    if not normalized:
        return "empty"

    tokens = _TOKEN_RE.findall(normalized)
    if len(tokens) < min_tokens and len(normalized) < (min_tokens + 1):
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
