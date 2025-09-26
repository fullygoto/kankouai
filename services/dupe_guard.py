"""Simple in-process guard to avoid duplicate responses."""
from __future__ import annotations

from time import time
from typing import Dict

_CACHE: Dict[str, float] = {}
WINDOW = 30.0


def seen_recent(key: str) -> bool:
    now = time()
    for k, ts in list(_CACHE.items()):
        if now - ts > WINDOW:
            _CACHE.pop(k, None)
    if key in _CACHE:
        return True
    _CACHE[key] = now
    return False
