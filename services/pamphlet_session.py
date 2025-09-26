from __future__ import annotations

import time
from typing import Any, Dict, MutableMapping, Optional

DEFAULT_TTL = 1800


class PamphletSession:
    """Helper to manage city/pending/followup session state with TTL."""

    def __init__(self, store: MutableMapping[str, Any], ttl: int = DEFAULT_TTL, *, now: Optional[float] = None) -> None:
        self._store = store
        self.ttl = max(60, int(ttl or DEFAULT_TTL))
        self.now = float(now if now is not None else time.time())

        legacy = store.setdefault("pamphlet", {})
        self._city = store.setdefault("pamphlet_city", legacy.setdefault("city", {}))
        self._pending = store.setdefault("pamphlet_pending", legacy.setdefault("pending", {}))
        self._follow = store.setdefault("pamphlet_followup", legacy.setdefault("followup", {}))

        self._cleanup()

    # Cleanup expired entries
    def _cleanup(self) -> None:
        for mapping in (self._city, self._pending, self._follow):
            stale_keys = []
            for key, value in list(mapping.items()):
                ts = _extract_ts(value)
                if ts is None or self.now - ts > self.ttl:
                    stale_keys.append(key)
            for key in stale_keys:
                mapping.pop(key, None)

    # --- City helpers -------------------------------------------------
    def get_city(self, user_id: str) -> Optional[str]:
        entry = self._city.get(user_id)
        info = _coerce_city(entry)
        if not info:
            return None
        info["ts"] = self.now
        self._city[user_id] = info
        return info.get("city")

    def set_city(self, user_id: str, city: str) -> None:
        if not user_id or not city:
            return
        self._city[user_id] = {"city": city, "ts": self.now}

    # --- Pending helpers ----------------------------------------------
    def get_pending(self, user_id: str) -> Optional[Dict[str, Any]]:
        entry = self._pending.get(user_id)
        if not entry:
            return None
        info = _coerce_pending(entry)
        if not info:
            self._pending.pop(user_id, None)
            return None
        info["ts"] = self.now
        self._pending[user_id] = info
        return dict(info)

    def set_pending(self, user_id: str, query: str, *, asked: Optional[bool] = None) -> Dict[str, Any]:
        info = self.get_pending(user_id) or {"query": "", "asked": False}
        if query:
            info["query"] = query
        if asked is not None:
            info["asked"] = bool(asked)
        info["ts"] = self.now
        self._pending[user_id] = info
        return dict(info)

    def clear_pending(self, user_id: str) -> None:
        self._pending.pop(user_id, None)

    # --- Follow-up helpers -------------------------------------------
    def get_followup(self, user_id: str) -> Optional[Dict[str, Any]]:
        entry = self._follow.get(user_id)
        if not entry:
            return None
        info = _coerce_follow(entry)
        if not info:
            self._follow.pop(user_id, None)
            return None
        info["ts"] = self.now
        self._follow[user_id] = info
        return dict(info)

    def set_followup(self, user_id: str, *, query: str, city: str) -> None:
        if not user_id:
            return
        self._follow[user_id] = {"query": query, "city": city, "ts": self.now}

    def clear_followup(self, user_id: str) -> None:
        self._follow.pop(user_id, None)


def _extract_ts(entry: Any) -> Optional[float]:
    if entry is None:
        return None
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        try:
            return float(entry[1])
        except (TypeError, ValueError):
            return None
    if isinstance(entry, dict):
        ts = entry.get("ts")
        try:
            return float(ts)
        except (TypeError, ValueError, KeyError):
            return None
    return None


def _coerce_city(entry: Any) -> Optional[Dict[str, Any]]:
    if isinstance(entry, dict):
        city = entry.get("city") or entry.get("value")
        ts = entry.get("ts") or entry.get("timestamp")
        if city:
            return {"city": str(city), "ts": float(ts or 0.0)}
        return None
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        city, ts = entry[0], entry[1]
        if city:
            return {"city": str(city), "ts": float(ts or 0.0)}
    if isinstance(entry, str):
        return {"city": entry, "ts": time.time()}
    return None


def _coerce_pending(entry: Any) -> Optional[Dict[str, Any]]:
    if isinstance(entry, dict):
        query = entry.get("query") or entry.get("text")
        asked = bool(entry.get("asked", entry.get("has_asked", False)))
        ts = _extract_ts(entry)
        if not query:
            query = ""
        if ts is None:
            ts = time.time()
        return {"query": str(query), "asked": asked, "ts": float(ts)}
    return None


def _coerce_follow(entry: Any) -> Optional[Dict[str, Any]]:
    if isinstance(entry, dict):
        query = entry.get("query") or entry.get("text")
        city = entry.get("city") or entry.get("city_key")
        ts = _extract_ts(entry)
        if not query or not city:
            return None
        if ts is None:
            ts = time.time()
        return {"query": str(query), "city": str(city), "ts": float(ts)}
    return None
