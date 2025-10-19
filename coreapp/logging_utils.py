"""Utility helpers for structured interaction logging."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class InteractionLogEntry:
    """Container representing a single structured log entry."""

    timestamp: str
    user_id: str
    channel: str
    intent: str
    hit_source: str
    query: str
    top_score: float | None
    model_used: str
    tokens: int
    latency_ms: float
    errors: List[str]

    @staticmethod
    def build(
        *,
        user_id: str,
        channel: str,
        intent: str,
        hit_source: str,
        query: str,
        top_score: float | None,
        model_used: str,
        tokens: int,
        latency_ms: float,
        errors: Iterable[str] | None = None,
        timestamp: Optional[_dt.datetime] = None,
    ) -> "InteractionLogEntry":
        """Create a new log entry hashing the provided user identifier."""

        ts = (timestamp or _dt.datetime.utcnow()).isoformat()
        hashed = hashlib.sha256(user_id.encode("utf-8")).hexdigest() if user_id else ""
        errs = [str(item) for item in (errors or []) if str(item)]
        return InteractionLogEntry(
            timestamp=ts,
            user_id=hashed,
            channel=str(channel),
            intent=str(intent),
            hit_source=str(hit_source),
            query=str(query),
            top_score=float(top_score) if top_score is not None else None,
            model_used=str(model_used),
            tokens=int(tokens),
            latency_ms=float(latency_ms),
            errors=errs,
        )

    def to_json(self) -> str:
        """Return the JSON representation of the entry."""

        return json.dumps(asdict(self), ensure_ascii=False)


def log_interaction(
    path: str | Path,
    *,
    user_id: str,
    channel: str,
    intent: str,
    hit_source: str,
    query: str,
    top_score: float | None,
    model_used: str,
    tokens: int,
    latency_ms: float,
    errors: Iterable[str] | None = None,
    timestamp: Optional[_dt.datetime] = None,
) -> InteractionLogEntry:
    """Append a structured interaction entry to ``path`` and return it."""

    entry = InteractionLogEntry.build(
        user_id=user_id,
        channel=channel,
        intent=intent,
        hit_source=hit_source,
        query=query,
        top_score=top_score,
        model_used=model_used,
        tokens=tokens,
        latency_ms=latency_ms,
        errors=errors,
        timestamp=timestamp,
    )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(entry.to_json() + "\n")

    return entry


__all__ = ["InteractionLogEntry", "log_interaction"]

