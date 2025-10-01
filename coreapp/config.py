"""Centralized configuration for the application."""
from __future__ import annotations

import os
import pathlib
from typing import Final


DATA_BASE_DIR: Final[str] = os.getenv("DATA_BASE_DIR", "/var/data")
PAMPHLET_BASE_DIR: Final[str] = os.getenv(
    "PAMPHLET_BASE_DIR",
    os.path.join(DATA_BASE_DIR, "pamphlets"),
)
SEED_PAMPHLET_DIR: Final[str] = os.getenv(
    "SEED_PAMPHLET_DIR",
    str(pathlib.Path(os.getcwd()) / "seeds" / "pamphlets"),
)

MODEL_DEFAULT: Final[str] = os.getenv("MODEL_DEFAULT", "gpt-4o-mini")
MODEL_HARD: Final[str] = os.getenv("MODEL_HARD", "gpt-5-mini")
MIN_QUERY_CHARS: Final[int] = int(os.getenv("MIN_QUERY_CHARS", "2"))
ENABLE_ENTRIES_2CHAR: Final[bool] = os.getenv(
    "ENABLE_ENTRIES_2CHAR", "1"
).lower() in {"1", "true", "on", "yes"}

THRESHOLD_SCORE_HARD: Final[float] = float(os.getenv("THRESHOLD_SCORE_HARD", "0.75"))
THRESHOLD_PIECES_HARD: Final[int] = int(os.getenv("THRESHOLD_PIECES_HARD", "6"))
THRESHOLD_REPROMPTS_HARD: Final[int] = int(os.getenv("THRESHOLD_REPROMPTS_HARD", "2"))


__all__ = [
    "DATA_BASE_DIR",
    "PAMPHLET_BASE_DIR",
    "SEED_PAMPHLET_DIR",
    "MODEL_DEFAULT",
    "MODEL_HARD",
    "MIN_QUERY_CHARS",
    "ENABLE_ENTRIES_2CHAR",
    "THRESHOLD_SCORE_HARD",
    "THRESHOLD_PIECES_HARD",
    "THRESHOLD_REPROMPTS_HARD",
]
