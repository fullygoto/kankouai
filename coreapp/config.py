"""Centralized configuration for the application."""
from __future__ import annotations

import os
from typing import Final


MODEL_DEFAULT: Final[str] = os.getenv("MODEL_DEFAULT", "gpt-4o-mini")
MODEL_HARD: Final[str] = os.getenv("MODEL_HARD", "gpt-5-mini")

THRESHOLD_SCORE_HARD: Final[float] = float(os.getenv("THRESHOLD_SCORE_HARD", "0.75"))
THRESHOLD_PIECES_HARD: Final[int] = int(os.getenv("THRESHOLD_PIECES_HARD", "6"))
THRESHOLD_REPROMPTS_HARD: Final[int] = int(os.getenv("THRESHOLD_REPROMPTS_HARD", "2"))


__all__ = [
    "MODEL_DEFAULT",
    "MODEL_HARD",
    "THRESHOLD_SCORE_HARD",
    "THRESHOLD_PIECES_HARD",
    "THRESHOLD_REPROMPTS_HARD",
]
