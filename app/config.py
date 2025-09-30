"""Centralized configuration for the application."""
from __future__ import annotations

import os
from typing import Final

# TODO: wire these constants into the runtime configuration once the new
# responder framework is implemented.
MODEL_DEFAULT: Final[str] = os.getenv("MODEL_DEFAULT", "gpt-4o-mini")
MODEL_HARD: Final[str] = os.getenv("MODEL_HARD", "gpt-5-mini")

__all__ = ["MODEL_DEFAULT", "MODEL_HARD"]
