"""Routing helpers for prioritising conversation handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple


Probe = Callable[[str], Any]


@dataclass
class RouteDecision:
    """Result returned by :func:`select_route`."""

    route: str
    payload: Any = None


def _interpret_probe_result(result: Any) -> Tuple[bool, Any]:
    """Normalise probe outputs.

    A probe may return a boolean, a payload, or ``(handled, payload)``.
    """

    if isinstance(result, tuple) and result:
        head = result[0]
        handled = bool(head)
        payload = result[1] if len(result) > 1 else None
        return handled, payload

    if isinstance(result, bool):
        return result, None

    handled = bool(result)
    return handled, result if handled else None


def select_route(
    text: str,
    *,
    special_probe: Optional[Probe] = None,
    tourism_probe: Optional[Probe] = None,
    pamphlet_probe: Optional[Probe] = None,
) -> RouteDecision:
    """Choose the handler route following the mandated priority.

    Priority: ``special`` → ``tourism`` → ``pamphlet`` → ``no_answer``.
    Each probe is invoked at most once; subsequent probes are skipped once
    a handler claims the message.  This keeps side effects (DB lookups,
    logging) deterministic and avoids accidental fallback execution.
    """

    ordered: Tuple[Tuple[str, Optional[Probe]], ...] = (
        ("special", special_probe),
        ("tourism", tourism_probe),
        ("pamphlet", pamphlet_probe),
    )

    for route, probe in ordered:
        if probe is None:
            continue
        handled, payload = _interpret_probe_result(probe(text))
        if handled:
            return RouteDecision(route=route, payload=payload)

    return RouteDecision(route="no_answer", payload=None)


__all__ = ["RouteDecision", "select_route"]
