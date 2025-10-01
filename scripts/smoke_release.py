#!/usr/bin/env python3
"""Release smoke-check helper.

The script exercises the critical readiness endpoints and a handful of
representative user journeys so that production rollouts can be verified
quickly.  It exits with a non-zero status on the first failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, Tuple

import requests


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _log_section(title: str, payload: Dict[str, Any]) -> None:
    logging.info("%s:\n%s", title, json.dumps(payload, ensure_ascii=False, indent=2))


def _ask(
    session: requests.Session,
    base_url: str,
    question: str,
    *,
    timeout: float,
    require_hit: bool,
    allow_safe: bool,
) -> Tuple[requests.Response, Dict[str, Any]]:
    url = f"{base_url}/ask"
    logging.info("POST %s question=%s", url, question)
    response = session.post(url, json={"question": question}, timeout=timeout)
    if response.status_code not in (200, 202):
        raise RuntimeError(f"/ask returned {response.status_code} for question {question!r}")

    payload = response.json()
    if require_hit and response.status_code != 200:
        raise RuntimeError(f"Expected hit response for {question!r}, got status {response.status_code}")
    if not allow_safe and response.status_code == 202:
        raise RuntimeError(f"Unexpected SAFE_MODE response for {question!r}")
    return response, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run release smoke checks against the API")
    parser.add_argument("--base-url", default="http://localhost:5000", help="Target service base URL")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    base_url = args.base_url.rstrip("/")
    session = requests.Session()

    # 1) /healthz
    health_url = f"{base_url}/healthz"
    logging.info("GET %s", health_url)
    health = session.get(health_url, timeout=args.timeout)
    if health.status_code != 200:
        raise RuntimeError(f"/healthz returned {health.status_code}")

    # 2) /readyz
    ready_url = f"{base_url}/readyz"
    logging.info("GET %s", ready_url)
    ready = session.get(ready_url, timeout=args.timeout)
    if ready.status_code != 200:
        raise RuntimeError(f"/readyz returned {ready.status_code}")

    ready_payload = ready.json()
    flags = ready_payload.get("details", {}).get("flags", {})
    dirs = {
        "data_base_dir": ready_payload.get("details", {}).get("data_base_dir"),
        "pamphlet_base_dir": ready_payload.get("details", {}).get("pamphlet_base_dir"),
    }
    pamphlet_count = ready_payload.get("details", {}).get("pamphlet_count")
    _log_section("readyz.flags", flags)
    _log_section("readyz.dirs", dirs)
    logging.info("pamphlet_count=%s", pamphlet_count)

    safe_mode = _truthy(flags.get("SAFE_MODE"))

    # 3) Smoke queries
    weather_resp, weather_payload = _ask(
        session,
        base_url,
        "天気",
        timeout=args.timeout,
        require_hit=True,
        allow_safe=False,
    )
    logging.info("weather meta=%s", json.dumps(weather_payload.get("meta", {}), ensure_ascii=False))

    entries_resp, entries_payload = _ask(
        session,
        base_url,
        "教会",
        timeout=args.timeout,
        require_hit=True,
        allow_safe=False,
    )
    if not entries_payload.get("answer"):
        raise RuntimeError("Entries query returned an empty answer")
    logging.info(
        "entries meta=%s",
        json.dumps(entries_payload.get("meta", {}), ensure_ascii=False),
    )

    pamphlet_resp, pamphlet_payload = _ask(
        session,
        base_url,
        "パンフレット",
        timeout=args.timeout,
        require_hit=not safe_mode,
        allow_safe=safe_mode,
    )
    logging.info(
        "pamphlet status=%s meta=%s",
        pamphlet_resp.status_code,
        json.dumps(pamphlet_payload.get("meta", {}), ensure_ascii=False),
    )
    if not safe_mode and not pamphlet_payload.get("answer"):
        raise RuntimeError("Pamphlet query returned an empty answer outside SAFE_MODE")

    fallback_resp, fallback_payload = _ask(
        session,
        base_url,
        "存在しないメニューです",
        timeout=args.timeout,
        require_hit=False,
        allow_safe=True,
    )
    logging.info(
        "fallback status=%s meta=%s",
        fallback_resp.status_code,
        json.dumps(fallback_payload.get("meta", {}), ensure_ascii=False),
    )

    logging.info("Smoke checks completed successfully")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
