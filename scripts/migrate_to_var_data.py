#!/usr/bin/env python3
"""Migrate legacy data directories into DATA_BASE_DIR (/var/data by default)."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Tuple

DEFAULT_TARGET = Path(os.getenv("DATA_BASE_DIR", "/var/data")).expanduser()

CANDIDATES: Tuple[Tuple[Path, Path], ...] = (
    (Path("./data/pamphlets"), Path("pamphlets")),
    (Path("./data/entries.json"), Path("entries/entries.json")),
    (Path("./data/synonyms.json"), Path("entries/synonyms.json")),
    (Path("./data/users.json"), Path("entries/users.json")),
    (Path("./data/notices.json"), Path("entries/notices.json")),
    (Path("./data/shop_infos.json"), Path("entries/shop_infos.json")),
    (Path("./data/paused_notice.json"), Path("entries/paused_notice.json")),
    (Path("./data/logs"), Path("logs/legacy")),
    (Path("./static/uploads"), Path("uploads/static")),
    (Path("./uploads"), Path("uploads")),
    (Path("/opt/render/project/src/uploads"), Path("uploads/legacy")),
)


def iter_sources() -> Iterable[Tuple[Path, Path]]:
    for source, target in CANDIDATES:
        resolved = source.expanduser().resolve()
        if resolved.exists():
            yield resolved, target


def copy_item(src: Path, dest: Path, *, execute: bool) -> str:
    if dest.exists():
        return f"SKIP {src} -> {dest} (already exists)"
    if not execute:
        return f"DRY-RUN copy {src} -> {dest}"
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return f"COPIED {src} -> {dest}"


def migrate(base_dir: Path, *, execute: bool) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for src, relative in iter_sources():
        dest = (base_dir / relative).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        message = copy_item(src, dest, execute=execute)
        print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="actually perform the copy")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="override DATA_BASE_DIR destination")
    args = parser.parse_args()

    migrate(args.target.expanduser(), execute=args.execute)


if __name__ == "__main__":
    main()
