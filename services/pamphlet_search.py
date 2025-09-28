"""Pamphlet search utilities for city-specific knowledge bases."""

from __future__ import annotations

import bisect
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from services.paths import get_data_base_dir


logger = logging.getLogger(__name__)

CITY_KEYS = ("goto", "shinkamigoto", "ojika", "uku")
CITY_LABELS = {
    "goto": "五島市",
    "shinkamigoto": "新上五島町",
    "ojika": "小値賀町",
    "uku": "宇久町",
}

# 追加の表記ゆれ（漢字から町/市を除いたりカナ表記など）
CITY_ALIASES: Dict[str, str] = {
    "五島": "goto",
    "五島市": "goto",
    "新上五島": "shinkamigoto",
    "新上五島町": "shinkamigoto",
    "小値賀": "ojika",
    "小値賀町": "ojika",
    "宇久": "uku",
    "宇久町": "uku",
}


def city_label(key: str) -> str:
    return CITY_LABELS.get(key, key)


@dataclass
class PamphletChunk:
    city: str
    source_file: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    line_start: int
    line_end: int


@dataclass
class SearchResult:
    chunk: PamphletChunk
    score: float


@dataclass
class CityIndexSnapshot:
    city: str
    chunks: List[PamphletChunk]
    last_mtime: Optional[float]
    last_files: List[str]


class _PamphletIndex:
    """Simple TF-IDF index for chunks within a single city."""

    def __init__(self, city: str, *, chunk_size: int, overlap: int) -> None:
        self.city = city
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[PamphletChunk] = []
        self.vectors: List[Dict[str, float]] = []
        self.df: Dict[str, int] = {}
        self.last_mtime: Optional[float] = None
        self.last_error: Optional[str] = None
        self.last_files: List[str] = []
        self._lock = threading.RLock()

    def rebuild(self, files: Iterable[Path]) -> None:
        file_list = list(files)
        with self._lock:
            self.chunks.clear()
            self.vectors.clear()
            self.df.clear()
            texts: List[str] = []
            chunk_meta: List[PamphletChunk] = []
            max_mtime: Optional[float] = None
            for file_path in file_list:
                try:
                    raw = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    raw = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as exc:
                    logger.warning("[pamphlet] failed to read %s: %s", file_path, exc)
                    continue
                chunks = _split_text(raw, self.chunk_size, self.overlap)
                for idx, segment in enumerate(chunks):
                    chunk_text, start_pos, end_pos, line_start, line_end = segment
                    chunk_meta.append(
                        PamphletChunk(
                            city=self.city,
                            source_file=file_path.name,
                            chunk_index=idx,
                            text=chunk_text,
                            char_start=start_pos,
                            char_end=end_pos,
                            line_start=line_start,
                            line_end=line_end,
                        )
                    )
                    texts.append(chunk_text)
                try:
                    stat = file_path.stat()
                    max_mtime = max(max_mtime or 0.0, stat.st_mtime)
                except OSError:
                    continue

            if not chunk_meta:
                self.chunks = []
                self.vectors = []
                self.df = {}
                self.last_mtime = max_mtime
                self.last_error = None
                self.last_files = sorted(f.name for f in file_list)
                return

            # Build TF counts first
            tf_list: List[Dict[str, int]] = []
            df_counter: Dict[str, int] = {}
            for meta, text in zip(chunk_meta, texts):
                tokens = _tokenize(text)
                counts: Dict[str, int] = {}
                for token in tokens:
                    counts[token] = counts.get(token, 0) + 1
                tf_list.append(counts)
                for token in counts:
                    df_counter[token] = df_counter.get(token, 0) + 1

            doc_count = len(tf_list)
            vectors: List[Dict[str, float]] = []
            for counts in tf_list:
                vec: Dict[str, float] = {}
                norm = 0.0
                for token, tf in counts.items():
                    idf = math.log((doc_count + 1) / (df_counter[token] + 1)) + 1.0
                    value = float(tf) * idf
                    vec[token] = value
                    norm += value * value
                if norm > 0:
                    norm_sqrt = math.sqrt(norm)
                    for token in list(vec):
                        vec[token] /= norm_sqrt
                vectors.append(vec)

            self.chunks = chunk_meta
            self.vectors = vectors
            self.df = df_counter
            self.last_mtime = max_mtime
            self.last_error = None
            self.last_files = sorted(f.name for f in file_list)

    def search(self, query: str, topk: int) -> List[SearchResult]:
        with self._lock:
            if not self.chunks or not query.strip():
                return []
            q_tokens = _tokenize(query)
            if not q_tokens:
                return []
            q_counts: Dict[str, int] = {}
            for token in q_tokens:
                q_counts[token] = q_counts.get(token, 0) + 1
            doc_count = len(self.chunks)
            q_vec: Dict[str, float] = {}
            norm = 0.0
            for token, tf in q_counts.items():
                df = self.df.get(token, 0)
                if df <= 0:
                    continue
                idf = math.log((doc_count + 1) / (df + 1)) + 1.0
                value = float(tf) * idf
                q_vec[token] = value
                norm += value * value
            if not q_vec:
                return []
            norm_sqrt = math.sqrt(norm)
            for token in list(q_vec):
                q_vec[token] /= norm_sqrt

            scored: List[SearchResult] = []
            for chunk, vec in zip(self.chunks, self.vectors):
                score = 0.0
                for token, weight in q_vec.items():
                    if token in vec:
                        score += weight * vec[token]
                if score > 0:
                    scored.append(SearchResult(chunk=chunk, score=score))
            scored.sort(key=lambda r: r.score, reverse=True)
            return scored[:topk]

    def snapshot(self) -> CityIndexSnapshot:
        with self._lock:
            return CityIndexSnapshot(
                city=self.city,
                chunks=list(self.chunks),
                last_mtime=self.last_mtime,
                last_files=list(self.last_files),
            )


class PamphletIndexManager:
    def __init__(self) -> None:
        self.base_dir = Path(os.getenv("PAMPHLET_BASE_DIR") or get_data_base_dir() / "pamphlets")
        self.chunk_size = int(os.getenv("PAMPHLET_CHUNK_SIZE", "700"))
        self.chunk_overlap = int(os.getenv("PAMPHLET_CHUNK_OVERLAP", "150"))
        self._indexes: Dict[str, _PamphletIndex] = {}
        self._lock = threading.RLock()
        self._status: Dict[str, Dict[str, Optional[str]]] = {}

    def configure(self, *, base_dir: Optional[str] = None, chunk_size: Optional[int] = None,
                  chunk_overlap: Optional[int] = None) -> None:
        with self._lock:
            if base_dir:
                self.base_dir = Path(base_dir)
            if chunk_size:
                self.chunk_size = int(chunk_size)
            if chunk_overlap is not None:
                self.chunk_overlap = int(chunk_overlap)

    def _city_dir(self, city: str) -> Path:
        root = self.base_dir
        path = (root / city).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            raise ValueError("Invalid city directory")
        return path

    def _collect_files(self, city: str) -> List[Path]:
        path = self._city_dir(city)
        if not path.exists() or not path.is_dir():
            return []
        files: List[Path] = []
        for item in sorted(path.iterdir()):
            if item.is_file() and item.suffix.lower() == ".txt":
                files.append(item)
        return files

    def load_index(self, city: str) -> None:
        if city not in CITY_KEYS:
            raise KeyError(city)
        files = self._collect_files(city)
        last_files = sorted(f.name for f in files)
        latest_mtime = 0.0
        for f in files:
            try:
                st = f.stat()
                latest_mtime = max(latest_mtime, st.st_mtime)
            except OSError:
                continue
        with self._lock:
            index = self._indexes.get(city)
            if index is None:
                index = _PamphletIndex(city, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
                self._indexes[city] = index
            needs = (
                index.last_mtime is None
                or latest_mtime > (index.last_mtime or 0.0)
                or last_files != index.last_files
            )
        if needs:
            try:
                index.rebuild(files)
                status = {"state": "ready", "message": None}
                if not index.chunks:
                    status = {"state": "empty", "message": "no documents"}
            except Exception as exc:
                index.last_error = str(exc)
                status = {"state": "error", "message": str(exc)}
                logger.exception("[pamphlet] rebuild failed for %s", city)
            with self._lock:
                self._status[city] = status
        else:
            with self._lock:
                if city not in self._status:
                    self._status[city] = {"state": "ready", "message": None}

    def search(self, city: str, query: str, topk: int) -> List[SearchResult]:
        if city not in CITY_KEYS:
            return []
        self.load_index(city)
        index = self._indexes.get(city)
        if not index:
            return []
        return index.search(query, topk)

    def snapshot(self, city: str) -> CityIndexSnapshot:
        if city not in CITY_KEYS:
            raise KeyError(city)
        self.load_index(city)
        index = self._indexes.get(city)
        if not index:
            return CityIndexSnapshot(city=city, chunks=[], last_mtime=None, last_files=[])
        return index.snapshot()

    def reindex_all(self) -> Dict[str, Dict[str, Optional[str]]]:
        results: Dict[str, Dict[str, Optional[str]]] = {}
        for city in CITY_KEYS:
            try:
                self.load_index(city)
                results[city] = self._status.get(city, {"state": "ready", "message": None})
            except Exception as exc:
                results[city] = {"state": "error", "message": str(exc)}
        return results

    def status(self) -> Dict[str, Dict[str, Optional[str]]]:
        with self._lock:
            out = {}
            for city in CITY_KEYS:
                out[city] = self._status.get(city, {"state": "unknown", "message": None})
            return out

    def overall_state(self) -> str:
        st = self.status()
        states = {info.get("state", "unknown") for info in st.values()}
        if "error" in states:
            return "error"
        if "unknown" in states:
            return "unknown"
        return "ready" if any(s in {"ready", "empty"} for s in states) else "unknown"


_manager = PamphletIndexManager()


def configure(settings: Optional[Dict[str, object]] = None) -> None:
    if not settings:
        return
    base_dir = settings.get("PAMPHLET_BASE_DIR") if isinstance(settings, dict) else None
    chunk_size = settings.get("PAMPHLET_CHUNK_SIZE") if isinstance(settings, dict) else None
    chunk_overlap = settings.get("PAMPHLET_CHUNK_OVERLAP") if isinstance(settings, dict) else None
    _manager.configure(base_dir=base_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def load_index(city: str) -> None:
    _manager.load_index(city)


def load_all() -> None:
    for key in CITY_KEYS:
        try:
            load_index(key)
        except Exception:
            logger.exception("[pamphlet] initial load failed for %s", key)


def search(city: str, query: str, topk: int) -> List[SearchResult]:
    return _manager.search(city, query, topk)


def snapshot(city: str) -> CityIndexSnapshot:
    return _manager.snapshot(city)


def reindex_all() -> Dict[str, Dict[str, Optional[str]]]:
    return _manager.reindex_all()


def status() -> Dict[str, Dict[str, Optional[str]]]:
    return _manager.status()


def overall_state() -> str:
    return _manager.overall_state()


def detect_city_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    for alias, key in CITY_ALIASES.items():
        if alias.lower() in lowered:
            return key
    return None


def available_cities() -> List[str]:
    return list(CITY_KEYS)


def city_choices() -> List[Dict[str, str]]:
    return [{"label": CITY_LABELS[c], "text": CITY_LABELS[c]} for c in CITY_KEYS]


def _split_text(text: str, chunk_size: int, overlap: int) -> List[tuple[str, int, int, int, int]]:
    """Split text into trimmed segments with positional metadata."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    line_starts = _build_line_starts(normalized)

    if chunk_size <= 0:
        stripped = normalized.strip()
        if not stripped:
            return []
        start_offset = normalized.find(stripped)
        end_offset = start_offset + len(stripped)
        return [
            (
                stripped,
                start_offset,
                end_offset,
                _line_number(line_starts, start_offset),
                _line_number(line_starts, end_offset),
            )
        ]

    step = max(1, chunk_size - max(0, overlap))
    segments: List[tuple[str, int, int, int, int]] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(length, start + chunk_size)
        window = normalized[start:end]
        stripped = window.strip()
        if stripped:
            local_offset = window.find(stripped)
            abs_start = start + max(0, local_offset)
            abs_end = abs_start + len(stripped)
            segments.append(
                (
                    stripped,
                    abs_start,
                    abs_end,
                    _line_number(line_starts, abs_start),
                    _line_number(line_starts, abs_end),
                )
            )
        if end >= length:
            break
        start += step
    return segments


def _build_line_starts(text: str) -> List[int]:
    starts = [0]
    for idx, char in enumerate(text):
        if char == "\n":
            starts.append(idx + 1)
    starts.append(len(text) + 1)
    return starts


def _line_number(line_starts: List[int], offset: int) -> int:
    """Return 1-indexed line number for a character offset."""

    return bisect.bisect_right(line_starts, offset)


def _tokenize(text: str) -> List[str]:
    import re

    text = text.lower()
    alnum = re.findall(r"[a-z0-9]+", text)
    text_wo_space = re.sub(r"\s+", "", text)
    bigrams = [text_wo_space[i:i + 2] for i in range(len(text_wo_space) - 1)]
    if len(text_wo_space) == 1:
        bigrams.append(text_wo_space)
    if not bigrams and text_wo_space:
        bigrams = [text_wo_space]
    return [tok for tok in alnum + bigrams if tok]
