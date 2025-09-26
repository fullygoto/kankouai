"""Hybrid RAG pipeline for pamphlet answers."""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - allow running without OpenAI
    OpenAI = None  # type: ignore

from . import pamphlet_search


logger = logging.getLogger(__name__)

_TOPK = int(os.getenv("PAMPHLET_TOPK", "12"))
_MMR_LAMBDA = float(os.getenv("PAMPHLET_MMR_LAMBDA", "0.4"))
_MMR_K = 8
_MIN_CONFIDENCE = float(os.getenv("PAMPHLET_MIN_CONFIDENCE", "0.42"))
_GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
_REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4o-mini")
_EMBED_MODEL = os.getenv("PAMPHLET_EMBED_MODEL", "text-embedding-3-small")

_CLIENT: Optional[Any] = None


@dataclass
class _EmbeddingStore:
    city: str
    key: Tuple[Optional[float], Tuple[str, ...], int]
    vectors: List[List[float]]
    chunk_index: Dict[Tuple[str, int], int]
    snapshot: pamphlet_search.CityIndexSnapshot


_EMBED_CACHE: Dict[str, _EmbeddingStore] = {}


_SRC_RE = re.compile(
    r"""
    ^(?P<city>[^/]+)/              # 五島市
    (?P<file>[^/]+?)               # 長崎五島観光ガイド.txt
    (?:\.(txt|md))?                # 拡張子は任意・あれば除去
    (?:/\d+(?:-\d+)?)?$            # /9-11 等の行範囲は捨てる
    """,
    re.X,
)


def normalize_sources(sources: Iterable[Any]) -> List[Tuple[str, str]]:
    """Normalize a sequence of source payloads to ``(city, stem)`` tuples."""

    seen = set()
    out: List[Tuple[str, str]] = []

    for raw in sources or []:
        city: Optional[str] = None
        file_: Optional[str] = None

        if isinstance(raw, str):
            match = _SRC_RE.match(raw.strip())
            if match:
                city = match.group("city")
                file_ = match.group("file")
        elif isinstance(raw, dict):
            city_val = raw.get("city") or raw.get("City")
            if city_val:
                city = pamphlet_search.city_label(str(city_val))

            file_val = raw.get("file") or raw.get("filename") or raw.get("path")
            if file_val:
                file_name = os.path.basename(str(file_val))
                file_ = re.sub(r"\.(txt|md)$", "", file_name, flags=re.I)

        if not city or not file_:
            continue

        key = (city, file_)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)

    return out


def format_sources_md(sources: Iterable[Any], heading: str = "### 出典") -> str:
    items = normalize_sources(sources)
    if not items:
        return ""
    body = "\n".join(f"- {city}/{file_}" for city, file_ in items)
    return f"{heading}\n{body}"


def configure(openai_client: Optional[Any]) -> None:
    global _CLIENT
    _CLIENT = openai_client


def _get_client() -> Optional[Any]:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if OpenAI is None:
        return None
    try:
        _CLIENT = OpenAI()
    except Exception:
        _CLIENT = None
    return _CLIENT


def answer_from_pamphlets(question: str, city: str) -> Dict[str, Any]:
    question = (question or "").strip()
    if not question:
        return {
            "answer": "資料に該当する記述が見当たりません。もう少し条件（市町/施設名/時期等）を教えてください。",
            "sources": [],
            "confidence": 0.0,
            "debug": {"reason": "empty question"},
        }

    queries = _rewrite_queries(question, city)
    # Always include original question first
    if question not in queries:
        queries = [question] + [q for q in queries if q != question]

    retrieval = _retrieve_chunks(city, queries)
    selected: List[_Candidate] = retrieval["selected"]
    confidence = retrieval["confidence"]
    debug_info = retrieval.get("debug", {})

    if confidence < _MIN_CONFIDENCE or not selected:
        return {
            "answer": "資料に該当する記述が見当たりません。もう少し条件（市町/施設名/時期等）を教えてください。",
            "sources": [],
            "confidence": confidence,
            "debug": debug_info,
        }

    prompt = _build_prompt(question, city, queries, selected)
    answer_text = _generate_answer(prompt)

    if not answer_text:
        answer_text = _fallback_answer(question, city, selected)

    sources = []
    for candidate in selected[:4]:
        chunk = candidate.chunk
        sources.append(
            {
                "city": city,
                "file": chunk.source_file,
                "line_from": chunk.line_start,
                "line_to": chunk.line_end,
                "snippet": chunk.text[:40],
            }
        )

    debug_payload = dict(debug_info)
    debug_payload["prompt"] = prompt

    return {
        "answer": answer_text,
        "sources": sources,
        "confidence": confidence,
        "debug": debug_payload,
    }


@dataclass
class _Candidate:
    chunk: pamphlet_search.PamphletChunk
    combined_score: float
    bm25_details: List[Dict[str, Any]]
    embed_details: List[Dict[str, Any]]
    vector_index: Optional[int]


def _rewrite_queries(question: str, city: str) -> List[str]:
    base = [question]
    client = _get_client()
    if not client or not os.getenv("OPENAI_API_KEY"):
        return base

    system = """あなたは長崎県五島列島の旅行案内スタッフです。利用者の質問意図を理解し、検索に適した日本語クエリを複数案出力します。"""
    user = (
        "質問を2〜4個の短い検索クエリに言い換えてください。\n"
        "- 同義語や正式名称、祭りなどの別名を含めます。\n"
        "- 質問の意図を補う補助キーワード（時期、エリア、交通手段など）があれば追加します。\n"
        "- 出力は箇条書きで、日本語のみ。英語表記が役立つ場合は括弧で併記してください。\n"
        f"市町: {pamphlet_search.city_label(city)}\n質問: {question}"
    )

    try:
        response = client.responses.create(
            model=_REWRITE_MODEL,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        text = getattr(response, "output_text", "")
    except Exception as exc:
        logger.warning("[pamphlet] query rewrite failed: %s", exc)
        return base

    if not text:
        return base

    lines = []
    for raw in text.splitlines():
        cleaned = raw.strip().lstrip("-・*●\t ")
        if cleaned:
            lines.append(cleaned)
    if not lines:
        return base

    uniq: List[str] = []
    for item in lines:
        if item not in uniq:
            uniq.append(item)
        if len(uniq) >= 4:
            break

    return uniq


def _retrieve_chunks(city: str, queries: Sequence[str]) -> Dict[str, Any]:
    bm25_limit = max(_TOPK * 2, _TOPK + 4)
    combined: Dict[Tuple[str, int], _Candidate] = {}
    query_vectors: Dict[str, List[float]] = {}
    embed_store = _ensure_embeddings(city)
    debug_details: Dict[str, Any] = {"queries": list(queries), "bm25": [], "embedding": []}

    for query in queries:
        bm25_results = pamphlet_search.search(city, query, bm25_limit)
        debug_details["bm25"].append(
            [
                {"file": res.chunk.source_file, "idx": res.chunk.chunk_index, "score": res.score}
                for res in bm25_results
            ]
        )
        for rank, res in enumerate(bm25_results, start=1):
            key = (res.chunk.source_file, res.chunk.chunk_index)
            existing = combined.get(key)
            score = 1.0 / (60 + rank)
            if not existing:
                combined[key] = _Candidate(
                    chunk=res.chunk,
                    combined_score=score,
                    bm25_details=[{"query": query, "rank": rank, "score": res.score}],
                    embed_details=[],
                    vector_index=_vector_index(embed_store, res.chunk),
                )
            else:
                existing.combined_score += score
                existing.bm25_details.append({"query": query, "rank": rank, "score": res.score})

        if not embed_store or not embed_store.vectors:
            continue
        query_vec = _embed_query(query)
        if not query_vec:
            continue
        query_vectors[query] = query_vec
        sims = []
        for idx, vector in enumerate(embed_store.vectors):
            score = _dot(vector, query_vec)
            sims.append((score, idx))
        sims.sort(key=lambda item: item[0], reverse=True)
        top_embed = sims[:bm25_limit]
        debug_details["embedding"].append(
            [
                {
                    "file": embed_store.snapshot.chunks[idx].source_file,
                    "idx": embed_store.snapshot.chunks[idx].chunk_index,
                    "score": float(score),
                }
                for score, idx in top_embed
            ]
        )
        for rank, (score, idx) in enumerate(top_embed, start=1):
            chunk = embed_store.snapshot.chunks[idx]
            key = (chunk.source_file, chunk.chunk_index)
            existing = combined.get(key)
            rrf_score = 1.0 / (60 + rank)
            if not existing:
                combined[key] = _Candidate(
                    chunk=chunk,
                    combined_score=rrf_score,
                    bm25_details=[],
                    embed_details=[{"query": query, "rank": rank, "score": float(score)}],
                    vector_index=idx,
                )
            else:
                existing.combined_score += rrf_score
                existing.embed_details.append({"query": query, "rank": rank, "score": float(score)})
                if existing.vector_index is None:
                    existing.vector_index = idx

    candidates = sorted(combined.values(), key=lambda c: c.combined_score, reverse=True)[:_TOPK]

    primary_query = queries[0] if queries else ""
    primary_vector = query_vectors.get(primary_query)
    mmr_selected = _apply_mmr(candidates, embed_store, primary_vector)

    confidence = _calculate_confidence(mmr_selected, embed_store, primary_vector)

    debug_details.update(
        {
            "combined": [
                {
                    "file": cand.chunk.source_file,
                    "idx": cand.chunk.chunk_index,
                    "score": cand.combined_score,
                    "bm25": cand.bm25_details,
                    "embedding": cand.embed_details,
                }
                for cand in candidates
            ],
            "selection": [
                {
                    "file": cand.chunk.source_file,
                    "idx": cand.chunk.chunk_index,
                }
                for cand in mmr_selected
            ],
            "confidence": confidence,
        }
    )

    return {
        "selected": mmr_selected,
        "confidence": confidence,
        "debug": debug_details,
    }


def _vector_index(store: Optional[_EmbeddingStore], chunk: pamphlet_search.PamphletChunk) -> Optional[int]:
    if not store:
        return None
    return store.chunk_index.get((chunk.source_file, chunk.chunk_index))


def _apply_mmr(
    candidates: List[_Candidate],
    store: Optional[_EmbeddingStore],
    query_vector: Optional[List[float]],
) -> List[_Candidate]:
    if not candidates:
        return []
    if not store or not query_vector:
        return candidates[:min(len(candidates), _MMR_K)]

    selected: List[_Candidate] = []
    remaining = candidates.copy()

    while remaining and len(selected) < _MMR_K:
        best_idx = None
        best_score = -float("inf")
        for idx, candidate in enumerate(remaining):
            vec_idx = candidate.vector_index
            if vec_idx is None:
                continue
            vector = store.vectors[vec_idx]
            relevance = _dot(vector, query_vector)
            if not selected:
                diversity = 0.0
            else:
                diversity = max(
                    _dot(vector, store.vectors[sel.vector_index])
                    for sel in selected
                    if sel.vector_index is not None
                )
            mmr_score = _MMR_LAMBDA * relevance - (1.0 - _MMR_LAMBDA) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(remaining.pop(best_idx))

    if not selected:
        return candidates[:min(len(candidates), _MMR_K)]
    return selected


def _calculate_confidence(
    selected: List[_Candidate],
    store: Optional[_EmbeddingStore],
    query_vector: Optional[List[float]],
) -> float:
    if not selected:
        return 0.0
    coverage = len({cand.chunk.source_file for cand in selected[:4]}) / max(1, min(4, len(selected)))
    if not store or not query_vector:
        return max(0.0, min(1.0, 0.5 * coverage))

    sims: List[float] = []
    for cand in selected[:4]:
        if cand.vector_index is None:
            continue
        sims.append(_dot(store.vectors[cand.vector_index], query_vector))
    if not sims:
        avg = 0.0
    else:
        avg = sum(max(0.0, s) for s in sims) / len(sims)
    confidence = 0.5 * coverage + 0.5 * min(1.0, max(0.0, avg))
    return max(0.0, min(1.0, confidence))


def _ensure_embeddings(city: str) -> Optional[_EmbeddingStore]:
    client = _get_client()
    if not client or not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        snapshot = pamphlet_search.snapshot(city)
    except Exception as exc:
        logger.warning("[pamphlet] snapshot failed for %s: %s", city, exc)
        return None
    key = (snapshot.last_mtime, tuple(snapshot.last_files), len(snapshot.chunks))
    cached = _EMBED_CACHE.get(city)
    if cached and cached.key == key:
        return cached
    if not snapshot.chunks:
        store = _EmbeddingStore(city=city, key=key, vectors=[], chunk_index={}, snapshot=snapshot)
        _EMBED_CACHE[city] = store
        return store

    texts = [chunk.text for chunk in snapshot.chunks]
    vectors = _embed_texts(texts)
    if not vectors or len(vectors) != len(snapshot.chunks):
        return None

    chunk_index = {
        (chunk.source_file, chunk.chunk_index): idx for idx, chunk in enumerate(snapshot.chunks)
    }
    store = _EmbeddingStore(city=city, key=key, vectors=vectors, chunk_index=chunk_index, snapshot=snapshot)
    _EMBED_CACHE[city] = store
    return store


def _embed_texts(texts: Sequence[str]) -> List[List[float]]:
    client = _get_client()
    if not client:
        return []
    vectors: List[List[float]] = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = client.embeddings.create(model=_EMBED_MODEL, input=list(batch))
        except Exception as exc:
            logger.warning("[pamphlet] embedding request failed: %s", exc)
            return []
        for item in response.data:
            vec = list(item.embedding)
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
    return vectors


def _embed_query(query: str) -> Optional[List[float]]:
    vectors = _embed_texts([query])
    if not vectors:
        return None
    return vectors[0]


def _dot(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    return float(sum(a * b for a, b in zip(vec_a, vec_b)))


def _build_prompt(
    question: str,
    city: str,
    queries: Sequence[str],
    selected: Sequence[_Candidate],
) -> str:
    lines = [
        "役割: あなたは長崎県五島列島の旅行案内専門スタッフです。",\
        "利用者の質問の意図を汲み、資料の該当箇所だけを根拠に整理します。",
        "以下の条件を必ず守ってください:",
        "1. まず質問の意図を一文で言い換えます。",
        "2. 資料から確認できた事実だけで2〜5行の要約を作成します。",
        "3. 数値・営業時間・アクセスなどは箇条書きで明確にします。",
        "4. 不明点は「資料に明記なし」と記載します。",
        "5. 最後に出典として「市町/ファイル名/行番号」の形式で最大4件列挙します。",
        "6. 出力は日本語・ですます調。原文の丸写しは避け、抜粋は10〜30字に留めます。",
    ]

    city_label = pamphlet_search.city_label(city)
    lines.append("")
    lines.append(f"質問: {question}")
    lines.append(f"対象市町: {city_label}")
    lines.append(f"検索クエリ候補: {', '.join(queries)}")
    lines.append("利用可能な資料抜粋:")

    for idx, cand in enumerate(selected, start=1):
        chunk = cand.chunk
        lines.append(
            f"[{idx}] {city_label}/{chunk.source_file} L{chunk.line_start}-{chunk.line_end}\n{chunk.text}"
        )

    lines.append("")
    lines.append("出力フォーマット:")
    lines.append("要約\n(2〜5行の文章)")
    lines.append("詳細\n- 箇条書き")
    lines.append("補足\n- 任意 (必要な場合のみ)")
    lines.append("出典\n- 市町/ファイル名/L開始-L終了")

    return "\n".join(lines)


def _generate_answer(prompt: str) -> str:
    client = _get_client()
    if not client or not os.getenv("OPENAI_API_KEY"):
        return ""
    try:
        response = client.responses.create(
            model=_GEN_MODEL,
            input=[
                {
                    "role": "system",
                    "content": "あなたは事実だけを整理する旅行案内スタッフです。出力指示に従い、日本語で端的に答えてください。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return getattr(response, "output_text", "").strip()
    except Exception as exc:
        logger.warning("[pamphlet] generation failed: %s", exc)
        return ""


def _fallback_answer(question: str, city: str, selected: Sequence[_Candidate]) -> str:
    city_label = pamphlet_search.city_label(city)
    lines = ["要約", "資料に基づく回答を作成できませんでした。以下は関連する抜粋です。", "", "詳細"]
    for cand in selected:
        chunk = cand.chunk
        snippet = chunk.text.replace("\n", " ")
        lines.append(f"- {city_label}/{chunk.source_file} (L{chunk.line_start}-{chunk.line_end}): {snippet[:80]}…")
    lines.append("")
    lines.append("出典")
    for cand in selected[:4]:
        chunk = cand.chunk
        lines.append(f"- {city_label}/{chunk.source_file}/L{chunk.line_start}-L{chunk.line_end}")
    return "\n".join(lines)
