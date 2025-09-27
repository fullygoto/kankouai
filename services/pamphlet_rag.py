"""Hybrid RAG pipeline for pamphlet answers."""

from __future__ import annotations

import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - allow running without OpenAI
    OpenAI = None  # type: ignore

from . import pamphlet_search
from .summary_config import SummaryBounds, get_summary_bounds, get_summary_style


logger = logging.getLogger(__name__)

_TOPK = int(os.getenv("PAMPHLET_TOPK", "12"))
_MMR_LAMBDA = float(os.getenv("PAMPHLET_MMR_LAMBDA", "0.4"))
_MMR_K = 8
_MIN_CONFIDENCE = float(os.getenv("PAMPHLET_MIN_CONFIDENCE", "0.42"))
_GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
_REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gpt-4o-mini")
_EMBED_MODEL = os.getenv("PAMPHLET_EMBED_MODEL", "text-embedding-3-small")
_CITATION_MIN_CHARS = int(os.getenv("CITATION_MIN_CHARS", "80"))
_CITATION_MIN_SCORE = float(os.getenv("CITATION_MIN_SCORE", "0.15"))

_LABEL_PATTERN = re.compile(r"\[\[(\d+)\]\]")


@dataclass
class _PromptConfig:
    prompt: str
    bounds: SummaryBounds


PROMPT_SUMMARY_POLITE_LONG = (
    "あなたは観光案内の編集者です。以下のコンテキスト【番号付き】だけを根拠に、\n"
    "日本語でやさしく丁寧に説明してください。\n\n"
    "制約:\n"
    "- 文章量は {length_note} 文字を目安（上限 {max_chars} 文字厳守）。\n"
    "- 各文の末尾に必ず [[番号]] を付与（与えたIDのみ使用）。\n"
    "- 外部知識や推測は厳禁。日時・数値・名称はコンテキストからのみ。\n"
    "- 箇条書きは避け、自然な段落1〜2個でまとめる。\n"
    "- 初めて読む人にも前提を補い、場所・見どころ・体験・注意点・アクセスなどを簡潔に補足する。\n"
    "- 同語反復を避け、言い換えで読みやすく。\n"
    "{extra_note}"
    "{question_line}"
    "{city_line}"
    "{query_line}\n\n"
    "【コンテキスト（番号付き）】\n"
    "{context}\n"
)


PROMPT_SUMMARY_TERSE_SHORT = (
    "あなたのタスクは、以下のコンテキストだけを根拠に日本語で要約を書くことです。\n"
    "ルール:\n"
    "- 各文末に必ず [[番号]] を付けてください（例：「〜であった。[[1]]」）。\n"
    "- [[番号]] は与えられた参照IDのみ使用し、推測や新規情報は禁止。\n"
    "- 出典数を増やすための不要な分割は禁止。簡潔に。\n"
    "- 指示されたセクション構成を守り、根拠は与えられた内容のみに限定する。\n\n"
    "{question_line}"
    "{city_line}"
    "{query_line}\n\n"
    "【コンテキスト（番号付き）】\n"
    "{context}\n\n"
    "出力フォーマット:\n"
    "### 要約\n"
    "質問に直接答える2〜4文。各文末に [[番号]] を付ける。\n\n"
    "### 詳細\n"
    "- 重要な追加事項を箇条書き（最大2行・各行末に [[番号]]）。必要なければ省略。\n\n"
    "### 出典\n"
    "- 市町/ファイル名（本文で使用した参照のみ）\n"
)

_CLIENT: Optional[Any] = None

_LAST_SUCCESS: Dict[Tuple[str, str], Dict[str, Any]] = {}
_SUCCESS_TTL = 1800.0


@dataclass
class _EmbeddingStore:
    city: str
    key: Tuple[Optional[float], Tuple[str, ...], int]
    vectors: List[List[float]]
    chunk_index: Dict[Tuple[str, int], int]
    snapshot: pamphlet_search.CityIndexSnapshot


_EMBED_CACHE: Dict[str, _EmbeddingStore] = {}


def _question_key(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "")).strip().lower()


def _prune_last_success(now: float) -> None:
    for key, info in list(_LAST_SUCCESS.items()):
        ts = float(info.get("ts", 0.0))
        if now - ts > _SUCCESS_TTL:
            _LAST_SUCCESS.pop(key, None)


def _store_last_success(city: str, question: str, candidates: Sequence["_Candidate"], queries: Sequence[str]) -> None:
    if not city or not question or not candidates:
        return
    now = time.time()
    _prune_last_success(now)
    key = (city, _question_key(question))
    payload = {
        "ts": now,
        "chunks": [(cand.chunk.source_file, cand.chunk.chunk_index) for cand in candidates[:4]],
        "queries": list(queries),
    }
    _LAST_SUCCESS[key] = payload


def _reuse_last_success(city: str, question: str) -> Optional[Dict[str, Any]]:
    if not city or not question:
        return None
    now = time.time()
    _prune_last_success(now)
    key = (city, _question_key(question))
    data = _LAST_SUCCESS.get(key)
    if not data:
        return None
    snapshot = None
    try:
        snapshot = pamphlet_search.snapshot(city)
    except Exception:
        return None
    index: Dict[Tuple[str, int], pamphlet_search.PamphletChunk] = {
        (chunk.source_file, chunk.chunk_index): chunk for chunk in snapshot.chunks
    }
    selected: List[_Candidate] = []
    for file_name, chunk_idx in data.get("chunks", []) or []:
        chunk = index.get((file_name, chunk_idx))
        if not chunk:
            continue
        selected.append(
            _Candidate(
                chunk=chunk,
                combined_score=1.0,
                bm25_details=[],
                embed_details=[],
                vector_index=None,
            )
        )
    if not selected:
        _LAST_SUCCESS.pop(key, None)
        return None
    return {"selected": selected, "queries": data.get("queries", [])}


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

    base_queries = [question]
    retrieval = _retrieve_chunks(city, base_queries)
    selected: List[_Candidate] = retrieval["selected"]
    confidence = retrieval["confidence"]
    debug_info = dict(retrieval.get("debug", {}) or {})
    used_queries = list(base_queries)

    if len(selected) < 2 and confidence < _MIN_CONFIDENCE:
        first_pass = dict(debug_info)
        rewritten = _rewrite_queries(question, city)
        expanded = [question] + [q for q in rewritten if q and q != question]
        if len(expanded) > 1:
            retry = _retrieve_chunks(city, expanded)
            selected = retry["selected"]
            confidence = retry["confidence"]
            debug_info = dict(retry.get("debug", {}) or {})
            if first_pass:
                debug_info["first_pass"] = first_pass
                debug_info["retry_reason"] = "low_confidence"
            used_queries = list(expanded)
        else:
            debug_info = first_pass

    reused_seed = False
    if (not selected or confidence < _MIN_CONFIDENCE) and question:
        reused = _reuse_last_success(city, question)
        if reused and reused.get("selected"):
            selected = reused["selected"]
            used_queries = reused.get("queries", used_queries)
            confidence = max(confidence, 0.5)
            reused_seed = True
            debug_info = dict(debug_info or {})
            debug_info["reuse_seed"] = True

    if not selected:
        return {
            "answer": "資料に該当する記述が見当たりません。もう少し条件（市町/施設名/時期等）を教えてください。",
            "sources": [],
            "confidence": confidence,
            "debug": debug_info,
        }

    context_text, id_map = build_context_with_labels(selected)
    prompt_cfg = _build_prompt(question, city, used_queries, context_text)
    answer_text = _generate_with_constraints(prompt_cfg)

    postprocessed = postprocess_answer(
        answer_text,
        id_map,
        min_chars=_CITATION_MIN_CHARS,
        min_score=_CITATION_MIN_SCORE,
    )

    if not postprocessed.answer_with_labels:
        fallback = _fallback_answer(question, city, selected)
        postprocessed = postprocess_answer(
            fallback,
            id_map,
            min_chars=_CITATION_MIN_CHARS,
            min_score=_CITATION_MIN_SCORE,
        )

    sources = [
        {
            "city": ref.city,
            "file": ref.title,
            "doc_id": ref.doc_id,
            "chunk_id": ref.chunk_id,
            "score": ref.score,
        }
        for ref in (id_map.get(label) for label in postprocessed.used_labels)
        if ref
    ]

    debug_payload = dict(debug_info or {})
    debug_payload["queries"] = used_queries
    debug_payload["prompt"] = prompt_cfg.prompt
    if postprocessed.invalid_labels:
        debug_payload["invalid_labels"] = sorted(postprocessed.invalid_labels)

    _store_last_success(city, question, selected, used_queries)

    return {
        "answer": postprocessed.answer_without_labels,
        "answer_with_labels": postprocessed.answer_with_labels,
        "citations": postprocessed.citations,
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


@dataclass
class CitationRef:
    doc_id: str
    chunk_id: str
    title: str
    city: str
    start_offset: int
    end_offset: int
    score: float
    text: str


@dataclass
class _PostProcessResult:
    answer_with_labels: str
    answer_without_labels: str
    used_labels: List[int]
    citations: List[Dict[str, Any]]
    invalid_labels: List[int]


def build_context_with_labels(selected: Sequence[_Candidate]) -> Tuple[str, Dict[int, CitationRef]]:
    parts: List[str] = []
    id_map: Dict[int, CitationRef] = {}

    for idx, candidate in enumerate(selected, start=1):
        chunk = candidate.chunk
        city_label = pamphlet_search.city_label(chunk.city)
        title = re.sub(r"\.(txt|md)$", "", chunk.source_file, flags=re.I)
        doc_id = f"{chunk.city}/{chunk.source_file}"
        chunk_id = f"{chunk.chunk_index}"
        ref = CitationRef(
            doc_id=doc_id,
            chunk_id=chunk_id,
            title=title,
            city=city_label,
            start_offset=chunk.char_start,
            end_offset=chunk.char_end,
            score=candidate.combined_score,
            text=chunk.text,
        )
        id_map[idx] = ref
        snippet = chunk.text.strip()
        header = f"[[{idx}]] {city_label}/{title} L{chunk.line_start}-{chunk.line_end}"
        parts.append(f"{header}\n{snippet}")

    return "\n\n".join(parts), id_map


def _extract_label_segments(answer_text: str, id_map: Dict[int, CitationRef]) -> Tuple[List[Tuple[int, str]], List[int], str, str]:
    used: List[Tuple[int, str]] = []
    invalid: List[int] = []
    labelled_parts: List[str] = []
    plain_parts: List[str] = []

    cursor = 0
    for match in _LABEL_PATTERN.finditer(answer_text or ""):
        start, end = match.span()
        text_before = (answer_text or "")[cursor:start]
        label = int(match.group(1))
        ref = id_map.get(label)
        if ref:
            used.append((label, text_before))
            labelled_parts.append(text_before + match.group(0))
            plain_parts.append(text_before)
        else:
            invalid.append(label)
            logger.warning("[pamphlet] drop sentence with undefined label: %s", label)
        cursor = end

    trailing = (answer_text or "")[cursor:]
    if trailing:
        labelled_parts.append(trailing)
        plain_parts.append(trailing)

    labelled = "".join(labelled_parts).strip()
    plain = "".join(plain_parts).strip()
    return used, invalid, labelled, plain


def _aggregate_citations(
    segments: List[Tuple[int, str]],
    id_map: Dict[int, CitationRef],
    *,
    min_chars: int,
    min_score: float,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    usage: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    valid_labels: List[int] = []

    for label, text in segments:
        ref = id_map.get(label)
        if not ref:
            continue
        doc_key = ref.doc_id
        if doc_key not in usage:
            usage[doc_key] = {
                "doc_id": ref.doc_id,
                "title": ref.title,
                "city": ref.city,
                "labels": set(),
                "chunk_ids": set(),
                "char_count": 0,
                "score": 0.0,
            }
            order.append(doc_key)
        payload = usage[doc_key]
        payload["labels"].add(label)
        payload["chunk_ids"].add(ref.chunk_id)
        payload["char_count"] += len(text.strip())
        payload["score"] += float(ref.score)
        valid_labels.append(label)

    citations: List[Dict[str, Any]] = []
    filtered_labels: List[int] = []
    for doc_key in order:
        payload = usage[doc_key]
        if payload["char_count"] < min_chars and payload["score"] < min_score:
            continue
        payload["labels"] = sorted(payload["labels"])
        payload["chunk_ids"] = sorted(payload["chunk_ids"], key=lambda x: int(x))
        payload["file"] = payload["title"]
        citations.append(payload)
        filtered_labels.extend(payload["labels"])

    return citations, filtered_labels


def postprocess_answer(
    answer_text: str,
    id_map: Dict[int, CitationRef],
    *,
    min_chars: int,
    min_score: float,
) -> _PostProcessResult:
    segments, invalid, labelled, plain = _extract_label_segments(answer_text or "", id_map)
    citations, allowed_labels = _aggregate_citations(segments, id_map, min_chars=min_chars, min_score=min_score)

    used_labels = allowed_labels
    if not labelled or not citations:
        note = "（注：根拠ラベルが見つかりませんでした）"
        if plain:
            if not plain.endswith(note):
                plain = plain + note
        else:
            plain = note
        labelled = ""
        citations = []
        used_labels = []

    return _PostProcessResult(
        answer_with_labels=labelled,
        answer_without_labels=plain,
        used_labels=used_labels,
        citations=citations,
        invalid_labels=invalid,
    )


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
    context_text: str,
) -> _PromptConfig:
    city_label = pamphlet_search.city_label(city)
    style = get_summary_style()
    bounds = get_summary_bounds(context_text)

    question_line = f"質問: {question}\n" if question else ""
    city_line = f"対象市町: {city_label}\n" if city_label else ""
    query_line = ""
    if queries:
        joined = ", ".join(q for q in queries if q)
        if joined:
            query_line = f"検索クエリ候補: {joined}\n"

    if style == "polite_long":
        length_note = f"{bounds.min_chars}〜{bounds.max_chars}"
        extra_note = "- コンテキストが短いため、無理に長文化せず自然な分量でまとめてください。" if bounds.is_short_context else ""
        extra_note_text = f"{extra_note}\n\n" if extra_note else "\n"
        prompt = PROMPT_SUMMARY_POLITE_LONG.format(
            length_note=length_note,
            max_chars=bounds.max_chars,
            extra_note=extra_note_text,
            question_line=question_line,
            city_line=city_line,
            query_line=query_line,
            context=context_text,
        )
        return _PromptConfig(prompt=prompt, bounds=bounds)

    prompt = PROMPT_SUMMARY_TERSE_SHORT.format(
        question_line=question_line,
        city_line=city_line,
        query_line=query_line,
        context=context_text,
    )
    return _PromptConfig(prompt=prompt, bounds=bounds)


def _generate_answer(
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> str:
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
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return getattr(response, "output_text", "").strip()
    except Exception as exc:
        logger.warning("[pamphlet] generation failed: %s", exc)
        return ""


def _normalize_generated_text(text: str) -> str:
    if not text:
        return ""
    lines = [(line or "").rstrip() for line in str(text).splitlines()]
    cleaned = "\n".join(lines).strip()
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned


def _truncate_sentences(text: str, limit: int) -> str:
    if not text:
        return ""
    if limit <= 0:
        return text.strip()
    trimmed = text.strip()
    if len(trimmed) <= limit:
        return trimmed
    matches = list(_LABEL_PATTERN.finditer(trimmed))
    if not matches:
        return trimmed[:limit].rstrip()
    for match in reversed(matches):
        end = match.end()
        candidate = trimmed[:end].rstrip()
        if len(candidate) <= limit:
            return candidate
    return trimmed[:limit].rstrip()


def _count_characters(text: str) -> int:
    if not text:
        return 0
    return len(text)


def _generate_with_constraints(prompt_cfg: _PromptConfig) -> str:
    style = get_summary_style()
    if style == "polite_long":
        gen_params = {
            "temperature": 0.5,
            "max_output_tokens": 1100,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.0,
        }
    else:
        gen_params = {
            "temperature": 0.1,
            "max_output_tokens": 700,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    attempts = 0
    last_text = ""
    while attempts < 2:
        raw = _generate_answer(prompt_cfg.prompt, **gen_params)
        cleaned = _normalize_generated_text(raw)
        trimmed = _truncate_sentences(cleaned, prompt_cfg.bounds.max_chars)
        char_count = _count_characters(trimmed)
        if not trimmed:
            attempts += 1
            last_text = trimmed
            continue
        if char_count < prompt_cfg.bounds.min_chars and attempts == 0:
            attempts += 1
            last_text = trimmed
            continue
        if char_count > prompt_cfg.bounds.max_chars:
            trimmed = _truncate_sentences(trimmed, prompt_cfg.bounds.max_chars)
        return trimmed

    return last_text


def _fallback_answer(question: str, city: str, selected: Sequence[_Candidate]) -> str:
    city_label = pamphlet_search.city_label(city)
    if get_summary_style() == "polite_long":
        if not selected:
            return "資料に該当する記述が見当たりませんでした。"

        primary = selected[0].chunk
        primary_title = re.sub(r"\.(txt|md)$", "", primary.source_file, flags=re.I)
        intro = f"{city_label}の資料「{primary_title}」にはご質問と関連する案内が掲載されています。[[1]]"

        detail_sentences: List[str] = []
        for idx, cand in enumerate(selected[:2], start=1):
            chunk = cand.chunk
            title = re.sub(r"\.(txt|md)$", "", chunk.source_file, flags=re.I)
            snippet = re.sub(r"\s+", " ", chunk.text).strip()
            if not snippet:
                continue
            excerpt = snippet[:120]
            detail_sentences.append(
                f"例えば「{title}」では「{excerpt}」と紹介され、旅の雰囲気をイメージできます。[[{idx}]]"
            )

        if not detail_sentences:
            detail_sentences.append(
                "資料を参照すると、交通や見どころの概要が分かり、行程づくりの助けになります。[[1]]"
            )

        body = " ".join(detail_sentences)
        return f"{intro}\n\n{body}".strip()

    if not selected:
        return "### 要約\n資料に該当する記述が見当たりませんでした。"

    primary = selected[0].chunk
    primary_title = re.sub(r"\.(txt|md)$", "", primary.source_file, flags=re.I)
    summary = f"{city_label}の資料「{primary_title}」に関連情報があります。[[1]]"

    lines = ["### 要約", summary, "", "### 詳細"]

    for idx, cand in enumerate(selected[:2], start=1):
        chunk = cand.chunk
        title = re.sub(r"\.(txt|md)$", "", chunk.source_file, flags=re.I)
        snippet = re.sub(r"\s+", " ", chunk.text).strip()
        if snippet:
            lines.append(f"- {title} の抜粋: {snippet[:80]}[[{idx}]]")

    lines.append("")
    lines.append("### 出典")
    for idx, cand in enumerate(selected[:4], start=1):
        chunk = cand.chunk
        title = re.sub(r"\.(txt|md)$", "", chunk.source_file, flags=re.I)
        lines.append(f"- {city_label}/{title}[[{idx}]]")

    return "\n".join(lines)
