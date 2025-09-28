"""Hybrid RAG pipeline for pamphlet answers."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - allow running without OpenAI
    OpenAI = None  # type: ignore

from . import pamphlet_search
from .pamphlet_planner import Plan, plan_answer
from .summary_config import SummaryBounds, get_summary_bounds, get_summary_style


logger = logging.getLogger(__name__)

_TOPK = int(os.getenv("PAMPHLET_TOPK", "12"))
_MMR_LAMBDA = float(os.getenv("PAMPHLET_MMR_LAMBDA", "0.4"))
_MMR_K = 8
_MIN_CONFIDENCE = float(os.getenv("PAMPHLET_MIN_CONFIDENCE", "0.42"))
_GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
_EMBED_MODEL = os.getenv("PAMPHLET_EMBED_MODEL", "text-embedding-3-small")
_CITATION_MIN_CHARS = int(os.getenv("CITATION_MIN_CHARS", "80"))
_CITATION_MIN_SCORE = float(os.getenv("CITATION_MIN_SCORE", "0.15"))

_LABEL_PATTERN = re.compile(r"\[\[(\d+)\]\]")
_NOISE_YEAR_LINE = re.compile(r"^[0-9０-９]{1,4}年(?:頃|代|台|以降)?$")
_NOISE_NUMBER_LINE = re.compile(r"^[0-9０-９]+$")
_NOISE_MARK_LINE = re.compile(r"^[\s・*●○◆◇▶▷■□◉◎]+$")
_NOISE_FOOTNOTE_LINE = re.compile(r"^\[[0-9０-９]+\]$")
_FOCUS_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9一-龠々〆ヵヶぁ-んァ-ンゝゞー]{2,}")
_FOCUS_BULLET_PREFIX = re.compile(r"^[\s\-・*●○◆◇▶▷■□◉◎]+")


@dataclass
class _PromptConfig:
    prompt: str
    bounds: SummaryBounds
    style: str


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

PROMPT_ADAPTIVE_TEMPLATE = (
    "あなたは観光案内の編集者です。以下の【パンフ本文抜粋（番号付き）】だけを根拠に日本語で回答します。\n\n"
    "ルール:\n"
    "- 質問の核心にまず1段落で答え、その後「### 詳細」で3〜5行の補足を必要な場合だけ付ける。\n"
    "- 不要な背景や別テーマ（施設年表など）は出力しない。質問と直接関係する事実のみ。\n"
    "- 固有名や年数は与えた抜粋内に明記があるものだけ。推測や一般知識は書かない。\n"
    "- 文末に [[番号]] を必ず付け、使った番号に対応する文献のみ「### 出典」に列挙。\n"
    "- 文字数は最大800字。短くて回答が完結するなら短くしてよい。\n\n"
    "【ユーザー質問】\n"
    "{question}\n\n"
    "【前提スコープ】自治体: {city}\n\n"
    "【パンフ本文抜粋（番号付き）】\n"
    "{context}\n"
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


@lru_cache(maxsize=1)
def _load_synonym_map() -> Dict[str, List[str]]:
    path = Path(__file__).resolve().parent.parent / "synonyms.json"
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        return {}

    cleaned: Dict[str, List[str]] = {}
    for key, value in (raw or {}).items():
        if not isinstance(key, str) or not isinstance(value, list):
            continue
        variants = [str(item) for item in value if isinstance(item, str)]
        cleaned[key] = variants
    return cleaned


def _clean_chunk_text(text: str) -> str:
    lines: List[str] = []
    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if _NOISE_MARK_LINE.match(stripped):
            continue
        if _NOISE_YEAR_LINE.fullmatch(stripped) or _NOISE_NUMBER_LINE.fullmatch(stripped):
            continue
        if _NOISE_FOOTNOTE_LINE.match(stripped) and len(stripped) <= 6:
            continue
        if len(stripped) <= 2:
            continue
        if not any(ch in stripped for ch in "。.!?！？"):
            separator_count = sum(stripped.count(ch) for ch in "、,／/・ ")
            if separator_count >= 3 and not re.search(r"(です|ます|する|した|れる|られる|開催|紹介|位置|伝わる)", stripped):
                continue
            if len(stripped) <= 6:
                continue
        lines.append(stripped)
    return "\n".join(lines)


def _extract_focus_tokens(plan: Plan, question: str) -> List[str]:
    tokens: List[str] = []
    for key in ("keywords", "time"):
        tokens.extend(plan.scope.get(key, []))
    tokens.extend(_FOCUS_TOKEN_PATTERN.findall(question or ""))

    split_tokens: List[str] = []
    separators = ["の", "は", "を", "が", "に", "で", "と", "へ", "より"]
    for token in tokens:
        for sep in separators:
            if sep in token:
                parts = [part.strip() for part in token.split(sep) if len(part.strip()) >= 2]
                split_tokens.extend(parts)
    tokens.extend(split_tokens)

    seen: List[str] = []
    for token in tokens:
        cleaned = (token or "").strip()
        if len(cleaned) < 2:
            continue
        if cleaned in seen:
            continue
        seen.append(cleaned)
    return seen


def _should_keep_focus_line(content: str, tokens: Sequence[str]) -> bool:
    lowered = content.lower()
    for token in tokens:
        norm = token.lower()
        if norm and norm in lowered:
            return True
    return False


def _filter_focus_text(labelled_text: str, tokens: Sequence[str]) -> str:
    lines = labelled_text.splitlines()
    if not lines:
        return labelled_text

    keep_flags: List[bool] = [False] * len(lines)
    keep_all_refs = False
    lowered_tokens = [tok.lower() for tok in tokens if tok]

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("### 出典"):
            keep_flags[idx] = True
            keep_all_refs = True
            continue
        if keep_all_refs:
            keep_flags[idx] = bool(stripped)
            continue
        if stripped.startswith("###"):
            keep_flags[idx] = True
            continue
        content = stripped
        if stripped.startswith("-"):
            content = _FOCUS_BULLET_PREFIX.sub("", stripped, count=1).strip()
        if _should_keep_focus_line(content, lowered_tokens):
            keep_flags[idx] = True

    for idx, line in enumerate(lines):
        if keep_flags[idx]:
            continue
        if line.strip():
            continue
        prev_keep = idx > 0 and keep_flags[idx - 1]
        next_keep = idx + 1 < len(lines) and keep_flags[idx + 1]
        if prev_keep or next_keep:
            keep_flags[idx] = True

    # Remove empty detail sections
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped.startswith("### 詳細"):
            j = idx + 1
            has_content = False
            while j < len(lines):
                if lines[j].strip().startswith("###"):
                    break
                if keep_flags[j] and lines[j].strip():
                    has_content = True
                    break
                j += 1
            if not has_content:
                for k in range(idx, j):
                    keep_flags[k] = False
            idx = j
            continue
        idx += 1

    filtered: List[str] = []
    last_blank = False
    for line, keep in zip(lines, keep_flags):
        if not keep:
            continue
        if not line.strip():
            if last_blank or not filtered:
                continue
            filtered.append("")
            last_blank = True
            continue
        filtered.append(line.rstrip())
        last_blank = False

    return "\n".join(filtered).strip()


def _replace_reference_section(text: str, entries: Sequence[str]) -> str:
    if not text:
        return text
    lines = text.splitlines()
    result: List[str] = []
    replaced = False
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped.startswith("### 出典"):
            replaced = True
            result.append("### 出典")
            idx += 1
            while idx < len(lines) and not lines[idx].strip().startswith("###"):
                idx += 1
            for entry in entries:
                result.append(entry)
            continue
        result.append(lines[idx].rstrip())
        idx += 1

    if not replaced:
        if result and result[-1].strip():
            result.append("")
        result.append("### 出典")
        for entry in entries:
            result.append(entry)

    cleaned: List[str] = []
    last_blank = False
    for line in result:
        stripped = line.strip()
        if not stripped:
            if last_blank or not cleaned:
                continue
            cleaned.append("")
            last_blank = True
            continue
        cleaned.append(line)
        last_blank = False

    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return "\n".join(cleaned)


def _normalize_reference_section(result: "_PostProcessResult") -> "_PostProcessResult":
    if not result.citations:
        return result

    entries: List[str] = []
    seen: set[str] = set()
    for citation in result.citations:
        doc_id = citation.get("doc_id")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        if "/" in doc_id:
            city_key, file_name = doc_id.split("/", 1)
        else:
            city_key, file_name = "", doc_id
        city_label = pamphlet_search.city_label(city_key)
        title = re.sub(r"\.(txt|md)$", "", file_name, flags=re.I)
        entries.append(f"- {city_label}/{title}")

    if not entries:
        return result

    result.answer_with_labels = _replace_reference_section(result.answer_with_labels, entries)
    result.answer_without_labels = _replace_reference_section(result.answer_without_labels, entries)
    return result


def _apply_focus_guard(
    result: "_PostProcessResult",
    tokens: Sequence[str],
    id_map: Dict[int, "CitationRef"],
) -> "_PostProcessResult":
    if not result.answer_with_labels or not tokens:
        return result

    filtered = _filter_focus_text(result.answer_with_labels, tokens)
    if not filtered or filtered == result.answer_with_labels:
        return result

    refreshed = postprocess_answer(
        filtered,
        id_map,
        min_chars=0,
        min_score=_CITATION_MIN_SCORE,
    )
    if not refreshed.answer_with_labels:
        return result
    return refreshed


def _ensure_city_reference(result: "_PostProcessResult", city_label: str) -> "_PostProcessResult":
    if not city_label:
        return result
    plain = result.answer_without_labels or ""
    if not plain.strip():
        return result
    if city_label in plain:
        return result
    labelled = result.answer_with_labels or ""
    if not labelled.strip():
        return result

    label_lines = labelled.splitlines()
    plain_lines = plain.splitlines()
    modified = False

    for idx, (lbl_line, plain_line) in enumerate(zip(label_lines, plain_lines)):
        stripped_lbl = lbl_line.strip()
        stripped_plain = plain_line.strip()
        if not stripped_lbl or stripped_lbl.startswith("###") or stripped_lbl.startswith("-"):
            continue
        injected_plain = plain_line.replace(stripped_plain, f"{city_label}の資料によると、{stripped_plain}", 1)
        injected_labelled = lbl_line.replace(stripped_lbl, f"{city_label}の資料によると、{stripped_lbl}", 1)
        plain_lines[idx] = injected_plain
        label_lines[idx] = injected_labelled
        modified = True
        break

    if modified:
        result.answer_with_labels = "\n".join(label_lines)
        result.answer_without_labels = "\n".join(plain_lines)

    return result


def _expand_with_synonyms(question: str, plan: Plan) -> List[str]:
    base = (question or "").strip()
    if not base:
        return []

    synonym_map = _load_synonym_map()
    seen: List[str] = []
    keywords = plan.scope.get("keywords", [])
    clusters: List[Tuple[str, List[str]]] = []
    for key, variants in synonym_map.items():
        cluster = [key] + variants
        if any(item in base for item in cluster) or any(item in keywords for item in cluster):
            clusters.append((key, cluster))

    for token in keywords:
        if token not in base:
            continue
        for key, cluster in clusters:
            if token in cluster:
                for alt in cluster:
                    if alt == token:
                        continue
                    replaced = base.replace(token, alt)
                    if replaced != base and replaced not in seen:
                        seen.append(replaced)

    for key, cluster in clusters:
        matches = [variant for variant in cluster if variant in base]
        if not matches:
            continue
        remainder = base
        for match in matches:
            remainder = remainder.replace(match, key)
        if remainder != base and remainder not in seen:
            seen.append(remainder)

    if not seen and keywords:
        for token in keywords:
            for key, cluster in clusters:
                if token in cluster:
                    others = [kw for kw in keywords if kw != token]
                    for alt in cluster:
                        if alt == token:
                            continue
                        candidate = " ".join([alt] + others)
                        candidate = candidate.strip()
                        if candidate and candidate not in seen:
                            seen.append(candidate)

    return seen[:4]


def _apply_scope(plan: Plan, candidates: Sequence[_Candidate]) -> List[_Candidate]:
    if not candidates:
        return []
    keywords = plan.scope.get("keywords", [])
    time_scope = plan.scope.get("time", [])
    tokens = [tok for tok in keywords if tok]
    tokens.extend(time_scope)
    unique_tokens = [tok for idx, tok in enumerate(tokens) if tok and tok not in tokens[:idx]]
    if not unique_tokens:
        return list(candidates)

    filtered: List[_Candidate] = []
    for cand in candidates:
        text = cand.chunk.text
        if any(token in text for token in unique_tokens):
            filtered.append(cand)

    if not filtered:
        return list(candidates)

    style = plan.style or "medium_structured"
    limit = {
        "short_direct": 4,
        "medium_structured": 6,
        "long_explanatory": 8,
    }.get(style, 6)
    return filtered[:limit]


def _format_plan_block(plan: Plan, city_label: str) -> str:
    intent = plan.intent or "overview"
    style = plan.style or "medium_structured"
    sections = " / ".join(plan.sections) if plan.sections else "結論"
    scope_parts: List[str] = []
    keywords = plan.scope.get("keywords", [])
    if keywords:
        scope_parts.append("キーワード=" + "、".join(keywords[:6]))
    time_scope = plan.scope.get("time", [])
    if time_scope:
        scope_parts.append("時期=" + "、".join(time_scope[:4]))
    if city_label:
        scope_parts.append(f"市町={city_label}")
    scope_line = " / ".join(scope_parts) if scope_parts else "指定なし"
    return (
        f"意図: {intent}\n"
        f"範囲: {scope_line}\n"
        f"スタイル: {style}\n"
        f"セクション: {sections}"
    )


def _resolve_bounds(plan: Plan, context_text: str) -> SummaryBounds:
    base = get_summary_bounds(context_text)
    if get_summary_style() != "adaptive":
        return base

    style = plan.style or "medium_structured"
    max_cap = min(800, base.max_chars)
    if style == "short_direct":
        max_len = max(120, min(220, max_cap))
        return SummaryBounds(min_chars=0, max_chars=max_len, is_short_context=True)
    if style == "medium_structured":
        max_len = max(220, min(420, max_cap))
        return SummaryBounds(min_chars=0, max_chars=max_len, is_short_context=base.is_short_context)
    if style == "long_explanatory":
        max_len = min(800, max_cap)
        min_len = 260 if max_len >= 260 else 0
        return SummaryBounds(min_chars=min_len, max_chars=max_len, is_short_context=base.is_short_context)
    return base


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
    raw_question = (question or "").strip()
    if not raw_question:
        return {
            "answer": "資料に該当する記述が見当たりません。もう少し条件（市町/施設名/時期等）を教えてください。",
            "sources": [],
            "confidence": 0.0,
            "debug": {"reason": "empty question"},
        }

    plan = plan_answer(raw_question, {"city": city})
    planned_question = plan.query or raw_question

    base_queries = [planned_question]
    synonym_queries = _expand_with_synonyms(planned_question, plan)
    for q in synonym_queries:
        if q and q not in base_queries:
            base_queries.append(q)

    retrieval = _retrieve_chunks(city, base_queries)
    selected: List[_Candidate] = retrieval["selected"]
    confidence = retrieval["confidence"]
    debug_info = dict(retrieval.get("debug", {}) or {})
    debug_info["plan"] = plan.to_dict()
    used_queries = list(base_queries)

    selected = _apply_scope(plan, selected)
    selected = [cand for cand in selected if _clean_chunk_text(cand.chunk.text)]

    if len(selected) < 2 and confidence < _MIN_CONFIDENCE:
        first_pass = dict(debug_info)
        rewritten = _expand_with_synonyms(planned_question, plan)
        expanded = [planned_question] + [q for q in rewritten if q and q != planned_question]
        if len(expanded) > 1:
            retry = _retrieve_chunks(city, expanded)
            selected = [cand for cand in retry["selected"] if _clean_chunk_text(cand.chunk.text)]
            confidence = retry["confidence"]
            debug_info = dict(retry.get("debug", {}) or {})
            if first_pass:
                debug_info["first_pass"] = first_pass
                debug_info["retry_reason"] = "low_confidence"
            used_queries = list(expanded)
        else:
            debug_info = first_pass

    selected = _apply_scope(plan, selected)
    selected = [cand for cand in selected if _clean_chunk_text(cand.chunk.text)]

    reused_seed = False
    if (not selected or confidence < _MIN_CONFIDENCE) and planned_question:
        reused = _reuse_last_success(city, planned_question)
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
    prompt_cfg = _build_prompt(plan, planned_question, city, used_queries, context_text)
    answer_text = _generate_with_constraints(prompt_cfg)

    postprocessed = postprocess_answer(
        answer_text,
        id_map,
        min_chars=_CITATION_MIN_CHARS,
        min_score=_CITATION_MIN_SCORE,
    )

    if not postprocessed.answer_with_labels:
        fallback = _fallback_answer(plan, planned_question, city, selected)
        postprocessed = postprocess_answer(
            fallback,
            id_map,
            min_chars=_CITATION_MIN_CHARS,
            min_score=_CITATION_MIN_SCORE,
        )

    postprocessed = _enforce_summary_bounds(
        postprocessed,
        bounds=prompt_cfg.bounds,
        question=planned_question,
        city=city,
        selected=selected,
        id_map=id_map,
        plan=plan,
    )

    focus_tokens = _extract_focus_tokens(plan, planned_question)
    postprocessed = _apply_focus_guard(postprocessed, focus_tokens, id_map)
    postprocessed = _normalize_reference_section(postprocessed)
    postprocessed = _ensure_city_reference(postprocessed, pamphlet_search.city_label(city))

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
    debug_payload["plan"] = plan.to_dict()
    if postprocessed.invalid_labels:
        debug_payload["invalid_labels"] = sorted(postprocessed.invalid_labels)

    _store_last_success(city, planned_question, selected, used_queries)

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
        cleaned_text = _clean_chunk_text(chunk.text)
        if not cleaned_text:
            continue
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
            text=cleaned_text,
        )
        id_map[idx] = ref
        snippet = cleaned_text.strip()
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

    allowed_set = set(allowed_labels)
    if allowed_set:
        filtered_labelled = []
        filtered_plain = []
        for label, text in segments:
            if label in allowed_set:
                filtered_labelled.append(f"{text}[[{label}]]")
                filtered_plain.append(text)
        rebuilt_labelled = "".join(filtered_labelled).strip()
        rebuilt_plain = "".join(filtered_plain).strip()
        if rebuilt_labelled:
            labelled = rebuilt_labelled
            plain = rebuilt_plain

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
    plan: Plan,
    question: str,
    city: str,
    queries: Sequence[str],
    context_text: str,
) -> _PromptConfig:
    city_label = pamphlet_search.city_label(city)
    style_mode = get_summary_style()
    bounds = _resolve_bounds(plan, context_text)

    question_line = f"質問: {question}\n" if question else ""
    city_line = f"対象市町: {city_label}\n" if city_label else ""
    query_line = ""
    if queries:
        joined = ", ".join(q for q in queries if q)
        if joined:
            query_line = f"検索クエリ候補: {joined}\n"

    if style_mode == "polite_long":
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
        return _PromptConfig(prompt=prompt, bounds=bounds, style="polite_long")

    if style_mode == "adaptive":
        question_text = (question or "").strip()
        city_text = city_label or city
        prompt = PROMPT_ADAPTIVE_TEMPLATE.format(
            question=question_text,
            city=city_text,
            context=context_text,
        )
        return _PromptConfig(prompt=prompt, bounds=bounds, style=plan.style)

    prompt = PROMPT_SUMMARY_TERSE_SHORT.format(
        question_line=question_line,
        city_line=city_line,
        query_line=query_line,
        context=context_text,
    )
    return _PromptConfig(prompt=prompt, bounds=bounds, style=style_mode)


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


def _append_sentence(base: str, addition: str) -> str:
    base = (base or "").strip()
    addition = (addition or "").strip()
    if not addition:
        return base
    if not base:
        return addition
    if base.endswith("\n"):
        return f"{base}{addition}"
    return f"{base}\n\n{addition}" if "\n" in base else f"{base} {addition}"


def _sentence_from_ref(ref: "CitationRef", label: int, *, desired_len: int) -> Tuple[str, str]:
    snippet = re.sub(r"\s+", " ", ref.text or "").strip()
    if not snippet:
        return "", ""
    target_len = max(desired_len, 120)
    segment = snippet[:target_len]
    if len(snippet) > target_len:
        segment = segment.rstrip("。")
    if not segment:
        segment = snippet[:80]
    segment = segment.strip()
    if not segment:
        return "", ""
    if not segment.endswith("。"):
        segment = f"{segment}。"
    labelled = f"{segment}[[{label}]]"
    return labelled, segment


def _enforce_summary_bounds(
    postprocessed: "_PostProcessResult",
    *,
    bounds: SummaryBounds,
    question: str,
    city: str,
    selected: Sequence["_Candidate"],
    id_map: Dict[int, "CitationRef"],
    plan: Plan,
) -> "_PostProcessResult":
    labelled = postprocessed.answer_with_labels or ""
    if not labelled:
        return postprocessed

    char_len = _count_characters(labelled)
    style_mode = get_summary_style()
    effective_min = bounds.min_chars
    if style_mode == "adaptive" and plan.style != "long_explanatory":
        effective_min = 0

    if effective_min > 0 and char_len < effective_min:
        fallback = _fallback_answer(plan, question, city, selected)
        replacement = postprocess_answer(
            fallback,
            id_map,
            min_chars=_CITATION_MIN_CHARS,
            min_score=_CITATION_MIN_SCORE,
        )
        if replacement.answer_with_labels:
            postprocessed = replacement
            labelled = postprocessed.answer_with_labels or ""
            char_len = _count_characters(labelled)

    if effective_min > 0 and postprocessed.used_labels and style_mode != "adaptive":
        label = postprocessed.used_labels[0]
        ref = id_map.get(label)
        while ref and char_len < bounds.min_chars:
            addition_labelled, addition_plain = _sentence_from_ref(
                ref,
                label,
                desired_len=bounds.min_chars - char_len,
            )
            if not addition_labelled:
                break
            postprocessed.answer_with_labels = _append_sentence(
                postprocessed.answer_with_labels,
                addition_labelled,
            )
            postprocessed.answer_without_labels = _append_sentence(
                postprocessed.answer_without_labels,
                addition_plain,
            )
            char_len = _count_characters(postprocessed.answer_with_labels)

    if bounds.max_chars and char_len > bounds.max_chars:
        postprocessed.answer_with_labels = _truncate_sentences(
            postprocessed.answer_with_labels,
            bounds.max_chars,
        )
        postprocessed.answer_without_labels = _truncate_sentences(
            postprocessed.answer_without_labels,
            bounds.max_chars,
        )

    return postprocessed


def _generate_with_constraints(prompt_cfg: _PromptConfig) -> str:
    style_mode = get_summary_style()
    if style_mode == "polite_long":
        gen_params = {
            "temperature": 0.5,
            "max_output_tokens": 1100,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.0,
        }
    elif style_mode == "adaptive":
        if prompt_cfg.style == "short_direct":
            gen_params = {
                "temperature": 0.32,
                "max_output_tokens": 320,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.0,
            }
        elif prompt_cfg.style == "long_explanatory":
            gen_params = {
                "temperature": 0.45,
                "max_output_tokens": 1100,
                "frequency_penalty": 0.4,
                "presence_penalty": 0.0,
            }
        else:
            gen_params = {
                "temperature": 0.35,
                "max_output_tokens": 700,
                "frequency_penalty": 0.35,
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
        trimmed = _truncate_sentences(cleaned, min(prompt_cfg.bounds.max_chars, 800))
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


def _fallback_answer(plan: Plan, question: str, city: str, selected: Sequence[_Candidate]) -> str:
    city_label = pamphlet_search.city_label(city)
    style_mode = get_summary_style()

    if style_mode == "adaptive":
        return _fallback_adaptive(plan, selected, city_label)

    if style_mode == "polite_long":
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


def _fallback_adaptive(plan: Plan, selected: Sequence[_Candidate], city_label: str) -> str:
    if not selected:
        return "### 要約\n資料に該当する記述が見当たりませんでした。\n\n### 出典"

    def _sentence_from_candidate(cand: _Candidate, label: int) -> str:
        text = _clean_chunk_text(cand.chunk.text)
        if not text:
            title = re.sub(r"\.(txt|md)$", "", cand.chunk.source_file, flags=re.I)
            prefix = city_label + "の資料" if city_label else "資料"
            text = f"{prefix}「{title}」に関連情報があります。"
        snippet = re.sub(r"\s+", " ", text).strip()
        snippet = snippet[:140].rstrip("。") or snippet[:80]
        snippet = (snippet or "関連情報が掲載されています").strip()
        if not snippet.endswith("。"):
            snippet += "。"
        prefix = ""
        if city_label and city_label not in snippet:
            prefix = f"{city_label}の資料によると、"
        return f"{prefix}{snippet}[[{label}]]"

    summary_line = _sentence_from_candidate(selected[0], 1)

    detail_lines: List[str] = []
    for idx, cand in enumerate(selected[1:3], start=2):
        detail_lines.append(f"- {_sentence_from_candidate(cand, idx)}")

    lines: List[str] = ["### 要約", summary_line]
    if detail_lines:
        lines.append("")
        lines.append("### 詳細")
        lines.extend(detail_lines)

    lines.append("")
    lines.append("### 出典")
    for idx, cand in enumerate(selected[:4], start=1):
        title = re.sub(r"\.(txt|md)$", "", cand.chunk.source_file, flags=re.I)
        lines.append(f"- {city_label}/{title}")

    return "\n".join(lines).strip()
