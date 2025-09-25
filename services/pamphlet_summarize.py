"""Summarization helpers for pamphlet fallback answers."""
from __future__ import annotations

import logging
import os
from typing import Iterable, List

from .pamphlet_search import PamphletChunk, SearchResult

logger = logging.getLogger(__name__)

_CLIENT = None


def configure(openai_client) -> None:
    """Register an OpenAI client instance used for summarisation."""
    global _CLIENT
    _CLIENT = openai_client


def summarize_with_gpt_nano(query: str, top_docs: Iterable[SearchResult], *, detailed: bool = False) -> str:
    """Summarise the provided chunks using GPT-5-nano if available."""
    docs = list(top_docs)
    if not docs:
        return ""

    goal = (
        "以下の資料を根拠に、質問者の求める情報を日本語でまとめてください。"
        "事実に基づき、観光案内所スタッフとして丁寧に説明します。"
    )
    length_note = "500〜700" if not detailed else "700〜900"
    instructions = (
        f"{goal}\n"
        f"- 回答は{length_note}字程度で、根拠のある内容のみ記述してください。\n"
        "- 不確かな推測は避け、資料内の情報のみで構成してください。\n"
        "- 箇条書きが適切であれば利用し、読みやすさを重視してください。\n"
    )

    context_parts: List[str] = []
    for item in docs:
        chunk: PamphletChunk = item.chunk
        context_parts.append(
            f"出典:{chunk.source_file}\n{chunk.text}"
        )
    context_text = "\n\n".join(context_parts)

    if _CLIENT and os.environ.get("OPENAI_API_KEY"):
        try:
            response = _CLIENT.responses.create(
                model="gpt-5-nano",
                input=[
                    {
                        "role": "system",
                        "content": instructions + "質問と資料を読み込み、回答のみを出力してください。",
                    },
                    {
                        "role": "user",
                        "content": f"質問: {query}\n\n資料:\n{context_text}",
                    },
                ],
                max_output_tokens=700 if detailed else 550,
                temperature=0.2,
            )
            text = getattr(response, "output_text", "")
            if text:
                return text.strip()
        except Exception as exc:
            logger.exception("[pamphlet] summarisation failed: %s", exc)

    # フォールバック：上位チャンクの冒頭部分を抜粋
    fallback: List[str] = []
    for item in docs:
        snippet = item.chunk.text.strip().replace("\n", " ")
        if not snippet:
            continue
        fallback.append(snippet[:200])
    if fallback:
        body = "\n".join(fallback)
        return f"{query}に関連しそうな内容です:\n{body}"
    return ""
