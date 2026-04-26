"""
evaluation/llm_judge.py
────────────────────────
LLM-as-judge for RAG answer quality evaluation.

Scores answers on three dimensions:
  - Faithfulness: Is every claim supported by the retrieved context?
  - Relevance:    Does it answer the question?
  - Completeness: Are all relevant details from context included?

Each dimension is scored 1–5. Uses the same LLM model as the RAG pipeline.
"""
import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import OPENAI_API_KEY, LLM_MODEL


_JUDGE_PROMPT = """\
You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.

Given a question, the retrieved context, and the system's answer, score the answer on three dimensions.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

SYSTEM ANSWER:
{answer}

Score each dimension from 1 to 5:
- faithfulness: Is every claim supported by the context? (5=fully grounded, 1=fabricated)
- relevance: Does it answer the question? (5=direct answer, 1=off-topic)
- completeness: Are all relevant details from context included? (5=comprehensive, 1=major gaps)

Return ONLY valid JSON: {{"faithfulness": N, "relevance": N, "completeness": N, "reasoning": "one sentence"}}"""


def judge_answer(
    question: str,
    answer: str,
    context: str,
) -> Dict:
    """
    Score a RAG answer using an LLM judge.

    Returns:
        {"faithfulness": int, "relevance": int, "completeness": int, "reasoning": str}
        On failure, all scores are 0 with reasoning explaining the error.
    """
    from langchain_openai import ChatOpenAI

    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=300,
        )
        prompt = _JUDGE_PROMPT.format(
            question=question,
            context=context[:3000],   # Cap context to avoid token limits
            answer=answer,
        )
        result = llm.invoke(prompt)
        raw = result.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        scores = json.loads(raw)

        return {
            "faithfulness":  int(scores.get("faithfulness", 0)),
            "relevance":     int(scores.get("relevance", 0)),
            "completeness":  int(scores.get("completeness", 0)),
            "reasoning":     scores.get("reasoning", ""),
        }

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        return {
            "faithfulness": 0, "relevance": 0, "completeness": 0,
            "reasoning": f"parse_error: {exc}",
        }
    except Exception as exc:
        return {
            "faithfulness": 0, "relevance": 0, "completeness": 0,
            "reasoning": f"error: {exc}",
        }
