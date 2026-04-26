"""
evaluation/run_rag_eval.py
───────────────────────────
RAG pipeline evaluation framework.

Runs a benchmark question set through the RAG pipeline and produces:
  - Per-question results with retrieved chunks, sources, latency
  - Aggregate metrics: retrieval hit rate, citation accuracy, avg latency,
    fallback rate, out-of-scope accuracy

Usage:
    python evaluation/run_rag_eval.py
    python evaluation/run_rag_eval.py --questions evaluation/benchmark_questions.json
    python evaluation/run_rag_eval.py --output evaluation/results/run_2026_03_13.json

Output:
    evaluation_results.json  (or --output path)
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ── Bootstrap path ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.rag_api              import run_query
from vectorstore.chroma_manager import ChromaManager
from config.settings          import VECTORSTORE_DIR


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def _source_hit(result: Dict, expected_doc: Optional[str]) -> bool:
    """
    Retrieval hit: did any retrieved chunk come from the expected document?
    """
    if not expected_doc:
        return True   # No ground truth — can't evaluate
    sources = [s["filename"].lower() for s in result.get("sources", [])]
    return any(expected_doc.lower() in s for s in sources)


def _citation_present(result: Dict) -> bool:
    """
    Citation accuracy: does the answer contain at least one [SOURCE N] citation?
    """
    answer = result.get("answer", "")
    return "[SOURCE" in answer and "|" in answer


def _fallback_correct(result: Dict, expect_fallback: bool) -> bool:
    """
    Out-of-scope accuracy: did the system correctly trigger / not trigger fallback?
    """
    triggered = result.get("fallback_triggered", False)
    return triggered == expect_fallback


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    questions_path: str = "evaluation/benchmark_questions.json",
    output_path:    str = "evaluation_results.json",
    collection:     str = "library",
    use_judge:      bool = False,
) -> Dict:
    """
    Run the full evaluation suite.

    Args:
        questions_path: Path to benchmark_questions.json
        output_path:    Where to write evaluation_results.json
        collection:     ChromaDB collection name to evaluate against

    Returns:
        Dict with per-question results and aggregate metrics.
    """
    # ── Load questions ────────────────────────────────────────────────────────
    questions_file = Path(ROOT) / questions_path
    if not questions_file.exists():
        questions_file = Path(questions_path)
    if not questions_file.exists():
        raise FileNotFoundError(f"Benchmark questions not found: {questions_path}")

    with open(questions_file, encoding="utf-8") as f:
        questions: List[Dict] = json.load(f)

    print(f"[eval] Loaded {len(questions)} benchmark questions from {questions_file}")

    # ── Load vector store ─────────────────────────────────────────────────────
    manager = ChromaManager(collection_name=collection, persist=True)
    try:
        chunk_count = manager.get_chunk_count()
        print(f"[eval] Vector store loaded — {chunk_count} chunks in '{collection}'")
    except Exception as e:
        print(f"[eval] ⚠ Could not load vector store: {e}")
        print("[eval] Run `python ingest.py` first.")
        sys.exit(1)

    vector_store = manager.load()

    # ── Run questions ─────────────────────────────────────────────────────────
    question_results: List[Dict] = []
    hit_count         = 0
    citation_count    = 0
    fallback_correct  = 0
    total_latency     = 0.0
    evaluatable       = 0

    print(f"\n{'─'*60}")
    for i, q in enumerate(questions, start=1):
        qid      = q.get("id", f"q{i:03d}")
        question = q["question"]
        exp_doc  = q.get("expected_document")
        exp_fb   = q.get("expect_fallback", False)

        print(f"[{i:02d}/{len(questions)}] {qid}: {question[:70]}…")

        result = run_query(vector_store=vector_store, query=question)

        # Compute per-question metrics
        hit       = _source_hit(result, exp_doc)
        cited     = _citation_present(result)
        fb_ok     = _fallback_correct(result, exp_fb)
        latency   = result.get("response_time", 0.0)

        hit_count        += int(hit)
        citation_count   += int(cited)
        fallback_correct += int(fb_ok)
        total_latency    += latency
        evaluatable      += 1

        question_results.append({
            "id":               qid,
            "question":         question,
            "category":         q.get("category", "unknown"),
            "expected_document": exp_doc,
            "expected_fallback": exp_fb,
            "answer":           result.get("answer", ""),
            "sources_retrieved": result.get("sources", []),
            "num_chunks":       result.get("num_chunks", 0),
            "fallback_triggered": result.get("fallback_triggered", False),
            "guardrail_reason": result.get("guardrail_reason", ""),
            "latency_ms":       round(latency * 1000, 1),
            "retrieval_ms":     round(result.get("retrieval_time", 0.0) * 1000, 1),
            "metrics": {
                "source_hit":       hit,
                "citation_present": cited,
                "fallback_correct": fb_ok,
            },
        })

        # ── LLM-as-judge (optional) ───────────────────────────────────────
        judge_scores = None
        if use_judge and not result.get("fallback_triggered"):
            from evaluation.llm_judge import judge_answer
            judge_scores = judge_answer(
                question=question,
                answer=result.get("answer", ""),
                context=result.get("context", ""),
            )
            question_results[-1]["judge_scores"] = judge_scores

        status = "✅" if hit and cited else "⚠️"
        judge_str = ""
        if judge_scores and judge_scores.get("faithfulness"):
            judge_str = (f" | F={judge_scores['faithfulness']} "
                         f"R={judge_scores['relevance']} "
                         f"C={judge_scores['completeness']}")
        print(f"       {status} hit={hit} cited={cited} fb_correct={fb_ok} "
              f"latency={latency*1000:.0f}ms{judge_str}")

    print(f"{'─'*60}\n")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    n = evaluatable or 1
    in_scope      = [r for r in question_results if not r["expected_fallback"]]
    out_of_scope  = [r for r in question_results if r["expected_fallback"]]

    aggregate = {
        "total_questions":          len(questions),
        "evaluated":                evaluatable,
        "retrieval_hit_rate":       round(hit_count        / n * 100, 1),
        "citation_accuracy":        round(citation_count   / n * 100, 1),
        "fallback_accuracy":        round(fallback_correct / n * 100, 1),
        "avg_latency_ms":           round(total_latency    / n * 1000, 1),
        "in_scope_questions":       len(in_scope),
        "out_of_scope_questions":   len(out_of_scope),
    }

    # ── Judge aggregates (if enabled) ────────────────────────────────────────
    if use_judge:
        judged = [r for r in question_results if r.get("judge_scores") and r["judge_scores"].get("faithfulness")]
        if judged:
            aggregate["avg_faithfulness"]  = round(sum(r["judge_scores"]["faithfulness"]  for r in judged) / len(judged), 2)
            aggregate["avg_relevance"]     = round(sum(r["judge_scores"]["relevance"]     for r in judged) / len(judged), 2)
            aggregate["avg_completeness"]  = round(sum(r["judge_scores"]["completeness"]  for r in judged) / len(judged), 2)
            aggregate["judged_questions"]  = len(judged)

    print("📊 EVALUATION SUMMARY")
    print(f"  Retrieval Hit Rate : {aggregate['retrieval_hit_rate']}%")
    print(f"  Citation Accuracy  : {aggregate['citation_accuracy']}%")
    print(f"  Fallback Accuracy  : {aggregate['fallback_accuracy']}%")
    print(f"  Avg Latency        : {aggregate['avg_latency_ms']} ms")
    if use_judge and "avg_faithfulness" in aggregate:
        print(f"  Avg Faithfulness   : {aggregate['avg_faithfulness']}/5")
        print(f"  Avg Relevance      : {aggregate['avg_relevance']}/5")
        print(f"  Avg Completeness   : {aggregate['avg_completeness']}/5")

    # ── Write output ──────────────────────────────────────────────────────────
    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "collection":    collection,
        "aggregate":     aggregate,
        "results":       question_results,
    }

    output_file = Path(ROOT) / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results written to: {output_file}")
    return output


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument(
        "--questions",
        default="evaluation/benchmark_questions.json",
        help="Path to benchmark questions JSON file.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Path to write evaluation results JSON.",
    )
    parser.add_argument(
        "--collection",
        default="library",
        help="ChromaDB collection name to evaluate.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-as-judge scoring (faithfulness, relevance, completeness).",
    )
    args = parser.parse_args()
    run_evaluation(args.questions, args.output, args.collection, use_judge=args.judge)
