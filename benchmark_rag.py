import argparse
import csv
import json
import os
from pathlib import Path

from src.rag_system import RAGSystem
from src.semantic_evaluation import GeminiSemanticEvaluator


DATASET_PATH = Path(__file__).with_name("evaluation_cases.json")


def normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def load_evaluation_cases(path: Path = DATASET_PATH):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_case(
    rag: RAGSystem,
    case: dict,
    semantic_evaluator: GeminiSemanticEvaluator | None = None,
) -> dict:
    question = case["question"]
    answer, sources = rag.answer(question, return_sources=True)
    answer_text = normalize_text(answer)
    source_text = normalize_text("\n".join((source.get("text") or "") for source in sources))

    expected_terms = case.get("must_have", [])
    answer_checks = []
    context_checks = []
    missing_answer_terms = []
    missing_context_terms = []
    for expected in expected_terms:
        term = normalize_text(expected)
        in_answer = term in answer_text
        in_context = term in source_text
        answer_checks.append(in_answer)
        context_checks.append(in_context)
        if not in_answer:
            missing_answer_terms.append(expected)
        if not in_context:
            missing_context_terms.append(expected)

    answer_coverage = round(sum(answer_checks) / len(answer_checks) * 100, 1) if answer_checks else 100.0
    context_coverage = round(sum(context_checks) / len(context_checks) * 100, 1) if context_checks else 100.0
    source_files = [source.get("source") for source in sources[:3] if source.get("source")]
    expected_sources = case.get("expected_sources", [])
    missing_sources = [source for source in expected_sources if source not in source_files]
    source_recall = (
        round((len(expected_sources) - len(missing_sources)) / len(expected_sources) * 100, 1)
        if expected_sources
        else 100.0
    )
    passed = not missing_answer_terms and not missing_sources
    semantic_result = (
        semantic_evaluator.evaluate(
            question,
            answer,
            [source.get("text", "") for source in sources],
        )
        if semantic_evaluator
        else {}
    )

    return {
        "id": case.get("id", question),
        "question": question,
        "passed": passed,
        "score": answer_coverage,
        "answer_coverage": answer_coverage,
        "context_coverage": context_coverage,
        "source_recall": source_recall,
        "answer_terms_found": sum(answer_checks),
        "answer_terms_total": len(answer_checks),
        "context_terms_found": sum(context_checks),
        "context_terms_total": len(context_checks),
        "expected_sources_found": len(expected_sources) - len(missing_sources),
        "expected_sources_total": len(expected_sources),
        "missing_terms": missing_answer_terms,
        "missing_answer_terms": missing_answer_terms,
        "missing_context_terms": missing_context_terms,
        "missing_sources": missing_sources,
        "source_files": source_files,
        "answer_preview": (answer or "")[:220],
        "faithfulness": semantic_result.get("faithfulness"),
        "faithfulness_supported_claims": semantic_result.get("faithfulness_supported_claims"),
        "faithfulness_total_claims": semantic_result.get("faithfulness_total_claims"),
        "faithfulness_claims": semantic_result.get("faithfulness_claims", []),
        "answer_relevancy": semantic_result.get("answer_relevancy"),
        "answer_relevancy_rationale": semantic_result.get("answer_relevancy_rationale"),
    }


def summarize_results(
    results: list[dict],
    min_pass_rate: float = 100.0,
    min_answer_coverage: float = 95.0,
    min_source_recall: float = 95.0,
    min_faithfulness: float | None = None,
    min_answer_relevancy: float | None = None,
) -> dict:
    thresholds = {
        "pass_rate": min_pass_rate,
        "answer_coverage": min_answer_coverage,
        "source_recall_at_k": min_source_recall,
    }
    thresholds.update(
        {
            name: threshold
            for name, threshold in (
                ("faithfulness", min_faithfulness),
                ("answer_relevancy", min_answer_relevancy),
            )
            if threshold is not None
        }
    )
    if any(not 0 <= threshold <= (1 if name in {"faithfulness", "answer_relevancy"} else 100)
           for name, threshold in thresholds.items()):
        raise ValueError("Semantic thresholds must be between 0 and 1; other thresholds must be between 0 and 100.")

    total_cases = len(results)
    passed_cases = sum(result["passed"] for result in results)
    total_answer_terms = sum(result["answer_terms_total"] for result in results)
    total_context_terms = sum(result["context_terms_total"] for result in results)
    total_expected_sources = sum(result["expected_sources_total"] for result in results)
    pass_rate = round(passed_cases / total_cases * 100, 1) if total_cases else 0.0
    answer_coverage = (
        round(sum(result["answer_terms_found"] for result in results) / total_answer_terms * 100, 1)
        if total_answer_terms
        else 100.0
    )
    context_coverage = (
        round(sum(result["context_terms_found"] for result in results) / total_context_terms * 100, 1)
        if total_context_terms
        else 100.0
    )
    source_recall = (
        round(sum(result["expected_sources_found"] for result in results) / total_expected_sources * 100, 1)
        if total_expected_sources
        else 100.0
    )
    semantic_results = [
        result for result in results if result.get("faithfulness") is not None
    ]
    faithfulness = (
        round(sum(result["faithfulness"] for result in semantic_results) / len(semantic_results), 3)
        if semantic_results
        else None
    )
    answer_relevancy = (
        round(sum(result["answer_relevancy"] for result in semantic_results) / len(semantic_results), 3)
        if semantic_results
        else None
    )
    metrics = {
        "pass_rate": pass_rate,
        "answer_coverage": answer_coverage,
        "source_recall_at_k": source_recall,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
    }
    failures = []
    for metric, threshold in thresholds.items():
        value = metrics[metric]
        if value is None or value < threshold:
            unit = "" if metric in {"faithfulness", "answer_relevancy"} else "%"
            failures.append(
                f"{metric} {value}{unit} is below the {threshold}{unit} threshold"
            )

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate": pass_rate,
        "answer_coverage": answer_coverage,
        "context_coverage": context_coverage,
        "source_recall_at_k": source_recall,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "thresholds": thresholds,
        "quality_gate_passed": not failures,
        "quality_gate_failures": failures,
        "results": results,
    }


def export_summary(summary: dict, output_path: str, output_format: str = "json") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format.lower() == "json":
        path = path.with_suffix(".json") if path.suffix.lower() != ".json" else path
        with path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
    elif output_format.lower() == "csv":
        path = path.with_suffix(".csv") if path.suffix.lower() != ".csv" else path
        fieldnames = [
            "id",
            "question",
            "passed",
            "score",
            "answer_coverage",
            "context_coverage",
            "source_recall",
            "faithfulness",
            "answer_relevancy",
            "source_files",
            "missing_terms",
            "missing_context_terms",
            "missing_sources",
            "answer_preview",
        ]
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for result in summary["results"]:
                writer.writerow(
                    {
                        "id": result["id"],
                        "question": result["question"],
                        "passed": result["passed"],
                        "score": result["score"],
                        "answer_coverage": result["answer_coverage"],
                        "context_coverage": result["context_coverage"],
                        "source_recall": result["source_recall"],
                        "faithfulness": result["faithfulness"],
                        "answer_relevancy": result["answer_relevancy"],
                        "source_files": "; ".join(result["source_files"]),
                        "missing_terms": "; ".join(result["missing_terms"]),
                        "missing_context_terms": "; ".join(result["missing_context_terms"]),
                        "missing_sources": "; ".join(result["missing_sources"]),
                        "answer_preview": result["answer_preview"],
                    }
                )
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    print(f"Benchmark report exported to: {path}")


def run_benchmark(
    output_path: str | None = None,
    output_format: str = "json",
    min_pass_rate: float = 100.0,
    min_answer_coverage: float = 95.0,
    min_source_recall: float = 95.0,
    semantic: bool = False,
    min_faithfulness: float | None = None,
    min_answer_relevancy: float | None = None,
) -> dict:
    if not semantic and (min_faithfulness is not None or min_answer_relevancy is not None):
        raise ValueError("Use --semantic when setting semantic metric thresholds.")

    rag = RAGSystem(docs_folder="data/documents", top_k=3)
    cases = load_evaluation_cases()
    semantic_evaluator = GeminiSemanticEvaluator() if semantic else None

    results = [evaluate_case(rag, case, semantic_evaluator) for case in cases]

    summary = summarize_results(
        results,
        min_pass_rate=min_pass_rate,
        min_answer_coverage=min_answer_coverage,
        min_source_recall=min_source_recall,
        min_faithfulness=min_faithfulness,
        min_answer_relevancy=min_answer_relevancy,
    )

    print("RAG evaluation summary")
    print("=" * 24)
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} | {result['question']}")
        print(f"  answer/context/source: {result['answer_coverage']}% / {result['context_coverage']}% / {result['source_recall']}%")
        if result["faithfulness"] is not None:
            print(f"  faithfulness/relevancy: {result['faithfulness']} / {result['answer_relevancy']}")
        print(f"  sources: {result['source_files']}")
        if result["missing_terms"]:
            print(f"  missing from answer: {result['missing_terms']}")
        if result["missing_context_terms"]:
            print(f"  missing from context: {result['missing_context_terms']}")
        if result["missing_sources"]:
            print(f"  missing expected sources: {result['missing_sources']}")
        print(f"  answer: {result['answer_preview']}...")
        print()

    print(f"Answer-term coverage: {summary['answer_coverage']}%")
    print(f"Retrieved-context coverage: {summary['context_coverage']}%")
    print(f"Expected-source recall@3: {summary['source_recall_at_k']}%")
    if summary["faithfulness"] is not None:
        print(f"Mean faithfulness: {summary['faithfulness']}")
        print(f"Mean answer relevancy: {summary['answer_relevancy']}")
    print(f"Benchmark result: {summary['passed_cases']}/{summary['total_cases']} passed ({summary['pass_rate']}%)")
    print("Quality gate: " + ("PASS" if summary["quality_gate_passed"] else "FAIL"))
    for failure in summary["quality_gate_failures"]:
        print(f"  - {failure}")

    if output_path:
        export_summary(summary, output_path, output_format)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and optionally export the RAG benchmark.")
    parser.add_argument("--output", type=str, help="Path for the benchmark export file.")
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Format for the exported benchmark results.",
    )
    parser.add_argument("--min-pass-rate", type=float, default=100.0)
    parser.add_argument("--min-answer-coverage", type=float, default=95.0)
    parser.add_argument("--min-source-recall", type=float, default=95.0)
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use Gemini as an LLM judge for faithfulness and answer relevancy (requires GEMINI_API_KEY).",
    )
    parser.add_argument("--min-faithfulness", type=float)
    parser.add_argument("--min-answer-relevancy", type=float)
    args = parser.parse_args()
    if args.semantic and not os.getenv("GEMINI_API_KEY"):
        parser.error("--semantic requires GEMINI_API_KEY to be set.")
    if not args.semantic and (
        args.min_faithfulness is not None or args.min_answer_relevancy is not None
    ):
        parser.error("--min-faithfulness and --min-answer-relevancy require --semantic.")

    summary = run_benchmark(
        output_path=args.output,
        output_format=args.format,
        min_pass_rate=args.min_pass_rate,
        min_answer_coverage=args.min_answer_coverage,
        min_source_recall=args.min_source_recall,
        semantic=args.semantic,
        min_faithfulness=args.min_faithfulness,
        min_answer_relevancy=args.min_answer_relevancy,
    )
    raise SystemExit(0 if summary["quality_gate_passed"] else 1)
