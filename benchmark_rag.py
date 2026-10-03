import argparse
import csv
import json
from pathlib import Path

from src.rag_system import RAGSystem


DATASET_PATH = Path(__file__).with_name("evaluation_cases.json")


def normalize_text(value: str) -> str:
    return " ".join((value or "").lower().split())


def load_evaluation_cases(path: Path = DATASET_PATH):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_case(rag: RAGSystem, case: dict) -> dict:
    question = case["question"]
    answer, sources = rag.answer(question, return_sources=True)
    answer_text = normalize_text(answer)
    source_text = normalize_text("\n".join((source.get("text") or "") for source in sources))

    checks = []
    missing_terms = []
    for expected in case["must_have"]:
        term = normalize_text(expected)
        present = term in answer_text or term in source_text
        checks.append(present)
        if not present:
            missing_terms.append(expected)

    passed = all(checks)
    score = round((sum(checks) / len(checks)) * 100, 1) if checks else 100.0

    return {
        "id": case.get("id", question),
        "question": question,
        "passed": passed,
        "score": score,
        "missing_terms": missing_terms,
        "source_files": [source.get("source") for source in sources[:3]],
        "answer_preview": (answer or "")[:220],
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
            "source_files",
            "missing_terms",
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
                        "source_files": "; ".join(result["source_files"]),
                        "missing_terms": "; ".join(result["missing_terms"]),
                        "answer_preview": result["answer_preview"],
                    }
                )
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    print(f"Benchmark report exported to: {path}")


def run_benchmark(output_path: str | None = None, output_format: str = "json") -> dict:
    rag = RAGSystem(docs_folder="data/documents", top_k=3)
    cases = load_evaluation_cases()

    results = [evaluate_case(rag, case) for case in cases]

    passed_cases = sum(1 for result in results if result["passed"])
    total_cases = len(results)
    overall_score = round((passed_cases / total_cases) * 100, 1) if total_cases else 0.0

    print("RAG evaluation summary")
    print("=" * 24)
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} | {result['question']}")
        print(f"  score: {result['score']}%")
        print(f"  sources: {result['source_files']}")
        if result["missing_terms"]:
            print(f"  missing: {result['missing_terms']}")
        print(f"  answer: {result['answer_preview']}...")
        print()

    summary = {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate": overall_score,
        "results": results,
    }

    print(f"Benchmark result: {passed_cases}/{total_cases} passed ({overall_score}%)")

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
    args = parser.parse_args()

    run_benchmark(output_path=args.output, output_format=args.format)
