"""Evaluation script to benchmark different solver approaches."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from src.orchestration.toolorchestra_adapter import create_aimo_orchestrator
from src.solvers.unified_solver import UnifiedSolver
from src.geometry.solver import GeometrySolver


def evaluate_solver(solver, problems: List[Dict], solver_name: str) -> Dict:
    """
    Evaluate a solver on a set of problems.

    Args:
        solver: Solver instance
        problems: List of problem dictionaries with 'statement' and 'answer'
        solver_name: Name of solver

    Returns:
        Dictionary with evaluation metrics
    """
    correct = 0
    total = 0
    total_time: float = 0.0
    errors = 0

    results = []

    for problem in problems:
        statement = problem.get("statement", "")
        expected_answer = problem.get("answer", 0)

        if not statement:
            continue

        try:
            start_time = time.time()
            predicted_answer = solver.solve(statement)
            elapsed = time.time() - start_time

            is_correct = predicted_answer == expected_answer
            if is_correct:
                correct += 1

            total += 1
            total_time += elapsed

            results.append({
                "statement": statement,
                "expected": expected_answer,
                "predicted": predicted_answer,
                "correct": is_correct,
                "time": elapsed,
            })

        except Exception as e:
            errors += 1
            results.append({
                "statement": statement,
                "expected": expected_answer,
                "predicted": 0,
                "correct": False,
                "error": str(e),
            })

    accuracy = correct / total if total > 0 else 0.0
    avg_time = total_time / total if total > 0 else 0.0

    return {
        "solver": solver_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "avg_time": avg_time,
        "total_time": total_time,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate solvers on AIMO problems")
    parser.add_argument(
        "--problems",
        type=Path,
        default=Path("data/generated_problems.json"),
        help="Path to problems JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation_results.json"),
        help="Output file for results",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["orchestrated", "unified", "geometry"],
        choices=["orchestrated", "unified", "geometry"],
        help="Solvers to evaluate",
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Maximum number of problems to evaluate",
    )

    args = parser.parse_args()

    # Load problems
    if not args.problems.exists():
        print(f"Error: Problems file not found: {args.problems}")
        return

    with open(args.problems) as f:
        problems_data = json.load(f)

    problems = problems_data if isinstance(problems_data, list) else problems_data.get("problems", [])

    if args.max_problems:
        problems = problems[:args.max_problems]

    print(f"Evaluating on {len(problems)} problems...")

    # Initialize solvers
    solvers = {}
    if "orchestrated" in args.solvers:
        solvers["orchestrated"] = create_aimo_orchestrator(use_toolorchestra=True)
    if "unified" in args.solvers:
        solvers["unified"] = UnifiedSolver()
    if "geometry" in args.solvers:
        solvers["geometry"] = GeometrySolver()

    # Evaluate each solver
    all_results = {}
    for solver_name, solver in solvers.items():
        print(f"\nEvaluating {solver_name} solver...")
        results = evaluate_solver(solver, problems, solver_name)
        all_results[solver_name] = results

        print(f"  Accuracy: {results['accuracy']:.2%}")
        print(f"  Correct: {results['correct']}/{results['total']}")
        print(f"  Errors: {results['errors']}")
        print(f"  Avg Time: {results['avg_time']:.3f}s")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Print comparison
    print("\n" + "=" * 60)
    print("SOLVER COMPARISON")
    print("=" * 60)
    for solver_name, results in all_results.items():
        print(f"{solver_name:15} | Accuracy: {results['accuracy']:6.2%} | "
              f"Time: {results['avg_time']:6.3f}s | "
              f"Correct: {results['correct']:3}/{results['total']:3}")


if __name__ == "__main__":
    main()

