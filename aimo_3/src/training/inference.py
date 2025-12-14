"""Inference pipeline for AIMO evaluation."""

from pathlib import Path
from typing import Dict, List, Optional, Protocol

from ..evaluation.api import AIMOEvaluator

class SupportsSolve(Protocol):
    def solve(self, problem_statement: str) -> int: ...


def create_default_solver() -> SupportsSolve:
    """
    Create the default solver implementation.

    Preference order:
    - ToolOrchestra orchestrator (if available)
    - UnifiedSolver
    - GeometrySolver
    - HybridSolver
    """
    try:
        from ..orchestration.toolorchestra_adapter import create_aimo_orchestrator
        return create_aimo_orchestrator(use_toolorchestra=True)
    except Exception:
        pass

    try:
        from ..solvers.unified_solver import UnifiedSolver
        return UnifiedSolver()
    except Exception:
        pass

    try:
        from ..geometry.solver import GeometrySolver
        return GeometrySolver()
    except Exception:
        from ..modeling.hybrid_solver import HybridSolver
        return HybridSolver()


class InferencePipeline:
    """Pipeline for running inference on AIMO problems."""

    def __init__(
        self,
        solver: Optional[SupportsSolve] = None,
        output_dir: Optional[Path] = None,
        use_geometry: bool = True,
    ):
        """
        Initialize inference pipeline.

        Args:
            solver: Solver instance (optional, will create GeometrySolver if None)
            output_dir: Directory to save outputs
            use_geometry: Whether to use GeometrySolver (default: True)
        """
        self.solver: SupportsSolve = solver if solver is not None else create_default_solver()
            
        self.output_dir = output_dir
        self.evaluator = AIMOEvaluator(output_dir=output_dir)

    def run_evaluation(self) -> Dict[str, int]:
        """
        Run evaluation on all problems using the evaluation API.

        Returns:
            Dictionary mapping problem_id to answer
        """
        print("Starting evaluation...")
        answers = self.evaluator.solve_all(self.solver)
        return answers

    def run_on_problems(self, problems: List[Dict]) -> Dict[str, int]:
        """
        Run inference on a list of problems.

        Args:
            problems: List of problem dictionaries

        Returns:
            Dictionary mapping problem_id to answer
        """
        answers = {}

        for problem in problems:
            problem_id = problem.get("problem_id", "unknown")
            statement = problem["statement"]

            try:
                answer = self.solver.solve(statement)
                answers[problem_id] = answer
            except Exception as e:
                print(f"Error solving problem {problem_id}: {e}")
                answers[problem_id] = 0

        return answers

    def generate_submission(self, answers: Dict[str, int], filename: str = "submission.csv") -> Path:
        """
        Generate submission file from answers.

        Args:
            answers: Dictionary mapping problem_id to answer
            filename: Output filename

        Returns:
            Path to submission file
        """
        self.evaluator.answers = answers
        return self.evaluator.generate_submission_file(filename)

