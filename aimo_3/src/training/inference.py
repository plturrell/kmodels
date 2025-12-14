"""Inference pipeline for AIMO evaluation."""

from pathlib import Path
from typing import Dict, List, Optional

from ..evaluation.api import AIMOEvaluator

# Try to import ToolOrchestraAdapter first (orchestrated), then UnifiedSolver, then fallbacks
try:
    from ..orchestration.toolorchestra_adapter import ToolOrchestraAdapter
    DEFAULT_SOLVER_CLASS = ToolOrchestraAdapter
    USE_ORCHESTRATION = True
except ImportError:
    USE_ORCHESTRATION = False
    try:
        from ..solvers.unified_solver import UnifiedSolver
        DEFAULT_SOLVER_CLASS = UnifiedSolver
    except ImportError:
        try:
            from ..geometry.solver import GeometrySolver
            DEFAULT_SOLVER_CLASS = GeometrySolver
        except ImportError:
            from ..modeling.hybrid_solver import HybridSolver
            DEFAULT_SOLVER_CLASS = HybridSolver


class InferencePipeline:
    """Pipeline for running inference on AIMO problems."""

    def __init__(
        self,
        solver=None,
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
        if solver is None:
            if DEFAULT_SOLVER_CLASS is not None:
                # Use ToolOrchestraAdapter (orchestrated) by default if available
                if USE_ORCHESTRATION:
                    from ..orchestration.toolorchestra_adapter import create_aimo_orchestrator
                    self.solver = create_aimo_orchestrator(use_toolorchestra=True)
                else:
                    self.solver = DEFAULT_SOLVER_CLASS()
            else:
                from ..modeling.hybrid_solver import HybridSolver
                self.solver = HybridSolver()
        else:
            self.solver = solver
            
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

