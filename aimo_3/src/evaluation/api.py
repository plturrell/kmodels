"""AIMO evaluation API wrapper for Kaggle submission environment."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# In Kaggle environment, the evaluation API is available
try:
    from kaggle_aimo3_evaluation import get_problem, submit_answer
    KAGGLE_ENV = True
except ImportError:
    KAGGLE_ENV = False
    # Mock for local testing
    def get_problem():
        return {"problem_id": "test", "statement": "Test problem"}
    def submit_answer(problem_id: str, answer: int):
        return {"status": "accepted"}


def validate_answer(answer: int) -> bool:
    """
    Validate that an answer is in the valid range [0, 99999].

    Args:
        answer: Answer to validate

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(answer, int):
        return False
    return 0 <= answer <= 99999


class AIMOEvaluator:
    """
    Wrapper for AIMO evaluation API.

    Handles problem retrieval, answer submission, and submission file generation.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize evaluator.

        Args:
            output_dir: Directory to save submission files (default: outputs/)
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "outputs"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.problems: List[Dict[str, Any]] = []
        self.answers: Dict[str, int] = {}
        self.is_kaggle = KAGGLE_ENV

    def get_next_problem(self) -> Optional[Dict[str, str]]:
        """
        Get the next problem from the evaluation API.

        Returns:
            Problem dictionary with 'problem_id' and 'statement', or None if done
        """
        if not self.is_kaggle:
            # Local testing mode
            if len(self.problems) == 0:
                return None
            return self.problems.pop(0)

        try:
            problem = get_problem()
            if problem is None:
                return None
            return problem
        except Exception as e:
            print(f"Error getting problem: {e}")
            return None

    def submit_answer_for_problem(self, problem_id: str, answer: int) -> bool:
        """
        Submit an answer for a problem.

        Args:
            problem_id: Problem identifier
            answer: Answer (must be integer in [0, 99999])

        Returns:
            True if submission successful, False otherwise
        """
        if not validate_answer(answer):
            print(f"Invalid answer: {answer} (must be integer in [0, 99999])")
            return False

        self.answers[problem_id] = answer

        if self.is_kaggle:
            try:
                result = submit_answer(problem_id, answer)
                return result.get("status") == "accepted"
            except Exception as e:
                print(f"Error submitting answer: {e}")
                return False
        else:
            # Local testing mode
            print(f"Submitted answer {answer} for problem {problem_id}")
            return True

    def solve_all(self, solver) -> Dict[str, int]:
        """
        Solve all problems using the provided solver.

        Args:
            solver: Solver object with a solve(problem_statement) method

        Returns:
            Dictionary mapping problem_id to answer
        """
        answers = {}

        while True:
            problem = self.get_next_problem()
            if problem is None:
                break

            problem_id = problem["problem_id"]
            statement = problem["statement"]

            print(f"Solving problem {problem_id}...")

            try:
                answer = solver.solve(statement)
                if validate_answer(answer):
                    self.submit_answer_for_problem(problem_id, answer)
                    answers[problem_id] = answer
                else:
                    print(f"Invalid answer from solver: {answer}")
            except Exception as e:
                print(f"Error solving problem {problem_id}: {e}")

        return answers

    def generate_submission_file(self, filename: str = "submission.csv") -> Path:
        """
        Generate submission file from collected answers.

        Args:
            filename: Output filename

        Returns:
            Path to submission file
        """
        submission_path = self.output_dir / filename

        # AIMO submission format: problem_id,answer
        with open(submission_path, "w") as f:
            f.write("problem_id,answer\n")
            for problem_id, answer in sorted(self.answers.items()):
                f.write(f"{problem_id},{answer}\n")

        print(f"Submission file generated: {submission_path}")
        return submission_path

    def calculate_penalized_accuracy(
        self, ground_truth: Dict[str, List[int]]
    ) -> float:
        """
        Calculate penalized accuracy score.

        For each problem:
        - Both answers correct: score = 1
        - One correct, one incorrect: score = 0.5
        - Both incorrect: score = 0

        Args:
            ground_truth: Dictionary mapping problem_id to list of two correct answers

        Returns:
            Total score
        """
        total_score = 0.0

        for problem_id, correct_answers in ground_truth.items():
            if problem_id not in self.answers:
                continue

            predicted = self.answers[problem_id]
            correct_count = sum(1 for ans in correct_answers if ans == predicted)

            if correct_count == 2:
                score = 1.0
            elif correct_count == 1:
                score = 0.5
            else:
                score = 0.0

            total_score += score

        return total_score

