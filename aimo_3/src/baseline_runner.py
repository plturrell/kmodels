
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluation.api import AIMOEvaluator
# from src.modeling.symbolic_solver import SymbolicSolver

class DummySolver:
    def solve(self, problem_statement: str) -> int:
        return 42

def main():
    print("Starting baseline runner...")
    
    # Initialize evaluator
    evaluator = AIMOEvaluator()
    
    # Initialize dummy solver
    solver = DummySolver()
    
    # Manually add a dummy problem for local testing if none exist
    if not evaluator.is_kaggle and len(evaluator.problems) == 0:
        print("Adding dummy problems for local testing...")
        evaluator.problems.append({
            "problem_id": "test_001", 
            "statement": "What is 2 + 2?"
        })
        # Add ground truth for score calculation simulation (not part of API but good for us)
        # Note: AIMOEvaluator doesn't store ground truth internally for local test, 
        # so we'll just check if it runs.
        
    # Solve all problems
    answers = evaluator.solve_all(solver)
    
    print(f"Solved {len(answers)} problems.")
    print(f"Answers: {answers}")
    
    # Generate submission file
    submission_path = evaluator.generate_submission_file()
    print(f"Submission generated at: {submission_path}")

if __name__ == "__main__":
    main()
