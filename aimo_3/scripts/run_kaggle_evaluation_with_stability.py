#!/usr/bin/env python3
"""
Run Kaggle-style evaluation locally with stability tracking.

This script mimics the Kaggle competition evaluation but adds full
stability tracking for debugging and validation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.kaggle_runner_with_stability import StabilityAwareKaggleRunner


def main():
    """Run local evaluation with stability tracking."""
    
    # Path to test data
    test_path = project_root / 'ai-mathematical-olympiad-progress-prize-3' / 'test.csv'
    
    if not test_path.exists():
        print(f"Error: Test file not found at {test_path}")
        print("Please ensure the Kaggle data is downloaded.")
        return
    
    # Read test data
    try:
        import polars as pl
        test = pl.read_csv(test_path)
        print(f"Loaded {len(test)} problems from {test_path}")
    except ImportError:
        print("Error: polars not installed. Install with: pip install polars")
        return
    except Exception as e:
        print(f"Error reading test data: {e}")
        return
    
    # Initialize runner with stability tracking
    print("\nInitializing stability-aware runner...")
    runner = StabilityAwareKaggleRunner(
        use_stability=True,
        track_orchestration=True,
        stability_output_dir='outputs/kaggle_stability'
    )
    
    # Solve problems
    print("\nSolving problems...")
    print("-" * 60)
    
    for i, row in enumerate(test.iter_rows(named=True), 1):
        problem_id = row['id']
        problem_statement = row['problem']
        
        print(f"[{i}/{len(test)}] {problem_id}... ", end='', flush=True)
        
        try:
            answer = runner.solve_problem(problem_id, problem_statement)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"ERROR: {e}")
            # Still record a default answer
            runner.problem_results.append({
                'problem_id': problem_id,
                'answer': 0
            })
    
    print("-" * 60)
    
    # Export submission
    print("\nExporting results...")
    runner.export_submission_csv('outputs/submission_with_stability.csv')
    
    # Export stability reports
    runner.export_stability_report()
    
    # Print summary
    runner.print_summary()
    
    print("\n✓ Evaluation complete!")
    print("\nGenerated files:")
    print("  - outputs/submission_with_stability.csv")
    print("  - outputs/kaggle_stability/aggregate_stability.json")
    print("  - outputs/kaggle_stability/tool_comparison.json")
    print("  - outputs/kaggle_stability/per_problem_stability.json")


if __name__ == '__main__':
    main()
