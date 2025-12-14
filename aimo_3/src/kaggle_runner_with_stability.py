"""
Kaggle submission runner with stability tracking.

Integrates stability-aware orchestrator with Kaggle-style evaluation,
enabling reliability tracking during competition testing.
"""

import json
from pathlib import Path
from typing import Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.orchestration import create_aimo_orchestrator


class StabilityAwareKaggleRunner:
    """
    Kaggle submission runner with integrated stability tracking.
    
    Wraps the ToolOrchestra adapter to enable stability measurement
    during Kaggle-style evaluation.
    """
    
    def __init__(
        self,
        use_stability: bool = True,
        track_orchestration: bool = True,
        stability_output_dir: str = 'outputs/kaggle_stability'
    ):
        """
        Initialize Kaggle runner with stability tracking.
        
        Args:
            use_stability: Whether to measure solver-level stability
            track_orchestration: Whether to track orchestration-level stability
            stability_output_dir: Directory to save stability reports
        """
        self.orchestrator = create_aimo_orchestrator(
            use_toolorchestra=True,
            measure_stability=use_stability,
            track_orchestration_stability=track_orchestration
        )
        
        self.stability_output_dir = Path(stability_output_dir)
        self.stability_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.problem_results = []
        
    def solve_problem(self, problem_id: str, problem_statement: str) -> int:
        """
        Solve a single problem with stability tracking.
        
        Args:
            problem_id: Problem identifier
            problem_statement: Problem text
            
        Returns:
            Integer answer
        """
        answer = self.orchestrator.solve(
            problem_statement,
            problem_id=problem_id
        )
        
        # Record result
        self.problem_results.append({
            'problem_id': problem_id,
            'answer': answer
        })
        
        return answer
        
    def export_stability_report(self) -> None:
        """Export stability metrics after evaluation."""
        if not self.orchestrator.stability_tracker:
            print("No stability tracking enabled")
            return
            
        # Aggregate stability report
        report = self.orchestrator.get_stability_report()
        report_path = self.stability_output_dir / 'aggregate_stability.json'
        
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        print(f"✓ Aggregate stability saved to {report_path}")
        
        # Per-tool comparison
        comparison = self.orchestrator.get_tool_comparison()
        comparison_path = self.stability_output_dir / 'tool_comparison.json'
        
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
            
        print(f"✓ Tool comparison saved to {comparison_path}")
        
        # Per-problem stability details
        if self.orchestrator.stability_tracker.problem_stabilities:
            problem_details_path = self.stability_output_dir / 'per_problem_stability.json'
            
            # Extract stability for each problem
            problem_stabilities = {}
            for problem_id, data in self.orchestrator.stability_tracker.problem_stabilities.items():
                problem_stabilities[problem_id] = {
                    'tools_used': [tool['tool'] for tool in data['tools']],
                    'stable': all(
                        tool['stability'] and tool['stability'].status == 'GREEN'
                        for tool in data['tools']
                        if tool['stability']
                    )
                }
            
            with open(problem_details_path, 'w') as f:
                json.dump(problem_stabilities, f, indent=2)
                
            print(f"✓ Per-problem stability saved to {problem_details_path}")
            
    def get_submission_dataframe(self):
        """
        Get submission in DataFrame format.
        
        Returns:
            polars DataFrame with problem_id and answer columns
        """
        try:
            import polars as pl
            return pl.DataFrame(self.problem_results)
        except ImportError:
            # Fallback to dict
            return self.problem_results
            
    def export_submission_csv(self, output_path: str = 'outputs/submission.csv') -> None:
        """
        Export submission file in Kaggle format.
        
        Args:
            output_path: Path to save submission CSV
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import polars as pl
            df = pl.DataFrame(self.problem_results)
            df.write_csv(output_path)
        except ImportError:
            # Fallback to manual CSV writing
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['problem_id', 'answer'])
                writer.writeheader()
                writer.writerows(self.problem_results)
                
        print(f"✓ Submission saved to {output_path}")
        
    def print_summary(self) -> None:
        """Print summary statistics."""
        print("\n" + "="*60)
        print("Evaluation Summary")
        print("="*60)
        print(f"Problems solved: {len(self.problem_results)}")
        
        if self.orchestrator.stability_tracker:
            report = self.orchestrator.get_stability_report()
            print(f"\nStability Status: {report.aggregate_status}")
            print(f"Problems tracked: {report.problem_count}")
            
            if report.routing_entropies:
                import numpy as np
                print(f"Mean routing entropy: {np.mean(report.routing_entropies):.3f}")
            
            # Tool usage
            print("\nTool Stability:")
            comparison = self.orchestrator.get_tool_comparison()
            for tool, stats in comparison.items():
                if 'green_pct' in stats:
                    print(f"  {tool}: {stats['green_pct']:.1f}% GREEN")
        
        print("="*60)
