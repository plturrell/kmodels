#!/usr/bin/env python3
"""
Demonstration of Lyapunov stability analysis for geometry solver.

This script shows how to use the integrated stability metrics
to measure reasoning robustness.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.geometry.solver import GeometrySolver


def demo_stability_aware_solving():
    """Demonstrate stability-aware geometry solving."""
    
    print("=" * 70)
    print("Lyapunov Stability Analysis for Geometry Solver")
    print("=" * 70)
    print()
    
    # Initialize solver with stability measurement enabled
    solver = GeometrySolver(
        max_search_iterations=100,
        max_depth=20,
        use_mcts=True,
        measure_stability=True,
        stability_horizon=15
    )
    
    # Test problem
    problem = """
    In right triangle ABC with right angle at C, if AC = 3 and BC = 4, find AB.
    """
    
    print("Problem:")
    print(problem)
    print()
    
    print("Solving with stability analysis enabled...")
    print()
    
    # Solve
    try:
        answer = solver.solve(problem, problem_id="pythagorean_demo")
        
        print(f"Answer: {answer}")
        print()
        
        # Retrieve proof token
        token = solver.get_last_proof_token()
        
        if token:
            print("Proof Metadata:")
            print(f"  Problem ID: {token.problem_id}")
            print(f"  Proof found: {token.proof_found}")
            print(f"  Proof length: {len(token.proof_sequence)}")
            print()
            
            if token.stability:
                print("Stability Metrics:")
                print(f"  λ_max (Lyapunov exponent): {token.stability.lambda_max:.4f}")
                print(f"  Entropy: {token.stability.entropy:.4f}")
                print(f"  Perturbation growth: {token.stability.perturbation_growth:.2f}x")
                print(f"  Status: {token.stability.status}")
                print(f"  Steps analyzed: {token.stability.num_steps}")
                print()
                
                # Interpret stability
                if token.stability.status == 'GREEN':
                    print("✓ STABLE: Reasoning is robust and reliable.")
                elif token.stability.status == 'AMBER':
                    print("⚠ MARGINAL: Reasoning shows some sensitivity.")
                else:
                    print("✗ UNSTABLE: Reasoning is chaotic and unreliable.")
            else:
                print("(No stability metrics - proof may not have been found)")
                
            print()
            
            # Export to JSON
            output_path = project_root / "outputs" / "proof_token_demo.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            solver.export_proof_token_json(str(output_path))
            print(f"Proof token exported to: {output_path}")
            
    except Exception as e:
        print(f"Error during solving: {e}")
        import traceback
        traceback.print_exc()

def main():
    demo_stability_aware_solving()

if __name__ == "__main__":
    main()
