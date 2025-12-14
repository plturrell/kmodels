#!/usr/bin/env python3
"""
Test G-JEPA integration: verify proof sequence extraction and trace building.

Usage:
    python scripts/test_gjepa_integration.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.geometry.solver import GeometrySolver
from src.geometry.scene_sequence import build_scene_sequence, extract_trace_from_solver
from src.geometry.scene_encoder import SceneEncoder
from src.geometry.jepa_dataset import GeometryJEPADataset
from src.geometry.generator import ProblemGenerator


def test_proof_sequence_extraction():
    """Test that we can extract proof sequences from solver."""
    print("=" * 60)
    print("Test 1: Proof Sequence Extraction")
    print("=" * 60)
    
    # Create solver
    solver = GeometrySolver(
        max_search_iterations=100,
        max_depth=10,
        use_mcts=True,
    )
    
    # Generate a simple problem
    generator = ProblemGenerator(seed=42)
    problem, _ = generator.generate_triangle_problem("right", "easy")
    
    print(f"Problem: {problem[:100]}...")
    
    # Solve
    answer = solver.solve(problem, problem_id="test_1")
    print(f"Answer: {answer}")
    
    # Extract proof states
    proof_states = solver.get_proof_states()
    print(f"Proof states extracted: {len(proof_states)}")
    
    if len(proof_states) > 0:
        print(f"  Initial state: {proof_states[0]}")
        if len(proof_states) > 1:
            print(f"  Final state: {proof_states[-1]}")
        print("✓ Proof sequence extraction works")
    else:
        print("⚠ No proof states found")
    
    return proof_states


def test_trace_building(proof_states):
    """Test trace building from proof states."""
    print("\n" + "=" * 60)
    print("Test 2: Trace Building")
    print("=" * 60)
    
    if not proof_states:
        print("⚠ Skipping: No proof states available")
        return None
    
    # Build trace
    trace = build_scene_sequence(
        proof_states=proof_states,
        trace_id="test_trace",
        problem_id="test_problem",
    )
    
    print(f"Trace ID: {trace.trace_id}")
    print(f"Number of scenes: {len(trace.scenes)}")
    print(f"Number of states: {len(trace.states)}")
    
    if len(trace.scenes) > 0:
        print(f"  First scene: {trace.scenes[0]}")
        if len(trace.scenes) > 1:
            print(f"  Last scene: {trace.scenes[-1]}")
        print("✓ Trace building works")
    else:
        print("⚠ Empty trace")
    
    return trace


def test_encoding(trace):
    """Test scene encoding."""
    print("\n" + "=" * 60)
    print("Test 3: Scene Encoding")
    print("=" * 60)
    
    if not trace:
        print("⚠ Skipping: No trace available")
        return None
    
    # Create encoder
    encoder = SceneEncoder(output_dim=256)
    
    # Encode single state
    if len(trace.states) > 0:
        latent = encoder.encode_state(trace.states[0])
        print(f"Single state encoding: shape {latent.shape}")
        print("✓ Single state encoding works")
    
    # Encode entire trace
    latent_seq = encoder.encode_trace(trace)
    print(f"Trace encoding: shape {latent_seq.shape}")
    print("✓ Trace encoding works")
    
    return latent_seq


def test_dataset(traces):
    """Test dataset creation."""
    print("\n" + "=" * 60)
    print("Test 4: Dataset Creation")
    print("=" * 60)
    
    if not traces:
        print("⚠ Skipping: No traces available")
        return None
    
    # Create encoder
    encoder = SceneEncoder(output_dim=256)
    
    # Create dataset
    dataset = GeometryJEPADataset(traces=traces, encoder=encoder)
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    if len(dataset) > 0:
        h_seq, mask_indices = dataset[0]
        print(f"Sample h_seq shape: {h_seq.shape}")
        print(f"Sample mask_indices shape: {mask_indices.shape}")
        print(f"Number of masked positions: {len(mask_indices)}")
        print("✓ Dataset creation works")
    
    return dataset


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("G-JEPA Integration Tests")
    print("=" * 60 + "\n")
    
    # Test 1: Extract proof sequence
    proof_states = test_proof_sequence_extraction()
    
    # Test 2: Build trace
    trace = test_trace_building(proof_states)
    
    # Test 3: Encode
    latent_seq = test_encoding(trace)
    
    # Test 4: Dataset (need multiple traces)
    if trace:
        traces = [trace]  # For testing, use single trace
        dataset = test_dataset(traces)
    
    print("\n" + "=" * 60)
    print("All Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

