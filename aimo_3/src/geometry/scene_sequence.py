"""
Scene sequence building for G-JEPA training.

Converts proof traces into sequences of geometric scenes.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from .state import State
from .scene_graph import GeometricSceneGraph


@dataclass
class SceneTrace:
    """
    A trace of geometric scenes from a proof.
    
    Represents a sequence of states: S_0 → S_1 → ... → S_n
    """
    scenes: List[GeometricSceneGraph]  # Sequence of scene graphs
    states: List[State]  # Corresponding states (for full context)
    trace_id: str = "unknown"
    problem_id: Optional[str] = None
    
    def __len__(self) -> int:
        """Length of trace (number of scenes)."""
        return len(self.scenes)
    
    def get_scene(self, index: int) -> GeometricSceneGraph:
        """Get scene at index."""
        if 0 <= index < len(self.scenes):
            return self.scenes[index]
        raise IndexError(f"Index {index} out of range for trace of length {len(self.scenes)}")


def build_scene_sequence(
    proof_states: List[State],
    trace_id: str = "unknown",
    problem_id: Optional[str] = None,
) -> SceneTrace:
    """
    Build scene sequence from proof states.
    
    Extracts the geometric scene graph from each state in the proof sequence.
    
    Args:
        proof_states: List of states from a proof (S_0, S_1, ..., S_n)
        trace_id: Identifier for this trace
        problem_id: Optional problem identifier
        
    Returns:
        SceneTrace with sequence of scene graphs
    """
    scenes = [state.graph for state in proof_states]
    
    return SceneTrace(
        scenes=scenes,
        states=proof_states,
        trace_id=trace_id,
        problem_id=problem_id,
    )


def extract_trace_from_solver(
    solver,
    problem_statement: str,
    problem_id: Optional[str] = None,
) -> Optional[SceneTrace]:
    """
    Extract proof trace from solver execution.
    
    Extracts the actual proof sequence states from the solver.
    
    Args:
        solver: GeometrySolver instance (must have solved the problem)
        problem_statement: Problem that was solved
        problem_id: Optional problem identifier
        
    Returns:
        SceneTrace if proof found, None otherwise
    """
    # Get proof states from solver
    proof_states = solver.get_proof_states()
    
    if not proof_states or len(proof_states) < 1:
        # Fallback: create minimal trace from initial state
        initial_graph = solver.parser.parse(problem_statement)
        initial_state = State(graph=initial_graph)
        proof_states = [initial_state]
    
    # Build trace from proof states
    trace = build_scene_sequence(
        proof_states=proof_states,
        trace_id=f"trace_{problem_id or 'unknown'}",
        problem_id=problem_id,
    )
    
    return trace

