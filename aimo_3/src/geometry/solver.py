"""Geometry Solver: Main solver class orchestrating the full pipeline."""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Any, Set, Dict

from .evaluation import EvaluationEngine
from .parser import GeometryParser, parse_problem
from .search import DeductiveSearch
from .state import State
from .theorems import Theorem, TheoremLibrary
from .theorems_extended import get_extended_theorem_library
from .theorems_advanced import get_advanced_theorem_library
from .stability_metrics import ReasoningStabilityMetrics
from .metadata_schema import ProofToken, create_proof_token


class GeometrySolver:
    """
    Main geometry solver implementing the full pipeline:
    I(S) → Search(I(S), T) → F(result) = k
    """

    def __init__(
        self,
        max_search_iterations: int = 1000,
        max_depth: int = 50,
        use_mcts: bool = True,
        measure_stability: bool = False,
        stability_horizon: int = 20,
        use_gjepa: bool = False,
    ):
        """
        Initialize geometry solver.

        Args:
            max_search_iterations: Maximum MCTS iterations
            max_depth: Maximum search depth
            use_mcts: Whether to use MCTS (True) or simple DFS (False)
            measure_stability: Whether to compute Lyapunov stability metrics
            stability_horizon: Number of steps to analyze for stability
            use_gjepa: Whether to use G-JEPA for heuristic guidance
        """
        self.parser = GeometryParser()
        # Use advanced theorem library with all theorems (25 total)
        self.theorem_library = get_advanced_theorem_library()
        self.max_search_iterations = max_search_iterations
        self.max_depth = max_depth
        self.use_mcts = use_mcts
        self.evaluator = EvaluationEngine()
        self.measure_stability = measure_stability
        self.stability_horizon = stability_horizon
        self.use_gjepa = use_gjepa
        self._last_proof_token: Optional[ProofToken] = None
        self._last_proof_states: List[State] = []  # Store proof states for trace extraction

    def solve(self, problem_statement: str, problem_id: str = "unknown") -> int:
        """
        Solve a geometric problem.

        Implements: k = F( Search( I(S), T ) )

        Args:
            problem_statement: LaTeX problem statement S
            problem_id: Problem identifier for metadata

        Returns:
            Integer answer k in [0, 99999]
        """
        # Stage 1: Parsing - I(S) → G_initial
        initial_graph = self.parser.parse(problem_statement)
        initial_state = State(graph=initial_graph)

        # Extract goal from problem statement
        goal_proposition = self._extract_goal(problem_statement)

        # Stage 2: Reasoning - Search(I(S), T) → sequence
        proof_states = [initial_state]  # Track proof states
        if self.use_mcts:
            # Load G-JEPA model if available and enabled
            gjepa_model = None
            encoder = None
            use_gjepa = getattr(self, 'use_gjepa', False)
            if use_gjepa:
                try:
                    from ..modeling.gjepa_model import GJEPA, create_gjepa_model
                    from .scene_encoder import SceneEncoder
                    import torch
                    from pathlib import Path
                    gjepa_path = Path(__file__).parent.parent.parent / "outputs" / "gjepa" / "gjepa_final.pt"
                    if gjepa_path.exists():
                        checkpoint = torch.load(gjepa_path, map_location='cpu')
                        # Create model and encoder
                        encoder = SceneEncoder(output_dim=256)
                        gjepa_model = create_gjepa_model(latent_dim=256)
                        if 'model_state_dict' in checkpoint:
                            gjepa_model.load_state_dict(checkpoint['model_state_dict'])
                        if 'encoder_state_dict' in checkpoint:
                            encoder.load_state_dict(checkpoint['encoder_state_dict'])
                        gjepa_model.encoder = encoder
                        gjepa_model.eval()
                        encoder.eval()
                        print("✓ G-JEPA model loaded for heuristic guidance")
                except Exception as e:
                    print(f"Warning: Could not load G-JEPA model: {e}")
                    gjepa_model = None
                    encoder = None
            
            search = DeductiveSearch(
                self.theorem_library,
                max_iterations=self.max_search_iterations,
                max_depth=self.max_depth,
                gjepa_model=gjepa_model,
                use_gjepa_heuristic=use_gjepa and gjepa_model is not None,
            )
            theorem_sequence = search.search(initial_state, goal_proposition)
            # Extract proof states from search
            proof_states = search.get_proof_states()
        else:
            # Simple depth-first search fallback
            theorem_sequence = self._simple_search(initial_state, goal_proposition)

        # Use proof states if available, otherwise apply theorem sequence
        if proof_states and len(proof_states) > 1:
            final_state = proof_states[-1]
        else:
            # Fallback: apply theorem sequence
            final_state = initial_state
            if theorem_sequence:
                for theorem, match in theorem_sequence:
                    final_state = final_state.apply_theorem(theorem, match)

        # Store proof states for trace extraction
        self._last_proof_states = proof_states
        
        # Stage 3: Evaluation - F(G_n, Φ_n) = k
        answer = self.evaluator.evaluate(final_state, problem_statement)
        
        # Optional: Measure stability and create proof token
        stability_metrics = None
        if self.measure_stability and self.use_mcts and theorem_sequence:
            search = DeductiveSearch(
                self.theorem_library,
                max_iterations=self.max_search_iterations,
                max_depth=self.max_depth,
            )
            stability_metrics = search.measure_stability(
                initial_state=initial_state,
                goal_proposition=goal_proposition,
                horizon_steps=self.stability_horizon
            )
        
        # Create proof token with metadata
        self._last_proof_token = create_proof_token(
            problem_id=problem_id,
            proof_sequence=theorem_sequence if theorem_sequence else [],
            answer=answer,
            stability_metrics=stability_metrics,
            search_config={
                'max_iterations': self.max_search_iterations,
                'max_depth': self.max_depth,
                'use_mcts': self.use_mcts
            }
        )

        return answer
    
    def get_proof_states(self) -> List[State]:
        """Get the states from the last proof sequence."""
        return self._last_proof_states.copy()

    def _extract_goal(self, problem_statement: str) -> Optional[str]:
        """
        Extract goal proposition from problem statement.

        Args:
            problem_statement: Problem statement

        Returns:
            Goal proposition or None
        """
        # Look for "Find", "Compute", "Determine" patterns
        goal_patterns = [
            r'find\s+(?:the\s+)?(?:length|measure|area|value|number)\s+of\s+(.+)',
            r'compute\s+(.+)',
            r'determine\s+(.+)',
            r'what\s+is\s+(.+)',
        ]

        import re
        for pattern in goal_patterns:
            match = re.search(pattern, problem_statement, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _simple_search(
        self, initial_state: State, goal_proposition: Optional[str]
    ) -> Optional[List[Tuple[Theorem, Dict[str, str]]]]:
        """
        Simple depth-first search (fallback when MCTS disabled).

        Args:
            initial_state: Initial state
            goal_proposition: Goal to find

        Returns:
            Theorem sequence or None
        """
        visited: set[int] = set()
        stack: List[Tuple[State, List[Tuple[Theorem, Dict[str, str]]]]] = [(initial_state, [])]

        while stack and len(stack[0][1]) < self.max_depth:
            state, sequence = stack.pop()

            # Check if goal reached
            if goal_proposition and state.has_proposition(goal_proposition):
                return sequence

            # Check if visited
            state_hash = hash(state)
            if state_hash in visited:
                continue
            visited.add(state_hash)

            # Get applicable theorems
            applicable = self.theorem_library.get_applicable_theorems(state)

            # Apply theorems
            for theorem, match in applicable:
                new_state = state.apply_theorem(theorem, match)
                new_sequence = sequence + [(theorem, match)]
                stack.append((new_state, new_sequence))

        return None

    def add_theorem(self, theorem) -> None:
        """Add a custom theorem to the library."""
        self.theorem_library.add_theorem(theorem)

    def get_theorem_library(self) -> TheoremLibrary:
        """Get the theorem library."""
        return self.theorem_library
    
    def get_last_proof_token(self) -> Optional[ProofToken]:
        """
        Get the proof token from the last solve() call.
        
        Returns:
            ProofToken with stability metadata, or None if no solve() called yet
        """
        return self._last_proof_token
    
    def export_proof_token_json(self, filepath: str) -> None:
        """
        Export the last proof token to a JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        if self._last_proof_token:
            with open(filepath, 'w') as f:
                f.write(self._last_proof_token.to_json())


