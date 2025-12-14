"""Deductive search: Search(S_initial, T) using MCTS."""

import random
import math
from typing import Any, Dict, List, Optional, Tuple

from .state import State
from .theorems import Theorem, TheoremLibrary
from .stability_metrics import (
    ReasoningStabilityMetrics,
    estimate_search_stability
)


class MCTSNode:
    """Node in Monte Carlo Tree Search."""

    def __init__(self, state: State, parent: Optional["MCTSNode"] = None, theorem: Optional[Theorem] = None, match: Optional[Dict[str, str]] = None):
        """
        Initialize MCTS node.

        Args:
            state: State at this node
            parent: Parent node
            theorem: Theorem that led to this state (if not root)
            match: Match used for theorem application
        """
        self.state = state
        self.parent = parent
        self.theorem = theorem
        self.match = match

        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0
        self.untried_theorems: List[Tuple[Theorem, Dict[str, str]]] = []
        self.gjepa_score: float = 0.0

    def is_fully_expanded(self) -> bool:
        """Check if all possible theorem applications have been tried."""
        return len(self.untried_theorems) == 0 and len(self.children) > 0

    def is_terminal(self, goal_proposition: Optional[str] = None) -> bool:
        """Check if this is a terminal (goal) state."""
        if goal_proposition:
            return self.state.has_proposition(goal_proposition)
        return False

    def select_child(
        self,
        exploration_constant: float = 1.414,
        gjepa_heuristic: Optional[float] = None,
    ) -> Optional["MCTSNode"]:
        """
        Select child using UCB1 formula, optionally enhanced with G-JEPA heuristic.

        Args:
            exploration_constant: Exploration parameter (default: √2)
            gjepa_heuristic: Optional G-JEPA heuristic score

        Returns:
            Selected child node
        """
        if not self.children:
            return None

        best_value = float("-inf")
        best_child: Optional["MCTSNode"] = None

        for child in self.children:
            if child.visits == 0:
                ucb_value = float("inf")
            else:
                exploitation = child.value / child.visits
                exploration = exploration_constant * (
                    (2 * self.visits) / child.visits
                ) ** 0.5
                ucb_value = exploitation + exploration
                
                # Add G-JEPA heuristic if available (use child's stored score)
                if hasattr(child, 'gjepa_score') and child.gjepa_score != 0.0:
                    ucb_value += child.gjepa_score * 0.5  # Weight heuristic
                elif gjepa_heuristic is not None:
                    ucb_value += gjepa_heuristic * 0.5

            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child

        return best_child

    def expand(self, theorem_library: TheoremLibrary) -> Optional["MCTSNode"]:
        """
        Expand node by applying a new theorem.

        Args:
            theorem_library: Library of available theorems

        Returns:
            New child node or None if no expansion possible
        """
        if not self.untried_theorems:
            # Get applicable theorems
            applicable = theorem_library.get_applicable_theorems(self.state)
            self.untried_theorems = applicable

        if not self.untried_theorems:
            return None

        # Select and apply a theorem
        theorem, match = self.untried_theorems.pop()
        new_state = self.state.apply_theorem(theorem, match)

        child = MCTSNode(new_state, parent=self, theorem=theorem, match=match)
        self.children.append(child)

        return child

    def update(self, value: float) -> None:
        """
        Update node statistics.

        Args:
            value: Value from simulation
        """
        self.visits += 1
        self.value += value


class DeductiveSearch:
    """
    Deductive search using Monte Carlo Tree Search.

    Implements Search(S_initial, T) → sequence (T₁, T₂, ..., Tₙ)
    
    Can optionally use G-JEPA for heuristic guidance.
    """

    def __init__(
        self,
        theorem_library: TheoremLibrary,
        max_iterations: int = 1000,
        max_depth: int = 50,
        exploration_constant: float = 1.414,
        gjepa_model: Optional[Any] = None,
        use_gjepa_heuristic: bool = False,
    ):
        """
        Initialize search.

        Args:
            theorem_library: Library of theorems
            max_iterations: Maximum MCTS iterations
            max_depth: Maximum search depth
            exploration_constant: UCB1 exploration parameter
            gjepa_model: Optional G-JEPA model for heuristic guidance
            use_gjepa_heuristic: Whether to use G-JEPA heuristic
        """
        self.theorem_library = theorem_library
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.gjepa_model = gjepa_model
        self.use_gjepa_heuristic = use_gjepa_heuristic and gjepa_model is not None
        self._last_search_trajectory: List[State] = []  # Track states for stability analysis
        self._proof_sequence_states: List[State] = []  # Track states in the actual proof sequence

    def search(
        self, initial_state: State, goal_proposition: Optional[str] = None
    ) -> Optional[List[Tuple[Theorem, Dict[str, str]]]]:
        """
        Search for a sequence of theorems that reach the goal.

        Args:
            initial_state: Initial state S_initial
            goal_proposition: Goal proposition to find

        Returns:
            Sequence of (theorem, match) tuples, or None if not found
        """
        root = MCTSNode(initial_state)
        self._last_search_trajectory = [initial_state]  # Reset trajectory tracking
        self._proof_sequence_states = [initial_state]  # Track states in proof sequence

        for _ in range(self.max_iterations):
            # Selection
            node = self._select(root, goal_proposition)

            # Expansion
            if not node.is_terminal(goal_proposition):
                child = node.expand(self.theorem_library)
                if child:
                    # Compute G-JEPA heuristic for new child if enabled
                    if self.use_gjepa_heuristic and self.gjepa_model is not None:
                        try:
                            goal_state = None  # Would need to construct from goal_proposition
                            child.gjepa_score = self.gjepa_model.compute_heuristic_score(
                                current_state=node.state,
                                candidate_state=child.state,
                                goal_state=goal_state,
                            )
                        except Exception:
                            child.gjepa_score = 0.0
                    else:
                        child.gjepa_score = 0.0
                    
                    node = child

            # Simulation
            value = self._simulate(node, goal_proposition)

            # Backpropagation
            self._backpropagate(node, value)
            
            # Track trajectory for stability analysis
            if len(self._last_search_trajectory) < 100:  # Cap trajectory length
                self._last_search_trajectory.append(node.state)

            # Check if goal found
            if root.is_terminal(goal_proposition):
                goal_sequence = self._extract_sequence(root)
                # Extract states from proof sequence
                self._extract_proof_states(root, initial_state)
                return goal_sequence

        # Return best sequence found
        best_sequence = self._extract_best_sequence(root, goal_proposition)
        if best_sequence:
            self._extract_proof_states(root, initial_state)
        return best_sequence

    def _select(self, root: MCTSNode, goal_proposition: Optional[str] = None) -> MCTSNode:
        """Select node using UCB1, optionally enhanced with G-JEPA."""
        node = root

        while node.is_fully_expanded() and node.children:
            # Compute G-JEPA heuristic if enabled
            gjepa_score = None
            if self.use_gjepa_heuristic and self.gjepa_model is not None:
                try:
                    # Get goal state if available (simplified)
                    goal_state = None
                    gjepa_score = self.gjepa_model.compute_heuristic_score(
                        current_state=node.state,
                        candidate_state=node.state,  # Will be computed per child
                        goal_state=goal_state,
                    )
                except Exception:
                    gjepa_score = None
            
            selected = node.select_child(self.exploration_constant, gjepa_heuristic=gjepa_score)
            if selected is None:
                break
            node = selected

        return node

    def _world_model_leaf_value(self, state: State) -> Optional[float]:
        """Estimate leaf value using the JEPA/world model, if available.

        Uses gjepa_model.compute_heuristic_score on the leaf state and
        maps the resulting score (negative loss) through a sigmoid to
        obtain a soft value in [0, 1]. Returns None if no model is
        available or if evaluation fails.
        """
        if self.gjepa_model is None:
            return None

        try:
            # Heuristic score: higher is better
            score = self.gjepa_model.compute_heuristic_score(
                current_state=state,
                candidate_state=state,
                goal_state=None,
            )
        except Exception:
            return None

        # Map score to [0, 1] with a sigmoid; clamp to avoid overflow
        # Large positive scores → value ≈ 1, large negative → value ≈ 0.
        x = max(min(score, 20.0), -20.0)
        value = 1.0 / (1.0 + math.exp(-x))
        return float(value)

    def _simulate(self, node: MCTSNode, goal_proposition: Optional[str]) -> float:
        """
        Simulate from node to terminal state.

        Args:
            node: Starting node
            goal_proposition: Goal to find

        Returns:
            Value (1.0 if goal found, 0.0 otherwise)
        """
        state = node.state.copy()
        depth = 0

        while depth < self.max_depth:
            if goal_proposition and state.has_proposition(goal_proposition):
                # Exact goal reached during rollout.
                return 1.0

            # Get applicable theorems
            applicable = self.theorem_library.get_applicable_theorems(state)
            if not applicable:
                break

            # Random selection for simulation
            theorem, match = random.choice(applicable)
            state = state.apply_theorem(theorem, match)
            depth += 1

        # Base heuristic value based on symbolic progress
        base_value = 0.0
        if goal_proposition:
            # If leaf already satisfies goal (but we hit depth/termination),
            # treat as partial success.
            if state.has_proposition(goal_proposition):
                base_value = 0.5

        # World-model-based leaf value (if available)
        wm_value = self._world_model_leaf_value(state)
        if wm_value is not None:
            if goal_proposition:
                # Blend symbolic and world-model signals.
                return 0.7 * base_value + 0.3 * wm_value
            else:
                # Pure world-model value when no explicit goal.
                return wm_value

        return base_value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.update(value)
            current = current.parent
            # Decay value as we go up
            value *= 0.9

    def _extract_sequence(self, root: MCTSNode) -> List[Tuple[Theorem, Dict[str, str]]]:
        """Extract theorem sequence from root to goal."""
        sequence: List[Tuple[Theorem, Dict[str, str]]] = []
        node = root

        while node.children:
            # Find child that leads to goal (simplified - would need proper tracking)
            best_child = max(node.children, key=lambda c: c.value / max(c.visits, 1))
            if best_child.theorem is not None and best_child.match is not None:
                sequence.append((best_child.theorem, best_child.match))
            node = best_child

        return sequence

    def _extract_best_sequence(
        self, root: MCTSNode, goal_proposition: Optional[str]
    ) -> Optional[List[Tuple[Theorem, Dict[str, str]]]]:
        """Extract best sequence found."""
        # Find best path to goal
        best_path = self._find_best_path_to_goal(root, goal_proposition)
        if best_path:
            seq: List[Tuple[Theorem, Dict[str, str]]] = []
            for node in best_path[1:]:
                if node.theorem is not None and node.match is not None:
                    seq.append((node.theorem, node.match))
            return seq
        return self._extract_sequence(root)
    
    def _find_best_path_to_goal(
        self, root: MCTSNode, goal_proposition: Optional[str]
    ) -> Optional[List[MCTSNode]]:
        """Find best path from root to goal node."""
        if not goal_proposition:
            return None
        
        # BFS to find goal node
        queue = [(root, [root])]
        visited = {root}
        
        while queue:
            node, path = queue.pop(0)
            
            if node.is_terminal(goal_proposition):
                return path
            
            # Explore children
            for child in node.children:
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        
        return None
    
    def _extract_proof_states(self, root: MCTSNode, initial_state: State) -> None:
        """Extract states from proof sequence."""
        sequence = self._extract_sequence(root)
        if not sequence:
            self._proof_sequence_states = [initial_state]
            return
        
        # Reconstruct states by applying theorems
        states = [initial_state]
        current_state = initial_state
        
        for theorem, match in sequence:
            try:
                current_state = current_state.apply_theorem(theorem, match)
                states.append(current_state)
            except Exception:
                # Skip if theorem application fails
                continue
        
        self._proof_sequence_states = states
    
    def get_proof_states(self) -> List[State]:
        """Get the states from the proof sequence."""
        return self._proof_sequence_states.copy()

    def measure_stability(
        self,
        initial_state: State,
        goal_proposition: Optional[str] = None,
        num_perturbations: int = 5,
        perturbation_epsilon: float = 0.1,
        horizon_steps: int = 20
    ) -> ReasoningStabilityMetrics:
        """
        Measure Lyapunov stability of proof search.
        
        Runs the search multiple times with slightly perturbed parameters
        and measures how trajectories diverge.
        
        Args:
            initial_state: Starting proof state
            goal_proposition: Goal to find
            num_perturbations: Number of perturbed runs
            perturbation_epsilon: Magnitude of parameter perturbations
            horizon_steps: Number of steps to analyze
            
        Returns:
            ReasoningStabilityMetrics with λ_max and status
        """
        # Create search function wrapper
        def search_fn(state: State, params: dict) -> Tuple[State, List[State]]:
            # Temporarily modify search parameters
            old_max_iterations = self.max_iterations
            old_exploration_constant = self.exploration_constant
            
            self.max_iterations = int(params.get('max_iterations', old_max_iterations))
            self.exploration_constant = float(params.get('exploration_constant', old_exploration_constant))
            
            # Run search
            self.search(state, goal_proposition)
            trajectory = list(self._last_search_trajectory)
            
            # Restore parameters
            self.max_iterations = old_max_iterations
            self.exploration_constant = old_exploration_constant
            
            final_state = trajectory[-1] if trajectory else state
            return final_state, trajectory
        
        # Current search parameters
        search_params = {
            'max_iterations': self.max_iterations,
            'exploration_constant': self.exploration_constant
        }
        
        # Estimate stability
        return estimate_search_stability(
            search_fn=search_fn,
            initial_state=initial_state,
            search_params=search_params,
            perturbation_epsilon=perturbation_epsilon,
            num_runs=num_perturbations,
            horizon_steps=horizon_steps
        )
    
    def search_with_stability(
        self,
        initial_state: State,
        goal_proposition: Optional[str] = None,
        measure_stability: bool = True
    ) -> Tuple[Optional[List[Tuple[Theorem, Dict[str, str]]]], Optional[ReasoningStabilityMetrics]]:
        """
        Run search and optionally measure stability.
        
        Args:
            initial_state: Starting state
            goal_proposition: Goal to find
            measure_stability: Whether to compute stability metrics
            
        Returns:
            (proof_sequence, stability_metrics) tuple
        """
        # Run primary search
        proof = self.search(initial_state, goal_proposition)
        
        # Optionally measure stability
        if measure_stability and proof is not None:
            metrics = self.measure_stability(initial_state, goal_proposition)
            return proof, metrics
        else:
            return proof, None


def search_for_proof(
    initial_state: State,
    theorem_library: TheoremLibrary,
    goal_proposition: Optional[str] = None,
    max_iterations: int = 1000,
) -> Optional[List[Tuple[Theorem, Dict[str, str]]]]:
    """
    Convenience function to search for a proof.

    Args:
        initial_state: Initial state
        theorem_library: Theorem library
        goal_proposition: Goal proposition
        max_iterations: Maximum search iterations

    Returns:
        Theorem sequence or None
    """
    search = DeductiveSearch(theorem_library, max_iterations=max_iterations)
    return search.search(initial_state, goal_proposition)

