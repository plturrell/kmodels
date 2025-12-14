"""State machine: State s = (G, Φ)."""

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from .scene_graph import GeometricSceneGraph

if TYPE_CHECKING:
    from .theorems import Theorem


class State:
    """
    State s = (G, Φ) where:
    - G is the Geometric Scene Graph
    - Φ is a set of derived propositions
    """

    def __init__(self, graph: Optional[GeometricSceneGraph] = None, propositions: Optional[Set[str]] = None):
        """
        Initialize state.

        Args:
            graph: Initial scene graph (default: empty graph)
            propositions: Initial proposition set (default: empty set)
        """
        self.graph = graph or GeometricSceneGraph()
        self.propositions = propositions or set()
        self.history: List[str] = []  # History of theorem applications

    def apply_theorem(self, theorem: 'Theorem', match: Dict[str, str]) -> "State":
        """
        Apply a theorem to transform the state.

        Args:
            theorem: Theorem to apply
            match: Mapping from pattern labels to graph labels

        Returns:
            New state after theorem application
        """
        # Get additions from theorem
        G_addition, Φ_addition = theorem.apply(self, match)

        # Create new state
        new_graph = self.graph.copy()
        new_graph.merge(G_addition)

        new_propositions = self.propositions.copy()
        new_propositions.update(Φ_addition)

        new_state = State(graph=new_graph, propositions=new_propositions)
        new_state.history = self.history + [f"{theorem.name}({match})"]

        return new_state

    def has_proposition(self, proposition: str) -> bool:
        """
        Check if a proposition is in the state.

        Args:
            proposition: Proposition to check

        Returns:
            True if proposition exists
        """
        # Exact match
        if proposition in self.propositions:
            return True

        # Pattern matching (simplified)
        for prop in self.propositions:
            if self._proposition_matches(prop, proposition):
                return True

        return False

    def _proposition_matches(self, prop: str, pattern: str) -> bool:
        """Check if proposition matches pattern."""
        # Simplified matching - would need more sophisticated logic
        return pattern.lower() in prop.lower()

    def get_goal_proposition(self, problem_goal: str) -> Optional[str]:
        """
        Extract goal proposition from problem statement.

        Args:
            problem_goal: Goal statement from problem

        Returns:
            Proposition string or None
        """
        # Try to find matching proposition
        for prop in self.propositions:
            if self._proposition_matches(prop, problem_goal):
                return prop

        return None

    def copy(self) -> "State":
        """Create a copy of the state."""
        new_state = State(
            graph=self.graph.copy(),
            propositions=self.propositions.copy(),
        )
        new_state.history = self.history.copy()
        return new_state

    def __eq__(self, other) -> bool:
        """Check equality of states."""
        if not isinstance(other, State):
            return False
        # Compare graphs and propositions
        # Simplified - would need proper graph isomorphism
        return self.propositions == other.propositions

    def __hash__(self) -> int:
        """Hash state for use in sets."""
        return hash((id(self.graph), tuple(sorted(self.propositions))))

    def __str__(self) -> str:
        """String representation."""
        props_str = ", ".join(sorted(self.propositions))[:100]
        return f"State(G={self.graph}, Φ=[{props_str}...])"

