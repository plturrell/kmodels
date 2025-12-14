"""Stability tracking for orchestration and proof tokens."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class StabilityStatus:
    """Stability status for a proof token or tool execution."""
    
    status: str  # "stable", "unstable", "marginally_stable"
    lyapunov_exponent: Optional[float] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProofToken:
    """A proof token with stability information."""
    
    token_id: str
    theorem_name: str
    stability: Optional[StabilityStatus] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationStabilityMetrics:
    """Aggregated stability metrics for orchestration."""
    
    total_problems: int = 0
    stable_routings: int = 0
    unstable_routings: int = 0
    average_confidence: float = 0.0
    tool_stability_scores: Dict[str, float] = field(default_factory=dict)
    routing_consistency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "total_problems": self.total_problems,
            "stable_routings": self.stable_routings,
            "unstable_routings": self.unstable_routings,
            "average_confidence": self.average_confidence,
            "tool_stability_scores": self.tool_stability_scores,
            "routing_consistency": self.routing_consistency,
            "metadata": self.metadata,
        }


class ToolComparisonStats(TypedDict):
    executions: int
    successes: int
    stability_scores: list[float]
    average_confidence: float


def _new_tool_comparison_stats() -> ToolComparisonStats:
    return {
        "executions": 0,
        "successes": 0,
        "stability_scores": [],
        "average_confidence": 0.0,
    }


class StabilityTracker:
    """
    Tracks stability metrics for orchestration and tool execution.
    
    Measures:
    - Routing stability (consistency of tool selection)
    - Tool execution stability (Lyapunov stability of proofs)
    - Overall orchestration stability
    """

    def __init__(self):
        """Initialize stability tracker."""
        self.routing_decisions: List[Dict[str, Any]] = []
        self.tool_executions: List[Dict[str, Any]] = []
        self.problem_tool_mapping: Dict[str, List[str]] = defaultdict(list)

    def record_routing_decision(
        self,
        problem_id: str,
        tool_scores: Dict[str, float],
        selected_tools: List[str],
    ) -> None:
        """
        Record a routing decision.

        Args:
            problem_id: Problem identifier
            tool_scores: Scores for each tool
            selected_tools: Tools selected for execution
        """
        decision = {
            "problem_id": problem_id,
            "tool_scores": tool_scores,
            "selected_tools": selected_tools,
            "timestamp": self._get_timestamp(),
        }
        self.routing_decisions.append(decision)
        self.problem_tool_mapping[problem_id] = selected_tools

    def record_tool_execution(
        self,
        tool_name: str,
        problem_id: str,
        stability_metrics: Optional[Dict[str, Any]],
        success: bool,
        answer: int,
    ) -> None:
        """
        Record tool execution with stability metrics.

        Args:
            tool_name: Name of tool executed
            problem_id: Problem identifier
            stability_metrics: Stability metrics from tool (if available)
            success: Whether execution was successful
            answer: Answer produced
        """
        execution = {
            "tool_name": tool_name,
            "problem_id": problem_id,
            "stability_metrics": stability_metrics,
            "success": success,
            "answer": answer,
            "timestamp": self._get_timestamp(),
        }
        self.tool_executions.append(execution)

    def get_aggregate_metrics(self) -> OrchestrationStabilityMetrics:
        """
        Compute aggregate stability metrics.

        Returns:
            OrchestrationStabilityMetrics with aggregated data
        """
        total_problems = len(set(d["problem_id"] for d in self.routing_decisions))
        
        # Compute routing stability (consistency)
        routing_consistency = self._compute_routing_consistency()
        
        # Compute tool stability scores
        tool_stability_scores = self._compute_tool_stability_scores()
        
        # Count stable vs unstable routings
        stable_routings = sum(
            1 for d in self.routing_decisions
            if self._is_stable_routing(d)
        )
        unstable_routings = total_problems - stable_routings
        
        # Average confidence
        confidences = [
            exec_data.get("stability_metrics", {}).get("confidence", 0.0)
            for exec_data in self.tool_executions
            if exec_data.get("stability_metrics")
        ]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OrchestrationStabilityMetrics(
            total_problems=total_problems,
            stable_routings=stable_routings,
            unstable_routings=unstable_routings,
            average_confidence=average_confidence,
            tool_stability_scores=tool_stability_scores,
            routing_consistency=routing_consistency,
            metadata={
                "total_routing_decisions": len(self.routing_decisions),
                "total_tool_executions": len(self.tool_executions),
            },
        )

    def get_tool_comparison(self) -> Dict[str, ToolComparisonStats]:
        """
        Get per-tool stability comparison.

        Returns:
            Dictionary mapping tool names to statistics
        """
        tool_stats: defaultdict[str, ToolComparisonStats] = defaultdict(_new_tool_comparison_stats)

        for execution in self.tool_executions:
            tool_name = str(execution["tool_name"])
            stats = tool_stats[tool_name]
            
            stats["executions"] += 1
            if bool(execution["success"]):
                stats["successes"] += 1

            stability_metrics = execution.get("stability_metrics")
            if stability_metrics:
                confidence = float(stability_metrics.get("confidence", 0.0))
                stats["stability_scores"].append(confidence)
                stats["average_confidence"] = sum(stats["stability_scores"]) / len(stats["stability_scores"])

        return dict(tool_stats)

    def _compute_routing_consistency(self) -> float:
        """
        Compute routing consistency score.

        Returns:
            Consistency score between 0 and 1
        """
        if len(self.routing_decisions) < 2:
            return 1.0

        # Group by problem type (simplified - would use problem similarity)
        # For now, compute consistency of tool selection patterns
        tool_patterns: dict[tuple[str, ...], list[str]] = {}
        for decision in self.routing_decisions:
            problem_id = str(decision["problem_id"])
            selected = tuple(sorted([str(t) for t in decision["selected_tools"]]))
            
            tool_patterns.setdefault(selected, []).append(problem_id)

        # Consistency = how often same pattern is used
        if not tool_patterns:
            return 0.0

        max_pattern_count = max(len(problems) for problems in tool_patterns.values())
        total_decisions = len(self.routing_decisions)
        
        return max_pattern_count / total_decisions if total_decisions > 0 else 0.0

    def _compute_tool_stability_scores(self) -> Dict[str, float]:
        """
        Compute stability scores for each tool.

        Returns:
            Dictionary mapping tool names to stability scores
        """
        tool_scores: defaultdict[str, list[float]] = defaultdict(list)

        for execution in self.tool_executions:
            tool_name = str(execution["tool_name"])
            stability_metrics = execution.get("stability_metrics")
            
            if stability_metrics:
                status = str(stability_metrics.get("stability_status", "unknown"))
                confidence = float(stability_metrics.get("confidence", 0.0))
                
                # Convert status to numeric score
                if status == "stable":
                    score = 1.0 * confidence
                elif status == "marginally_stable":
                    score = 0.5 * confidence
                else:  # unstable
                    score = 0.0
                
                tool_scores[tool_name].append(score)

        # Average scores per tool
        return {
            tool_name: sum(scores) / len(scores) if scores else 0.0
            for tool_name, scores in tool_scores.items()
        }

    def _is_stable_routing(self, decision: Dict[str, Any]) -> bool:
        """
        Check if routing decision is stable.

        Args:
            decision: Routing decision dictionary

        Returns:
            True if routing is considered stable
        """
        tool_scores = decision.get("tool_scores", {})
        if not tool_scores:
            return False

        # Stable if clear winner (score difference > threshold)
        sorted_scores = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) < 2:
            return True

        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        # Stable if top score is significantly higher
        return (top_score - second_score) >= 1.0

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def export_to_json(self, filepath: str) -> None:
        """
        Export all tracking data to JSON.

        Args:
            filepath: Output file path
        """
        data = {
            "routing_decisions": self.routing_decisions,
            "tool_executions": self.tool_executions,
            "aggregate_metrics": self.get_aggregate_metrics().to_dict(),
            "tool_comparison": self.get_tool_comparison(),
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
