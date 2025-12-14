"""Adapter for integrating ToolOrchestra with AIMO solvers."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add ToolOrchestra to path
vendor_path = Path(__file__).parent.parent.parent / "vendor" / "ToolOrchestra"
if vendor_path.exists():
    sys.path.insert(0, str(vendor_path))

from ..orchestration.aimo_tools import get_aimo_tools

try:
    from .stability_tracker import StabilityTracker
    STABILITY_AVAILABLE = True
except ImportError:
    STABILITY_AVAILABLE = False
    StabilityTracker = None


class ToolOrchestraAdapter:
    """
    Adapter for ToolOrchestra to work with AIMO problem solving.
    
    Integrates ToolOrchestra's orchestration framework with our domain solvers.
    """

    def __init__(
        self,
        orchestrator_model: Optional[str] = None,
        use_toolorchestra: bool = True,
        measure_stability: bool = False,
        track_orchestration_stability: bool = False,
    ):
        """
        Initialize ToolOrchestra adapter.

        Args:
            orchestrator_model: Path to orchestrator model checkpoint
            use_toolorchestra: Whether to use ToolOrchestra (False = direct solver)
            measure_stability: Whether to compute stability in individual solvers
            track_orchestration_stability: Whether to track orchestration-level stability
        """
        self.use_toolorchestra = use_toolorchestra
        self.orchestrator_model = orchestrator_model
        self.measure_stability = measure_stability
        self.tools = get_aimo_tools(measure_stability=measure_stability)
        
        # Initialize stability tracker
        if track_orchestration_stability and STABILITY_AVAILABLE:
            self.stability_tracker = StabilityTracker()
        else:
            self.stability_tracker = None

        if use_toolorchestra and self._check_toolorchestra_available():
            self._setup_orchestrator()
        else:
            # Fallback to direct solver routing
            from ..solvers.unified_solver import UnifiedSolver
            self.fallback_solver = UnifiedSolver()

    def _check_toolorchestra_available(self) -> bool:
        """Check if ToolOrchestra is available."""
        try:
            import sys
            vendor_path = Path(__file__).parent.parent.parent / "vendor" / "ToolOrchestra"
            if not vendor_path.exists():
                return False

            # Check for key dependencies
            try:
                import vllm
                import transformers
                return True
            except ImportError:
                print("ToolOrchestra dependencies not installed. Using fallback.")
                return False
        except Exception:
            return False

    def _setup_orchestrator(self):
        """Set up ToolOrchestra orchestrator."""
        try:
            # Import ToolOrchestra components
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vendor" / "ToolOrchestra"))

            # Set up LLM call function (simplified - would use actual ToolOrchestra API)
            self.orchestrator_available = True
        except Exception as e:
            print(f"Failed to set up ToolOrchestra: {e}")
            self.orchestrator_available = False

    def solve(self, problem_statement: str, problem_id: str = "unknown") -> int:
        """
        Solve problem using ToolOrchestra orchestration.

        Args:
            problem_statement: Problem statement
            problem_id: Problem identifier for metadata

        Returns:
            Integer answer
        """
        if self.use_toolorchestra and self.orchestrator_available:
            return self._solve_with_orchestration(problem_statement, problem_id)
        else:
            # Fallback to direct solver
            return self.fallback_solver.solve(problem_statement)

    def _solve_with_orchestration(self, problem_statement: str, problem_id: str) -> int:
        """
        Solve using ToolOrchestra orchestration.

        Args:
            problem_statement: Problem statement
            problem_id: Problem identifier

        Returns:
            Integer answer
        """
        # Simplified orchestration logic
        # In full implementation, would use ToolOrchestra's RL-trained orchestrator

        # Step 1: Analyze problem to determine which tools might be needed
        problem_lower = problem_statement.lower()

        # Step 2: Try tools in order of relevance
        tool_order = self._determine_tool_order(problem_lower)
        
        # NEW: Compute tool scores and track routing stability
        tool_scores = self._compute_tool_scores(problem_lower)
        
        if self.stability_tracker:
            self.stability_tracker.record_routing_decision(
                problem_id=problem_id,
                tool_scores=tool_scores,
                selected_tools=tool_order
            )

        # Step 3: Execute tools and collect results
        results = []
        for tool_name in tool_order:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                result = tool(problem_statement)
                if result.get("success"):
                    results.append(result)
                    
                    # NEW: Record tool execution with stability
                    if self.stability_tracker:
                        stability_metrics = result.get("stability")
                        self.stability_tracker.record_tool_execution(
                            tool_name=tool_name,
                            problem_id=problem_id,
                            stability_metrics=stability_metrics,
                            success=result["success"],
                            answer=result["answer"]
                        )

        # Step 4: Combine results (could use ToolOrchestra's learned combination)
        if results:
            # Use result with highest confidence
            best_result = max(results, key=lambda r: r.get("confidence", 0))
            return best_result.get("answer", 0)

        # Fallback
        return self.fallback_solver.solve(problem_statement)

    def _determine_tool_order(self, problem_lower: str) -> List[str]:
        """
        Determine order of tools to try based on problem content.

        Args:
            problem_lower: Lowercase problem statement

        Returns:
            List of tool names in order of relevance
        """
        tool_scores = {}

        # Geometry keywords
        geometry_keywords = [
            "triangle", "circle", "angle", "length", "area", "perimeter",
            "parallel", "perpendicular", "tangent", "inscribed", "circumscribed",
        ]
        geometry_score = sum(1 for kw in geometry_keywords if kw in problem_lower)
        if geometry_score > 0:
            tool_scores["geometry_solver"] = geometry_score

        # Algebra keywords
        algebra_keywords = [
            "equation", "solve", "polynomial", "factor", "expand",
            "quadratic", "linear", "system", "variable",
        ]
        algebra_score = sum(1 for kw in algebra_keywords if kw in problem_lower)
        if algebra_score > 0:
            tool_scores["algebra_solver"] = algebra_score

        # Number theory keywords
        nt_keywords = [
            "mod", "modulo", "remainder", "divisible", "gcd", "lcm",
            "prime", "digit", "integer",
        ]
        nt_score = sum(1 for kw in nt_keywords if kw in problem_lower)
        if nt_score > 0:
            tool_scores["number_theory_solver"] = nt_score

        # Combinatorics keywords
        comb_keywords = [
            "permutation", "combination", "choose", "arrange", "probability",
            "ways", "factorial", "select",
        ]
        comb_score = sum(1 for kw in comb_keywords if kw in problem_lower)
        if comb_score > 0:
            tool_scores["combinatorics_solver"] = comb_score

        # Graph keywords
        graph_keywords = ["graph", "vertex", "edge", "path", "tree", "cycle"]
        graph_score = sum(1 for kw in graph_keywords if kw in problem_lower)
        if graph_score > 0:
            tool_scores["graph_solver"] = graph_score

        # Analysis keywords
        analysis_keywords = [
            "limit", "derivative", "integral", "differentiate", "integrate",
            "sequence", "series", "calculus",
        ]
        analysis_score = sum(1 for kw in analysis_keywords if kw in problem_lower)
        if analysis_score > 0:
            tool_scores["analysis_solver"] = analysis_score

        # Sort by score (highest first)
        sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
        return [tool_name for tool_name, _ in sorted_tools]
    
    def _compute_tool_scores(self, problem_lower: str) -> Dict[str, float]:
        """
        Compute numeric scores for each tool.
        
        Args:
            problem_lower: Lowercase problem statement
            
        Returns:
            Dictionary mapping tool names to scores
        """
        tool_scores = {}

        # Geometry keywords
        geometry_keywords = [
            "triangle", "circle", "angle", "length", "area", "perimeter",
            "parallel", "perpendicular", "tangent", "inscribed", "circumscribed",
        ]
        geometry_score = sum(1 for kw in geometry_keywords if kw in problem_lower)
        if geometry_score > 0:
            tool_scores["geometry_solver"] = float(geometry_score)

        # Algebra keywords
        algebra_keywords = [
            "equation", "solve", "polynomial", "factor", "expand",
            "quadratic", "linear", "system", "variable",
        ]
        algebra_score = sum(1 for kw in algebra_keywords if kw in problem_lower)
        if algebra_score > 0:
            tool_scores["algebra_solver"] = float(algebra_score)

        # Number theory keywords
        nt_keywords = [
            "mod", "modulo", "remainder", "divisible", "gcd", "lcm",
            "prime", "digit", "integer",
        ]
        nt_score = sum(1 for kw in nt_keywords if kw in problem_lower)
        if nt_score > 0:
            tool_scores["number_theory_solver"] = float(nt_score)

        # Combinatorics keywords
        comb_keywords = [
            "permutation", "combination", "choose", "arrange", "probability",
            "ways", "factorial", "select",
        ]
        comb_score = sum(1 for kw in comb_keywords if kw in problem_lower)
        if comb_score > 0:
            tool_scores["combinatorics_solver"] = float(comb_score)

        # Graph keywords
        graph_keywords = ["graph", "vertex", "edge", "path", "tree", "cycle"]
        graph_score = sum(1 for kw in graph_keywords if kw in problem_lower)
        if graph_score > 0:
            tool_scores["graph_solver"] = float(graph_score)
            
        return tool_scores

    def solve_with_metadata(self, problem_statement: str) -> Dict[str, Any]:
        """
        Solve with full metadata.

        Args:
            problem_statement: Problem statement

        Returns:
            Dictionary with answer and metadata
        """
        answer = self.solve(problem_statement)
        return {
            "answer": answer,
            "method": "toolorchestra" if self.use_toolorchestra else "direct",
            "tools_used": self._determine_tool_order(problem_statement.lower()),
        }
    
    def get_stability_report(self):
        """
        Get aggregated stability metrics.
        
        Returns:
            OrchestrationStabilityMetrics or None if tracking disabled
        """
        if self.stability_tracker:
            return self.stability_tracker.get_aggregate_metrics()
        return None
    
    def get_tool_comparison(self):
        """
        Get per-tool stability comparison.
        
        Returns:
            Dictionary with tool statistics or None if tracking disabled
        """
        if self.stability_tracker:
            return self.stability_tracker.get_tool_comparison()
        return None
        
    def export_stability_json(self, filepath: str):
        """
        Export stability metrics to JSON file.
        
        Args:
            filepath: Output file path
        """
        import json
        
        if self.stability_tracker:
            metrics = self.stability_tracker.get_aggregate_metrics()
            with open(filepath, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)


def create_aimo_orchestrator(
    orchestrator_model: Optional[str] = None,
    use_toolorchestra: bool = True,
    measure_stability: bool = False,
    track_orchestration_stability: bool = False,
) -> ToolOrchestraAdapter:
    """
    Create AIMO orchestrator using ToolOrchestra.

    Args:
        orchestrator_model: Path to orchestrator model
        use_toolorchestra: Whether to use ToolOrchestra
        measure_stability: Whether to compute stability in individual solvers
        track_orchestration_stability: Whether to track orchestration stability

    Returns:
        ToolOrchestraAdapter instance
    """
    return ToolOrchestraAdapter(
        orchestrator_model=orchestrator_model,
        use_toolorchestra=use_toolorchestra,
        measure_stability=measure_stability,
        track_orchestration_stability=track_orchestration_stability,
    )
