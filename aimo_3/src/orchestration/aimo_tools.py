"""AIMO-specific tools for ToolOrchestra integration."""

from typing import Any, Dict, Optional

from ..solvers.algebra_solver import AlgebraSolver
from ..solvers.analysis_solver import AnalysisSolver
from ..solvers.combinatorics_solver import CombinatoricsSolver, GraphSolver
from ..solvers.geometry_solver_wrapper import GeometrySolverWrapper
from ..solvers.number_theory_solver import NumberTheorySolver


class AIMOGeometryTool:
    """Geometry solver as a ToolOrchestra tool."""

    def __init__(self, measure_stability: bool = False):
        """Initialize geometry tool.
        
        Args:
            measure_stability: Whether to compute Lyapunov stability metrics
        """
        self.solver = GeometrySolverWrapper(measure_stability=measure_stability)
        self.name = "geometry_solver"
        self.description = "Solves geometric problems using formal reasoning with scene graphs and theorems"
        self.measure_stability = measure_stability

    def __call__(self, problem_statement: str) -> Dict[str, Any]:
        """
        Execute geometry solving.

        Args:
            problem_statement: Problem statement

        Returns:
            Dictionary with answer and metadata
        """
        try:
            answer = self.solver.solve(problem_statement)
            
            result = {
                "answer": answer,
                "success": True,
                "method": "geometry",
                "confidence": 1.0 if self.solver.can_solve(problem_statement) else 0.5,
            }
            
            # Include stability metrics if measured
            if self.measure_stability:
                token = self.solver.get_last_proof_token()
                if token and token.stability:
                    # Convert stability to dict format for orchestration tracker
                    stability_dict = {
                        "status": token.stability.status,
                        "confidence": token.stability.confidence,
                    }
                    if hasattr(token.stability, 'lyapunov_exponent') and token.stability.lyapunov_exponent is not None:
                        stability_dict["lyapunov_exponent"] = token.stability.lyapunov_exponent
                    if hasattr(token.stability, 'metadata'):
                        stability_dict.update(token.stability.metadata)
                    
                    result["stability"] = stability_dict
                    result["stability_status"] = token.stability.status
                    
            return result
        except Exception as e:
            return {
                "answer": 0,
                "success": False,
                "error": str(e),
                "method": "geometry",
            }


class AIMOAlgebraTool:
    """Algebra solver as a ToolOrchestra tool."""

    def __init__(self):
        """Initialize algebra tool."""
        self.solver = AlgebraSolver()
        self.name = "algebra_solver"
        self.description = "Solves algebraic problems using SymPy for symbolic manipulation and equation solving"

    def __call__(self, problem_statement: str) -> Dict[str, Any]:
        """
        Execute algebra solving.

        Args:
            problem_statement: Problem statement

        Returns:
            Dictionary with answer and metadata
        """
        try:
            answer = self.solver.solve(problem_statement)
            return {
                "answer": answer,
                "success": True,
                "method": "algebra",
                "confidence": 1.0 if self.solver.can_solve(problem_statement) else 0.5,
            }
        except Exception as e:
            return {
                "answer": 0,
                "success": False,
                "error": str(e),
                "method": "algebra",
            }


class AIMONumberTheoryTool:
    """Number theory solver as a ToolOrchestra tool."""

    def __init__(self):
        """Initialize number theory tool."""
        self.solver = NumberTheorySolver()
        self.name = "number_theory_solver"
        self.description = "Solves number theory problems including modular arithmetic, GCD/LCM, and prime operations"

    def __call__(self, problem_statement: str) -> Dict[str, Any]:
        """
        Execute number theory solving.

        Args:
            problem_statement: Problem statement

        Returns:
            Dictionary with answer and metadata
        """
        try:
            answer = self.solver.solve(problem_statement)
            return {
                "answer": answer,
                "success": True,
                "method": "number_theory",
                "confidence": 1.0 if self.solver.can_solve(problem_statement) else 0.5,
            }
        except Exception as e:
            return {
                "answer": 0,
                "success": False,
                "error": str(e),
                "method": "number_theory",
            }


class AIMOSymbolicTool:
    """Symbolic computation tool using SymPy."""

    def __init__(self):
        """Initialize symbolic tool."""
        self.name = "symbolic_computation"
        self.description = "Performs symbolic mathematical computations using SymPy"

    def __call__(self, expression: str) -> Dict[str, Any]:
        """
        Evaluate symbolic expression.

        Args:
            expression: Mathematical expression to evaluate

        Returns:
            Dictionary with result
        """
        try:
            import sympy as sp
            result = sp.sympify(expression)
            if isinstance(result, (int, sp.Integer)):
                value = int(result)
            elif isinstance(result, (float, sp.Float)):
                value = int(round(result))
            else:
                value = int(round(float(result.evalf())))

            return {
                "result": value,
                "success": True,
                "expression": expression,
            }
        except Exception as e:
            return {
                "result": 0,
                "success": False,
                "error": str(e),
            }


class AIMOCombinatoricsTool:
    """Combinatorics solver as a ToolOrchestra tool."""

    def __init__(self):
        """Initialize combinatorics tool."""
        self.solver = CombinatoricsSolver()
        self.name = "combinatorics_solver"
        self.description = "Solves combinatorics problems including permutations, combinations, and probability"

    def __call__(self, problem_statement: str) -> Dict[str, Any]:
        """Execute combinatorics solving."""
        try:
            answer = self.solver.solve(problem_statement)
            return {
                "answer": answer,
                "success": True,
                "method": "combinatorics",
                "confidence": 1.0 if self.solver.can_solve(problem_statement) else 0.5,
            }
        except Exception as e:
            return {
                "answer": 0,
                "success": False,
                "error": str(e),
                "method": "combinatorics",
            }


class AIMOGraphTool:
    """Graph theory solver as a ToolOrchestra tool."""

    def __init__(self):
        """Initialize graph tool."""
        self.solver = GraphSolver()
        self.name = "graph_solver"
        self.description = "Solves graph theory problems including path counting and graph enumeration"

    def __call__(self, problem_statement: str) -> Dict[str, Any]:
        """Execute graph solving."""
        try:
            answer = self.solver.solve(problem_statement)
            return {
                "answer": answer,
                "success": True,
                "method": "graph",
                "confidence": 1.0 if self.solver.can_solve(problem_statement) else 0.5,
            }
        except Exception as e:
            return {
                "answer": 0,
                "success": False,
                "error": str(e),
                "method": "graph",
            }


class AIMOAnalysisTool:
    """Analysis/calculus solver as a ToolOrchestra tool."""

    def __init__(self):
        """Initialize analysis tool."""
        self.solver = AnalysisSolver()
        self.name = "analysis_solver"
        self.description = "Solves calculus problems including limits, derivatives, and integrals"

    def __call__(self, problem_statement: str) -> Dict[str, Any]:
        """Execute analysis solving."""
        try:
            answer = self.solver.solve(problem_statement)
            return {
                "answer": answer,
                "success": True,
                "method": "analysis",
                "confidence": 1.0 if self.solver.can_solve(problem_statement) else 0.5,
            }
        except Exception as e:
            return {
                "answer": 0,
                "success": False,
                "error": str(e),
                "method": "analysis",
            }


def get_aimo_tools(measure_stability: bool = False) -> Dict[str, Any]:
    """
    Get all AIMO tools for ToolOrchestra.
    
    Args:
        measure_stability: Whether to enable stability measurement in solvers

    Returns:
        Dictionary mapping tool names to tool instances
    """
    return {
        "geometry_solver": AIMOGeometryTool(measure_stability=measure_stability),
        "algebra_solver": AIMOAlgebraTool(),
        "number_theory_solver": AIMONumberTheoryTool(),
        "combinatorics_solver": AIMOCombinatoricsTool(),
        "graph_solver": AIMOGraphTool(),
        "analysis_solver": AIMOAnalysisTool(),
        "symbolic_computation": AIMOSymbolicTool(),
    }
