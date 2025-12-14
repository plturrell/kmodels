"""Geometry reasoning system with formal scene graphs and theorem application."""

from .generator import ProblemGenerator
from .parser import GeometryParser, parse_problem
from .primitives import Circle, Line, Point
from .relations import Relation, RelationType
from .scene_graph import GeometricSceneGraph
from .solver import GeometrySolver
from .state import State
from .theorems import Theorem, TheoremLibrary
from .theorems_extended import get_extended_theorem_library
from .theorems_advanced import get_advanced_theorem_library

# G-JEPA module (optional, requires PyTorch)
try:
    from .gjepa import GJEPA, GraphEncoder, ContextEncoder, Predictor
    __all__ = [
        "ProblemGenerator",
        "GeometryParser",
        "parse_problem",
        "Circle",
        "Line",
        "Point",
        "Relation",
        "RelationType",
        "GeometricSceneGraph",
        "GeometrySolver",
        "State",
        "Theorem",
        "TheoremLibrary",
        "get_extended_theorem_library",
        "get_advanced_theorem_library",
        "GJEPA",
        "GraphEncoder",
        "ContextEncoder",
        "Predictor",
    ]
except ImportError:
    __all__ = [
        "ProblemGenerator",
        "GeometryParser",
        "parse_problem",
        "Circle",
        "Line",
        "Point",
        "Relation",
        "RelationType",
        "GeometricSceneGraph",
        "GeometrySolver",
        "State",
        "Theorem",
        "TheoremLibrary",
        "get_extended_theorem_library",
        "get_advanced_theorem_library",
    ]
