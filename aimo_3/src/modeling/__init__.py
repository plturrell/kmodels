"""Modeling components for AIMO 3."""

from .answer_extractor import AnswerExtractor
from .ensemble import EnsembleSolver
from .hybrid_solver import HybridSolver
from .llm_base import LLMSolver
from .prompt_engineer import PromptEngineer
from .sandbox import SandboxExecutor
from .symbolic_solver import SymbolicSolver

# G-JEPA model (optional, requires PyTorch)
try:
    from .gjepa_model import GJEPA, create_gjepa_model
    __all__ = [
        "AnswerExtractor",
        "EnsembleSolver",
        "HybridSolver",
        "LLMSolver",
        "PromptEngineer",
        "SandboxExecutor",
        "SymbolicSolver",
        "GJEPA",
        "create_gjepa_model",
    ]
except ImportError:
    __all__ = [
        "AnswerExtractor",
        "EnsembleSolver",
        "HybridSolver",
        "LLMSolver",
        "PromptEngineer",
        "SandboxExecutor",
        "SymbolicSolver",
    ]
