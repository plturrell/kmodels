"""Configuration for ToolOrchestra integration."""

from pathlib import Path
from typing import Dict, List

# ToolOrchestra paths
VENDOR_PATH = Path(__file__).parent.parent.parent / "vendor" / "ToolOrchestra"
TOOLORCHESTRA_AVAILABLE = VENDOR_PATH.exists()

# AIMO tool configuration for ToolOrchestra
AIMO_TOOL_CONFIG = {
    "tools": [
        {
            "name": "geometry_solver",
            "description": "Solves geometric problems using formal reasoning with scene graphs and theorems",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_statement": {
                        "type": "string",
                        "description": "The geometric problem statement in LaTeX format",
                    }
                },
                "required": ["problem_statement"],
            },
        },
        {
            "name": "algebra_solver",
            "description": "Solves algebraic problems using SymPy for symbolic manipulation",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_statement": {
                        "type": "string",
                        "description": "The algebraic problem statement",
                    }
                },
                "required": ["problem_statement"],
            },
        },
        {
            "name": "number_theory_solver",
            "description": "Solves number theory problems including modular arithmetic, GCD/LCM",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem_statement": {
                        "type": "string",
                        "description": "The number theory problem statement",
                    }
                },
                "required": ["problem_statement"],
            },
        },
        {
            "name": "symbolic_computation",
            "description": "Performs symbolic mathematical computations using SymPy",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    ]
}

# ToolOrchestra evaluation configuration
TOOLORCHESTRA_EVAL_CONFIG = {
    "model_name": "nvidia/Nemotron-Orchestrator-8B",  # Default orchestrator
    "max_turns": 10,  # Maximum tool-calling turns
    "temperature": 0.0,  # Deterministic for math problems
    "max_tokens": 2048,
}

# Training configuration (if training custom orchestrator)
TOOLORCHESTRA_TRAINING_CONFIG = {
    "output_dir": "outputs/orchestrator_training",
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-5,
    "warmup_steps": 100,
}


def get_tool_config() -> Dict:
    """Get tool configuration for ToolOrchestra."""
    return AIMO_TOOL_CONFIG


def get_eval_config() -> Dict:
    """Get evaluation configuration."""
    return TOOLORCHESTRA_EVAL_CONFIG.copy()


def get_training_config() -> Dict:
    """Get training configuration."""
    return TOOLORCHESTRA_TRAINING_CONFIG.copy()

