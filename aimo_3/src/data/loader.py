"""Load and parse LaTeX problem statements."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _safe_project_path(path: Path) -> Path:
    """
    Resolve a user-provided path and ensure it stays within the project root.
    """
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = (_PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_relative_to(_PROJECT_ROOT):
        raise ValueError(f"Refusing to access paths outside project root: {candidate}")
    return candidate


def load_problem(problem_path: Union[str, Path]) -> Dict[str, str]:
    """
    Load a single problem from a file.

    Args:
        problem_path: Path to problem file (JSON or text)

    Returns:
        Dictionary with 'problem_id' and 'statement' keys
    """
    problem_path = _safe_project_path(Path(problem_path))

    if problem_path.suffix == ".json":
        with open(problem_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    else:
        # Assume text file with problem statement
        with open(problem_path, "r", encoding="utf-8") as f:
            statement = f.read()
            return {
                "problem_id": problem_path.stem,
                "statement": statement,
            }


def load_problems(
    data_dir: Union[str, Path],
    pattern: str = "*.json",
) -> List[Dict[str, str]]:
    """
    Load multiple problems from a directory.

    Args:
        data_dir: Directory containing problem files
        pattern: File pattern to match (default: "*.json")

    Returns:
        List of problem dictionaries
    """
    data_dir = _safe_project_path(Path(data_dir))
    problems = []

    for problem_file in data_dir.glob(pattern):
        try:
            problem = load_problem(problem_file)
            problems.append(problem)
        except Exception as e:
            print(f"Warning: Failed to load {problem_file}: {e}")

    return problems


def parse_problem_statement(statement: str) -> Dict[str, str]:
    """
    Parse a LaTeX problem statement into components.

    Args:
        statement: Raw LaTeX problem statement

    Returns:
        Dictionary with parsed components (problem text, equations, etc.)
    """
    # Basic parsing - can be extended
    return {
        "raw": statement,
        "text": statement,  # Placeholder for more sophisticated parsing
    }

