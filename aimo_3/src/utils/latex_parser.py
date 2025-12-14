"""LaTeX parsing and normalization utilities."""

import re
from typing import Dict, List


def parse_latex(text: str) -> Dict[str, any]:
    """
    Parse LaTeX text into components.

    Args:
        text: LaTeX text

    Returns:
        Dictionary with parsed components
    """
    return {
        "raw": text,
        "text": text,
        "math_expressions": extract_math_expressions(text),
    }


def normalize_latex(text: str) -> str:
    """
    Normalize LaTeX text.

    Args:
        text: Raw LaTeX text

    Returns:
        Normalized text
    """
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Normalize line breaks
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def extract_math_expressions(text: str) -> List[str]:
    """
    Extract mathematical expressions from LaTeX text.

    Args:
        text: LaTeX text

    Returns:
        List of mathematical expressions
    """
    patterns = [
        r"\$([^$]+)\$",  # Inline math
        r"\\\[([^\]]+)\\\]",  # Display math \[...\]
        r"\$\$([^$]+)\$\$",  # Display math $$...$$
    ]

    expressions = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        expressions.extend(matches)

    return expressions


def clean_latex_command(text: str, command: str) -> str:
    """
    Clean a specific LaTeX command.

    Args:
        text: LaTeX text
        command: Command name (e.g., "textbf", "emph")

    Returns:
        Text with command cleaned
    """
    pattern = rf"\\{command}\{{([^}}]+)\}}"
    return re.sub(pattern, r"\1", text)

