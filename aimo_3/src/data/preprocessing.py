"""Text preprocessing and LaTeX normalization."""

import re
from typing import List


def normalize_latex_text(text: str) -> str:
    """
    Normalize LaTeX text for processing.

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
    # Match inline math $...$ and display math \[...\] or $$...$$
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


def clean_problem_text(text: str) -> str:
    """
    Clean problem text for model input.

    Args:
        text: Raw problem text

    Returns:
        Cleaned text
    """
    # Normalize LaTeX
    text = normalize_latex_text(text)
    # Remove excessive formatting
    text = re.sub(r"\\[a-zA-Z]+\{([^}]+)\}", r"\1", text)  # Simplify commands
    return text

