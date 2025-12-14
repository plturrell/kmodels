"""Submission file generation utilities."""

import csv
from pathlib import Path
from typing import Dict


def generate_submission(
    answers: Dict[str, int],
    output_path: Path,
) -> Path:
    """
    Generate submission CSV file.

    Args:
        answers: Dictionary mapping problem_id to answer
        output_path: Path to output CSV file

    Returns:
        Path to generated submission file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem_id", "answer"])

        for problem_id, answer in sorted(answers.items()):
            writer.writerow([problem_id, answer])

    print(f"Submission file generated: {output_path}")
    return output_path

