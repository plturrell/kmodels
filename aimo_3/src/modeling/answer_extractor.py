"""Advanced answer extraction from LLM responses."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class AnswerExtractor:
    """
    Extracts integer answers from LLM responses using multiple strategies.
    """

    def __init__(self, use_structured_output: bool = True):
        """
        Initialize answer extractor.

        Args:
            use_structured_output: Whether to prefer structured output (JSON)
        """
        self.use_structured_output = use_structured_output

    def extract(self, response: str, problem_statement: Optional[str] = None) -> Tuple[int, float]:
        """
        Extract answer from LLM response.

        Args:
            response: LLM response text
            problem_statement: Original problem statement (for context)

        Returns:
            (answer, confidence) tuple where answer is int [0, 99999] and confidence is [0, 1]
        """
        # Try structured output first
        if self.use_structured_output:
            answer, confidence = self._extract_structured(response)
            if answer is not None:
                return answer, confidence

        # Try JSON extraction
        answer, confidence = self._extract_json(response)
        if answer is not None:
            return answer, confidence

        # Try pattern matching
        answer, confidence = self._extract_patterns(response)
        if answer is not None:
            return answer, confidence

        # Try reasoning chain extraction
        answer, confidence = self._extract_reasoning_chain(response)
        if answer is not None:
            return answer, confidence

        # Fallback: return 0 with low confidence
        return 0, 0.0

    def _extract_structured(self, response: str) -> Tuple[Optional[int], float]:
        """Extract from structured output (JSON)."""
        # Look for JSON blocks
        json_patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"\{[^{}]*\"answer\"[^{}]*\}",
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if "answer" in data:
                        answer = int(data["answer"])
                        confidence = data.get("confidence", 0.8)
                        if 0 <= answer <= 99999:
                            return answer, float(confidence)
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue

        return None, 0.0

    def _extract_json(self, response: str) -> Tuple[Optional[int], float]:
        """Extract JSON from response."""
        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                answer = int(data.get("answer", 0))
                if 0 <= answer <= 99999:
                    return answer, 0.7
            except (json.JSONDecodeError, ValueError):
                pass

        return None, 0.0

    def _extract_patterns(self, response: str) -> Tuple[Optional[int], float]:
        """Extract using pattern matching."""
        # High confidence patterns
        high_confidence_patterns = [
            r"(?:answer|result|solution)[:\s]*(\d{1,5})\b",
            r"(?:the answer is|final answer is|answer =)[:\s]*(\d{1,5})\b",
            r"\[answer[:\s]*(\d{1,5})\s*\]",
            r"<answer>(\d{1,5})</answer>",
        ]

        for pattern in high_confidence_patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                try:
                    answer = int(matches[-1])  # Take last match
                    if 0 <= answer <= 99999:
                        return answer, 0.8
                except ValueError:
                    continue

        # Medium confidence patterns
        medium_confidence_patterns = [
            r"\b(\d{1,5})\b",  # Any 1-5 digit number
        ]

        for pattern in medium_confidence_patterns:
            matches = re.findall(pattern, response)
            if matches:
                # Filter out common non-answer numbers
                filtered = [m for m in matches if not self._is_likely_non_answer(m, response)]
                if filtered:
                    try:
                        answer = int(filtered[-1])
                        if 0 <= answer <= 99999:
                            return answer, 0.5
                    except ValueError:
                        continue

        return None, 0.0

    def _extract_reasoning_chain(self, response: str) -> Tuple[Optional[int], float]:
        """Extract answer from reasoning chain (e.g., "Step 1: ... Step 2: ... Answer: 42")."""
        # Look for reasoning steps ending with answer
        reasoning_patterns = [
            r"(?:step\s+\d+|step|therefore|thus|hence|so)[:\s]+.*?(?:answer|result)[:\s]*(\d{1,5})\b",
            r"(?:conclusion|final|result)[:\s]+.*?(\d{1,5})\b",
        ]

        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response.lower(), re.DOTALL)
            if matches:
                try:
                    answer = int(matches[-1])
                    if 0 <= answer <= 99999:
                        return answer, 0.6
                except ValueError:
                    continue

        return None, 0.0

    def _is_likely_non_answer(self, number: str, context: str) -> bool:
        """Check if a number is likely not the answer (e.g., year, step number)."""
        num = int(number)
        
        # Years (1900-2100)
        if 1900 <= num <= 2100:
            return True
        
        # Step numbers (1-20)
        if 1 <= num <= 20 and f"step {num}" in context.lower():
            return True
        
        # Common mathematical constants that might appear
        common_constants = {3, 14, 141, 1415, 271, 2718, 1618, 618}
        if num in common_constants:
            return False  # Could be answer
        
        return False


def extract_answer(response: str, use_structured: bool = True) -> int:
    """
    Convenience function to extract answer from response.

    Args:
        response: LLM response text
        use_structured: Whether to prefer structured output

    Returns:
        Extracted integer answer [0, 99999]
    """
    extractor = AnswerExtractor(use_structured_output=use_structured)
    answer, _ = extractor.extract(response)
    return answer

