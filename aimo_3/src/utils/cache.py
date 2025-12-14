"""Caching utilities for performance optimization."""

import hashlib
import json
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False


class ProblemCache:
    """
    Cache for problem solutions to avoid recomputation.
    
    Uses file-based caching with hash-based keys.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache files (default: outputs/cache)
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "outputs" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for fast access
        self.memory_cache: Dict[str, int] = {}

    def _get_cache_key(self, problem_statement: str) -> str:
        """
        Generate cache key from problem statement.

        Args:
            problem_statement: Problem statement

        Returns:
            Cache key (hash)
        """
        # Normalize problem statement
        normalized = problem_statement.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, problem_statement: str) -> Optional[int]:
        """
        Get cached answer for problem.

        Args:
            problem_statement: Problem statement

        Returns:
            Cached answer or None if not found
        """
        cache_key = self._get_cache_key(problem_statement)

        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    answer = data.get("answer")
                    if answer is not None:
                        # Update memory cache
                        self.memory_cache[cache_key] = answer
                        return answer
            except Exception:
                pass

        return None

    def set(self, problem_statement: str, answer: int) -> None:
        """
        Cache answer for problem.

        Args:
            problem_statement: Problem statement
            answer: Answer to cache
        """
        cache_key = self._get_cache_key(problem_statement)

        # Update memory cache
        self.memory_cache[cache_key] = answer

        # Update file cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({"answer": answer, "problem": problem_statement[:100]}, f)
        except Exception:
            pass

    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        # Optionally clear file cache
        # for cache_file in self.cache_dir.glob("*.json"):
        #     cache_file.unlink()


def cached_solve(cache: Optional[ProblemCache] = None):
    """
    Decorator to cache solver results.

    Args:
        cache: ProblemCache instance (creates new if None)

    Returns:
        Decorated function
    """
    if cache is None:
        cache = ProblemCache()

    def decorator(func: Callable[[str], int]) -> Callable[[str], int]:
        @wraps(func)
        def wrapper(problem_statement: str) -> int:
            # Check cache
            cached_answer = cache.get(problem_statement)
            if cached_answer is not None:
                return cached_answer

            # Solve and cache
            answer = func(problem_statement)
            cache.set(problem_statement, answer)
            return answer

        return wrapper

    return decorator

