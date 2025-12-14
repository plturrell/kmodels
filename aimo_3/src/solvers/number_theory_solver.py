"""Number theory solver for modular arithmetic and divisibility."""

import re
from typing import List, Optional

from .base import BaseSolver


class NumberTheorySolver(BaseSolver):
    """
    Solver for number theory problems.
    
    Handles:
    - Modular arithmetic
    - Divisibility
    - GCD/LCM
    - Prime operations
    - Digit problems
    """

    def __init__(self):
        """Initialize number theory solver."""
        super().__init__("number_theory")

    def solve(self, problem_statement: str) -> int:
        """
        Solve number theory problem.

        Args:
            problem_statement: Problem statement

        Returns:
            Integer answer
        """
        problem_lower = problem_statement.lower()

        # Modular arithmetic
        if "mod" in problem_lower or "modulo" in problem_lower or "remainder" in problem_lower:
            return self._solve_modular(problem_statement)

        # Divisibility
        if "divisible" in problem_lower or "divides" in problem_lower:
            return self._solve_divisibility(problem_statement)

        # GCD/LCM
        if "gcd" in problem_lower or "greatest common divisor" in problem_lower:
            return self._solve_gcd(problem_statement)
        if "lcm" in problem_lower or "least common multiple" in problem_lower:
            return self._solve_lcm(problem_statement)

        # Prime problems
        if "prime" in problem_lower:
            return self._solve_prime(problem_statement)

        # Digit problems
        if "digit" in problem_lower:
            return self._solve_digit(problem_statement)

        return 0

    def can_solve(self, problem_statement: str) -> bool:
        """Check if problem is number theory."""
        problem_lower = problem_statement.lower()
        nt_keywords = [
            "mod", "modulo", "remainder", "divisible", "divides",
            "gcd", "lcm", "prime", "digit", "integer",
        ]
        return any(keyword in problem_lower for keyword in nt_keywords)

    def _solve_modular(self, problem_statement: str) -> int:
        """Solve modular arithmetic problem."""
        # Pattern: "a mod m" or "a modulo m" or "remainder when a is divided by m"
        mod_patterns = [
            r'(\d+)\s+mod\s+(\d+)',
            r'(\d+)\s+modulo\s+(\d+)',
            r'remainder\s+when\s+(\d+)\s+is\s+divided\s+by\s+(\d+)',
            r'(\d+)\s*%\s*(\d+)',
        ]

        for pattern in mod_patterns:
            match = re.search(pattern, problem_statement, re.IGNORECASE)
            if match:
                a = int(match.group(1))
                m = int(match.group(2))
                return a % m

        # Pattern: "find x such that x ≡ a (mod m)"
        congruence_pattern = r'x\s*≡\s*(\d+)\s*\(mod\s*(\d+)\)'
        match = re.search(congruence_pattern, problem_statement, re.IGNORECASE)
        if match:
            a = int(match.group(1))
            m = int(match.group(2))
            # Return smallest positive solution
            return a % m

        return 0

    def _solve_divisibility(self, problem_statement: str) -> int:
        """Solve divisibility problem."""
        # Extract numbers
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if len(numbers) >= 2:
            # Check if first is divisible by second
            if numbers[0] % numbers[1] == 0:
                return 1
            else:
                return 0

        return 0

    def _solve_gcd(self, problem_statement: str) -> int:
        """Solve GCD problem."""
        import math

        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if len(numbers) >= 2:
            result = math.gcd(numbers[0], numbers[1])
            for num in numbers[2:]:
                result = math.gcd(result, num)
            return result

        return 0

    def _solve_lcm(self, problem_statement: str) -> int:
        """Solve LCM problem."""
        import math

        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if len(numbers) >= 2:
            result = math.lcm(numbers[0], numbers[1])
            for num in numbers[2:]:
                result = math.lcm(result, num)
            return result

        return 0

    def _solve_prime(self, problem_statement: str) -> int:
        """Solve prime-related problem."""
        # Extract numbers
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if not numbers:
            return 0

        # Check if number is prime
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        # Find primes or check primality
        if "prime" in problem_statement.lower():
            for num in numbers:
                if is_prime(num):
                    return num

        return 0

    def _solve_digit(self, problem_statement: str) -> int:
        """Solve digit problem."""
        # Extract numbers
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if not numbers:
            return 0

        # Sum of digits
        if "sum of digits" in problem_statement.lower():
            num = numbers[0]
            return sum(int(d) for d in str(num))

        # Product of digits
        if "product of digits" in problem_statement.lower():
            num = numbers[0]
            digits = [int(d) for d in str(num) if d != '0']
            if digits:
                result = 1
                for d in digits:
                    result *= d
                return result

        return 0

