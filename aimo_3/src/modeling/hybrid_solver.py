"""Hybrid solver combining LLM reasoning with symbolic execution."""

from typing import Optional

from .llm_base import LLMSolver
from .symbolic_solver import SymbolicSolver


class HybridSolver:
    """
    Hybrid solver that combines LLM reasoning with symbolic code generation.
    """

    def __init__(
        self,
        llm_solver: Optional[LLMSolver] = None,
        symbolic_solver: Optional[SymbolicSolver] = None,
        strategy: str = "fallback",  # "fallback", "ensemble", "reasoning_first"
    ):
        """
        Initialize hybrid solver.

        Args:
            llm_solver: LLM solver instance
            symbolic_solver: Symbolic solver instance
            strategy: Combination strategy
                - "fallback": Try LLM first, fallback to symbolic
                - "ensemble": Use both and combine answers
                - "reasoning_first": Use LLM to reason, then generate code
        """
        self.llm_solver = llm_solver or LLMSolver()
        self.symbolic_solver = symbolic_solver or SymbolicSolver(llm_solver=self.llm_solver)
        self.strategy = strategy

    def solve(self, problem_statement: str) -> int:
        """
        Solve problem using hybrid approach.

        Args:
            problem_statement: LaTeX problem statement

        Returns:
            Answer as integer in [0, 99999]
        """
        if self.strategy == "fallback":
            return self._solve_fallback(problem_statement)
        elif self.strategy == "ensemble":
            return self._solve_ensemble(problem_statement)
        elif self.strategy == "reasoning_first":
            return self._solve_reasoning_first(problem_statement)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _solve_fallback(self, problem_statement: str) -> int:
        """Try LLM first, fallback to symbolic if needed."""
        try:
            answer = self.llm_solver.solve(problem_statement)
            # Validate answer
            if 0 <= answer <= 99999:
                return answer
        except Exception as e:
            print(f"LLM solver failed: {e}")

        # Fallback to symbolic
        try:
            answer = self.symbolic_solver.solve(problem_statement)
            return answer
        except Exception as e:
            print(f"Symbolic solver failed: {e}")
            return 0

    def _solve_ensemble(self, problem_statement: str) -> int:
        """Use both solvers and combine answers."""
        answers = []

        # Get LLM answer
        try:
            llm_answer = self.llm_solver.solve(problem_statement)
            if 0 <= llm_answer <= 99999:
                answers.append(llm_answer)
        except Exception as e:
            print(f"LLM solver failed: {e}")

        # Get symbolic answer
        try:
            symbolic_answer = self.symbolic_solver.solve(problem_statement)
            if 0 <= symbolic_answer <= 99999:
                answers.append(symbolic_answer)
        except Exception as e:
            print(f"Symbolic solver failed: {e}")

        if not answers:
            return 0

        # Return most common answer, or first if all different
        from collections import Counter
        counter = Counter(answers)
        return counter.most_common(1)[0][0]

    def _solve_reasoning_first(self, problem_statement: str) -> int:
        """Use LLM to reason, then generate code based on reasoning."""
        # Step 1: Get LLM reasoning
        reasoning_prompt = f"""Analyze this mathematical problem and explain the approach to solve it.

Problem:
{problem_statement}

Explain the key steps and mathematical concepts needed."""

        try:
            if hasattr(self.llm_solver, "_call_api"):
                reasoning = self.llm_solver._call_api(reasoning_prompt)
            else:
                reasoning = ""
        except Exception as e:
            print(f"Reasoning generation failed: {e}")
            reasoning = ""

        # Step 2: Generate code with reasoning context
        code_prompt = f"""Based on this reasoning:
{reasoning}

Problem:
{problem_statement}

Write Python code to solve the problem."""

        # Update symbolic solver's prompt
        original_solver = self.symbolic_solver.llm_solver
        if original_solver:
            # Temporarily modify prompt
            code = self.symbolic_solver._generate_code(code_prompt)
        else:
            code = self.symbolic_solver._generate_code(problem_statement)

        # Step 3: Execute code
        answer = self.symbolic_solver._execute_code(code)
        return answer

