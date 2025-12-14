"""Symbolic solver using code generation and execution."""

import re
from typing import Optional

from .sandbox import RestrictedCodeExecutor


class SymbolicSolver:
    """
    Solver that generates Python code to solve problems and executes it.
    """

    def __init__(
        self,
        llm_solver: Optional[object] = None,
        timeout: int = 30,
        use_sympy: bool = True,
        use_sandbox: bool = True,
    ):
        """
        Initialize symbolic solver.

        Args:
            llm_solver: LLM solver to use for code generation (optional)
            timeout: Timeout for code execution in seconds
            use_sympy: Whether to use SymPy for symbolic math
            use_sandbox: Whether to use sandboxed execution (recommended)
        """
        self.llm_solver = llm_solver
        self.timeout = timeout
        self.use_sympy = use_sympy
        self.use_sandbox = use_sandbox
        self.executor = RestrictedCodeExecutor(timeout=timeout) if use_sandbox else None

    def solve(self, problem_statement: str) -> int:
        """
        Solve problem by generating and executing Python code.

        Args:
            problem_statement: LaTeX problem statement

        Returns:
            Answer as integer in [0, 99999]
        """
        code = self._generate_code(problem_statement)
        answer = self._execute_code(code)
        return answer

    def _generate_code(self, problem_statement: str) -> str:
        """
        Generate Python code to solve the problem.

        Args:
            problem_statement: Problem statement

        Returns:
            Python code string
        """
        if self.llm_solver:
            prompt = self._create_code_prompt(problem_statement)
            code = self.llm_solver._call_api(prompt) if hasattr(self.llm_solver, "_call_api") else ""
            # Extract code block if present
            code = self._extract_code_block(code)
            return code
        else:
            # Basic template - can be extended
            return self._create_basic_code(problem_statement)

    def _create_code_prompt(self, problem_statement: str) -> str:
        """Create prompt for code generation."""
        return f"""Solve the following mathematical problem by writing Python code.

Problem:
{problem_statement}

Write Python code that:
1. Parses the problem
2. Performs the necessary calculations
3. Prints the final answer as an integer

Your code should output a single integer between 0 and 99999.

Code:"""

    def _extract_code_block(self, text: str) -> str:
        """Extract Python code from markdown code block."""
        # Look for ```python ... ``` or ``` ... ```
        pattern = r"```(?:python)?\n?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return text.strip()

    def _create_basic_code(self, problem_statement: str) -> str:
        """Create basic code template."""
        return f"""
# Problem: {problem_statement[:100]}...
# TODO: Implement solution
answer = 0
print(answer)
"""

    def _execute_code(self, code: str) -> int:
        """
        Execute Python code safely and extract answer.

        Args:
            code: Python code to execute

        Returns:
            Extracted integer answer
        """
        if self.use_sandbox and self.executor:
            return self._execute_with_sandbox(code)
        else:
            return self._execute_with_subprocess(code)

    def _execute_with_sandbox(self, code: str) -> int:
        """Execute code using sandbox."""
        # Add necessary imports to code
        full_code = "import math\n"
        full_code += "from fractions import Fraction\n"
        full_code += "from decimal import Decimal\n"
        full_code += "from itertools import combinations, permutations, product\n"
        full_code += "from collections import Counter, defaultdict\n"
        full_code += code
        full_code += "\nif 'answer' not in locals():\n    answer = 0\nprint(answer)"

        result = self.executor.execute(full_code)

        if not result["success"]:
            print(f"Code execution error: {result.get('error', 'Unknown error')}")
            return 0

        # Extract answer from result or output
        if result.get("result") is not None:
            try:
                answer = int(result["result"])
                if 0 <= answer <= 99999:
                    return answer
            except (ValueError, TypeError):
                pass

        # Fallback to output parsing
        output = result.get("output", "").strip()
        return self._extract_answer_from_output(output)

    def _execute_with_subprocess(self, code: str) -> int:
        """Execute code using subprocess (legacy, less secure)."""
        import subprocess
        import tempfile

        # Add necessary imports
        full_code = "import math\n"
        if self.use_sympy:
            full_code += "from sympy import *\n"
        full_code += code
        full_code += "\nif 'answer' in locals():\n    print(answer)"

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                temp_file = f.name

            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                print(f"Code execution error: {result.stderr}")
                return 0

            # Extract answer from output
            output = result.stdout.strip()
            answer = self._extract_answer_from_output(output)

            return answer

        except subprocess.TimeoutExpired:
            print("Code execution timed out")
            return 0
        except Exception as e:
            print(f"Error executing code: {e}")
            return 0

    def _extract_answer_from_output(self, output: str) -> int:
        """Extract integer answer from code output."""
        # Look for integer in output
        import re

        # Try to find last integer in output
        numbers = re.findall(r"\b(\d{1,5})\b", output)
        if numbers:
            try:
                answer = int(numbers[-1])
                if 0 <= answer <= 99999:
                    return answer
            except ValueError:
                pass

        return 0

