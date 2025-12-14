"""Sandbox for safe code execution."""

import ast
import io
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class RestrictedCodeExecutor:
    """
    Restricted code executor that only allows safe operations.
    
    Blocks dangerous operations like file I/O, network access, imports, etc.
    """

    # Allowed builtins (safe operations only)
    ALLOWED_BUILTINS = {
        "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod", "enumerate",
        "filter", "float", "format", "frozenset", "hex", "int", "isinstance",
        "issubclass", "len", "list", "map", "max", "min", "oct", "ord", "pow",
        "print", "range", "repr", "reversed", "round", "set", "slice", "sorted",
        "str", "sum", "tuple", "type", "zip", "range",
        # Math operations
        "abs", "round", "pow", "divmod",
    }

    # Allowed modules (math, fractions, decimal, etc.)
    ALLOWED_MODULES = {
        "math": [
            "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh",
            "ceil", "comb", "copysign", "cos", "cosh", "degrees", "dist",
            "erf", "erfc", "exp", "expm1", "fabs", "factorial", "floor",
            "fmod", "frexp", "fsum", "gamma", "gcd", "hypot", "isclose",
            "isfinite", "isinf", "isnan", "isqrt", "ldexp", "lgamma", "log",
            "log10", "log1p", "log2", "modf", "nextafter", "perm", "pow",
            "prod", "radians", "remainder", "sin", "sinh", "sqrt", "tan",
            "tanh", "trunc", "ulp"
        ],
        "fractions": ["Fraction"],
        "decimal": ["Decimal"],
        "itertools": ["combinations", "permutations", "product"],
        "collections": ["Counter", "defaultdict"],
    }

    def __init__(self, timeout: int = 30, max_memory: Optional[int] = None):
        """
        Initialize restricted executor.

        Args:
            timeout: Maximum execution time in seconds
            max_memory: Maximum memory usage in bytes (optional)
        """
        self.timeout = timeout
        self.max_memory = max_memory

    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code before execution.

        Args:
            code: Python code to validate

        Returns:
            (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
            validator = CodeValidator()
            validator.visit(tree)
            return True, None
        except SecurityError as e:
            return False, str(e)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute code in restricted environment.

        Args:
            code: Python code to execute

        Returns:
            Dictionary with 'output', 'result', 'error', 'success'
        """
        # Validate code first
        is_valid, error = self.validate_code(code)
        if not is_valid:
            return {
                "success": False,
                "error": error,
                "output": "",
                "result": None,
            }

        # Create restricted environment
        restricted_globals = self._create_restricted_globals()
        restricted_locals: Dict[str, Any] = {}

        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, restricted_globals, restricted_locals)

            output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            # Try to extract result (look for 'answer' variable)
            result = restricted_locals.get("answer", None)

            return {
                "success": True,
                "error": stderr_output if stderr_output else None,
                "output": output,
                "result": result,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": stdout_capture.getvalue(),
                "result": None,
            }

    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted global namespace."""
        import builtins
        builtins_dict: Dict[str, Any] = builtins.__dict__
        restricted_globals: Dict[str, Any] = {
            "__builtins__": {name: builtins_dict[name] for name in self.ALLOWED_BUILTINS if name in builtins_dict},
        }

        # Add allowed modules
        import math
        import fractions
        import decimal
        import itertools
        from collections import Counter, defaultdict

        restricted_globals["math"] = math
        restricted_globals["fractions"] = fractions
        restricted_globals["decimal"] = decimal
        restricted_globals["itertools"] = itertools
        restricted_globals["Counter"] = Counter
        restricted_globals["defaultdict"] = defaultdict

        return restricted_globals


class SecurityError(Exception):
    """Raised when code violates security restrictions."""


class CodeValidator(ast.NodeVisitor):
    """AST visitor that checks for dangerous operations."""

    FORBIDDEN_NAMES = {
        "open", "file", "input", "raw_input", "exec", "eval", "compile",
        "__import__", "reload", "importlib", "subprocess", "os", "sys",
        "shutil", "pickle", "marshal", "ctypes", "socket", "urllib",
        "http", "requests", "multiprocessing", "threading",
    }

    FORBIDDEN_ATTRIBUTES = {
        "open", "file", "exec", "eval", "compile", "__import__",
    }

    def visit_Import(self, node):
        """Block all imports."""
        for alias in node.names:
            if alias.name.split(".")[0] not in ["math", "fractions", "decimal", "itertools", "collections"]:
                raise SecurityError(f"Import of '{alias.name}' is not allowed")

    def visit_ImportFrom(self, node):
        """Block unsafe imports."""
        if node.module and node.module.split(".")[0] not in ["math", "fractions", "decimal", "itertools", "collections"]:
            raise SecurityError(f"Import from '{node.module}' is not allowed")

    def visit_Call(self, node):
        """Check for dangerous function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_NAMES:
                raise SecurityError(f"Call to '{node.func.id}' is not allowed")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.FORBIDDEN_ATTRIBUTES:
                raise SecurityError(f"Call to '{node.func.attr}' is not allowed")

    def visit_Attribute(self, node):
        """Check for dangerous attribute access."""
        if isinstance(node.attr, str) and node.attr.startswith("_"):
            if node.attr not in ["__builtins__", "__name__", "__doc__"]:
                raise SecurityError(f"Access to private attribute '{node.attr}' is not allowed")


# Backwards-compatible alias expected by src.modeling.__init__
SandboxExecutor = RestrictedCodeExecutor

