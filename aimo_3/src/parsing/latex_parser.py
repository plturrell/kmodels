"""Advanced LaTeX parser using pylatexenc for proper AST construction."""

import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

try:
    from pylatexenc.latexwalker import LatexWalker, LatexCharsNode, LatexMacroNode, LatexGroupNode
    from pylatexenc.macrospec import MacroSpec, LatexContextDb
    PYLATEXENC_AVAILABLE = True
except ImportError:
    PYLATEXENC_AVAILABLE = False

from ..geometry.primitives import Circle, Line, Point
from ..geometry.relations import Relation, RelationType
from ..geometry.scene_graph import GeometricSceneGraph


class LaTeXASTNode:
    """Node in LaTeX Abstract Syntax Tree."""

    def __init__(self, node_type: str, content: Any, children: Optional[List["LaTeXASTNode"]] = None):
        self.node_type = node_type  # 'text', 'math', 'macro', 'group', 'fraction', 'sum', etc.
        self.content = content
        self.children = children or []

    def __repr__(self) -> str:
        return f"LaTeXASTNode({self.node_type}, {self.content})"


class AdvancedLaTeXParser:
    """
    Advanced LaTeX parser using pylatexenc for proper AST construction.
    
    Handles complex LaTeX expressions, nested structures, and special notation.
    """

    def __init__(self):
        """Initialize parser."""
        if PYLATEXENC_AVAILABLE:
            self.context_db = LatexContextDb()
            self.walker = LatexWalker("", latex_context=self.context_db)
        else:
            self.walker = None

    def parse(self, latex_string: str) -> LaTeXASTNode:
        """
        Parse LaTeX string into AST.

        Args:
            latex_string: LaTeX problem statement

        Returns:
            Root AST node
        """
        if PYLATEXENC_AVAILABLE and self.walker:
            return self._parse_with_pylatexenc(latex_string)
        else:
            return self._parse_fallback(latex_string)

    def _parse_with_pylatexenc(self, latex_string: str) -> LaTeXASTNode:
        """Parse using pylatexenc."""
        walker = LatexWalker(latex_string, latex_context=self.context_db)
        nodes, pos, len_ = walker.get_latex_nodes()

        root = LaTeXASTNode("root", latex_string)
        root.children = self._process_nodes(nodes)

        return root

    def _process_nodes(self, nodes: List) -> List[LaTeXASTNode]:
        """Process pylatexenc nodes into AST nodes."""
        ast_nodes = []

        for node in nodes:
            if isinstance(node, LatexCharsNode):
                # Text content
                ast_nodes.append(LaTeXASTNode("text", node.chars))
            elif isinstance(node, LatexMacroNode):
                # Macro (command)
                macro_name = node.macroname
                if macro_name == "frac":
                    # Fraction: \frac{a}{b}
                    children = self._process_nodes(node.nodeargs[0] if node.nodeargs else [])
                    ast_nodes.append(LaTeXASTNode("fraction", macro_name, children))
                elif macro_name in ["sum", "prod", "int"]:
                    # Summation, product, integral
                    children = self._process_nodes(node.nodeargs[0] if node.nodeargs else [])
                    ast_nodes.append(LaTeXASTNode("operator", macro_name, children))
                else:
                    ast_nodes.append(LaTeXASTNode("macro", macro_name))
            elif isinstance(node, LatexGroupNode):
                # Group: {content}
                children = self._process_nodes(node.nodelist)
                ast_nodes.append(LaTeXASTNode("group", "", children))
            else:
                # Other node types
                ast_nodes.append(LaTeXASTNode("unknown", str(node)))

        return ast_nodes

    def _parse_fallback(self, latex_string: str) -> LaTeXASTNode:
        """Fallback parser using regex (when pylatexenc not available)."""
        root = LaTeXASTNode("root", latex_string)

        # Extract math mode
        math_pattern = r'\$([^$]+)\$|\\\[([^\]]+)\\]|\\begin\{equation\}(.*?)\\end\{equation\}'
        math_matches = re.finditer(math_pattern, latex_string, re.DOTALL)

        for match in math_matches:
            math_content = match.group(1) or match.group(2) or match.group(3)
            root.children.append(LaTeXASTNode("math", math_content))

        # Extract text
        text_parts = re.split(math_pattern, latex_string)
        for text_part in text_parts:
            if text_part and not text_part.strip().startswith("$"):
                root.children.append(LaTeXASTNode("text", text_part))

        return root

    def extract_mathematical_expressions(self, ast: LaTeXASTNode) -> List[str]:
        """
        Extract all mathematical expressions from AST.

        Args:
            ast: Root AST node

        Returns:
            List of mathematical expressions
        """
        expressions = []

        def traverse(node: LaTeXASTNode):
            if node.node_type == "math":
                expressions.append(node.content)
            for child in node.children:
                traverse(child)

        traverse(ast)
        return expressions

    def extract_numerical_values(self, latex_string: str) -> List[float]:
        """
        Extract numerical values from LaTeX string.

        Args:
            latex_string: LaTeX string

        Returns:
            List of numerical values found
        """
        # Find numbers (integers and decimals)
        number_pattern = r'\b(\d+\.?\d*)\b'
        matches = re.findall(number_pattern, latex_string)
        return [float(m) for m in matches]

    def extract_variables(self, latex_string: str) -> List[str]:
        """
        Extract variable names from LaTeX string.

        Args:
            latex_string: LaTeX string

        Returns:
            List of variable names
        """
        # Find single letters (variables)
        var_pattern = r'\b([a-zA-Z])\b'
        matches = re.findall(var_pattern, latex_string)
        # Filter out common words
        common_words = {"the", "and", "or", "is", "are", "find", "what", "when", "where"}
        return [v for v in set(matches) if v.lower() not in common_words]

    def parse_problem_structure(self, latex_string: str) -> "ProblemStructure":
        """
        Parse problem structure: given, find, constraints.

        Args:
            latex_string: Problem statement

        Returns:
            Dictionary with 'given', 'find', 'constraints'
        """
        structure: ProblemStructure = {
            "given": [],
            "find": [],
            "constraints": [],
        }

        # Extract "given" information
        given_patterns = [
            r'given\s+that\s+(.+?)(?:\.|,|$)',
            r'if\s+(.+?)(?:\.|,|$)',
            r'suppose\s+(.+?)(?:\.|,|$)',
        ]
        for pattern in given_patterns:
            matches = re.findall(pattern, latex_string, re.IGNORECASE)
            structure["given"].extend(matches)

        # Extract "find" information
        find_patterns = [
            r'find\s+(.+?)(?:\.|,|$)',
            r'what\s+is\s+(.+?)(?:\.|,|$)',
            r'compute\s+(.+?)(?:\.|,|$)',
            r'determine\s+(.+?)(?:\.|,|$)',
        ]
        for pattern in find_patterns:
            matches = re.findall(pattern, latex_string, re.IGNORECASE)
            structure["find"].extend(matches)

        # Extract constraints (equations, inequalities)
        constraint_patterns = [
            r'([A-Za-z0-9\s\+\-\*\/=<>≤≥]+)\s*=\s*([A-Za-z0-9\s\+\-\*\/]+)',
            r'([A-Za-z0-9\s\+\-\*\/]+)\s*[<>≤≥]\s*([A-Za-z0-9\s\+\-\*\/]+)',
        ]
        for pattern in constraint_patterns:
            matches = re.findall(pattern, latex_string)
            structure["constraints"].extend(matches)

        return structure


class ProblemStructure(TypedDict):
    given: List[str]
    find: List[str]
    constraints: List[Tuple[str, str]]


def parse_latex(latex_string: str) -> LaTeXASTNode:
    """
    Convenience function to parse LaTeX.

    Args:
        latex_string: LaTeX string

    Returns:
        AST root node
    """
    parser = AdvancedLaTeXParser()
    return parser.parse(latex_string)

