"""Advanced prompt engineering with token-oriented object notation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenObject:
    """
    Token-oriented object for structured prompt construction.
    
    Represents a semantic unit in the prompt that can be composed
    with other tokens to build complex prompts.
    """
    token_type: str  # e.g., "instruction", "example", "constraint"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Importance weight for this token

    def to_string(self, format_style: str = "markdown") -> str:
        """
        Convert token to string representation.

        Args:
            format_style: Format style ("markdown", "plain", "json")

        Returns:
            Formatted string
        """
        if format_style == "markdown":
            return self._to_markdown()
        elif format_style == "json":
            return self._to_json()
        else:
            return self.content

    def _to_markdown(self) -> str:
        """Convert to markdown format."""
        if self.token_type == "instruction":
            return f"## {self.content}\n"
        elif self.token_type == "example":
            return f"### Example\n{self.content}\n"
        elif self.token_type == "constraint":
            return f"**Constraint**: {self.content}\n"
        elif self.token_type == "reasoning":
            return f"**Reasoning**: {self.content}\n"
        else:
            return f"{self.content}\n"

    def _to_json(self) -> str:
        """Convert to JSON format."""
        import json
        return json.dumps({
            "type": self.token_type,
            "content": self.content,
            "metadata": self.metadata,
        })


class PromptBuilder:
    """
    Builder for constructing prompts using token-oriented object notation.
    """

    def __init__(self):
        """Initialize prompt builder."""
        self.tokens: List[TokenObject] = []

    def add_instruction(self, instruction: str, weight: float = 1.0) -> "PromptBuilder":
        """Add instruction token."""
        self.tokens.append(TokenObject("instruction", instruction, weight=weight))
        return self

    def add_example(self, problem: str, solution: str, answer: int, weight: float = 1.0) -> "PromptBuilder":
        """Add example token."""
        example_content = f"Problem: {problem}\nSolution: {solution}\nAnswer: {answer}"
        self.tokens.append(TokenObject("example", example_content, {"answer": answer}, weight=weight))
        return self

    def add_constraint(self, constraint: str, weight: float = 1.0) -> "PromptBuilder":
        """Add constraint token."""
        self.tokens.append(TokenObject("constraint", constraint, weight=weight))
        return self

    def add_reasoning_step(self, step: str, weight: float = 1.0) -> "PromptBuilder":
        """Add reasoning step token."""
        self.tokens.append(TokenObject("reasoning", step, weight=weight))
        return self

    def add_token(self, token: TokenObject) -> "PromptBuilder":
        """Add custom token."""
        self.tokens.append(token)
        return self

    def build(self, format_style: str = "markdown", filter_by_weight: Optional[float] = None) -> str:
        """
        Build final prompt from tokens.

        Args:
            format_style: Output format style
            filter_by_weight: Only include tokens with weight >= this value

        Returns:
            Complete prompt string
        """
        tokens_to_use = self.tokens
        if filter_by_weight is not None:
            tokens_to_use = [t for t in self.tokens if t.weight >= filter_by_weight]

        # Sort by weight (descending) and type
        sorted_tokens = sorted(
            tokens_to_use,
            key=lambda t: (-t.weight, t.token_type),
        )

        parts = [token.to_string(format_style) for token in sorted_tokens]
        return "\n".join(parts)

    def clear(self) -> "PromptBuilder":
        """Clear all tokens."""
        self.tokens = []
        return self


class AIMPOPromptEngineer:
    """
    Advanced prompt engineer for AIMO problems using token-oriented object notation.
    """

    def __init__(self, use_few_shot: bool = True, num_examples: int = 3):
        """
        Initialize prompt engineer.

        Args:
            use_few_shot: Whether to include few-shot examples
            num_examples: Number of examples to include
        """
        self.use_few_shot = use_few_shot
        self.num_examples = num_examples
        self.examples = self._load_default_examples()

    def create_prompt(self, problem_statement: str, strategy: str = "standard") -> str:
        """
        Create optimized prompt for problem.

        Args:
            problem_statement: Problem statement in LaTeX
            strategy: Prompting strategy ("standard", "chain_of_thought", "self_consistency")

        Returns:
            Complete prompt
        """
        builder = PromptBuilder()

        # Add main instruction
        builder.add_instruction(
            "Solve the following mathematical problem. Your answer must be an integer between 0 and 99999.",
            weight=2.0,
        )

        # Add constraints
        builder.add_constraint("The answer must be an integer in the range [0, 99999].", weight=2.0)
        builder.add_constraint("Show your reasoning step by step.", weight=1.5)

        # Add few-shot examples
        if self.use_few_shot:
            for example in self.examples[:self.num_examples]:
                builder.add_example(
                    example["problem"],
                    example["solution"],
                    example["answer"],
                    weight=1.5,
                )

        # Add problem
        builder.add_token(TokenObject("problem", problem_statement, weight=3.0))

        # Add output format instruction
        if strategy == "chain_of_thought":
            builder.add_instruction(
                "Think step by step, then provide your final answer in the format: [ANSWER: <number>]",
                weight=2.0,
            )
        elif strategy == "self_consistency":
            builder.add_instruction(
                "Generate multiple reasoning paths and choose the most consistent answer.",
                weight=1.5,
            )

        return builder.build(format_style="markdown")

    def create_code_generation_prompt(self, problem_statement: str, reasoning: Optional[str] = None) -> str:
        """
        Create prompt for code generation.

        Args:
            problem_statement: Problem statement
            reasoning: Optional reasoning from LLM

        Returns:
            Code generation prompt
        """
        builder = PromptBuilder()

        builder.add_instruction(
            "Write Python code to solve this mathematical problem.",
            weight=2.0,
        )

        if reasoning:
            builder.add_reasoning_step(reasoning, weight=1.5)

        builder.add_token(TokenObject("problem", problem_statement, weight=3.0))

        builder.add_constraint(
            "Your code should: 1) Parse the problem, 2) Perform calculations, 3) Print the final answer as an integer.",
            weight=2.0,
        )

        builder.add_constraint("Use only safe operations (math, basic data structures).", weight=1.5)

        return builder.build(format_style="markdown")

    def _load_default_examples(self) -> List[Dict[str, Any]]:
        """Load default few-shot examples."""
        return [
            {
                "problem": "What is $2 + 2$?",
                "solution": "Simple addition: 2 + 2 = 4",
                "answer": 4,
            },
            {
                "problem": "What is the remainder when $17$ is divided by $5$?",
                "solution": "17 ÷ 5 = 3 remainder 2, so the answer is 2",
                "answer": 2,
            },
            {
                "problem": "Calculate $3!$ (3 factorial).",
                "solution": "3! = 3 × 2 × 1 = 6",
                "answer": 6,
            },
        ]


def create_aimo_prompt(problem: str, strategy: str = "standard") -> str:
    """
    Convenience function to create AIMO prompt.

    Args:
        problem: Problem statement
        strategy: Prompting strategy

    Returns:
        Formatted prompt
    """
    engineer = AIMPOPromptEngineer()
    return engineer.create_prompt(problem, strategy=strategy)


# Backwards-compatible alias expected by src.modeling.__init__
PromptEngineer = AIMPOPromptEngineer

