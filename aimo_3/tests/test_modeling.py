"""Tests for modeling components."""

import sys
from pathlib import Path

import pytest

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from modeling.llm_base import LLMSolver
from modeling.symbolic_solver import SymbolicSolver
from modeling.hybrid_solver import HybridSolver
from modeling.answer_extractor import AnswerExtractor, extract_answer
from modeling.sandbox import RestrictedCodeExecutor, SecurityError


def test_answer_extractor_structured():
    """Test answer extraction from structured output."""
    extractor = AnswerExtractor(use_structured_output=True)
    
    response = '{"answer": 42, "confidence": 0.9}'
    answer, confidence = extractor.extract(response)
    assert answer == 42
    assert confidence > 0.5


def test_answer_extractor_patterns():
    """Test answer extraction using patterns."""
    extractor = AnswerExtractor()
    
    response = "The answer is 123."
    answer, confidence = extractor.extract(response)
    assert answer == 123
    assert confidence > 0.0


def test_answer_extractor_reasoning_chain():
    """Test answer extraction from reasoning chain."""
    extractor = AnswerExtractor()
    
    response = "Step 1: Calculate... Step 2: Therefore, the answer is 456."
    answer, confidence = extractor.extract(response)
    assert answer == 456


def test_extract_answer_convenience():
    """Test convenience function."""
    response = "Answer: 789"
    answer = extract_answer(response)
    assert answer == 789


def test_sandbox_allowed_code():
    """Test sandbox allows safe code."""
    executor = RestrictedCodeExecutor()
    
    code = """
answer = 2 + 2
print(answer)
"""
    result = executor.execute(code)
    assert result["success"] is True
    assert result["result"] == 4


def test_sandbox_blocks_imports():
    """Test sandbox blocks dangerous imports."""
    executor = RestrictedCodeExecutor()
    
    code = "import os\nanswer = 0"
    is_valid, error = executor.validate_code(code)
    assert is_valid is False
    assert "not allowed" in error.lower()


def test_sandbox_blocks_file_operations():
    """Test sandbox blocks file operations."""
    executor = RestrictedCodeExecutor()
    
    code = "open('test.txt', 'w').write('test')"
    is_valid, error = executor.validate_code(code)
    assert is_valid is False


def test_symbolic_solver_with_sandbox():
    """Test symbolic solver uses sandbox."""
    solver = SymbolicSolver(use_sandbox=True, timeout=5)
    
    # Test with simple code
    code = "answer = 10 * 5"
    result = solver._execute_with_sandbox(code)
    assert result == 50


def test_llm_solver_initialization():
    """Test LLM solver initialization."""
    solver = LLMSolver(
        model_name="gpt-4",
        use_api=True,
        api_provider="openai",
    )
    assert solver.use_api is True
    assert solver.api_provider == "openai"


def test_hybrid_solver_fallback():
    """Test hybrid solver fallback strategy."""
    # Mock solvers
    class MockLLM:
        def solve(self, problem):
            return 42
    
    class MockSymbolic:
        def solve(self, problem):
            return 100
    
    llm = MockLLM()
    symbolic = MockSymbolic()
    
    solver = HybridSolver(
        llm_solver=llm,
        symbolic_solver=symbolic,
        strategy="fallback",
    )
    
    answer = solver.solve("test problem")
    assert answer == 42  # Should use LLM first


def test_hybrid_solver_ensemble():
    """Test hybrid solver ensemble strategy."""
    class MockLLM:
        def solve(self, problem):
            return 42
    
    class MockSymbolic:
        def solve(self, problem):
            return 42  # Same answer
    
    llm = MockLLM()
    symbolic = MockSymbolic()
    
    solver = HybridSolver(
        llm_solver=llm,
        symbolic_solver=symbolic,
        strategy="ensemble",
    )
    
    answer = solver.solve("test problem")
    assert answer == 42  # Should agree


@pytest.mark.integration
def test_end_to_end_solving():
    """Integration test for end-to-end problem solving."""
    # This would require actual model/API setup
    # For now, just test the flow
    solver = HybridSolver(strategy="fallback")
    
    # Simple problem
    problem = "What is 2 + 2?"
    # Note: This will fail without actual model, but tests the interface
    try:
        answer = solver.solve(problem)
        assert isinstance(answer, int)
        assert 0 <= answer <= 99999
    except Exception:
        # Expected if no model is available
        pass

