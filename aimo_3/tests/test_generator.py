"""Tests for problem generator."""

import pytest

from ..src.geometry.generator import ProblemGenerator, generate_problems


def test_generator_initialization():
    """Test generator initialization."""
    generator = ProblemGenerator(seed=42)
    assert generator is not None


def test_generate_right_triangle():
    """Test right triangle problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_triangle_problem("right", "easy")

    assert isinstance(problem, str)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999
    assert "triangle" in problem.lower() or "ABC" in problem


def test_generate_equilateral_triangle():
    """Test equilateral triangle problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_triangle_problem("equilateral", "medium")

    assert isinstance(problem, str)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999


def test_generate_isosceles_triangle():
    """Test isosceles triangle problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_triangle_problem("isosceles", "medium")

    assert isinstance(problem, str)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999


def test_generate_scalene_triangle():
    """Test scalene triangle problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_triangle_problem("scalene", "hard")

    assert isinstance(problem, str)
    assert isinstance(answer, int)
    assert 0 <= answer <= 180  # Angle measure


def test_generate_inscribed_circle():
    """Test inscribed circle problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_circle_problem("inscribed", "easy")

    assert isinstance(problem, str)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999


def test_generate_circumscribed_circle():
    """Test circumscribed circle problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_circle_problem("circumscribed", "medium")

    assert isinstance(problem, str)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999


def test_generate_tangent_circle():
    """Test tangent circle problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_circle_problem("tangent", "medium")

    assert isinstance(problem, str)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999


def test_generate_chord():
    """Test chord problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_circle_problem("chord", "hard")

    assert isinstance(problem, str)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999


def test_generate_distance():
    """Test distance problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_coordinate_problem("distance", "easy")

    assert isinstance(problem, str)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999


def test_generate_midpoint():
    """Test midpoint problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_coordinate_problem("midpoint", "medium")

    assert isinstance(problem, str)
    assert isinstance(answer, int)


def test_generate_slope():
    """Test slope problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_coordinate_problem("slope", "medium")

    assert isinstance(problem, str)
    assert isinstance(answer, int)


def test_generate_area():
    """Test area problem generation."""
    generator = ProblemGenerator(seed=42)
    problem, answer = generator.generate_coordinate_problem("area", "hard")

    assert isinstance(problem, str)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999


def test_generate_batch():
    """Test batch problem generation."""
    generator = ProblemGenerator(seed=42)

    problem_types = [
        {"family": "triangle", "type": "right", "difficulty": "easy"},
        {"family": "circle", "type": "inscribed", "difficulty": "medium"},
        {"family": "coordinate", "type": "distance", "difficulty": "hard"},
    ]

    problems = generator.generate_batch(problem_types, num_problems=5)

    assert len(problems) == 5
    for problem, answer, metadata in problems:
        assert isinstance(problem, str)
        assert isinstance(answer, int)
        assert isinstance(metadata, dict)
        assert "family" in metadata
        assert "type" in metadata
        assert "difficulty" in metadata


def test_generate_problems_convenience():
    """Test convenience function."""
    problems = generate_problems(
        family="triangle",
        problem_type="right",
        difficulty="easy",
        num_problems=3,
        seed=42,
    )

    assert len(problems) == 3
    for problem, answer in problems:
        assert isinstance(problem, str)
        assert isinstance(answer, int)


def test_all_difficulties():
    """Test all difficulty levels."""
    generator = ProblemGenerator(seed=42)

    for difficulty in ["easy", "medium", "hard"]:
        problem, answer = generator.generate_triangle_problem("right", difficulty)
        assert isinstance(problem, str)
        assert isinstance(answer, int)


def test_reproducibility():
    """Test that generator is reproducible with same seed."""
    gen1 = ProblemGenerator(seed=42)
    gen2 = ProblemGenerator(seed=42)

    p1, a1 = gen1.generate_triangle_problem("right", "easy")
    p2, a2 = gen2.generate_triangle_problem("right", "easy")

    assert p1 == p2
    assert a1 == a2

