# Formal Geometry Reasoning System

## Overview

The formal geometry reasoning system implements a rigorous mathematical framework for solving geometric problems in the AIMO 3 competition. The system replaces the hybrid LLM/symbolic approach with a three-stage pipeline:

1. **Parsing**: LaTeX problem statement → Geometric Scene Graph
2. **Reasoning**: State transitions via theorem application using MCTS search
3. **Evaluation**: Symbolic constraint solving → Integer answer

## Mathematical Foundation

The system implements:
- **Formal Language L**: Vocabulary of geometric primitives (points P, lines L, circles C, relations R)
- **Interpretation Function I**: S → G, where G = (V, E) is a Geometric Scene Graph
- **State Machine**: State s = (G, Φ) with theorem production rules T
- **Deductive Search**: Search(S_initial, T) using Monte Carlo Tree Search (MCTS)
- **Evaluation Function**: F(G_n, Φ_n) → ℝ (integer answer in [0, 99999])

## Architecture

### Core Components

1. **`primitives.py`**: Geometric primitives (Point, Line, Circle)
2. **`relations.py`**: Geometric relations (incidence, congruence, tangency, etc.)
3. **`scene_graph.py`**: Geometric Scene Graph G = (V, E) implementation
4. **`parser.py`**: LaTeX parser implementing I: S → G
5. **`state.py`**: State machine with (G, Φ) structure
6. **`theorems.py`**: Theorem system with pattern matching and application
7. **`search.py`**: MCTS-based deductive search
8. **`evaluation.py`**: Evaluation function F: (G_n, Φ_n) → ℝ
9. **`solver.py`**: Main GeometrySolver orchestrating the full pipeline

## Usage

### Basic Usage

```python
from aimo_3.src.geometry.solver import GeometrySolver

# Initialize solver
solver = GeometrySolver(
    max_search_iterations=1000,
    max_depth=50,
    use_mcts=True,
)

# Solve a problem
problem = "In right triangle ABC with right angle at C, if AC = 3 and BC = 4, find AB."
answer = solver.solve(problem)  # Returns integer answer
```

### Integration with Inference Pipeline

The GeometrySolver is integrated as the default solver in the inference pipeline:

```python
from aimo_3.src.training.inference import InferencePipeline

# Uses GeometrySolver by default
pipeline = InferencePipeline(use_geometry=True)
answers = pipeline.run_evaluation()
```

## Key Features

### 1. Geometric Scene Graph
- Directed, labeled multigraph representation
- Vertices: geometric primitives (points, lines, circles)
- Edges: geometric relations
- Supports subgraph matching for theorem application

### 2. Theorem System
- Production rules: T: (G_pattern, Φ_condition) → (G_addition, Φ_addition)
- Pattern matching for theorem application
- Extensible theorem library (Pythagorean, Angle Sum, Congruence, etc.)

### 3. MCTS Search
- Monte Carlo Tree Search for finding theorem sequences
- UCB1 selection for exploration/exploitation balance
- Heuristic-guided rollout policies
- Configurable depth and iteration limits

### 4. Symbolic Evaluation
- Constraint extraction from final state
- SymPy-based symbolic solving
- Automatic answer extraction and validation

## Implementation Details

### Parsing Stage
- Extracts points, lines, circles from LaTeX
- Identifies geometric relations
- Constructs initial scene graph G_initial

### Reasoning Stage
- Applies theorems to transform state
- Uses MCTS to search for proof sequences
- Tracks derived propositions in Φ

### Evaluation Stage
- Extracts equations from propositions and graph
- Solves constraint system symbolically
- Computes integer answer in valid range [0, 99999]

## Testing

Comprehensive test suite in `tests/test_geometry.py`:
- Primitive creation and operations
- Scene graph operations
- Parser functionality
- State machine and theorem application
- Full pipeline integration tests

## Dependencies

- `networkx`: Graph operations
- `sympy`: Symbolic mathematics
- `pylatexenc`: LaTeX parsing (already included)

## Problem Generator

The system includes a comprehensive problem generator that can synthesize novel geometric problems for training and testing.

### Usage

```python
from aimo_3.src.geometry.generator import ProblemGenerator, generate_problems

# Initialize generator
generator = ProblemGenerator(seed=42)

# Generate triangle problems
problem, answer = generator.generate_triangle_problem("right", "medium")

# Generate circle problems
problem, answer = generator.generate_circle_problem("inscribed", "hard")

# Generate coordinate problems
problem, answer = generator.generate_coordinate_problem("distance", "easy")

# Generate batch of problems
problems = generator.generate_batch(
    problem_types=[
        {"family": "triangle", "type": "right", "difficulty": "easy"},
        {"family": "circle", "type": "inscribed", "difficulty": "medium"},
    ],
    num_problems=100,
)
```

### Supported Problem Families

1. **Triangles**: right, equilateral, isosceles, scalene
2. **Circles**: inscribed, circumscribed, tangent, chord
3. **Coordinate**: distance, midpoint, slope, area

### Training Data Generation

Use the provided script to generate training datasets:

```bash
python scripts/generate_training_data.py \
    --output data/generated_problems.json \
    --num_problems 1000 \
    --distribution balanced \
    --seed 42
```

## Future Enhancements

1. **Extended Theorem Library**: Add more geometric theorems
2. **Improved Parsing**: Better LaTeX understanding for complex problems
3. **Enhanced Search**: Domain-specific heuristics for theorem selection
4. **Coordinate System**: Automatic coordinate assignment for easier solving
5. **Visualization**: Graph visualization for debugging
6. **Advanced Problem Generation**: More complex problem types and multi-step problems

## Notes

- The system is designed to be extensible: new theorems can be easily added
- MCTS parameters can be tuned for different problem types
- The parser handles common geometric notation but may need extension for edge cases
- The evaluation engine uses pattern matching for proposition extraction, which can be improved

