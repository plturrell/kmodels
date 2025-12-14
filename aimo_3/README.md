# AIMO 3 Competition Workspace

Local workspace scaffold for the [AI Mathematical Olympiad 3 (AIMO 3)](https://www.kaggle.com/competitions/aimo3) Kaggle competition.

## Overview

This competition challenges participants to create algorithms that can solve olympiad-level math problems written in LaTeX format. Answers must be integers between 0 and 99999.

## Getting Started

### 1. Prerequisites

- Python 3.9+
- Kaggle API credentials at `~/.kaggle/kaggle.json`
- GPU recommended for training (optional)

### 2. Installation

```bash
cd aimo_3
pip install -r requirements.txt
```

### 3. Download Competition Data

```bash
python -m aimo_3.src.data.download --extract
```

This downloads competition data to `data/raw/`.

### 4. Quick Start

#### Using LLM Solver

```python
from aimo_3.src.modeling.llm_base import LLMSolver
from aimo_3.src.evaluation.api import AIMOEvaluator

# Initialize solver
solver = LLMSolver(
    model_name="meta-llama/Llama-2-7b-hf",
    use_api=False,  # Set to True for API-based models
)

# Solve a problem
problem = "What is $2 + 2$?"
answer = solver.solve(problem)
print(f"Answer: {answer}")
```

#### Using Geometry Solver (Primary)

```python
from aimo_3.src.geometry.solver import GeometrySolver

# Initialize geometry solver (formal reasoning system)
solver = GeometrySolver(
    max_search_iterations=1000,
    max_depth=50,
)

# Solve problems - automatically parses, reasons, and evaluates
answer = solver.solve(problem_statement)
```

#### Using Hybrid Solver (Fallback)

```python
from aimo_3.src.modeling.hybrid_solver import HybridSolver
from aimo_3.src.modeling.llm_base import LLMSolver

# Initialize hybrid solver
llm_solver = LLMSolver(model_name="meta-llama/Llama-2-7b-hf")
hybrid_solver = HybridSolver(
    llm_solver=llm_solver,
    strategy="fallback",  # or "ensemble", "reasoning_first"
)

# Solve problems
answer = hybrid_solver.solve(problem_statement)
```

#### Running Evaluation

```python
from aimo_3.src.training.inference import InferencePipeline
from aimo_3.src.geometry.solver import GeometrySolver

# Initialize pipeline (uses GeometrySolver by default)
pipeline = InferencePipeline(use_geometry=True)

# Run evaluation (in Kaggle environment)
answers = pipeline.run_evaluation()

# Generate submission file
submission_path = pipeline.generate_submission(answers)
```

## Project Structure

```
aimo_3/
├── src/
│   ├── data/          # Data loading and preprocessing
│   ├── evaluation/    # Evaluation API wrapper
│   ├── config/        # Configuration management
│   ├── modeling/      # Model architectures
│   ├── training/      # Training and inference pipelines
│   └── utils/         # Utility functions
├── notebooks/         # Jupyter notebooks for exploration
├── outputs/           # Training runs and submissions
├── data/              # Raw and processed data
├── configs/            # YAML configuration files
└── tests/              # Test suite
```

## Configuration

Configuration files are stored in `configs/`. Load a configuration:

```python
from aimo_3.src.config import load_config

config = load_config("baseline")
print(config.model.model_name)
```

Available configurations:
- `baseline.yaml` - Basic LLM setup
- `hybrid.yaml` - Hybrid LLM + symbolic reasoning

## Training

### Fine-tune a Model

```bash
python -m aimo_3.src.training.baseline \
    --config baseline \
    --data-dir data/raw \
    --output-dir outputs/baseline
```

### Custom Training

```python
from aimo_3.src.training.trainer import AIMOTrainer
from aimo_3.src.data.loader import load_problems

# Load data
problems = load_problems("data/raw")

# Initialize trainer
trainer = AIMOTrainer(
    model_name="meta-llama/Llama-2-7b-hf",
    output_dir="outputs/my_experiment",
)

# Train
trainer.train(problems[:100], problems[100:110])
```

## Evaluation API

AIMO uses a special evaluation API that serves problems one-by-one. The `AIMOEvaluator` class handles this:

```python
from aimo_3.src.evaluation.api import AIMOEvaluator

evaluator = AIMOEvaluator()

# In Kaggle environment, problems are served automatically
while True:
    problem = evaluator.get_next_problem()
    if problem is None:
        break
    
    answer = solver.solve(problem["statement"])
    evaluator.submit_answer_for_problem(problem["problem_id"], answer)

# Generate submission
evaluator.generate_submission_file()
```

## Model Architectures

### Geometry Solver (Primary)

Formal geometry reasoning system using scene graphs, theorem application, and MCTS search. See `GEOMETRY_SYSTEM.md` for details.

### Problem Generator

Synthesize novel geometric problems for training and testing:

```python
from aimo_3.src.geometry.generator import ProblemGenerator

generator = ProblemGenerator(seed=42)

# Generate problems in different families
problem, answer = generator.generate_triangle_problem("right", "medium")
problem, answer = generator.generate_circle_problem("inscribed", "hard")
problem, answer = generator.generate_coordinate_problem("distance", "easy")

# Generate training dataset
python scripts/generate_training_data.py --num_problems 1000
```

### LLM Solver

Uses language models (HuggingFace or API-based) to solve problems directly.

### Symbolic Solver

Generates Python code to solve problems and executes it safely.

### Hybrid Solver

Combines LLM reasoning with symbolic code generation:
- **fallback**: Try LLM first, fallback to symbolic
- **ensemble**: Use both and combine answers
- **reasoning_first**: Use LLM to reason, then generate code

### Ensemble Solver

Combines multiple solvers using voting or averaging.

## Submission

### Generate Submission File

```python
from aimo_3.src.utils.submission import generate_submission

answers = {"problem_001": 42, "problem_002": 100}
submission_path = generate_submission(answers, "outputs/submission.csv")
```

### Check Leaderboard

```bash
python -m aimo_3.src.utils.leaderboard \
    --competition aimo3 \
    --team "Your Team Name" \
    --top-n 20
```

## Kaggle Notebook Submission

To submit to Kaggle, create a notebook that:

1. Imports the evaluation API (provided by Kaggle)
2. Uses your solver to solve problems
3. Generates the submission file

Example template:

```python
# In Kaggle notebook
from kaggle_aimo3_evaluation import get_problem, submit_answer
from aimo_3.src.modeling.hybrid_solver import HybridSolver

solver = HybridSolver()

while True:
    problem = get_problem()
    if problem is None:
        break
    
    answer = solver.solve(problem["statement"])
    submit_answer(problem["problem_id"], answer)
```

## GPU with Brev (Optional)

Use the project wrapper `./brev_gpu.sh` for GPU training:

```bash
./brev_gpu.sh create      # Create GPU instance
./brev_gpu.sh sync-up    # Sync files
./brev_gpu.sh shell      # Open shell
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## LaTeX Handling

The competition uses specific LaTeX conventions. The `latex_parser` module handles:
- Normalization of LaTeX text
- Extraction of mathematical expressions
- Parsing of special notation (factorials, binomials, etc.)

## Evaluation Metrics

- **Public leaderboard**: Unnormalized accuracy (number of correct answers)
- **Private leaderboard**: Penalized accuracy
  - Both answers correct: score = 1
  - One correct, one incorrect: score = 0.5
  - Both incorrect: score = 0

## Next Steps

1. Explore the problem format using the EDA notebook
2. Train/fine-tune an initial LLM baseline
3. Implement symbolic code generation
4. Combine into hybrid solver
5. Iterate on evaluation API integration
6. Submit to Kaggle notebooks

## Resources

- [Competition Page](https://www.kaggle.com/competitions/aimo3)
- [Evaluation API Documentation](https://www.kaggle.com/competitions/aimo3/overview/evaluation)
- [Reference Problems PDF](https://www.kaggle.com/competitions/aimo3/data)

## License

This workspace is for competition use. See competition rules for licensing requirements.

