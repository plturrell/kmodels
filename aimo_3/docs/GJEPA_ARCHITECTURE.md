# G-JEPA Architecture Documentation

## Overview

Clean implementation of G-JEPA following the specified architecture:
- **Trace → sequence of scenes** (`build_scene_sequence`)
- **Encoder methods** (`encode_state`, `encode_trace`)
- **Dataset** (`GeometryJEPADataset`)
- **GJEPA Model** (Transformer + predictor)

## Architecture

### 1. Scene Sequence Building

**File:** `src/geometry/scene_sequence.py`

```python
from src.geometry.scene_sequence import build_scene_sequence, SceneTrace

# Build trace from proof states
trace = build_scene_sequence(
    proof_states=[state_0, state_1, ..., state_n],
    trace_id="trace_1",
    problem_id="problem_1",
)
```

**SceneTrace** contains:
- `scenes`: List[GeometricSceneGraph] - sequence of scene graphs
- `states`: List[State] - corresponding states
- `trace_id`: str
- `problem_id`: Optional[str]

### 2. Scene Encoder

**File:** `src/geometry/scene_encoder.py`

**Methods:**
- `encode_state(scene_or_trace) → [D]`: Encode single scene/state/trace to latent vector
- `encode_trace(trace) → [T+1, D]`: Encode entire trace to sequence of latents

```python
from src.geometry.scene_encoder import SceneEncoder

encoder = SceneEncoder(output_dim=256)

# Encode single state
latent = encoder.encode_state(state)  # [256]

# Encode entire trace
latent_seq = encoder.encode_trace(trace)  # [T+1, 256]
```

**Architecture:**
- Graph Neural Network (GNN) with message passing
- Graph-level mean pooling
- Output: fixed-dimensional latent vector [D]

### 3. Geometry JEPA Dataset

**File:** `src/geometry/jepa_dataset.py`

```python
from src.geometry.jepa_dataset import GeometryJEPADataset

dataset = GeometryJEPADataset(
    traces=traces,
    encoder=encoder,
    mask_ratio=0.3,
)

# Returns (h_seq, mask_indices) suitable for JEPA loss
h_seq, mask_indices = dataset[0]
# h_seq: [T+1, D]
# mask_indices: [num_masked]
```

**Features:**
- Automatic encoding of traces
- Random masking (configurable ratio)
- Collate function for batching

### 4. GJEPA Model

**File:** `src/modeling/gjepa_model.py`

```python
from src.modeling.gjepa_model import GJEPA, create_gjepa_model

model = create_gjepa_model(
    latent_dim=256,
    hidden_dim=512,
    num_layers=4,
    num_heads=8,
)

# Forward pass
predicted, targets = model(h_seq, mask_indices)
# predicted: [total_masked, D]
# targets: [total_masked, D]

# Compute loss
loss = model.compute_loss(h_seq, mask_indices)
```

**Architecture:**
- **Input Projection**: [D] → [hidden_dim]
- **Positional Encoding**: Learnable positional embeddings
- **Transformer Encoder**: Multi-layer transformer with attention
- **Predictor**: MLP that predicts masked latents
- **Loss**: MSE between predicted and target latents

## Training Flow

```python
# 1. Generate traces
traces = generate_training_traces(num_problems=1000)

# 2. Create encoder
encoder = SceneEncoder(output_dim=256)

# 3. Create dataset
dataset = GeometryJEPADataset(traces=traces, encoder=encoder)

# 4. Create model
model = create_gjepa_model(latent_dim=256)

# 5. Train
trainer = GJEPATrainer(model, encoder)
trainer.train(dataloader, num_epochs=10)
```

## Integration with MCTS

The trained G-JEPA model can be used to guide MCTS search:

```python
# Load trained model
checkpoint = torch.load("outputs/gjepa/gjepa_final.pt")
model.load_state_dict(checkpoint['model_state_dict'])
encoder.load_state_dict(checkpoint['encoder_state_dict'])

# Use in solver
solver = GeometrySolver(use_gjepa=True)
answer = solver.solve(problem_statement)
```

## File Structure

```
src/
├── geometry/
│   ├── scene_sequence.py      # Trace building
│   ├── scene_encoder.py        # Encoder (encode_state, encode_trace)
│   └── jepa_dataset.py         # Dataset (h_seq, mask_indices)
└── modeling/
    └── gjepa_model.py          # GJEPA model (Transformer + predictor)
```

## Compilation Check

All files compile cleanly:
```bash
python3 -m compileall src/geometry/scene_sequence.py
python3 -m compileall src/geometry/scene_encoder.py
python3 -m compileall src/geometry/jepa_dataset.py
python3 -m compileall src/modeling/gjepa_model.py
```

## Next Steps

1. **Extract real proof sequences** from solver (currently simplified)
2. **Train on GPU** for efficient training
3. **Integrate with MCTS** for heuristic guidance
4. **Evaluate performance** improvements

