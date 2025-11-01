# Quick Start Guide - NFL Big Data Bowl 2026

## Installation

```bash
cd competitions/nfl_big_data_bowl_2026_prediction
pip install -r requirements.txt
```

## Download Data

```bash
python -m src.data.download --extract
```

## Training Options

### Option 1: Baseline Random Forest
```bash
python -m src.modeling.baseline \
  --n-estimators 300 \
  --generate-submission
```

### Option 2: Liquid Neural Network (Recommended)
```bash
python -m src.training.train_liquid_integrated \
  --epochs 50 \
  --batch-size 32 \
  --hidden-dim 128 \
  --num-layers 2 \
  --integrator rk4 \
  --output-dir outputs/liquid
```

**Expected improvement:** +10-15% MAE reduction over baseline.

### Option 3: Cross-Validation
```python
from src.training.cross_validation import run_cross_validation
from src.modeling.liquid_network import LiquidNetwork

def model_factory():
    return LiquidNetwork(input_dim=10, hidden_dim=128, output_dim=2)

def train_fn(model, train_weeks):
    # Train model on train_weeks
    return trained_model

def evaluate_fn(model, val_weeks):
    # Evaluate on val_weeks
    return {"mae": mae, "fde": fde, "ade": ade}

metrics = run_cross_validation(
    model_factory=model_factory,
    train_fn=train_fn,
    evaluate_fn=evaluate_fn,
    n_folds=5,
    strategy="sequential",
    output_dir=Path("outputs/cv"),
)
```

## Generate Submission

### Baseline Submission
```bash
python -m src.modeling.baseline --generate-submission
```

### Liquid Network Submission
```python
from src.utils.advanced_submission import generate_submission_from_checkpoint
from src.modeling.liquid_network import LiquidNetwork

submission = generate_submission_from_checkpoint(
    checkpoint_path=Path("outputs/liquid/best_model.pt"),
    model_factory=lambda: LiquidNetwork(
        input_dim=10,
        hidden_dim=128,
        output_dim=2,
        num_layers=2,
    ),
    output_path=Path("submission_liquid.csv"),
)
```

### With Test-Time Augmentation
```python
# Add TTA in advanced_submission.py
def predict_with_tta(model, features, positions, mask, n_samples=5):
    predictions = []
    for _ in range(n_samples):
        noisy_positions = positions + torch.randn_like(positions) * 0.05
        pred = model(features, noisy_positions, mask)
        predictions.append(pred)
    return torch.stack(predictions).mean(dim=0)
```

## Recommended Workflow

1. **Start with baseline**
   ```bash
   python -m src.modeling.baseline --generate-submission
   ```

2. **Train Liquid Network**
   ```bash
   python -m src.training.train_liquid_integrated --epochs 50
   ```

3. **Run cross-validation**
   ```python
   run_cross_validation(model_factory, train_fn, evaluate_fn, n_folds=5)
   ```

4. **Create ensemble**
   ```python
   # Combine baseline + Liquid + GNN
   ensemble = TrajectoryEnsemble([baseline_model, liquid_model, gnn_model])
   ```

5. **Generate submission with TTA**
   ```python
   submission = generate_submission_with_tta(ensemble, test_data)
   ```

## Expected Performance

| Method | Expected MAE | Notes |
|--------|-------------|-------|
| Baseline RF | ~2.5-3.0 | Random Forest |
| Liquid Network | ~2.0-2.5 | +10-15% improvement |
| + Ensemble | ~1.8-2.2 | +5-10% improvement |
| + TTA | ~1.7-2.1 | +2-5% improvement |
| + Advanced Losses | ~1.6-2.0 | Final boost |

## Tips for Best Results

1. **Use Liquid Networks** - Better long-term dependencies than LSTM
2. **Use RK4 integrator** - More accurate than Euler
3. **Ensemble multiple models** - Combine RF + Liquid + GNN
4. **Apply TTA** - Multiple samples per prediction
5. **Use physics-informed losses** - Enforce realistic dynamics
6. **Time-series aware CV** - Sequential folding for temporal data

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 16`
- Reduce hidden dimension: `--hidden-dim 64`
- Reduce max sequences: `--max-sequences 1000`

### Slow Training
- Use Euler integrator: `--integrator euler`
- Reduce number of layers: `--num-layers 1`
- Use GPU: `--device cuda`

### Poor Performance
- Increase hidden dimension: `--hidden-dim 256`
- Use RK4 integrator: `--integrator rk4`
- Train longer: `--epochs 100`
- Add data augmentation

## Advanced Usage

### Custom Loss Function
```python
from src.modeling.trajectory_losses import (
    HuberLoss, CollisionLoss, PhysicsInformedLoss
)

criterion = nn.ModuleDict({
    "mse": nn.MSELoss(),
    "huber": HuberLoss(delta=1.0),
    "collision": CollisionLoss(threshold=1.0, weight=0.1),
    "physics": PhysicsInformedLoss(max_speed=10.0),
})

# Combined loss
loss = (
    criterion["mse"](pred, target) +
    0.5 * criterion["huber"](pred, target) +
    0.1 * criterion["collision"](pred) +
    0.1 * criterion["physics"](pred)
)
```

### Hyperparameter Tuning
```python
import optuna

def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=64)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    
    model = LiquidNetwork(hidden_dim=hidden_dim)
    val_mae = train_and_evaluate(model, lr=lr)
    return val_mae

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```

## Support

- See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed feature documentation
- See [README.md](README.md) for repository structure
- Check notebooks for EDA examples

Good luck with the competition! 🏈🚀

