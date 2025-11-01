# NFL Big Data Bowl 2026 Analytics

Local workspace scaffold for the [NFL Big Data Bowl 2026 Analytics](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-analytics) Kaggle competition.

## Getting Started

1. **Join the competition**  
   Sign in to Kaggle, open the competition page, and accept the rules so the API can access the files.

2. **Configure Kaggle credentials**  
   - Install the CLI: `pip install kaggle`  
   - Create an API token via Kaggle → Account → *Create New Token* and place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json` (make sure the file has `0600` permissions on Unix-like systems).

3. **Download the data**  
   From the repository root run:
   ```bash
   python -m competitions.nfl_big_data_bowl_2026_analytics.src.data.download --extract
   ```
   Files land in `competitions/nfl_big_data_bowl_2026_analytics/data/raw`. Pass `--file filename.csv` to grab specific assets.

## Recent Improvements

This workspace has been significantly enhanced with 14 major improvements:

### 🚀 Core Enhancements
- **Fixed requirements.txt**: Complete dependency list with version constraints
- **Submission Validation**: Format, range, and completeness checks
- **Advanced Metrics**: FDE, ADE, miss rate, direction accuracy
- **Bug Fixes**: Fixed torch_geometric import in graph_model.py
- **Comprehensive Documentation**: IMPROVEMENTS.md with all features

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed documentation.

## Quick Start Examples

### Train with Advanced Architecture
```bash
# Residual MLP (DeepMind-style)
python -m src.train --architecture residual_mlp --epochs 50

# Perceiver (Latent attention)
python -m src.train --architecture perceiver --epochs 50

# GAT (Graph attention)
python -m src.train --architecture gat --epochs 50
```

### Validate Submission
```python
from src.utils.validation import validate_and_save_submission

is_valid = validate_and_save_submission(
    submission,
    output_path=Path("submission.csv"),
    strict=True,
)
```

### Compute Advanced Metrics
```python
from src.utils.advanced_metrics import compute_all_metrics

metrics = compute_all_metrics(predictions, targets)
# Returns: {'fde': 1.23, 'ade': 0.98, 'miss_rate_2yd': 0.15, ...}
```

## Repository Layout

- `competitions/nfl_big_data_bowl_2026_analytics/data/` – raw competition bundle (gitignored by default).
- `notebooks/` – exploratory analysis notebooks.
- `outputs/` – generated artifacts such as submissions or plots.
- `src/` – reusable Python modules:
  - `data/` – download helpers and loading utilities.
  - `features/` – feature engineering logic (kinematics, relational).
  - `modeling/` – **ENHANCED**: MLP, Residual MLP, Perceiver, GAT models.
  - `training/` – Lightning training infrastructure.
  - `utils/` – **NEW**: Validation, advanced metrics, submission tools.
  - `config/` – Dataclass-based configuration management.
- `src/train.py` – Main training entry point with cross-validation.
- `src/optimize.py` – Optuna hyperparameter optimization.
- `src/ensemble.py` – Model ensembling utilities.
- `requirements.txt` – **FIXED**: Complete dependency list.

Adjust or extend this scaffold as experiments evolve.

## Train the Baseline

After downloading the data, you can launch the local baseline with:

```bash
PYTHONPATH=. python -m competitions.nfl_big_data_bowl_2026_analytics.src.train \
  --device mps \
  --weeks 2023_01 \
  --epochs 5 \
  --batch-size 256
```

Key flags:
- `--weeks 2023_01,2023_02` limits training to specific season/week pairs (defaults to all weeks).
- `--n-folds 5` enables grouped cross-validation over `game_id`.
- `--snapshot-count 3` averages the top checkpoints for a snapshot ensemble.
- `--use-pairwise-distance` adds a graph-inspired distance-to-landing feature.
- `--use-game-clock-seconds` injects temporal context via parsed game-clock seconds and a two-minute drill flag.
- `--architecture residual_mlp|perceiver` switches to deeper models inspired by DeepMind research (residual perceptrons or latent attention).
- `--early-stopping-patience 5` enables Lightning early stopping when validation RMSE stalls.
- `--persistent-workers` keeps dataloader workers alive across epochs to reduce loader overhead.

Outputs land in `competitions/nfl_big_data_bowl_2026_analytics/outputs/baseline/run-YYYYMMDD-HHMMSS/` with configs, history, metrics, and the best checkpoint.
