# NFL Big Data Bowl 2026 Prediction

Local workspace scaffold for the [NFL Big Data Bowl 2026 Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) Kaggle competition.

## Getting Started

1. **Join the competition**  
   Sign in to Kaggle, open the competition page, and accept the rules so the API can access the files.

2. **Configure Kaggle credentials**  
   - Install the CLI: `pip install kaggle`  
   - Create an API token via Kaggle → Account → *Create New Token* and place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json` (make sure the file has `0600` permissions on Unix-like systems).

3. **Download the data**  
From the repository root run:
```bash
python -m competitions.nfl_big_data_bowl_2026_prediction.src.data.download --extract
```
Files land in `competitions/nfl_big_data_bowl_2026_prediction/data/raw`. Pass `--file filename.csv` to grab specific assets.

4. **Check leaderboard standing**  
   After submitting to Kaggle, sync the public leaderboard with:
   ```bash
   python -m competitions.nfl_big_data_bowl_2026_prediction.src.utils.leaderboard \
       --metrics-file competitions/nfl_big_data_bowl_2026_prediction/outputs/baseline_metrics.json
   ```
   The script fetches the current leaderboard, highlights your team's rank, and juxtaposes it with the selected local metric.

## Recent Improvements

This workspace has been significantly enhanced with 14 major improvements:

### 🚀 Core Enhancements
- **Fixed requirements.txt**: Complete dependency list
- **Integrated Liquid Network**: Connected to real NFL data pipeline
- **Advanced Submission**: Generation for Liquid/GNN/Diffusion models
- **Cross-Validation**: Time-series aware K-fold CV
- **Ensemble Methods**: Model and fold ensembling (documented)
- **Advanced Metrics**: FDE, ADE, miss rate, collision rate (documented)

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed documentation.

## Quick Start Examples

### Train Liquid Network
```bash
python -m src.training.train_liquid_integrated \
  --epochs 50 \
  --batch-size 32 \
  --hidden-dim 128 \
  --output-dir outputs/liquid
```

### Generate Submission (Liquid Network)
```python
from src.utils.advanced_submission import generate_submission_from_checkpoint
from src.modeling.liquid_network import LiquidNetwork

submission = generate_submission_from_checkpoint(
    checkpoint_path=Path("outputs/liquid/best_model.pt"),
    model_factory=lambda: LiquidNetwork(input_dim=10, hidden_dim=128, output_dim=2),
    output_path=Path("submission_liquid.csv"),
)
```

### Run Cross-Validation
```python
from src.training.cross_validation import run_cross_validation

metrics = run_cross_validation(
    model_factory=lambda: build_model(),
    train_fn=train_function,
    evaluate_fn=evaluate_function,
    n_folds=5,
    strategy="sequential",
)
```

## GPU with Brev (optional)

Need GPU capacity? Use the project wrapper `./brev_gpu.sh` to manage a Brev workspace when required.

- `./brev_gpu.sh create` (first run) provisions the instance sized for this project.
- `./brev_gpu.sh sync-up` uploads the project directory.
- `./brev_gpu.sh shell` opens the remote GPU environment.

See `../docs/brev_gpu_workflow.md` for the complete workflow and advanced overrides.

## Repository Layout

- `data/` – storage for raw and processed datasets (gitignored by default).
- `notebooks/` – exploratory analysis notebooks.
- `outputs/` – generated artifacts such as submissions or plots.
- `src/` – reusable Python modules:
  - `data/` – download helpers, loading utilities, trajectory dataset
  - `features/` – feature engineering logic
  - `modeling/` – **NEW**: Liquid Networks, GNN, Diffusion models
  - `training/` – **NEW**: Integrated training, cross-validation
  - `utils/` – **NEW**: Advanced submission generation
- `train_liquid.py` – Liquid Network training script
- `requirements.txt` – **FIXED**: Complete dependency list

Adjust or extend this scaffold as experiments evolve.
