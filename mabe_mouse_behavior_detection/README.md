# MABe Mouse Behavior Detection

Local experimentation scaffold for the [MABe Mouse Behavior Detection](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection) Kaggle competition.

## Getting Started
- Join the competition on Kaggle and accept the rules so the API can access the data.
- Install dependencies: `pip install -r competitions/mabe_mouse_behavior_detection/requirements.txt`
- Configure the Kaggle CLI (place `kaggle.json` at `~/.kaggle/kaggle.json` with `0600` permissions).

### Download the dataset

```bash
python -m competitions.mabe_mouse_behavior_detection.src.data.download --extract
```

All files land in `data/raw/`. Pass `--file filename.ext` to fetch a specific asset and `--download-dir` to override the target directory.

## Repository Layout
- `data/`: Kaggle downloads under `raw/` plus any derived artifacts in `processed/`.
- `src/`: Python package with data loaders, model pipelines, and utilities.
- `notebooks/`: Scratch space for Jupyter experiments (ignored by Git).
- `outputs/`: Training runs, metrics, submissions, and diagnostics.
- `docs/`: Notes, checklists, and design sketches shared across the team.

## Train the Baseline (Lightning)

The Lightning reimplementation of the GRU pose classifier lives at `src/training/baseline.py`. From the repository root, point it at the metadata CSV that references pose/keypoint tensors:

```bash
python -m competitions.mabe_mouse_behavior_detection.src.training.baseline \
  --train-csv data/raw/train.csv \
  --asset-root data/raw \
  --asset-column pose_path \
  --target-column behavior \
  --id-column clip_id \
  --sequence-length 128 \
  --epochs 20 \
  --output-dir competitions/mabe_mouse_behavior_detection/outputs/lightning_baseline
```

- `--pose-array` selects the array key inside `.npz` pose files (default `keypoints`).
- `--test-csv` and `--submission-path` will emit predictions for the Kaggle test split.
- Outputs now follow the Lightning pattern: timestamped `run-*` directories with `config.json`, `history.json`, `summary.json`, a `checkpoints/` folder, and optional submissions.

## Build a Submission

Use the helper to align run outputs with Kaggle's sample submission ordering:

```bash
python -m competitions.mabe_mouse_behavior_detection.src.utils.submission \
  --run-dir competitions/mabe_mouse_behavior_detection/outputs/lightning_baseline/run-YYYYMMDD-HHMMSS \
  --sample-submission data/raw/sample_submission.csv \
  --id-column clip_id \
  --prediction-column prediction \
  --output competitions/mabe_mouse_behavior_detection/outputs/lightning_baseline/submission.csv
```

## Recent Improvements

This workspace has been significantly enhanced with 14 major improvements:

### 🚀 Core Enhancements
- **Advanced Architectures**: Transformer, LSTM, TCN (CPU-compatible)
- **Data Augmentation**: Temporal and spatial augmentation (documented)
- **Cross-Validation**: K-fold with stratification (documented)
- **Ensemble Methods**: Model and fold ensembling (documented)
- **Advanced Metrics**: F1, precision, recall, confusion matrix (documented)

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed documentation.

### Quick Examples

**Train with Transformer:**
```python
from src.modeling.advanced import TransformerSequenceClassifier, TransformerConfig

config = TransformerConfig(input_dim=24, num_classes=7, d_model=256, nhead=8)
model = TransformerSequenceClassifier(config)
```

**Train with TCN:**
```python
from src.modeling.advanced import TCNSequenceClassifier, TCNConfig

config = TCNConfig(input_dim=24, num_classes=7, num_channels=[64, 128, 256])
model = TCNSequenceClassifier(config)
```

## GPU with Brev (optional)

Launch the project wrapper `./brev_gpu.sh` when you need to burst onto a Brev GPU.

- `./brev_gpu.sh create` (one-time) provisions a GPU workspace dedicated to this project.
- `./brev_gpu.sh sync-up` uploads the project tree before running experiments.
- `./brev_gpu.sh shell` opens the GPU environment for installs and training.

Full workflow details live in `../docs/brev_gpu_workflow.md`.

## Other Utilities
- Track leaderboard progress: `python -m competitions.mabe_mouse_behavior_detection.src.utils.leaderboard --top 10`
- **NEW**: Advanced architectures in `src/modeling/advanced.py` (Transformer, LSTM, TCN)
- Drop exploratory notebooks in `notebooks/` (ignored by Git).

Pull requests should keep scripts runnable as modules (e.g. `python -m competitions.mabe_mouse_behavior_detection.src.training.baseline`) so experiments stay reproducible.
