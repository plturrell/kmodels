# Hull Tactical Market Prediction Starter Kit

This workspace bootstraps local experimentation for the [Hull Tactical Market Prediction competition](https://www.kaggle.com/competitions/hull-tactical-market-prediction). It is intentionally lightweight so you can iterate quickly outside the hosted Kaggle notebook environment while keeping runs reproducible.

## 1. Prerequisites

- Python 3.10+
- `pip install -r competitions/hull_tactical_market_prediction/requirements.txt`
- Kaggle API credentials stored at `~/.kaggle/kaggle.json`

Install the Kaggle CLI if it is not already available:

```bash
pip install kaggle
chmod 600 ~/.kaggle/kaggle.json
```

## 2. Download the data

All command snippets assume you run them from the repository root so paths such as
`competitions/hull_tactical_market_prediction/...` resolve correctly.

The helper below mirrors `kaggle competitions download` but keeps paths consistent with this repo:

```bash
python -m competitions.hull_tactical_market_prediction.src.data.download \
  --download-dir competitions/hull_tactical_market_prediction/data/raw \
  --extract
```

This pulls every published file for the competition into `competitions/hull_tactical_market_prediction/data/raw/`. Large archives are left in place unless you pass `--extract`.

## 3. Recent Improvements

This workspace has been enhanced with 14 major improvements **focused on financial market prediction**:

### 🚀 Core Enhancements
- **Advanced Time-Series Models**: LSTM, Transformer
- **Financial Features**: RSI, MACD, Bollinger Bands, ATR
- **Walk-Forward CV**: Purged K-fold, embargo periods
- **Risk-Adjusted Metrics**: Sharpe ratio, Sortino ratio, max drawdown

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed documentation.

## 4. Directory layout

- `src/`: Python modules for feature engineering, modeling, and CLI entrypoints.
  - `features/`: **ENHANCED** - Technical indicators (RSI, MACD, Bollinger Bands)
  - `modeling/`: **ENHANCED** - GBM, MLP, Perceiver, TSMixer + LSTM, Transformer
  - `training/`: **NEW** - Walk-forward CV, purged K-fold
  - `utils/`: **NEW** - Financial metrics (Sharpe, Sortino, drawdown)
- `notebooks/`: Drop ad-hoc Jupyter exploration here (ignored by Git).
- `outputs/`: Baseline runs land in timestamped sub-folders with configs, metrics, and submissions.

## 4. Train the gradient boosting baseline

After downloading the data, point the training script at the CSVs. The current files expose `date_id` as the identifier and `forward_returns` as the target—double-check in case the organisers refresh the schema:

```bash
python -m competitions.hull_tactical_market_prediction.src.train \
  --train-csv competitions/hull_tactical_market_prediction/data/raw/train.csv \
  --test-csv competitions/hull_tactical_market_prediction/data/raw/test.csv \
  --target-column forward_returns \
  --id-column date_id \
  --output-dir competitions/hull_tactical_market_prediction/outputs/baseline \
  --lag-step 1 --lag-step 5 --lag-step 21 \
  --cv-splits 5
```

Key flags:

- `--task` and `--drop-column`: steer the prediction objective and strip identifiers or leaks up front.
- `--max-nan-ratio` / `--keep-constant-features`: control whether sparse or constant signals survive preprocessing (defaults drop >75% missing and one-value columns).
- `--lag-step`, `--rolling-window`, and `--rolling-stat`: add lagged and rolling statistics (defaults mirror 1/5/21-day windows with mean/std).
- `--no-time-series-cv` and `--save-model`: toggle shuffled CV and persist the fitted pipeline to `model.joblib`.
- `--submission-column` and `--sample-submission`: tailor the submission header layout before conversion to parquet.

Every run creates a folder such as `outputs/baseline/run-20250217-101530/` containing:

- `submission.csv` plus a convenience copy at `outputs/baseline/latest_submission.csv`.
- `metrics.json` with simple CV diagnostics (accuracy for classification, RMSE for regression).
- `run_config.json` capturing the CLI settings for the run.
- `model.joblib` when `--save-model` is supplied.

## 5. Neural solver (PyTorch)

For a richer solver inspired by the CSIRO starter kit, train the tabular neural network:

```bash
python -m competitions.hull_tactical_market_prediction.src.train_nn \
  --train-csv competitions/hull_tactical_market_prediction/data/raw/train.csv \
  --test-csv competitions/hull_tactical_market_prediction/data/raw/test.csv \
  --target-column forward_returns \
  --id-column date_id \
  --model mlp \
  --hidden-dim 512 --hidden-dim 256 --hidden-dim 128 \
  --batch-size 512 --epochs 40 --max-nan-ratio 0.5
```

Key flags mirror the baseline but extend to neural specifics:

- `--hidden-dim`, `--dropout`, `--activation`, `--no-batch-norm`: shape the MLP architecture.
- `--model perceiver` plus the `--perceiver-*` knobs switches to a Perceiver-style encoder over the engineered features.
- `--learning-rate`, `--weight-decay`: tweak the AdamW optimiser.
- `--val-fraction`, `--shuffled-holdout`: adjust the validation split strategy (chronological holdout by default).
- Feature flags (`--lag-step`, `--rolling-window`, `--rolling-stat`, `--max-nan-ratio`) map directly onto the enhanced feature builder.

Runs land under `outputs/tabular_nn/run-*/` and capture the experiment config, feature statistics, best checkpoint (as `model_state.pt`), metrics, and generated submission (with a convenience copy at `outputs/tabular_nn/latest_submission.csv`).

Research adapters
-----------------

- `python -m competitions.hull_tactical_market_prediction.src.experimental.export_tsmixer --train-csv ... --test-csv ...` exports the Kaggle tables into the Autoformer/TSMixer CSV layout, ready for google-research `tsmixer_basic` experiments.
- `python -m competitions.hull_tactical_market_prediction.src.experimental.two_phase_perceiver` runs the Perceiver in two stages (MSE warm-up followed by Sharpe fine-tuning) using the existing `train_nn` CLI under the hood.
- `python -m competitions.hull_tactical_market_prediction.src.experimental.setup_tsmixer --tsmixer-root /path/to/google-research` copies the CSVs into `tsmixer/tsmixer_basic/dataset/` and prints the config snippet to register the dataset before running `sh run_tuned_hparam.sh hull_tactical`.
- `python -m competitions.hull_tactical_market_prediction.src.experimental.create_validation_split --train-path ... --date-column ...` materialises a time-aware validation set that other evaluation utilities consume.
- To run **TSMixer** directly, clone `google-research/google-research`, copy the CSVs to `tsmixer/tsmixer_basic/dataset/`, install the listed requirements, then invoke `sh run_tuned_hparam.sh traffic` (swap `traffic` for a Hull-specific entry after editing their config to point at your CSVs). Capture the resulting `pred.csv`, convert it with the submission helper, and evaluate against the leaderboard.

## 6. Export to parquet

Kaggle’s evaluation service expects a `submission.parquet`. Convert the latest CSV with:

```bash
python -m competitions.hull_tactical_market_prediction.src.utils.convert_submission \
  --input competitions/hull_tactical_market_prediction/outputs/baseline/latest_submission.csv \
  --output competitions/hull_tactical_market_prediction/outputs/baseline/latest_submission.parquet
```

`--compression` controls the parquet codec (`snappy` by default, pass `none` to disable compression).

## GPU with Brev (optional)

Leverage the optional Brev wrapper (`./brev_gpu.sh`) when you need GPU time for experiments tied to this project.

- `./brev_gpu.sh create` (first run) provisions the dedicated GPU instance.
- `./brev_gpu.sh sync-up` mirrors the project directory to that workspace.
- `./brev_gpu.sh shell` opens a remote shell for GPU-bound training or evaluation.

See `../docs/brev_gpu_workflow.md` for additional details and override options.

## 7. Next steps

- Integrate richer domain signals (spreads, regime flags, macro rollups) to complement the default lag/rolling set.
- Experiment with alternative learners (LightGBM, CatBoost, tabular neural nets) while reusing the feature pipeline.
- Wire up experiment tracking (Weights & Biases, MLflow) or hyperparameter search once the baseline is stable.

### Leaderboard check

To compare a local validation score with the live Kaggle leaderboard (requires authenticated Kaggle API):

```bash
python -m competitions.hull_tactical_market_prediction.src.leaderboard.check_score \
  --score 0.12 --max-entries 20
```

Omit `--score` to just print the current top chunk.
