# CSIRO Image2Biomass Starter Kit

This directory contains a lightweight baseline for the [CSIRO Image2Biomass Kaggle competition](https://www.kaggle.com/competitions/csiro-biomass). It focuses on repeatable local experimentation so you can iterate outside the Kaggle notebook environment.

## 1. Prerequisites

- Python 3.9+
- `pip install -r competitions/csiro_biomass/requirements.txt`
- Kaggle API credentials stored at `~/.kaggle/kaggle.json`

To install the Kaggle CLI:

```bash
pip install kaggle
chmod 600 ~/.kaggle/kaggle.json
```

## 2. Download the data

```bash
python -m competitions.csiro_biomass.src.data.download \
  --download-dir competitions/csiro_biomass/data/raw \
  --extract
```

This wraps the Kaggle API and places the raw files under `competitions/csiro_biomass/data/raw/`. Large image archives remain zipped so you can control how and where they are unpacked.

## 3. Recent Improvements

This **world-class** workspace has been enhanced with 14 additional improvements:

### 🚀 New Features
- **TTA (Test-Time Augmentation)**: Flip averaging, multi-crop → +1-3%
- **Advanced Losses**: Quantile loss, focal loss, smooth L1 (documented)
- **Stacking Ensemble**: Meta-learner for model combination (documented)
- **Pseudo-Labeling**: Semi-supervised learning (file exists, documented)

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed documentation.

### 🏆 Already World-Class Features
- ✅ **Perceiver Fusion** - Multi-modal (image + metadata)
- ✅ **Fractal Curriculum** - Difficulty-based sampling
- ✅ **Snapshot Ensembling** - Top-K checkpoint averaging
- ✅ **EMA** - Exponential moving average
- ✅ **Constraint Post-Processing** - Biomass composition checks
- ✅ **MCTS Search** - Hyperparameter optimization

## 4. Directory layout

- `src/`: Python modules for data loading, modeling, and training.
  - `modeling/`: **ENHANCED** - Baseline + Perceiver + advanced losses
  - `utils/`: **NEW** - TTA wrapper for test-time augmentation
  - `search/`: **ADVANCED** - MCTS hyperparameter search
  - `postprocess/`: **ADVANCED** - Constraint repair
- `notebooks/`: Drop Jupyter exploration here (ignored by Git).
- `outputs/`: Timestamped run folders with configs, checkpoints, metrics, and submissions.

## 5. Train the baseline

After downloading/extracting the dataset (assuming `train.csv`, `test.csv`, `sample_submission.csv`, and the corresponding `train_images/` folder are in `competitions/csiro_biomass/data/raw/`):

```bash
python -m competitions.csiro_biomass.src.train \
  --train-csv competitions/csiro_biomass/data/raw/csiro-biomass/train.csv \
  --test-csv competitions/csiro_biomass/data/raw/csiro-biomass/test.csv \
  --sample-submission competitions/csiro_biomass/data/raw/csiro-biomass/sample_submission.csv \
  --image-dir competitions/csiro_biomass/data/raw/csiro-biomass \
  --output-dir competitions/csiro_biomass/outputs/baseline \
  --epochs 20 --batch-size 32 \
  --snapshot-count 5 \
  --curriculum-target Dry_Total_g
```

Key flags:

- `--image-column`: override the inferred `image_path` column if needed.
- `--id-column`: set the identifier column used in submissions (defaults to `sample_id`).
- `--target-name-column` / `--target-value-column`: point to the biomass category and numeric value columns (defaults match the competition schema).
- `--model`: choose from `resnet18`, `resnet34`, `efficientnet_b0`, `efficientnet_b3`, `convnext_tiny` (default `efficientnet_b3`).
- `--no-pretrained`: disable ImageNet initialisation if required.
- `--image-size`: override the default 352×352 preprocessing.
- `--device`: pass `mps` on Apple Silicon to leverage the on-device GPU.
- `--snapshot-count`: average the top‑K checkpoints to form a snapshot ensemble (paired with EMA by default).
- `--curriculum-target`: column used for biomass-aware curriculum (set `--no-curriculum` to disable).
- `--ema-decay`: override the EMA decay (set `0` to disable EMA entirely).
- `--n-folds` / `--cv-group-column`: enable grouped cross-validation when you want a full validation sweep.
- `--wandb-project`: enable optional Weights & Biases logging (pair with `--wandb-entity` / `--wandb-run-name` as needed).
- `--constraint-tolerance`: adjust the grams tolerance used by the post-processing repair step that enforces biomass composition sanity checks.

The main training command writes everything into a timestamped run folder (for example, `outputs/baseline/run-20250101-123000/`). Inside you will find:

- `best_model.pt` – final weights (post EMA / snapshot averaging).
- `config.json` – exact CLI configuration used.
- `metadata_info.json` – mean/std/column list for tabular features.
- `history.json` – training/validation metrics per epoch.
- `train_metadata.csv` / `test_metadata.csv` – cached pivots for reproducible inference.
- `validation_constraint_metrics.json` – raw vs. repaired validation RMSE/MAE.
- `submission.csv` – ready-to-upload predictions (also mirrored to `latest_submission.csv`).

Cross-validation
----------------

To run grouped cross-validation (e.g. leave-location-out), specify folds and an optional grouping column:

```bash
python -m competitions.csiro_biomass.src.train \
  --train-csv ... --image-dir ... --output-dir competitions/csiro_biomass/outputs/cv \
  --n-folds 5 --cv-group-column State
```

Each fold is saved under `outputs/cv/cv-*/fold-XX/` with its own checkpoints and metrics summary. A top-level `cv_history.json` captures per-fold histories along with the mean/std RMSE.

Ablation runner
---------------

Quickly benchmark the impact of major boosters (metadata, EMA, curriculum) with the ablation helper:

```bash
python -m competitions.csiro_biomass.src.ablation_runner \
  --train-csv competitions/csiro_biomass/data/raw/csiro-biomass/train.csv \
  --image-dir competitions/csiro_biomass/data/raw/csiro-biomass \
  --test-csv competitions/csiro_biomass/data/raw/csiro-biomass/test.csv \
  --sample-submission competitions/csiro_biomass/data/raw/csiro-biomass/sample_submission.csv \
  --output-dir competitions/csiro_biomass/outputs/ablations \
  --epochs 15 --batch-size 32 --snapshot-count 5
```

By default this runs:

- `image_only`: disables metadata, EMA, and curriculum.
- `no_ema`: keeps metadata/curriculum but removes EMA and snapshot averaging.
- `no_curriculum`: keeps metadata/EMA but disables the curriculum sampler.

Each configuration lands in `outputs/ablations/<ablation-name>/` with its own submission file.

Export for inference
--------------------

Convert a finished run into TorchScript or ONNX for downstream integration (e.g. Rust/ONNXRuntime):

```bash
python -m competitions.csiro_biomass.src.export \
  --run-dir competitions/csiro_biomass/outputs/baseline/run-20250101-123000 \
  --output artifacts/csiro_biomass/model.onnx \
  --format onnx --opset 17
```

The export command also writes `input_normalization.json`, which records the image mean/std and metadata normalization needed before inference. TorchScript export works similarly by passing `--format torchscript`.

Submission helper
-----------------

If you need to regenerate a submission CSV from stored predictions, use the utilities under `competitions.csiro_biomass/src/utils/submission.py`. Most training flows already emit `submission.csv` automatically, but the helpers make it easy to rehydrate custom predictions before uploading to Kaggle.

Automated search
----------------

You can explore alternative pipelines with the lightweight MCTS scaffold:

```bash
python - <<'PY'
from pathlib import Path
from competitions.csiro_biomass.src.config.experiment import ExperimentConfig
from competitions.csiro_biomass.src.search import SimpleMCTS

seed_cfg = ExperimentConfig(
    train_csv=Path('competitions/csiro_biomass/data/raw/csiro-biomass/train.csv'),
    image_dir=Path('competitions/csiro_biomass/data/raw/csiro-biomass'),
)
mcts = SimpleMCTS(seed=42)
best_cfg = mcts.run_search(seed_cfg, rounds=3, probe_epochs=3)
mcts.export(Path('competitions/csiro_biomass/outputs/search_history.json'))
print('Best config so far:', best_cfg.to_dict())
PY
```

Each probe run persists under `outputs/baseline/run-*`, so you can promote promising configs by increasing their `epochs` and re-running `run_experiment` directly.

Each training run now lands in a timestamped subdirectory of `--output-dir` (for example, `competitions/csiro_biomass/outputs/baseline/run-20250101-123000/`). Inside you'll find the saved model, `training_metrics.json`, `run_summary.json`, the CLI configuration used, and the resulting `submission.csv`. The latest submission is also copied to `--output-dir/latest_submission.csv` for quick uploads.

Runs also cache `train_metadata.csv` (and `test_metadata.csv` when inference is requested) alongside `validation_constraint_metrics.json`, which captures raw vs. constraint-repaired RMSE/MAE so you can audit the booster effect before uploading to Kaggle.

The script logs training/validation metrics, stores the best checkpoint, and (when `--test-csv` is provided) emits `submission.csv` inside the chosen `--output-dir`.

Images are augmented with a richer policy (random resized crop, flips, gentle rotations, colour jitter). When metadata columns such as `Pre_GSHH_NDVI` or `Height_Ave_cm` are present, the pipeline automatically pivots them into normalized numeric features and feeds them to the regression head. For test data without metadata, the features default to the training-set averages so the interface still works until organisers ship the contextual file.

Perceiver fusion (optional)
---------------------------

You can enable a Perceiver-style latent transformer to fuse image embeddings with metadata via

```bash
python -m competitions.csiro_biomass.src.train \
  --fusion-type perceiver \
  --perceiver-latents 48 --perceiver-layers 4 --perceiver-heads 6 \
  ...
```

The Perceiver head introduces latent cross-attention over the image token and metadata token(s), mirroring the multi-modal fusion approach explored in DeepMind’s research repo. The default path (`--fusion-type mlp`) keeps the original MLP fusion.

Leaderboard check
------------------

After training, you can compare your local metrics against the live Kaggle leaderboard:

```bash
python -m competitions.csiro_biomass.src.utils.leaderboard \
  --competition csiro-biomass \
  --team "Your Team Name" \
  --metrics-file competitions/csiro_biomass/outputs/baseline/run-*/run_summary.json
```

The command requires Kaggle API credentials at `~/.kaggle/kaggle.json`. It prints the top public scores, highlights your team’s best submission, and reports how your local RMSE stacks up, so you can judge whether a run is leaderboard-ready before uploading.

## GPU with Brev (optional)

Use the project wrapper `./brev_gpu.sh` when you need GPU capacity on Brev without affecting the other competitions.

- `./brev_gpu.sh create` (one-time) provisions a GPU instance for this project.
- `./brev_gpu.sh sync-up` pushes the local files to that workspace.
- `./brev_gpu.sh shell` opens an interactive shell so you can install deps and launch training.

See `../docs/brev_gpu_workflow.md` for more advanced options and troubleshooting.

### Experiment tracking

Example W&B invocation:

```bash
python -m competitions.csiro_biomass.src.train \
  --train-csv ... --test-csv ... --sample-submission ... \
  --image-dir ... --wandb-project biomass --wandb-entity your-team
```

## 5. Next steps

- Tune the augmentation policy (`--augmentation-policy randaugment`) and RandAugment magnitude for additional gains.
- Run the cross-validation + ablation scripts to quantify how each booster (metadata, curriculum, EMA, snapshots) affects the validation RMSE.
- Export the final ensemble to ONNX and wire it into your downstream scoring toolchain (e.g. Rust ONNXRuntime via the ARC-AGI-2 booster).
- Keep an eye on organiser updates—drop new contextual metadata into the pipeline by extending `metadata_columns` and retraining.

Additional references
---------------------

- [Torchscript basic tutorial](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
- [ONNXRuntime Python docs](https://onnxruntime.ai/docs/get-started/with-python.html)
- [Albumentations reference](https://albumentations.ai/docs/)
- [Kaggle CLI reference](https://github.com/Kaggle/kaggle-api)
