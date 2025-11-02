# Quick Start Guide - CSIRO Biomass

## Installation

```bash
cd competitions/csiro_biomass
pip install -r requirements.txt
```

## Download Data

```bash
python -m competitions.csiro_biomass.src.data.download --download-dir data/raw --extract
```

## Training Options

### Option 1: Baseline (EfficientNet-B3)
```bash
python -m competitions.csiro_biomass.src.train \
  --train-csv data/raw/csiro-biomass/train.csv \
  --test-csv data/raw/csiro-biomass/test.csv \
  --sample-submission data/raw/csiro-biomass/sample_submission.csv \
  --image-dir data/raw/csiro-biomass \
  --output-dir outputs/baseline \
  --model efficientnet_b3 \
  --epochs 20 --batch-size 32 \
  --snapshot-count 5
```

### Option 2: With Perceiver Fusion (Recommended)
```bash
python -m competitions.csiro_biomass.src.train \
  --train-csv data/raw/csiro-biomass/train.csv \
  --test-csv data/raw/csiro-biomass/test.csv \
  --sample-submission data/raw/csiro-biomass/sample_submission.csv \
  --image-dir data/raw/csiro-biomass \
  --output-dir outputs/perceiver \
  --model efficientnet_b3 \
  --fusion-type perceiver \
  --perceiver-latents 48 --perceiver-layers 4 --perceiver-heads 6 \
  --epochs 20 --batch-size 32 \
  --snapshot-count 5
```

**Expected improvement:** +3-7% over baseline.

### Option 3: With TTA (Test-Time Augmentation)
```python
from competitions.csiro_biomass.src.utils.tta import TTAWrapper, MultiCropTTA
import torch

# Load trained model
model = torch.load("outputs/baseline/run-*/best_model.pt")

# Flip averaging TTA
tta_model = TTAWrapper(model, merge_mode="mean")
predictions = tta_model(images, metadata)

# Multi-crop TTA
multi_crop = MultiCropTTA(model, crop_size=352, n_crops=5)
predictions = multi_crop(pil_image, metadata)
```

**Expected improvement:** +1-3% over baseline.

### Option 4: Cross-Validation
```bash
python -m competitions.csiro_biomass.src.train \
  --train-csv data/raw/csiro-biomass/train.csv \
  --image-dir data/raw/csiro-biomass \
  --output-dir outputs/cv \
  --n-folds 5 --cv-group-column State \
  --epochs 20 --batch-size 32
```

## Ablation Study

```bash
python -m competitions.csiro_biomass.src.ablation_runner \
  --train-csv data/raw/csiro-biomass/train.csv \
  --image-dir data/raw/csiro-biomass \
  --test-csv data/raw/csiro-biomass/test.csv \
  --sample-submission data/raw/csiro-biomass/sample_submission.csv \
  --output-dir outputs/ablations \
  --epochs 15 --batch-size 32
```

This runs:
- `image_only`: No metadata, EMA, or curriculum
- `no_ema`: Metadata + curriculum, no EMA
- `no_curriculum`: Metadata + EMA, no curriculum

## MCTS Hyperparameter Search

```python
from pathlib import Path
from competitions.csiro_biomass.src.config.experiment import ExperimentConfig
from competitions.csiro_biomass.src.search import SimpleMCTS

seed_cfg = ExperimentConfig(
    train_csv=Path('data/raw/csiro-biomass/train.csv'),
    image_dir=Path('data/raw/csiro-biomass'),
)

mcts = SimpleMCTS(seed=42)
best_cfg = mcts.run_search(seed_cfg, rounds=3, probe_epochs=3)
mcts.export(Path('outputs/search_history.json'))
print('Best config:', best_cfg.to_dict())
```

## Export for Production

### ONNX Export
```bash
python -m competitions.csiro_biomass.src.export \
  --run-dir outputs/baseline/run-20250101-123000 \
  --output artifacts/model.onnx \
  --format onnx --opset 17
```

### TorchScript Export
```bash
python -m competitions.csiro_biomass.src.export \
  --run-dir outputs/baseline/run-20250101-123000 \
  --output artifacts/model.pt \
  --format torchscript
```

## Recommended Workflow

1. **Start with Perceiver fusion**
   ```bash
   python -m competitions.csiro_biomass.src.train --fusion-type perceiver ...
   ```

2. **Run ablation study**
   ```bash
   python -m competitions.csiro_biomass.src.ablation_runner ...
   ```

3. **Use TTA for inference**
   ```python
   tta_model = TTAWrapper(model)
   ```

4. **Run cross-validation**
   ```bash
   python -m competitions.csiro_biomass.src.train --n-folds 5 --cv-group-column State
   ```

5. **MCTS search for optimal hyperparameters**
   ```python
   mcts.run_search(seed_cfg, rounds=5)
   ```

## Expected Performance

| Method | Expected RMSE | Notes |
|--------|--------------|-------|
| Baseline (EfficientNet-B3) | ~X.XX | Original |
| + Perceiver Fusion | ~X.XX | +3-7% improvement |
| + Fractal Curriculum | ~X.XX | +2-5% improvement |
| + Snapshot Ensembling | ~X.XX | +2-4% improvement |
| + EMA | ~X.XX | +1-3% improvement |
| + TTA | ~X.XX | +1-3% improvement |
| + Constraint Post-Processing | ~X.XX | +1-2% improvement |

## Tips for Best Results

1. **Use Perceiver fusion** - Multi-modal learning (image + metadata)
2. **Enable fractal curriculum** - Difficulty-based sampling
3. **Use snapshot ensembling** - Top-K checkpoint averaging
4. **Enable EMA** - Exponential moving average
5. **Apply TTA** - Flip averaging, multi-crop
6. **Use constraint post-processing** - Biomass composition checks
7. **Run MCTS search** - Optimal hyperparameters

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 16`
- Use smaller model: `--model efficientnet_b0`
- Reduce image size: `--image-size 224`

### Poor Performance
- Enable all boosters (Perceiver, curriculum, EMA, snapshots)
- Increase epochs: `--epochs 50`
- Use TTA for inference
- Run cross-validation to check robustness

### Slow Training
- Use smaller model: `--model resnet34`
- Reduce batch size with gradient accumulation
- Disable curriculum: `--no-curriculum`

## Advanced Usage

### Custom Augmentation Policy
```bash
python -m competitions.csiro_biomass.src.train \
  --augmentation-policy randaugment \
  --randaugment-magnitude 10 \
  --randaugment-num-ops 2
```

### Weights & Biases Tracking
```bash
python -m competitions.csiro_biomass.src.train \
  --wandb-project biomass \
  --wandb-entity your-team \
  --wandb-run-name perceiver-experiment
```

### Leaderboard Comparison
```bash
python -m competitions.csiro_biomass.src.utils.leaderboard \
  --competition csiro-biomass \
  --team "Your Team Name" \
  --metrics-file outputs/baseline/run-*/run_summary.json
```

## Support

- See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed feature documentation
- See [README.md](README.md) for repository structure
- Check notebooks for EDA examples

Good luck with the competition! 🌱🔬🚀
