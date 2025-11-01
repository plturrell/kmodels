# PhysioNet ECG Image Digitization

Local workspace scaffold for the [PhysioNet ECG Image Digitization](https://www.kaggle.com/competitions/physionet-ecg-image-digitization) Kaggle competition.

## Getting Started

1. **Join the competition**  
   Sign in to Kaggle, join the competition, and accept the terms so the API can access the files.

2. **Configure Kaggle credentials**  
   - Install the CLI: `pip install kaggle`  
   - Create an API token via Kaggle → Account → *Create New Token* and place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json` (use `chmod 600` on Unix-like systems).

3. **Download the data**  
   From the repository root run:
   ```bash
   python -m competitions.physionet_ecg_image_digitization.src.data.download --extract
   ```
   Files land in `data/raw`. Use `--file <name>` to pull specific assets.

4. **Check leaderboard standing**  
   After submitting to Kaggle, sync the public leaderboard with:
   ```bash
   python -m competitions.physionet_ecg_image_digitization.src.utils.leaderboard \
       --metrics-file outputs/latest_metrics.json
   ```
   The script fetches the current leaderboard, highlights your team's position (if available), and juxtaposes it with a local metric file.

## Repository Layout

- `data/` – raw downloads (`raw/`) and processed intermediates (`processed/`).
- `notebooks/` – exploratory analysis notebooks or lightweight scripts.
- `outputs/` – generated artefacts such as submissions or evaluation reports.
- `src/` – reusable Python modules:
  - `data/` – download helpers and dataset access utilities.
  - `features/` – feature engineering logic (add your transforms here).
  - `modeling/` – model architectures and inference helpers.
  - `training/` – training loops, cross-validation harnesses, schedulers.
  - `utils/` – auxiliary helpers (leaderboard sync, submissions, etc.).

Adjust or extend this scaffold as experiments evolve.

## Recent Improvements

This workspace has been significantly enhanced with 14 major improvements:

### 🚀 Core Enhancements
- **Submission Generation**: Complete test inference pipeline with TTA support
- **Signal Post-Processing**: Baseline correction, smoothing, wavelet denoising
- **Advanced Loss Functions**: Smooth L1, perceptual, frequency domain losses
- **Cross-Validation**: K-fold CV with stratification by lead type
- **Requirements**: Comprehensive dependency list

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed documentation.

## Quick Start Examples

### Generate Submission with TTA
```bash
python -m src.utils.submission \
  --checkpoint outputs/run/checkpoints/best.ckpt \
  --test-dir data/test_images \
  --output submission.csv \
  --use-tta
```

### Train with Advanced Loss
```python
from src.modeling.losses import CombinedLoss

criterion = CombinedLoss(
    mse_weight=1.0,
    mae_weight=0.5,
    frequency_weight=0.3,
)
```

### Post-Process Predictions
```python
from src.features.signal_processing import smooth_signal, remove_baseline_wander

pred_clean = remove_baseline_wander(pred, sampling_rate=500)
pred_smooth = smooth_signal(pred_clean, method="savgol")
```

### Multi-Lead Smoke Test

Use the lightweight synthetic dataset below to verify that the multi-channel
pipeline (data loading ➜ Lightning training ➜ cross-validation) works end to end:

```bash
# 1) Create a handful of synthetic image/signal pairs (writes to data/…/train_*)
python3 - <<'PY'
from pathlib import Path
import csv
import numpy as np
from PIL import Image

root = Path('data/physionet_ecg_image_digitization/raw/physionet-ecg-image-digitization')
image_dir = root / 'train_images'
signal_dir = root / 'train_signals'
for subdir in [image_dir / 'lead_I', image_dir / 'lead_II', signal_dir]:
    subdir.mkdir(parents=True, exist_ok=True)

samples = [
    ('sample_001', 'lead_I/sample_001.png'),
    ('sample_002', 'lead_II/sample_002.png'),
    ('sample_003', 'lead_I/sample_003.png'),
]

for ecg_id, rel_path in samples:
    img_path = image_dir / rel_path
    img_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.linspace(0, 255, num=64*64*3, dtype=np.float32) % 255).astype(np.uint8).reshape(64, 64, 3)
    Image.fromarray(arr).save(img_path)

    signal = np.stack([
        np.sin(np.linspace(0, 3.14, 100)) + 0.1,
        np.cos(np.linspace(0, 3.14, 100)) - 0.1,
    ])
    np.save(signal_dir / f'{ecg_id}.npy', signal.astype('float32'))

csv_path = root / 'train.csv'
with csv_path.open('w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=['ecg_id', 'image', 'signal', 'lead'])
    writer.writeheader()
    for ecg_id, rel_path in samples:
        writer.writerow({
            'ecg_id': ecg_id,
            'image': rel_path,
            'signal': f'{ecg_id}.npy',
            'lead': 'I' if 'lead_I' in rel_path else 'II',
        })
print(f'Created sample dataset at {root}')
PY

# 2) Run a minimal Lightning training loop on CPU
PYTHONPATH=physionet_ecg_image_digitization \
python3 -m src.training.baseline \
  --train-csv data/physionet_ecg_image_digitization/raw/physionet-ecg-image-digitization/train.csv \
  --image-dir data/physionet_ecg_image_digitization/raw/physionet-ecg-image-digitization/train_images \
  --signal-dir data/physionet_ecg_image_digitization/raw/physionet-ecg-image-digitization/train_signals \
  --epochs 1 --batch-size 2 --num-workers 0 --limit-val-batches 1 --max-train-steps 2 \
  --accelerator cpu --devices 1 --precision 32 --output-dir outputs/smoke_run --preload-signals

# 3) Exercise the cross-validation harness against the same fixtures
PYTHONPATH=physionet_ecg_image_digitization python3 - <<'PY'
from pathlib import Path

from src.config.training import AugmentationConfig, OptimizerConfig
from src.data.dataset import load_samples_from_metadata
from src.modeling.baseline import BaselineModelConfig, build_baseline_model
from src.training.cross_validation import run_cross_validation

root = Path('data/physionet_ecg_image_digitization/raw/physionet-ecg-image-digitization')
samples = load_samples_from_metadata(
    root / 'train.csv',
    image_root=root / 'train_images',
    signal_root=root / 'train_signals',
)

def factory(signal_length: int, signal_channels: int):
    cfg = BaselineModelConfig(
        signal_length=signal_length,
        signal_channels=signal_channels,
        pretrained=False,
        hidden_dim=32,
    )
    return build_baseline_model(cfg)

metrics = run_cross_validation(
    samples,
    factory,
    n_folds=2,
    output_dir=Path('outputs/smoke_cv'),
    epochs=1,
    batch_size=2,
    num_workers=0,
    optimizer_cfg=OptimizerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        use_scheduler=False,
        scheduler_t_max=1,
        warmup_epochs=0,
        gradient_clip_val=None,
    ),
    augmentation=AugmentationConfig(image_size=64, random_rotation=0.0),
    preload_signals=True,
    accelerator='cpu',
    devices=1,
    precision='32',
)

print(metrics)
PY
```

## Baseline tooling

- `python -m physionet_ecg_image_digitization.notebooks.eda_overview`
  Prints a quick summary of discovered leads, sample counts, and waveform lengths.
- `python -m competitions.physionet_ecg_image_digitization.src.training.baseline \
    --train-csv data/raw/physionet-ecg-image-digitization/train.csv \
    --image-dir data/raw/physionet-ecg-image-digitization/train_images \
    --signal-dir data/raw/physionet-ecg-image-digitization/train_signals \
    --output-dir competitions/physionet_ecg_image_digitization/outputs/lightning_baseline`
  Trains the Lightning baseline (CNN → waveform regression) and writes checkpoints, configs, and metrics to a timestamped run directory under `outputs/lightning_baseline/`.
