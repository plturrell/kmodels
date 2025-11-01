# RecoDAI LUC Scientific Image Forgery Detection

Local workspace scaffold for the [RecoDAI LUC Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection) Kaggle competition.

## Getting Started

1. **Join the competition**  
   Sign in to Kaggle, open the competition page, and accept the rules so the API can access the files.

2. **Configure Kaggle credentials**  
   - Install the CLI: `pip install kaggle`  
   - Create an API token via Kaggle → Account → *Create New Token* and place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json` (make sure the file has `0600` permissions on Unix-like systems).

3. **Download the data**  
   From the repository root run:
   ```bash
   python -m competitions.recodai_luc_scientific_image_forgery_detection.src.data.download --extract
   ```
   Files land in `data/raw/recodai-luc-scientific-image-forgery-detection`. Pass `--file filename` to grab specific assets.


4. **Inspect the dataset quickly**  
   ```bash
   python -m competitions.recodai_luc_scientific_image_forgery_detection.src.data.summary --max-samples-per-class 200
   ```
   A JSON snapshot with label counts, sample shapes, and mask metadata is written to `outputs/dataset_overview.json`. The companion notebook `notebooks/001_dataset_overview.ipynb` includes the same sanity checks in an interactive format.

5. **Train the Lightning baseline**  
   ```bash
   python -m competitions.recodai_luc_scientific_image_forgery_detection.src.train \
     --epochs 5 \
     --batch-size 4 \
     --output-dir outputs/baseline
   ```
   This wires the joint segmentation + classification model into PyTorch Lightning. Check `outputs/baseline/` for checkpoints, metrics, and configs.

6. **Extract tabular features**  
   ```bash
   python3 -m competitions.recodai_luc_scientific_image_forgery_detection.src.features.tabular --output outputs/features/tabular_features.csv
   ```
   Computes per-image color and edge statistics plus mask coverage for quick baselines.

7. **Train the tabular baseline**  
   ```bash
   python3 -m competitions.recodai_luc_scientific_image_forgery_detection.src.modeling.tabular_baseline --features-csv outputs/features/tabular_features.csv
   ```
   Runs a gradient-boosted classifier across StratifiedKFold splits. Outputs metrics under `outputs/tabular_baseline/`.

8. **Check leaderboard standings**  
   ```bash
   python3 -m competitions.recodai_luc_scientific_image_forgery_detection.src.utils.leaderboard_check --top 10
   ```
   Requires Kaggle API credentials and prints the top public ranks alongside your best submission gap.

## Recent Improvements

This workspace has been significantly enhanced with 14 major improvements:

### 🚀 Core Enhancements
- **Augmentations**: Albumentations pipeline with forgery-specific transforms (JPEG compression, noise, blur)
- **Pre-trained Encoders**: ResNet, EfficientNet, etc. via segmentation_models_pytorch
- **Advanced Losses**: Focal, Dice, Tversky, and combined losses
- **Cross-Validation**: K-fold CV for deep learning models
- **Ensemble Methods**: Model and fold ensembling with weighted averaging
- **TTA**: Test-time augmentation for robust predictions
- **Enhanced Submission**: Complete pipeline with TTA and ensemble support

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed documentation.

## Quick Start Examples

### Train with Pre-trained Encoder
```python
from src.modeling.pretrained import build_pretrained_model

model = build_pretrained_model(
    architecture="Unet",
    encoder_name="resnet34",
    encoder_weights="imagenet",
)
```

### Run Cross-Validation
```bash
python -m src.training.cross_validation --config default --n-folds 5
```

### Generate Submission with TTA
```bash
python -m src.utils.submission \
  --run-dir outputs/baseline/run-latest \
  --output submission.csv \
  --use-tta
```

## Repository Structure
- `src/data/dataset.py` - PyTorch Dataset with stratified splitting
- `src/data/transforms.py` - **NEW**: Albumentations augmentation pipeline
- `src/modeling/baseline.py` - Joint segmentation + classification network
- `src/modeling/pretrained.py` - **NEW**: Pre-trained encoder models
- `src/modeling/losses.py` - **NEW**: Advanced loss functions
- `src/modeling/ensemble.py` - **NEW**: Model ensembling
- `src/training/cross_validation.py` - **NEW**: K-fold CV framework
- `src/train.py` - Lightning training orchestration
- `src/features/tabular.py` - Tabular feature extraction
- `src/modeling/tabular_baseline.py` - Gradient boosting baseline
- `src/utils/submission.py` - **ENHANCED**: Submission generation with TTA
- `src/utils/metrics.py` - Dice/IoU + accuracy metrics
- `src/config/training.py` - Configuration dataclasses
- `docs/csiro_audit.md` - CSIRO biomass pipeline patterns
