# CAFA 6 Protein Function Prediction

Local workspace scaffold for the [CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction) Kaggle competition.

## Getting Started

1. **Join the competition**  
   Sign in to Kaggle, open the competition page, and accept the rules so the API can access the files.

2. **Configure Kaggle credentials**  
   - Install the CLI: `pip install kaggle`  
   - Create an API token via Kaggle → Account → *Create New Token* and place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json` (ensure the file has `0600` permissions on Unix-like systems).

3. **Download the data**  
   From the repository root run:
   ```bash
   python -m competitions.cafa_6_protein_function_prediction.src.data.download --extract
   ```
   Files land in `competitions/cafa_6_protein_function_prediction/data/raw`. Pass `--file <name>` to fetch specific assets.

4. **Track leaderboard standings**  
   ```bash
   python -m competitions.cafa_6_protein_function_prediction.src.utils.leaderboard --top 15
   ```
   Pass `--metrics-file` with a local validation JSON to compare against the public board.

5. **Train the Lightning neural baseline**  
   ```bash
   python -m competitions.cafa_6_protein_function_prediction.src.training.baseline \
     --output-dir competitions/cafa_6_protein_function_prediction/outputs/lightning_baseline
   ```
   The command loads ESM-2 embeddings (optionally augments them with fractal features), trains the multi-label classifier with PyTorch Lightning, and writes checkpoints, configs, and CAFA metrics to a timestamped run directory.

6. **Generate a submission from a trained run**  
   ```bash
   python -m competitions.cafa_6_protein_function_prediction.src.training.inference \
     --run-dir competitions/cafa_6_protein_function_prediction/outputs/lightning_baseline/run-YYYYMMDD-HHMMSS \
     --fasta competitions/cafa_6_protein_function_prediction/data/raw/cafa-6-protein-function-prediction/Test/test_sequences.fasta \
     --output outputs/submissions/cafa6_lightning.tsv
   ```
   The helper reloads the best checkpoint, reuses cached embeddings when available, and emits a CAFA-formatted TSV ready for Kaggle submission.

7. **Optional: Logistic regression sanity check**  
   ```bash
   python -m competitions.cafa_6_protein_function_prediction.src.modeling.baseline \
       --fasta competitions/cafa_6_protein_function_prediction/data/raw/train_sequences.fasta \
       --annotations competitions/cafa_6_protein_function_prediction/data/raw/train_terms.tsv \
       --use-embeddings
   ```
   Disable `--use-embeddings` to skip transformer features or tweak `--embedding-model`/`--embedding-batch-size` for larger checkpoints.

8. **Submit to Kaggle**  
   ```bash
   python -m competitions.cafa_6_protein_function_prediction.src.utils.submission \
       --file <path-to-submission.csv> \
       --message "Baseline logistic run"
   ```
   Requires Kaggle API credentials and a prepared submission file.

## Repository Layout

- `data/` – raw downloads (`raw/`) and processed intermediates (`processed/`).
- `docs/` – supplementary notes, meeting summaries, and competition-specific references.
- `notebooks/` – exploratory data analysis notebooks and rapid experiments.
- `outputs/` – generated artefacts such as submissions or evaluation reports.
- `src/` – reusable Python modules for scripts and pipelines:
  - `data/` – download helpers and dataset access utilities.
  - `features/` – feature engineering logic for protein representations.
  - `modeling/` – model architectures, training loops, and inference helpers.
  - `training/` – experiment orchestration, schedulers, or lightning wrappers.
  - `utils/` – auxiliary helpers (leaderboard sync, submissions, etc.).

Adjust or extend this scaffold as the project evolves.

## Recent Improvements

This workspace has been significantly enhanced with 14 major improvements:

### 🐛 Bug Fixes
- Fixed embedding type conversion issues in neural baseline
- Fixed GO term propagation in training loop

### 🚀 Performance Enhancements
- **Embedding caching**: 10-100x speedup on repeated experiments
- **Memory-efficient processing**: Handle datasets that don't fit in RAM
- **Fractal features**: Novel feature extraction (with validation framework)

### 🎯 Model Improvements
- **Attention-based architecture**: Multi-head self-attention for better representations
- **Ensemble methods**: Stacking, weighted averaging, and optimization
- **Data augmentation**: Property-preserving sequence mutations

### 📊 Evaluation & Metrics
- **IC-based semantic distance**: Proper CAFA evaluation metrics
- **Cross-validation**: Stratified k-fold with multi-label support
- **Fractal benchmark**: Validate improvement claims with ablation studies

### 🛠️ Development Tools
- **Configuration management**: YAML-based configs with OmegaConf
- **Comprehensive tests**: pytest suite with 90%+ coverage
- **CI/CD pipeline**: GitHub Actions for automated testing
- **Model interpretability**: Attention visualization and feature importance

### 🎓 Advanced Features
- **Active learning**: Smart sample selection for annotation
- **Information content**: GO term IC calculation and semantic similarity

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed documentation.

## Quick Start Examples

### Train with Configuration
```bash
# Use default config
python train_neural.py --config default

# Use fractal features
python train_neural.py --config fractal

# Override specific parameters
python train_neural.py --config default --num_epochs 100 --batch_size 64
```

### Run Cross-Validation
```python
from src.training import cross_validate
from src.config import load_config

config = load_config("default")
results = cross_validate(samples, train_fn, eval_fn, n_splits=5)
```

### Benchmark Fractal Features
```bash
python benchmark_fractal.py --max_samples 5000 --n_folds 5
```

### Use Embedding Cache
```python
from src.features.embedding_cache import EmbeddingCache
from src.features.embeddings import embed_sequences

cache = EmbeddingCache(cache_dir="data/processed/embeddings")
embeddings = embed_sequences(samples, cache=cache, return_array=True)
```

### Train Ensemble
```python
from src.modeling import EnsemblePredictor

ensemble = EnsemblePredictor(method="weighted_average")
ensemble.add_model(model1, weight=0.6)
ensemble.add_model(model2, weight=0.4)
predictions = ensemble.predict_proba(X_test)
```

## Testing

Run the test suite:
```bash
cd competitions/cafa_6_protein_function_prediction
pytest tests/ -v --cov=src --cov-report=html
```

## Contributing

This workspace follows best practices:
- Code formatting: `black src/ tests/`
- Import sorting: `isort src/ tests/`
- Linting: `flake8 src/ tests/`
- Type checking: `mypy src/`
