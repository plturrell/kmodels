#!/usr/bin/env python
"""Train neural network baseline for CAFA 6 protein function prediction.

This script trains a deep learning model using ESM-2 protein embeddings
and evaluates it using CAFA metrics.

Example usage:
    # Quick test (small subset)
    python train_neural.py --max_samples 1000 --num_epochs 10
    
    # Full training
    python train_neural.py --num_epochs 50 --batch_size 32
    
    # With GO hierarchy
    python train_neural.py --use_go_hierarchy --num_epochs 50
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from src.data import (
    build_samples,
    load_go_terms_long_format,
    load_sequences_from_fasta,
    train_val_split,
)
from src.data.go_ontology import parse_obo_file
from src.modeling.neural_baseline import train_neural_baseline
from src.utils.cafa_metrics import evaluate_cafa_metrics
from src.utils.submission_generator import create_submission_from_classifier
from src.features.fractal_features import FractalProteinFeatures, combine_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train neural baseline for CAFA 6")
    
    # Data paths
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/raw/cafa-6-protein-function-prediction/Train"),
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/neural_baseline"),
        help="Output directory for model and results",
    )
    
    # Data options
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)",
    )
    parser.add_argument(
        "--min_go_terms",
        type=int,
        default=3,
        help="Minimum GO terms per protein",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Validation set fraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    # Model options
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="ESM-2 model name",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate",
    )
    parser.add_argument(
        "--use_fractal_features",
        action="store_true",
        help="Add fractal features to embeddings (expected +5-10%% Fmax)",
    )
    parser.add_argument(
        "--fractal_max_iter",
        type=int,
        default=100,
        help="Maximum iterations for fractal computation",
    )

    # Training options
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    
    # GO hierarchy
    parser.add_argument(
        "--use_go_hierarchy",
        action="store_true",
        help="Use GO hierarchy for label propagation",
    )
    parser.add_argument(
        "--go_obo_path",
        type=Path,
        default=Path("data/raw/cafa-6-protein-function-prediction/go-basic.obo"),
        help="Path to GO OBO file",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(args.output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    
    LOGGER.info("=" * 80)
    LOGGER.info("CAFA 6 Neural Network Baseline Training")
    LOGGER.info("=" * 80)
    
    # Load data
    LOGGER.info("Loading data...")
    fasta_path = args.data_dir / "train_sequences.fasta"
    terms_path = args.data_dir / "train_terms.tsv"
    
    sequences = load_sequences_from_fasta(fasta_path)
    annotations = load_go_terms_long_format(terms_path)
    all_samples = build_samples(sequences, annotations)
    
    LOGGER.info(f"Loaded {len(all_samples)} samples")
    
    # Filter samples
    filtered_samples = [s for s in all_samples if len(s.go_terms) >= args.min_go_terms]
    LOGGER.info(f"Filtered to {len(filtered_samples)} samples with >= {args.min_go_terms} GO terms")
    
    # Limit samples if requested
    if args.max_samples:
        filtered_samples = filtered_samples[:args.max_samples]
        LOGGER.info(f"Limited to {len(filtered_samples)} samples for testing")
    
    # Load GO hierarchy if requested
    ontology = None
    if args.use_go_hierarchy and args.go_obo_path.exists():
        LOGGER.info("Loading GO ontology...")
        ontology = parse_obo_file(args.go_obo_path)

        # Propagate annotations up the hierarchy
        LOGGER.info("Propagating GO annotations up the hierarchy...")
        from dataclasses import replace
        for i, sample in enumerate(filtered_samples):
            propagated = ontology.propagate_annotations(set(sample.go_terms))
            # Update sample (need to recreate since it's frozen)
            filtered_samples[i] = replace(sample, go_terms=tuple(sorted(propagated)))
    
    # Split data
    LOGGER.info("Splitting data...")
    train_samples, val_samples = train_val_split(
        filtered_samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    
    LOGGER.info(f"Training samples: {len(train_samples)}")
    LOGGER.info(f"Validation samples: {len(val_samples)}")
    
    # Train model
    LOGGER.info("\nTraining neural network...")
    if args.use_fractal_features:
        LOGGER.info("🌟 Using fractal features (expected +5-10% Fmax improvement)")

    results = train_neural_baseline(
        train_samples=train_samples,
        val_samples=val_samples,
        model_name=args.model_name,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        output_dir=args.output_dir,
        use_fractal_features=args.use_fractal_features,
        fractal_max_iter=args.fractal_max_iter,
    )
    
    LOGGER.info(f"\nBest validation loss: {results['best_val_loss']:.4f}")
    LOGGER.info(f"Number of GO terms: {results['num_labels']}")
    
    # Evaluate with CAFA metrics
    LOGGER.info("\nEvaluating with CAFA metrics...")
    
    # Generate predictions for validation set
    model = results['model']
    mlb = results['mlb']
    device = next(model.parameters()).device
    
    # Get embeddings and predictions
    from src.features.embeddings import embed_sequences
    val_embeddings = embed_sequences(val_samples, model_name=args.model_name, device=str(device), return_array=True)
    
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(val_embeddings).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Convert to CAFA format
    predictions = {}
    for idx, sample in enumerate(val_samples):
        predictions[sample.accession] = {}
        for term_idx, go_term in enumerate(mlb.classes_):
            predictions[sample.accession][go_term] = float(probs[idx, term_idx])
    
    ground_truth = {s.accession: set(s.go_terms) for s in val_samples}
    
    # Calculate CAFA metrics
    cafa_metrics = evaluate_cafa_metrics(predictions, ground_truth, ontology)
    
    # Save metrics
    with open(args.output_dir / "cafa_metrics.json", "w") as f:
        json.dump(cafa_metrics, f, indent=2)
    
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Training Complete!")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Model saved to: {args.output_dir}")
    LOGGER.info(f"Fmax: {cafa_metrics['fmax']:.4f}")
    LOGGER.info(f"Coverage: {cafa_metrics['coverage']:.4f}")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()

