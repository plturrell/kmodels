#!/usr/bin/env python
"""
Benchmark script to validate fractal features improvement claims.

Usage:
    python benchmark_fractal.py --max_samples 5000 --n_folds 5
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.data import (
    build_samples,
    load_go_terms_long_format,
    load_sequences_from_fasta,
)
from src.features.embeddings import embed_sequences
from src.features.fractal_features import FractalProteinFeatures, combine_features
from src.features.fractal_benchmark import FractalFeatureBenchmark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark fractal features")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/raw/cafa-6-protein-function-prediction/Train"),
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/fractal_benchmark"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum samples for benchmark (for speed)",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="ESM-2 model name",
    )
    parser.add_argument(
        "--fractal_max_iter",
        type=int,
        default=100,
        help="Maximum iterations for fractal computation",
    )
    
    args = parser.parse_args()
    
    LOGGER.info("=" * 80)
    LOGGER.info("Fractal Features Benchmark")
    LOGGER.info("=" * 80)
    
    # Load data
    LOGGER.info("Loading data...")
    fasta_path = args.data_dir / "train_sequences.fasta"
    terms_path = args.data_dir / "train_terms.tsv"
    
    sequences = load_sequences_from_fasta(fasta_path)
    annotations = load_go_terms_long_format(terms_path)
    all_samples = build_samples(sequences, annotations)
    
    # Filter and limit
    filtered_samples = [s for s in all_samples if len(s.go_terms) >= 3]
    if args.max_samples:
        filtered_samples = filtered_samples[:args.max_samples]
    
    LOGGER.info(f"Using {len(filtered_samples)} samples for benchmark")
    
    # Prepare labels
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform([s.go_terms for s in filtered_samples])
    LOGGER.info(f"Number of GO terms: {len(mlb.classes_)}")
    
    # Generate base embeddings
    LOGGER.info("Generating ESM-2 embeddings...")
    base_embeddings = embed_sequences(
        filtered_samples,
        model_name=args.model_name,
        batch_size=8,
        return_array=True,
    )
    LOGGER.info(f"Base embedding shape: {base_embeddings.shape}")
    
    # Generate fractal features
    LOGGER.info("Generating fractal features...")
    fractal_extractor = FractalProteinFeatures(max_iter=args.fractal_max_iter)
    fractal_features = fractal_extractor.extract_batch(base_embeddings)
    LOGGER.info(f"Fractal features shape: {fractal_features.shape}")
    
    # Run benchmark
    LOGGER.info("\nRunning ablation study...")
    benchmark = FractalFeatureBenchmark(output_dir=args.output_dir)
    
    def classifier_factory():
        return OneVsRestClassifier(
            LogisticRegression(max_iter=200, solver="lbfgs"),
            n_jobs=1,  # CV already parallelizes
        )
    
    results = benchmark.run_ablation_study(
        samples=filtered_samples,
        base_embeddings=base_embeddings,
        fractal_features=fractal_features,
        labels=labels,
        classifier_factory=classifier_factory,
        n_folds=args.n_folds,
    )
    
    # Save results
    benchmark.save_results()
    
    # Print summary
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("BENCHMARK RESULTS")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Baseline F1:        {results['baseline']['mean']:.4f} ± {results['baseline']['std']:.4f}")
    LOGGER.info(f"With Fractal F1:    {results['with_fractal']['mean']:.4f} ± {results['with_fractal']['std']:.4f}")
    LOGGER.info(f"Fractal Only F1:    {results['fractal_only']['mean']:.4f} ± {results['fractal_only']['std']:.4f}")
    LOGGER.info(f"Improvement:        {results['improvement']['absolute']:.4f} ({results['improvement']['percentage']:.2f}%)")
    LOGGER.info(f"Statistical Sig:    p={results['significance']['p_value']:.4f} (significant={results['significance']['significant']})")
    LOGGER.info("=" * 80)
    
    # Validate claim
    claimed_improvement = 5.0  # Minimum claimed improvement
    actual_improvement = results['improvement']['percentage']
    if actual_improvement >= claimed_improvement and results['significance']['significant']:
        LOGGER.info("✓ Fractal features claim VALIDATED")
    else:
        LOGGER.warning("✗ Fractal features claim NOT validated")
        LOGGER.warning(f"  Expected: ≥{claimed_improvement}% improvement")
        LOGGER.warning(f"  Actual: {actual_improvement:.2f}% improvement")


if __name__ == "__main__":
    main()

