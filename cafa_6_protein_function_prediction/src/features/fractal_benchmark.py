"""
Benchmark and ablation study for fractal features.

This module provides tools to validate the claimed +5-10% Fmax improvement
from fractal features through rigorous ablation studies.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer

from ..data import ProteinSample

logger = logging.getLogger(__name__)


class FractalFeatureBenchmark:
    """Benchmark fractal features against baselines."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize benchmark.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir or Path("outputs/fractal_benchmark")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, Dict] = {}
    
    def run_ablation_study(
        self,
        samples: Sequence[ProteinSample],
        base_embeddings: np.ndarray,
        fractal_features: np.ndarray,
        labels: np.ndarray,
        classifier_factory,
        n_folds: int = 5,
    ) -> Dict[str, float]:
        """Run ablation study comparing different feature combinations.
        
        Args:
            samples: Protein samples
            base_embeddings: Base embeddings (e.g., ESM-2)
            fractal_features: Fractal features
            labels: Multi-label binary matrix
            classifier_factory: Function that returns a new classifier instance
            n_folds: Number of cross-validation folds
        
        Returns:
            Dictionary with results for each configuration
        """
        logger.info("Running ablation study with %d-fold CV", n_folds)
        
        results = {}
        
        # 1. Baseline: Only base embeddings
        logger.info("Testing baseline (base embeddings only)...")
        baseline_scores = cross_val_score(
            classifier_factory(),
            base_embeddings,
            labels,
            cv=n_folds,
            scoring='f1_micro',
            n_jobs=-1,
        )
        results['baseline'] = {
            'mean': float(np.mean(baseline_scores)),
            'std': float(np.std(baseline_scores)),
            'scores': baseline_scores.tolist(),
        }
        logger.info("Baseline F1: %.4f ± %.4f", results['baseline']['mean'], results['baseline']['std'])
        
        # 2. With fractal features
        logger.info("Testing with fractal features...")
        combined = np.concatenate([base_embeddings, fractal_features], axis=1)
        fractal_scores = cross_val_score(
            classifier_factory(),
            combined,
            labels,
            cv=n_folds,
            scoring='f1_micro',
            n_jobs=-1,
        )
        results['with_fractal'] = {
            'mean': float(np.mean(fractal_scores)),
            'std': float(np.std(fractal_scores)),
            'scores': fractal_scores.tolist(),
        }
        logger.info("With fractal F1: %.4f ± %.4f", results['with_fractal']['mean'], results['with_fractal']['std'])
        
        # 3. Calculate improvement
        improvement = results['with_fractal']['mean'] - results['baseline']['mean']
        improvement_pct = (improvement / results['baseline']['mean']) * 100
        results['improvement'] = {
            'absolute': float(improvement),
            'percentage': float(improvement_pct),
        }
        logger.info("Improvement: %.4f (%.2f%%)", improvement, improvement_pct)
        
        # 4. Statistical significance test (paired t-test)
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(fractal_scores, baseline_scores)
        results['significance'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
        }
        logger.info("Statistical significance: p=%.4f (significant=%s)", p_value, p_value < 0.05)
        
        # 5. Only fractal features (to check if they're informative alone)
        logger.info("Testing fractal features only...")
        fractal_only_scores = cross_val_score(
            classifier_factory(),
            fractal_features,
            labels,
            cv=n_folds,
            scoring='f1_micro',
            n_jobs=-1,
        )
        results['fractal_only'] = {
            'mean': float(np.mean(fractal_only_scores)),
            'std': float(np.std(fractal_only_scores)),
            'scores': fractal_only_scores.tolist(),
        }
        logger.info("Fractal only F1: %.4f ± %.4f", results['fractal_only']['mean'], results['fractal_only']['std'])
        
        self.results['ablation'] = results
        return results
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        import json
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info("Saved benchmark results to %s", output_path)


__all__ = [
    "FractalFeatureBenchmark",
]

