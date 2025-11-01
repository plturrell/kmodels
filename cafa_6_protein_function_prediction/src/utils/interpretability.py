"""
Model interpretability tools for protein function prediction.

Provides attention visualization and feature importance analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LOGGER = logging.getLogger(__name__)


class AttentionVisualizer:
    """Visualize attention weights from attention-based models."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir or Path("outputs/interpretability")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        protein_id: str,
        sequence: Optional[str] = None,
        top_k: int = 50,
    ) -> Path:
        """Plot attention weights as heatmap.
        
        Args:
            attention_weights: Attention weight matrix
            protein_id: Protein identifier
            sequence: Optional protein sequence for labels
            top_k: Show only top_k positions if sequence is long
        
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Limit to top_k if sequence is long
        if attention_weights.shape[0] > top_k:
            # Get top_k positions by average attention
            avg_attention = attention_weights.mean(axis=1)
            top_indices = np.argsort(avg_attention)[-top_k:]
            attention_weights = attention_weights[top_indices][:, top_indices]
        
        # Plot heatmap
        sns.heatmap(
            attention_weights,
            cmap='viridis',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'},
        )
        
        ax.set_title(f'Attention Weights: {protein_id}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Position')
        
        # Save figure
        output_path = self.output_dir / f"attention_{protein_id}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        LOGGER.info(f"Saved attention heatmap to {output_path}")
        return output_path
    
    def plot_attention_distribution(
        self,
        attention_weights: np.ndarray,
        protein_id: str,
    ) -> Path:
        """Plot distribution of attention weights.
        
        Args:
            attention_weights: Attention weight matrix
            protein_id: Protein identifier
        
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Flatten attention weights
        flat_weights = attention_weights.flatten()
        
        # Histogram
        axes[0].hist(flat_weights, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Attention Weight')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Attention Weight Distribution')
        axes[0].grid(alpha=0.3)
        
        # Box plot by position
        position_weights = attention_weights.mean(axis=1)
        axes[1].boxplot([position_weights])
        axes[1].set_ylabel('Average Attention Weight')
        axes[1].set_title('Attention by Position')
        axes[1].grid(alpha=0.3)
        
        plt.suptitle(f'Attention Analysis: {protein_id}')
        
        # Save figure
        output_path = self.output_dir / f"attention_dist_{protein_id}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        LOGGER.info(f"Saved attention distribution to {output_path}")
        return output_path


class FeatureImportanceAnalyzer:
    """Analyze feature importance for predictions."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize analyzer.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir or Path("outputs/interpretability")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_permutation_importance(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
    ) -> Dict[str, float]:
        """Calculate permutation feature importance.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            feature_names: Optional feature names
            n_repeats: Number of permutation repeats
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        from sklearn.inspection import permutation_importance
        
        LOGGER.info("Calculating permutation importance...")
        
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
        )
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        importance_dict = {
            name: float(importance)
            for name, importance in zip(feature_names, result.importances_mean)
        }
        
        # Sort by importance
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True,
        ))
        
        return importance_dict
    
    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_k: int = 20,
    ) -> Path:
        """Plot feature importance.
        
        Args:
            importance_dict: Dictionary of feature importances
            top_k: Number of top features to show
        
        Returns:
            Path to saved figure
        """
        # Get top k features
        items = list(importance_dict.items())[:top_k]
        features, importances = zip(*items)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_k} Feature Importances')
        ax.grid(axis='x', alpha=0.3)
        
        # Save figure
        output_path = self.output_dir / "feature_importance.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        LOGGER.info(f"Saved feature importance plot to {output_path}")
        return output_path


__all__ = [
    "AttentionVisualizer",
    "FeatureImportanceAnalyzer",
]

