"""Real-time performance monitoring for adaptive ensemble."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .adaptive_ensemble import AdaptiveEnsemble


class PerformanceMonitor:
    """Real-time monitoring of ensemble performance and adaptation."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize performance monitor.
        
        Args:
            output_dir: Directory to save monitoring reports (optional)
        """
        self.output_dir = output_dir
        self.weight_history: List[Dict] = []
        self.regime_history: List[Dict] = []
        self.sharpe_history: List[Dict] = []
        self.performance_metrics: List[Dict] = []
    
    def log_ensemble_state(
        self,
        adaptive_ensemble: AdaptiveEnsemble,
        current_regime: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log current ensemble state for analysis.
        
        Args:
            adaptive_ensemble: Adaptive ensemble instance
            current_regime: Current market regime
            timestamp: Timestamp for logging (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get current weights
        weights = adaptive_ensemble.calculate_adaptive_weights(current_regime=current_regime)
        
        # Get recent Sharpe ratios
        model_stats = adaptive_ensemble.get_model_stats()
        recent_sharpes = {
            row['model_id']: row['recent_sharpe']
            for _, row in model_stats.iterrows()
        }
        
        # Get other metrics
        metrics_dict = {}
        for _, row in model_stats.iterrows():
            metrics_dict[row['model_id']] = {
                'recent_sharpe': row['recent_sharpe'],
                'max_drawdown': row['max_drawdown'],
                'hit_rate': row['hit_rate'],
                'n_periods': row['n_periods'],
            }
        
        # Log weight history
        self.weight_history.append({
            'timestamp': timestamp.isoformat(),
            'weights': weights.copy(),
            'regime': current_regime,
            'sharpes': recent_sharpes.copy(),
            'metrics': metrics_dict.copy(),
        })
        
        # Log regime history
        self.regime_history.append({
            'timestamp': timestamp.isoformat(),
            'regime': current_regime,
        })
        
        # Log Sharpe history
        self.sharpe_history.append({
            'timestamp': timestamp.isoformat(),
            'sharpes': recent_sharpes.copy(),
        })
    
    def generate_performance_report(self) -> str:
        """Generate markdown performance report.
        
        Returns:
            Markdown-formatted performance report
        """
        if not self.weight_history:
            return "# Performance Report\n\nNo performance data yet. Start logging ensemble state.\n"
        
        latest = self.weight_history[-1]
        report_lines = [
            "# Ensemble Performance Report",
            "",
            f"**Latest Update**: {latest['timestamp']}",
            f"**Current Regime**: {latest.get('regime', 'Unknown')}",
            "",
            "## Model Weights & Performance",
            "",
        ]
        
        # Sort models by weight (descending)
        sorted_weights = sorted(
            latest['weights'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for model_id, weight in sorted_weights:
            sharpe = latest['sharpes'].get(model_id, 0.0)
            metrics = latest['metrics'].get(model_id, {})
            max_dd = metrics.get('max_drawdown', 0.0)
            hit_rate = metrics.get('hit_rate', 0.0)
            
            report_lines.append(
                f"- **{model_id}**: {weight:.1%} "
                f"(Sharpe: {sharpe:.3f}, Max DD: {max_dd:.4f}, Hit Rate: {hit_rate:.2%})"
            )
        
        # Add weight evolution summary
        if len(self.weight_history) > 1:
            report_lines.extend([
                "",
                "## Weight Evolution",
                "",
            ])
            
            # Calculate weight changes
            first_weights = self.weight_history[0]['weights']
            for model_id in first_weights.keys():
                initial = first_weights.get(model_id, 0.0)
                final = latest['weights'].get(model_id, 0.0)
                change = final - initial
                change_pct = change * 100
                
                if abs(change_pct) > 0.1:  # Only show significant changes
                    direction = "↑" if change > 0 else "↓"
                    report_lines.append(
                        f"- **{model_id}**: {initial:.1%} → {final:.1%} "
                        f"({direction}{abs(change_pct):.1f}%)"
                    )
        
        # Add regime distribution
        if self.regime_history:
            regime_counts = {}
            for entry in self.regime_history:
                regime = entry.get('regime', 'Unknown')
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            total_regimes = len(self.regime_history)
            if regime_counts:
                report_lines.extend([
                    "",
                    "## Regime Distribution",
                    "",
                ])
                for regime, count in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True):
                    pct = (count / total_regimes) * 100
                    report_lines.append(f"- **{regime}**: {count} ({pct:.1f}%)")
        
        return "\n".join(report_lines)
    
    def save_report(self, filename: str = "performance_report.md") -> Path:
        """Save performance report to file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved report
        """
        report = self.generate_performance_report()
        
        if self.output_dir:
            output_path = self.output_dir / filename
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(filename)
        
        output_path.write_text(report)
        return output_path
    
    def get_weight_dataframe(self) -> pd.DataFrame:
        """Get weight history as DataFrame for analysis.
        
        Returns:
            DataFrame with columns: timestamp, model_id, weight, regime, sharpe
        """
        if not self.weight_history:
            return pd.DataFrame()
        
        rows = []
        for entry in self.weight_history:
            timestamp = entry['timestamp']
            regime = entry.get('regime', 'Unknown')
            weights = entry['weights']
            sharpes = entry.get('sharpes', {})
            
            for model_id, weight in weights.items():
                rows.append({
                    'timestamp': pd.to_datetime(timestamp),
                    'model_id': model_id,
                    'weight': weight,
                    'regime': regime,
                    'sharpe': sharpes.get(model_id, 0.0),
                })
        
        return pd.DataFrame(rows)
    
    def analyze_regime_specialization(self, adaptive_ensemble: AdaptiveEnsemble) -> Dict[str, str]:
        """Analyze which models perform best in which regimes.
        
        Args:
            adaptive_ensemble: Adaptive ensemble instance
            
        Returns:
            Dictionary mapping model_id to best-performing regime
        """
        specialization = {}
        
        # Iterate directly over models
        for model_id, diag in adaptive_ensemble.models.items():
            if diag and diag.regime_correlation:
                # Find regime with highest average performance
                regime_scores = {}
                for regime, sharpe_list in diag.regime_correlation.items():
                    if sharpe_list:
                        regime_scores[regime] = sum(sharpe_list) / len(sharpe_list)
                
                if regime_scores:
                    best_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
                    specialization[model_id] = best_regime
        
        return specialization

