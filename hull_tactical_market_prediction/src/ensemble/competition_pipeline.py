"""Competition submission pipeline with metadata and performance tracking."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .adaptive_ensemble import AdaptiveEnsemble
from .performance_monitor import PerformanceMonitor


class CompetitionAdaptor:
    """Adapt ensemble for competition timeline and leaderboard feedback."""
    
    def __init__(
        self,
        initial_ensemble: AdaptiveEnsemble,
        output_dir: Path,
        ensemble_metadata: Optional[Dict] = None,
    ):
        """Initialize competition adaptor.
        
        Args:
            initial_ensemble: Initial adaptive ensemble instance
            output_dir: Directory for outputs
            ensemble_metadata: Metadata about ensemble configuration
        """
        self.ensemble = initial_ensemble
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ensemble_metadata = ensemble_metadata or {}
        
        self.submission_count = 0
        self.leaderboard_scores: List[Dict] = []
        self.monitor = PerformanceMonitor(output_dir=self.output_dir)
    
    def create_submission(
        self,
        predictions: pd.DataFrame,
        model_metadata: Optional[Dict] = None,
        current_regime: Optional[str] = None,
        performance_report: Optional[str] = None,
    ) -> Dict[Path, str]:
        """Create comprehensive competition submission package.
        
        Args:
            predictions: DataFrame with predictions (must have date_id and prediction columns)
            model_metadata: Additional model metadata
            current_regime: Current market regime
            performance_report: Pre-generated performance report (optional)
            
        Returns:
            Dictionary mapping file paths to descriptions
        """
        self.submission_count += 1
        timestamp = datetime.now().isoformat()
        
        # 1. Save predictions (main submission file)
        submission_path = self.output_dir / "submission.csv"
        predictions.to_csv(submission_path, index=False)
        
        # 2. Generate and save performance report
        self.monitor.log_ensemble_state(self.ensemble, current_regime)
        if performance_report is None:
            performance_report = self.monitor.generate_performance_report()
        
        report_path = self.monitor.save_report(f"performance_report_{self.submission_count}.md")
        
        # 3. Save model metadata
        metadata = {
            'ensemble_type': 'adaptive_regime_aware',
            'submission_number': self.submission_count,
            'timestamp': timestamp,
            'base_models': len(self.ensemble.models),
            'blend_factor': self.ensemble_metadata.get('blend_factor', '70% meta-model, 30% adaptive'),
            'regime_detection': True,
            'adaptive_weights': True,
            'lookback_window': self.ensemble_metadata.get('lookback_window', 63),
            'current_regime': current_regime,
            'model_weights': self.ensemble.calculate_adaptive_weights(current_regime),
            **(model_metadata or {}),
        }
        
        metadata_path = self.output_dir / f"model_metadata_{self.submission_count}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 4. Save regime specialization analysis
        specialization = self.monitor.analyze_regime_specialization(self.ensemble)
        if specialization:
            spec_path = self.output_dir / f"regime_specialization_{self.submission_count}.json"
            with open(spec_path, 'w') as f:
                json.dump(specialization, f, indent=2)
        
        # 5. Create submission summary
        summary_path = self.output_dir / f"submission_summary_{self.submission_count}.md"
        summary = self._generate_submission_summary(
            submission_path,
            metadata_path,
            report_path,
            specialization,
            timestamp
        )
        summary_path.write_text(summary)
        
        return {
            submission_path: "Main submission file (predictions)",
            metadata_path: "Model metadata and configuration",
            report_path: "Performance analysis report",
            summary_path: "Submission summary",
        }
    
    def _generate_submission_summary(
        self,
        submission_path: Path,
        metadata_path: Path,
        report_path: Path,
        specialization: Dict[str, str],
        timestamp: str,
    ) -> str:
        """Generate submission summary markdown."""
        weights = self.ensemble.calculate_adaptive_weights()
        
        lines = [
            f"# Submission #{self.submission_count}",
            "",
            f"**Generated**: {timestamp}",
            "",
            "## Files Generated",
            "",
            f"- **Predictions**: `{submission_path.name}`",
            f"- **Metadata**: `{metadata_path.name}`",
            f"- **Performance Report**: `{report_path.name}`",
            "",
            "## Ensemble Configuration",
            "",
        ]
        
        lines.append(f"- **Base Models**: {len(self.ensemble.models)}")
        lines.append(f"- **Lookback Window**: {self.ensemble_metadata.get('lookback_window', 63)} days")
        lines.append(f"- **Blend Factor**: {self.ensemble_metadata.get('blend_factor', '70/30')}")
        lines.append("")
        
        lines.extend([
            "## Current Model Weights",
            "",
        ])
        for model_id, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{model_id}**: {weight:.1%}")
        
        if specialization:
            lines.extend([
                "",
                "## Model Regime Specialization",
                "",
            ])
            for model_id, regime in specialization.items():
                lines.append(f"- **{model_id}**: Best in `{regime}` regime")
        
        return "\n".join(lines)
    
    def update_based_on_leaderboard(
        self,
        public_score: float,
        private_score: Optional[float] = None,
    ) -> Dict:
        """Adjust ensemble strategy based on competition feedback.
        
        Args:
            public_score: Public leaderboard score
            private_score: Private leaderboard score (if available)
            
        Returns:
            Dictionary with adjustment recommendations
        """
        self.leaderboard_scores.append({
            'submission': self.submission_count,
            'public_score': public_score,
            'private_score': private_score,
            'timestamp': datetime.now().isoformat(),
        })
        
        recommendations = {
            'status': 'stable',
            'recommendations': [],
        }
        
        # Analyze trend
        if len(self.leaderboard_scores) > 1:
            recent_trend = (
                self.leaderboard_scores[-1]['public_score'] -
                self.leaderboard_scores[-2]['public_score']
            )
            
            if recent_trend > 0:  # Score getting worse (higher RMSE)
                recommendations['status'] = 'degrading'
                recommendations['recommendations'].append(
                    "Consider increasing adaptive blend ratio"
                )
                recommendations['recommendations'].append(
                    "Review regime detection thresholds"
                )
            elif recent_trend < 0:  # Score improving (lower RMSE)
                recommendations['status'] = 'improving'
                recommendations['recommendations'].append(
                    "Current strategy is working well"
                )
        
        # Save leaderboard history
        leaderboard_path = self.output_dir / "leaderboard_history.json"
        with open(leaderboard_path, 'w') as f:
            json.dump(self.leaderboard_scores, f, indent=2)
        
        return recommendations

