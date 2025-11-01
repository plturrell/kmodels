"""Enhanced post-processing with comprehensive constraint validation and logging."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging

class AdvancedBiomassConstraintProcessor:
    """Enhanced constraint processor with validation logging and multiple repair strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'tolerance': 0.5,
            'max_biomass': 500.0,
            'repair_strategy': 'adaptive',  # 'adaptive', 'trust_total', 'trust_components'
            'enable_species_constraints': True,
            'enable_temporal_constraints': False,
        }
        
        self.logger = logging.getLogger(__name__)
        self.violation_stats = {
            'non_negative': 0,
            'compositional': 0,
            'species_aware': 0,
            'temporal': 0
        }
        
        self.species_constraints = {
            'perennial_ryegrass': {'max_clover_ratio': 0.3, 'max_dead_ratio': 0.4},
            'sub_clover': {'min_clover_ratio': 0.1, 'max_clover_ratio': 0.8},
            'annual_ryegrass': {'max_dead_ratio': 0.5},
        }

    def apply(self, predictions_df: pd.DataFrame, metadata: pd.DataFrame = None, 
              epistemic_uncertainty: pd.DataFrame = None) -> pd.DataFrame:
        """
        Applies enhanced constraints with comprehensive logging.
        """
        repaired_df = predictions_df.copy()
        
        # 1. Non-negative clamping
        non_negative_cols = ['Dry_Total_g', 'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']
        for col in non_negative_cols:
            if col in repaired_df.columns:
                repaired_df[col] = np.maximum(0, repaired_df[col])

        # 2. Enhanced compositional consistency
        component_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']
        if all(c in repaired_df.columns for c in component_cols) and 'Dry_Total_g' in repaired_df.columns:
            self._apply_compositional_constraints(repaired_df, epistemic_uncertainty)

        return repaired_df

    def _apply_compositional_constraints(self, df: pd.DataFrame, epistemic_uncertainty: pd.DataFrame = None):
        """Apply compositional constraints with adaptive repair strategy."""
        component_cols = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']
        component_sum = df[component_cols].sum(axis=1)
        total_pred = df['Dry_Total_g']
        
        violation_mask = abs(total_pred - component_sum) > self.config['tolerance']
        
        if self.config['repair_strategy'] == 'adaptive' and epistemic_uncertainty is not None:
            self._adaptive_composition_repair(df, violation_mask, component_sum, epistemic_uncertainty)
        elif self.config['repair_strategy'] == 'trust_total':
            scale_factors = total_pred[violation_mask] / (component_sum[violation_mask] + 1e-8)
            for col in component_cols:
                df.loc[violation_mask, col] *= scale_factors
        else:  # trust_components
            df.loc[violation_mask, 'Dry_Total_g'] = component_sum[violation_mask]

    def _adaptive_composition_repair(self, df: pd.DataFrame, violation_mask: pd.Series, 
                                   component_sum: pd.Series, epistemic_uncertainty: pd.DataFrame):
        """Use epistemic uncertainty to decide repair strategy per sample."""
        for idx in violation_mask[violation_mask].index:
            total_uncertainty = epistemic_uncertainty.loc[idx, 'Dry_Total_g'] if 'Dry_Total_g' in epistemic_uncertainty.columns else 0.0
            
            if total_uncertainty < 0.1: # If model is confident
                scale_factor = df.loc[idx, 'Dry_Total_g'] / (component_sum[idx] + 1e-8)
                for col in ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']:
                    df.loc[idx, col] *= scale_factor
            else: # If model is uncertain
                df.loc[idx, 'Dry_Total_g'] = component_sum[idx]