"""Advanced confidence-gated augmentation with risk detection and policy evaluation."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import albumentations as A
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

class AdvancedRiskySampleDetector:
    """Advanced risk detection using multiple signals and anomaly detection."""
    
    def __init__(self, config: Dict[str, Any] = None, typical_fractal_dimensions: Dict[str, float] = None):
        self.config = config or {
            'ndvi_threshold': 0.3,
            'species_rarity_threshold': 0.05,
            'error_threshold': 0.8,  # Relative to mean error
            'use_anomaly_detection': True,
            'contamination': 0.1,  # For isolation forest
        }
        
        self.isolation_forest = IsolationForest(
            contamination=self.config['contamination'],
            random_state=42
        )
        self.scaler = StandardScaler()
        self.fitted = False
        self.logger = logging.getLogger(__name__)

        if typical_fractal_dimensions:
            self.typical_fractal_dimensions = typical_fractal_dimensions
        else:
            self.typical_fractal_dimensions = {
                'perennial_ryegrass': 1.5,
                'sub_clover': 1.6,
                'annual_ryegrass': 1.45,
                'default': 1.5
            }

    def fit(self, predictions: np.ndarray, targets: np.ndarray, metadata: pd.DataFrame):
        """Fit risk detection models on validation data."""
        risk_features = self._extract_risk_features(predictions, targets, metadata)
        
        if len(risk_features) > 1:  # Need multiple samples for fitting
            self.scaler.fit(risk_features)
            scaled_features = self.scaler.transform(risk_features)
            self.isolation_forest.fit(scaled_features)
            self.fitted = True
            self.logger.info("Risky sample detector fitted successfully")
        else:
            self.logger.warning("Insufficient data to fit risky sample detector")

    def is_risky(self, metadata_row: pd.Series, 
                 prediction: Optional[np.ndarray] = None,
                 target: Optional[np.ndarray] = None) -> bool:
        """
        Enhanced risk detection using multiple signals.
        
        Args:
            metadata_row: Sample metadata
            prediction: Model prediction (optional)
            target: Ground truth target (optional)
            
        Returns:
            True if sample is determined to be risky
        """
        risks = []
        
        # 1. Traditional threshold-based risks
        if 'NDVI' in metadata_row:
            risks.append(metadata_row['NDVI'] < self.config['ndvi_threshold'])
        
        if 'species_frequency' in metadata_row:
            risks.append(metadata_row['species_frequency'] < self.config['species_rarity_threshold'])

        # Fractal-based risk
        if 'fractal_dimension' in metadata_row and 'species' in metadata_row:
            species = metadata_row['species']
            typical_fd = self.typical_fractal_dimensions.get(species, self.typical_fractal_dimensions['default'])
            current_fd = metadata_row['fractal_dimension']
            risks.append(abs(current_fd - typical_fd) > 0.1) # 0.1 is a configurable threshold

        
        # 2. Error-based risk (if prediction and target available)
        if prediction is not None and target is not None:
            error = np.abs(prediction - target).mean()
            if hasattr(self, 'mean_error'):
                relative_error = error / (self.mean_error + 1e-8)
                risks.append(relative_error > self.config['error_threshold'])
        
        # 3. Anomaly-based risk (if detector is fitted)
        if self.fitted and prediction is not None:
            risk_features = self._extract_risk_features(
                prediction.reshape(1, -1), 
                target.reshape(1, -1) if target is not None else np.ones((1, prediction.shape[0])),
                pd.DataFrame([metadata_row])
            )
            scaled_features = self.scaler.transform(risk_features)
            anomaly_score = self.isolation_forest.decision_function(scaled_features)[0]
            risks.append(anomaly_score < np.percentile(self.isolation_forest.decision_function(
                self.scaler.transform(self._extract_risk_features(
                    np.ones((1, prediction.shape[0])), 
                    np.ones((1, prediction.shape[0])),
                    pd.DataFrame([metadata_row])
                ))), 20))
        
        return any(risks)

    def detect_risky_batch(self, predictions: np.ndarray, metadata: pd.DataFrame,
                          targets: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect risky samples in a batch.
        
        Returns:
            Boolean array indicating risky samples
        """
        risky_mask = np.zeros(len(predictions), dtype=bool)
        
        for i in range(len(predictions)):
            metadata_row = metadata.iloc[i] if i < len(metadata) else pd.Series()
            pred = predictions[i] if i < len(predictions) else None
            target = targets[i] if targets is not None and i < len(targets) else None
            
            risky_mask[i] = self.is_risky(metadata_row, pred, target)
        
        risky_count = risky_mask.sum()
        if risky_count > 0:
            self.logger.info(f"Detected {risky_count}/{len(risky_mask)} risky samples")
            
        return risky_mask

    def _extract_risk_features(self, predictions: np.ndarray, targets: np.ndarray, 
                             metadata: pd.DataFrame) -> np.ndarray:
        """Extract multi-dimensional risk features."""
        errors = np.abs(predictions - targets)
        relative_errors = errors / (np.mean(errors, axis=0) + 1e-8)
        mean_relative_error = np.mean(relative_errors, axis=1)
        
        risk_features = []
        for idx in range(len(predictions)):
            features = []
            
            # Metadata-based features
            if idx < len(metadata):
                meta_row = metadata.iloc[idx]
                features.extend([
                    meta_row.get('NDVI', 0),
                    meta_row.get('species_rarity', 0),
                    meta_row.get('image_quality_score', 1.0),
                ])
            else:
                features.extend([0, 0, 1.0])
            
            # Error-based features
            features.extend([
                mean_relative_error[idx],
                np.mean(predictions[idx]) / (np.mean(predictions) + 1e-8),  # Prediction magnitude
            ])
            
            risk_features.append(features)
        
        return np.array(risk_features)

class AugmentationOracle:
    """Evaluates augmentation policies using validation performance."""
    
    def __init__(self, validation_data: tuple, config: Dict[str, Any] = None):
        self.validation_data = validation_data
        self.config = config or {
            'min_improvement': 0.02,
            'patience': 3,
            'evaluation_batch_size': 32,
        }
        
        self.failed_policies = set()
        self.policy_scores = {}
        self.logger = logging.getLogger(__name__)

    def create_augmentation_policies(self) -> Dict[str, A.Compose]:
        """Create targeted augmentation policies for different risk types."""
        return {
            'low_ndvi': A.Compose([
                A.HueSaturationValue(
                    hue_shift_limit=40, 
                    sat_shift_limit=50, 
                    val_shift_limit=30, 
                    p=0.8
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.4, 
                    contrast_limit=0.4, 
                    p=0.7
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            ], p=1.0),
            
            'rare_species': A.Compose([
                A.Rotate(limit=30, p=0.7),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.CoarseDropout(
                    max_holes=8, 
                    max_height=16, 
                    max_width=16, 
                    p=0.3
                ),
            ], p=1.0),
            
            'high_error': A.Compose([
                A.HueSaturationValue(
                    hue_shift_limit=20, 
                    sat_shift_limit=30, 
                    val_shift_limit=20, 
                    p=0.9
                ),
                A.Blur(blur_limit=3, p=0.4),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=1.0)
        }

    def evaluate_policy(self, model, policy: A.Compose, policy_name: str, 
                       risky_indices: np.ndarray) -> bool:
        """
        Evaluate if an augmentation policy improves validation performance.
        
        Returns:
            True if policy should be accepted
        """
        if policy_name in self.failed_policies:
            return False
        
        X_val, y_val = self.validation_data
        if len(risky_indices) == 0:
            return False
            
        # Select risky samples from validation set
        val_risky_indices = risky_indices[risky_indices < len(X_val)]
        if len(val_risky_indices) == 0:
            return False
            
        X_val_risky = X_val[val_risky_indices]
        y_val_risky = y_val[val_risky_indices]
        
        try:
            # Get original predictions
            with torch.no_grad():
                model.eval()
                pred_original = model(X_val_risky)
                if isinstance(pred_original, tuple):
                    pred_original = pred_original[0]  # Unpack if model returns (predictions, confidence)
                
            # Apply augmentation and get new predictions
            X_val_aug = self._apply_augmentation_batch(X_val_risky, policy)
            
            with torch.no_grad():
                pred_augmented = model(X_val_aug)
                if isinstance(pred_augmented, tuple):
                    pred_augmented = pred_augmented[0]
            
            # Calculate improvement
            mae_original = np.mean(np.abs(pred_original.cpu().numpy() - y_val_risky))
            mae_augmented = np.mean(np.abs(pred_augmented.cpu().numpy() - y_val_risky))
            improvement = mae_original - mae_augmented
            
            self.policy_scores[policy_name] = improvement
            
            if improvement >= self.config['min_improvement']:
                self.logger.info(f"Policy {policy_name} accepted with improvement: {improvement:.4f}")
                return True
            else:
                self.logger.info(f"Policy {policy_name} rejected with improvement: {improvement:.4f}")
                self.failed_policies.add(policy_name)
                return False
                
        except Exception as e:
            self.logger.warning(f"Error evaluating policy {policy_name}: {e}")
            self.failed_policies.add(policy_name)
            return False

    def _apply_augmentation_batch(self, images: np.ndarray, policy: A.Compose) -> np.ndarray:
        """Apply augmentation to a batch of images."""
        augmented = []
        for img in images:
            # Ensure image is in correct format for albumentations
            if img.shape[0] == 3:  # CHW to HWC
                img = img.transpose(1, 2, 0)
            augmented_img = policy(image=img)['image']
            if augmented_img.shape[2] == 3:  # HWC to CHW
                augmented_img = augmented_img.transpose(2, 0, 1)
            augmented.append(augmented_img)
        return np.array(augmented)

    def get_policy_scores(self) -> Dict[str, float]:
        """Get scores for all evaluated policies."""
        return self.policy_scores.copy()

# Maintain backward compatibility
class RiskySampleDetector(AdvancedRiskySampleDetector):
    """Compatibility wrapper for the original RiskySampleDetector interface."""
    
    def __init__(self, ndvi_threshold: float = 0.3, species_rarity_threshold: float = 0.05):
        super().__init__(config={
            'ndvi_threshold': ndvi_threshold,
            'species_rarity_threshold': species_rarity_threshold,
            'use_anomaly_detection': False,  # Disable advanced features for compatibility
        })