"""
Robustness evaluation framework for baseline models.

Implements perturbation analysis and robustness metrics per protocol:
- Gaussian noise (σ ∈ {0.01, 0.05, 0.1})
- Feature dropout (missingness)
- Temporal delay (lag)
- FGSM adversarial (for deep models)

Computes Area under Degradation curve (AUC-D) as robustness metric.
"""

import numpy as np
from typing import Dict, Tuple, List, Callable, Any
import logging

logger = logging.getLogger(__name__)


class PerturbationFactory:
    """Factory for creating perturbation functions."""

    @staticmethod
    def gaussian_noise(X: np.ndarray, sigma: float) -> np.ndarray:
        """
        Add Gaussian noise to observations.
        
        Args:
            X: Observations (n_samples, timesteps, n_features)
            sigma: Noise standard deviation
        
        Returns:
            X_noisy: Perturbed observations
        """
        noise = np.random.normal(0, sigma, X.shape)
        X_noisy = X + noise
        return X_noisy

    @staticmethod
    def feature_dropout(X: np.ndarray, dropout_rate: float) -> np.ndarray:
        """
        Simulate missing features (dropout).
        
        Args:
            X: Observations (n_samples, timesteps, n_features)
            dropout_rate: Fraction of values to replace with MEDIAN
        
        Returns:
            X_dropped: Observations with missing features
        """
        X_dropped = X.copy()
        mask = np.random.random(X.shape) < dropout_rate
        
        # Replace with feature median
        for feat_idx in range(X.shape[-1]):
            feat_median = np.nanmedian(X[:, :, feat_idx])
            X_dropped[:, :, feat_idx][mask[:, :, feat_idx]] = feat_median
        
        return X_dropped

    @staticmethod
    def temporal_delay(X: np.ndarray, lag_steps: int) -> np.ndarray:
        """
        Introduce temporal delay (lag) in observations.
        
        Shifts each feature back in time by lag_steps.
        
        Args:
            X: Observations (n_samples, timesteps, n_features)
            lag_steps: Number of timesteps to shift
        
        Returns:
            X_delayed: Delayed observations
        """
        X_delayed = X.copy()
        
        if lag_steps > 0:
            # Shift backward: future values become current
            for i in range(X.shape[-1]):
                X_delayed[:, :-lag_steps, i] = X[:, lag_steps:, i]
                # Pad with forward-fill
                X_delayed[:, -lag_steps:, i] = X[:, -1:, i]
        
        return X_delayed

    @staticmethod
    def fgsm_adversarial(
        X: np.ndarray,
        model_grad_fn: Callable,
        epsilon: float = 0.01,
    ) -> np.ndarray:
        """
        FGSM (Fast Gradient Sign Method) adversarial perturbation.
        
        For deep models only.
        
        Args:
            X: Input observations (n_samples, ...)
            model_grad_fn: Function that computes gradient w.r.t. input
            epsilon: Perturbation magnitude
        
        Returns:
            X_adv: Adversarial examples
        """
        gradients = model_grad_fn(X)
        X_adv = X + epsilon * np.sign(gradients)
        return X_adv


class RobustnessEvaluator:
    """Evaluate model robustness under perturbations."""

    def __init__(self, X_test: np.ndarray, y_test: np.ndarray):
        """Initialize with test data."""
        self.X_test = X_test
        self.y_test = y_test
        self.perturbations: Dict[str, List[Tuple[float, np.ndarray]]] = {}

    def apply_perturbations(self) -> Dict[str, List[Tuple[float, np.ndarray]]]:
        """
        Apply all perturbation types per protocol.
        
        Returns:
            Dictionary:
            {
                'gaussian_noise': [(sigma_value, X_perturbed), ...],
                'feature_dropout': [(dropout_rate, X_perturbed), ...],
                'temporal_delay': [(lag_steps, X_perturbed), ...],
            }
        """
        perturbations = {}
        
        # =====================================================================
        # Gaussian Noise: σ ∈ {0.01, 0.05, 0.1}
        # =====================================================================
        logger.info("Applying Gaussian noise perturbations...")
        perturbations["gaussian_noise"] = []
        
        for sigma in [0.01, 0.05, 0.1]:
            X_noisy = PerturbationFactory.gaussian_noise(self.X_test, sigma)
            perturbations["gaussian_noise"].append((sigma, X_noisy))
            logger.info(f"  [OK] Gaussian noise sigma={sigma:.3f}")
        
        # =====================================================================
        # Feature Dropout: rate ∈ {0.1, 0.2, 0.3}
        # =====================================================================
        logger.info("Applying feature dropout perturbations...")
        perturbations["feature_dropout"] = []
        
        for dropout_rate in [0.1, 0.2, 0.3]:
            X_dropped = PerturbationFactory.feature_dropout(self.X_test, dropout_rate)
            perturbations["feature_dropout"].append((dropout_rate, X_dropped))
            logger.info(f"  [OK] Feature dropout rate={dropout_rate:.1%}")
        
        # =====================================================================
        # Temporal Delay: lag_steps ∈ {1, 2, 5}
        # =====================================================================
        logger.info("Applying temporal delay perturbations...")
        perturbations["temporal_delay"] = []
        
        max_lag = min(5, self.X_test.shape[1] // 2)
        for lag_steps in [1, min(2, max_lag), min(5, max_lag)]:
            X_delayed = PerturbationFactory.temporal_delay(self.X_test, lag_steps)
            perturbations["temporal_delay"].append((lag_steps, X_delayed))
            logger.info(f"  [OK] Temporal delay lag={lag_steps} steps")
        
        self.perturbations = perturbations
        return perturbations

    @staticmethod
    def compute_degradation(
        metric_clean: float,
        metric_perturbed: float,
    ) -> float:
        """
        Compute degradation percentage.
        
        Degradation = (metric_clean - metric_perturbed) / metric_clean * 100%
        
        Args:
            metric_clean: Metric on clean data
            metric_perturbed: Metric on perturbed data
        
        Returns:
            Degradation percentage (higher = worse)
        """
        if abs(metric_clean) < 1e-10:
            return 0.0
        
        degradation = (metric_clean - metric_perturbed) / abs(metric_clean)
        return float(np.clip(degradation * 100, -100, 100))

    @staticmethod
    def compute_auc_degradation(
        perturbation_intensities: List[float],
        degradation_values: List[float],
    ) -> float:
        """
        Compute Area Under Degradation curve (AUC-D).
        
        x-axis: perturbation intensity
        y-axis: degradation %
        
        Lower AUC-D = more robust (better)
        
        Args:
            perturbation_intensities: Sorted intensity values
            degradation_values: Corresponding degradations
        
        Returns:
            AUC_D: float >= 0
        """
        if len(perturbation_intensities) < 2:
            return 0.0
        
        # Sort by intensity
        sorted_pairs = sorted(zip(perturbation_intensities, degradation_values))
        intensities = np.array([p[0] for p in sorted_pairs])
        degradations = np.array([p[1] for p in sorted_pairs])
        
        # Trapezoidal integration
        auc_d = float(np.trapz(degradations, intensities))
        
        return max(auc_d, 0.0)  # AUC should be non-negative


class RobustnessAnalyzer:
    """Analyze and report robustness across perturbation types."""

    def __init__(self):
        """Initialize analyzer."""
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate_model_robustness(
        self,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        predict_fn: Callable,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        metric_name: str = "accuracy",
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model robustness under all perturbation types.
        
        Args:
            model_name: Name of model being evaluated
            X_test: Test observations
            y_test: Test labels
            predict_fn: Function that returns predictions (n_samples,)
            metric_fn: Function that computes metric from y_test, y_pred
            metric_name: Name of metric for logging
        
        Returns:
            Dictionary with robustness results:
            {
                'gaussian_noise': {'auc_d': float, 'degradations': [...]},
                'feature_dropout': {...},
                'temporal_delay': {...},
            }
        """
        logger.info(f"\nEvaluating robustness for {model_name}")
        logger.info("="*60)
        
        evaluator = RobustnessEvaluator(X_test, y_test)
        perturbations = evaluator.apply_perturbations()
        
        # Baseline metric (clean data)
        y_pred_clean = predict_fn(X_test)
        metric_clean = metric_fn(y_test, y_pred_clean)
        logger.info(f"Clean {metric_name}: {metric_clean:.4f}")
        
        robustness_results = {}
        
        for perturb_type, perturb_list in perturbations.items():
            logger.info(f"\n{perturb_type}:")
            
            intensities = []
            degradations = []
            
            for intensity, X_perturbed in perturb_list:
                # Predict on perturbed data
                y_pred_perturbed = predict_fn(X_perturbed)
                metric_perturbed = metric_fn(y_test, y_pred_perturbed)
                
                # Compute degradation
                degradation = RobustnessEvaluator.compute_degradation(
                    metric_clean, metric_perturbed
                )
                
                intensities.append(intensity)
                degradations.append(degradation)
                
                logger.info(
                    f"  Intensity {intensity:.3f}: {metric_name}={metric_perturbed:.4f}, "
                    f"degradation={degradation:.1f}%"
                )
            
            # Compute AUC-D
            auc_d = RobustnessEvaluator.compute_auc_degradation(intensities, degradations)
            
            robustness_results[perturb_type] = {
                "auc_d": auc_d,
                "intensities": intensities,
                "degradations": degradations,
                "metric_clean": metric_clean,
            }
            
            logger.info(f"  AUC-D: {auc_d:.2f}")
        
        logger.info("="*60)
        self.results[model_name] = robustness_results
        
        return robustness_results

    def summarize_robustness(self) -> Dict[str, float]:
        """
        Summarize robustness across all models.
        
        Returns:
            Dictionary with average AUC-D per perturbation type.
        """
        summary = {}
        
        for perturb_type in ["gaussian_noise", "feature_dropout", "temporal_delay"]:
            auc_d_values = []
            
            for model_name, robustness_data in self.results.items():
                if perturb_type in robustness_data:
                    auc_d_values.append(robustness_data[perturb_type]["auc_d"])
            
            if auc_d_values:
                summary[f"{perturb_type}_mean_auc_d"] = float(np.mean(auc_d_values))
                summary[f"{perturb_type}_std_auc_d"] = float(np.std(auc_d_values))
        
        return summary
