"""
Core metrics for baseline evaluation - Sprint 1.

Focused on Lead Time as primary metric.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_lead_time(
    T_event: np.ndarray,
    T_alert: np.ndarray,
    T_start: float = 0.0
) -> np.ndarray:
    """
    Compute Normalized Lead Time.
    
    LT_norm = (T_event - T_alert) / (T_event - T_start)
    
    Interpretation:
    - LT > 0: Alert BEFORE event (good!)
    - LT = 0: Alert AT event time
    - LT < 0: Alert AFTER event (failure)
    - LT = 1: Alert at start (perfect early warning)
    
    Args:
        T_event: True event times (n_samples,)
        T_alert: Predicted alert times (n_samples,)
        T_start: Start time (default 0.0)
    
    Returns:
        lead_times: Normalized lead time for each sample (n_samples,)
    """
    T_event = np.asarray(T_event)
    T_alert = np.asarray(T_alert)
    
    # Avoid division by zero
    denominator = np.maximum(T_event - T_start, 1e-10)
    
    lead_times = (T_event - T_alert) / denominator
    
    return lead_times


def compute_lead_time_stats(lead_times: np.ndarray) -> Dict[str, float]:
    """
    Aggregate lead time statistics.
    
    Args:
        lead_times: Array of normalized lead times
    
    Returns:
        Dictionary with mean, std, median, positive_rate, etc.
    """
    lead_times = np.asarray(lead_times)
    
    return {
        'mean': float(np.mean(lead_times)),
        'std': float(np.std(lead_times)),
        'median': float(np.median(lead_times)),
        'min': float(np.min(lead_times)),
        'max': float(np.max(lead_times)),
        'q25': float(np.percentile(lead_times, 25)),
        'q75': float(np.percentile(lead_times, 75)),
        'positive_rate': float(np.mean(lead_times > 0)),  # fraction with early warning
        'perfect_rate': float(np.mean(lead_times > 0.5)),  # very early warnings
    }


def false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    False Positive Rate for binary predictions.
    
    FPR = FP / (FP + TN)
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities or scores
        threshold: Decision threshold
    
    Returns:
        FPR value
    """
    y_pred_binary = (y_pred > threshold).astype(int)
    
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    
    if fp + tn == 0:
        return 0.0
    
    return float(fp) / float(fp + tn)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE for regression."""
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE for regression."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class BaselineEvaluator:
    """
    Simple evaluator for baseline models.
    
    Handles different model types and computes Lead Time as primary metric.
    """
    
    def __init__(self, metric_name: str = 'lead_time'):
        self.metric_name = metric_name
        self.results = {}
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        T_test: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate a baseline model.
        
        Args:
            model: Fitted baseline model
            X_test: Test sequences (n_samples, seq_len, n_features)
            T_test: True time-to-event (n_samples,)
            model_name: Name for logging
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        try:
            # Predict based on model type
            if hasattr(model, 'predict_time_to_collapse'):
                # Markov Chain
                T_pred = model.predict_time_to_collapse(X_test)
            elif hasattr(model, 'predict_alert_time_batch'):
                # Linear Threshold
                T_pred = model.predict_alert_time_batch(X_test)
            elif hasattr(model, 'predict_median_time'):
                # Kaplan-Meier (returns scalar, broadcast to batch)
                median = model.predict_median_time()
                # X_test can be None for KM (doesn't use covariates)
                n_samples = len(T_test) if X_test is None else len(X_test)
                T_pred = np.full(n_samples, median)
            else:
                raise ValueError(f"Unknown model type for {model_name}")
            
            # Compute lead times
            lead_times = compute_lead_time(T_test, T_pred)
            stats = compute_lead_time_stats(lead_times)
            
            # Compute regression metrics
            mae = mean_absolute_error(T_test, T_pred)
            rmse = root_mean_squared_error(T_test, T_pred)
            
            results = {
                'model': model_name,
                'lead_time': stats,
                'mae': mae,
                'rmse': rmse,
                'n_samples': len(T_test),
            }
            
            logger.info(f"{model_name}: Lead Time = {stats['mean']:.3f} ± {stats['std']:.3f}, "
                       f"MAE = {mae:.3f}, RMSE = {rmse:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
            return {
                'model': model_name,
                'error': str(e),
                'n_samples': len(T_test),
            }
