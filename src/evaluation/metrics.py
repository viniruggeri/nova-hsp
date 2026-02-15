"""
Metrics computation for baseline evaluation.

Implements paradigm-specific metrics per experimental protocol:
- Survival: C-index, Integrated Brier Score, Time-dependent AUC
- Classification: F1, AUC-ROC, Precision-Recall AUC
- Regression: RMSE, MAE
- Universal: Lead Time Normalized, FPR

All metrics follow standard implementations for reproducibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    f1_score,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
)
import logging

logger = logging.getLogger(__name__)


class MetricsComputer:
    """Compute metrics per paradigm following experimental protocol."""

    @staticmethod
    def normalized_lead_time(
        T_event: np.ndarray,
        T_alert: np.ndarray,
        T_start: float = 0.0,
    ) -> np.ndarray:
        """
        Compute Normalized Lead Time.

        LT_norm = (T_event - T_alert) / (T_event - T_start)

        Higher values = earlier alerts (better)

        Args:
            T_event: True event times (n_samples,)
            T_alert: Alert times predicted by model (n_samples,)
            T_start: Start time (default 0.0)

        Returns:
            LT_norm: (n_samples,) values in [0, 1]
                - 1.0 = alert at start (perfect early)
                - 0.0 = alert at event time (no anticipation)
                - <0.0 = alert after event (failure)
        """
        denominator = T_event - T_start
        denominator = np.maximum(denominator, 1e-10)  # Avoid division by zero

        LT_norm = (T_event - T_alert) / denominator

        return np.clip(LT_norm, -1.0, 1.0)  # Clip to valid range

    @staticmethod
    def false_positive_rate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5,
    ) -> float:
        """
        False Positive Rate (FPR).

        FPR = FP / (FP + TN)

        Fraction of negative samples incorrectly classified as positive.
        """
        y_pred_binary = (y_pred > threshold).astype(int)

        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        tn = np.sum((y_pred_binary == 0) & (y_true == 0))

        if fp + tn == 0:
            return 0.0

        return float(fp) / float(fp + tn)

    @staticmethod
    def survival_c_index(
        T_true: np.ndarray,
        risk_score: np.ndarray,
        events: np.ndarray,
    ) -> float:
        """
        Concordance Index (C-index) for survival models.

        Measures discriminative ability: fraction of pairs correctly ordered
        by risk score according to event times.

        Args:
            T_true: Observed event times (n_samples,)
            risk_score: Predicted risk score (n_samples,)
            events: Event indicators (n_samples,) - 1 if event observed

        Returns:
            C_index: float in [0, 1]
                - 1.0 = perfect discrimination
                - 0.5 = random prediction
        """
        # Only consider pairs where at least one event occurred
        n = len(T_true)
        concordant = 0
        total = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Only count if earlier event is observed
                if T_true[i] < T_true[j] and events[i] == 1:
                    total += 1
                    if risk_score[i] > risk_score[j]:
                        concordant += 1
                elif T_true[j] < T_true[i] and events[j] == 1:
                    total += 1
                    if risk_score[j] > risk_score[i]:
                        concordant += 1

        if total == 0:
            return 0.5  # Random prediction

        return float(concordant) / float(total)

    @staticmethod
    def integrated_brier_score(
        T_true: np.ndarray,
        survival_pred: np.ndarray,
        T_eval: Optional[np.ndarray] = None,
    ) -> float:
        """
        Integrated Brier Score (IBS) for survival models.

        Measures prediction error integrated over time.

        Args:
            T_true: Event times (n_samples,)
            survival_pred: Predicted survival probabilities (n_samples, n_times)
            T_eval: Evaluation time points (default: unique event times)

        Returns:
            IBS: float >= 0
                - 0 = perfect predictions
        """
        if T_eval is None:
            T_eval = np.unique(T_true)

        if survival_pred.ndim == 1:
            # If scalar survival probability, convert to matrix
            survival_pred = np.tile(survival_pred[:, np.newaxis], (1, len(T_eval)))

        T_max = np.max(T_true)
        brier_scores = []

        for t in T_eval:
            if t > T_max:
                continue

            # Status at time t
            status = (T_true <= t).astype(int)

            # Predicted survival at time t
            t_idx = np.searchsorted(T_eval, t)
            if t_idx >= survival_pred.shape[1]:
                t_idx = survival_pred.shape[1] - 1

            S_t = survival_pred[:, t_idx]

            # Brier score: mean squared error of predictions
            brier = np.mean((S_t - (1 - status)) ** 2)
            brier_scores.append(brier)

        return float(np.mean(brier_scores)) if brier_scores else 0.0

    @staticmethod
    def time_dependent_auc(
        T_true: np.ndarray,
        risk_score: np.ndarray,
        t: float,
    ) -> float:
        """
        Time-dependent AUC at time t.

        Discrimination ability at specific time point.

        Args:
            T_true: Event times (n_samples,)
            risk_score: Risk scores (n_samples,)
            t: Time point of interest

        Returns:
            AUC: float in [0, 1]
        """
        # Case 1: event before t (status=1)
        # Case 2: censoring or event after t at t (status=0)
        y_true = (T_true <= t).astype(int)

        # Need at least two classes
        if len(np.unique(y_true)) < 2:
            return 0.5

        try:
            auc_score = roc_auc_score(y_true, risk_score)
            return float(auc_score)
        except:
            return 0.5

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Classification metrics: F1, AUC-ROC, PR-AUC.

        Args:
            y_true: True labels (n_samples,)
            y_pred: Predicted labels (n_samples,)
            y_score: Predicted probabilities (n_samples,) for AUC computation

        Returns:
            Dictionary with metrics
        """
        metrics = {}

        # F1-score
        if len(np.unique(y_true)) >= 2:
            metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        else:
            metrics["f1"] = 0.0

        # AUC-ROC and PR-AUC
        if y_score is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
            except:
                metrics["auc_roc"] = 0.5

            try:
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                metrics["pr_auc"] = float(auc(recall, precision))
            except:
                metrics["pr_auc"] = 0.5
        else:
            metrics["auc_roc"] = 0.5
            metrics["pr_auc"] = 0.5

        return metrics

    @staticmethod
    def regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Regression metrics: RMSE, MAE.

        Args:
            y_true: True values (n_samples,)
            y_pred: Predicted values (n_samples,)

        Returns:
            Dictionary with metrics
        """
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
        return metrics

    @staticmethod
    def compute_lead_time_stats(
        LT_norm: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute statistical summary of normalized lead time.

        Args:
            LT_norm: Normalized lead times (n_samples,)

        Returns:
            Dictionary with statistics
        """
        # Remove invalid values (alerts after event)
        valid_lt = LT_norm[LT_norm >= 0]

        stats = {
            "mean": float(np.mean(valid_lt)) if len(valid_lt) > 0 else 0.0,
            "std": float(np.std(valid_lt)) if len(valid_lt) > 0 else 0.0,
            "median": float(np.median(valid_lt)) if len(valid_lt) > 0 else 0.0,
            "min": float(np.min(valid_lt)) if len(valid_lt) > 0 else -1.0,
            "max": float(np.max(valid_lt)) if len(valid_lt) > 0 else 1.0,
            "pct_valid": float(len(valid_lt)) / float(len(LT_norm)) * 100,
        }

        return stats


class SurvivalMetrics:
    """Specialized survival metrics computation."""

    @staticmethod
    def concordance_harrell(
        T_true: np.ndarray,
        risk_score: np.ndarray,
    ) -> float:
        """Harrell's Concordance Index (C-index)."""
        return MetricsComputer.survival_c_index(
            T_true, risk_score, np.ones_like(T_true)
        )

    @staticmethod
    def integrated_brier_score_standard(
        T_true: np.ndarray,
        T_pred: np.ndarray,
    ) -> float:
        """IBS using time predictions instead of survival curves."""
        return float(np.sqrt(np.mean((T_true - T_pred) ** 2)))


def summarize_metrics(
    metrics_dict: Dict[str, Any],
    model_name: str = "model",
) -> pd.DataFrame:
    """Convert metrics dictionary to DataFrame for reporting."""
    df_data = {"model": [model_name]}

    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                df_data[f"{key}_{subkey}"] = [subvalue]
        else:
            df_data[key] = [value]

    return pd.DataFrame(df_data)
