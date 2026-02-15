"""
Linear Threshold Heuristic for early warning detection.

Simple rule-based method: weighted sum of features triggers alert when
threshold is crossed persistently.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class LinearThresholdHeuristic(BaseEstimator):
    """
    Linear threshold heuristic for collapse detection.

    Computes: score(t) = w^T * x(t)
    Alerts when: score(t) < threshold for k_steps consecutive steps

    Usage:
        model = LinearThresholdHeuristic(threshold=0.5, k_steps=3)
        model.fit(X_train, T_train)  # learns weights
        t_alert = model.predict_alert_time(X_test)
    """

    def __init__(
        self,
        weights: np.ndarray = None,
        threshold: float = 0.5,
        k_steps: int = 3,
        normalize: bool = True,
    ):
        """
        Args:
            weights: Feature weights (if None, learned from data)
            threshold: Alert threshold for normalized score
            k_steps: Required persistence (consecutive steps below threshold)
            normalize: Whether to standardize features
        """
        self.weights = weights
        self.threshold = threshold
        self.k_steps = k_steps
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, T: np.ndarray = None):
        """
        Fit heuristic by learning weights from data.

        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            T: Time-to-event for each sample (optional, for supervised weighting)

        Returns:
            self
        """
        if X.ndim == 3:
            # Flatten sequences for weight estimation
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X

        # Fit scaler
        if self.normalize:
            self.scaler.fit(X_flat)

        # Learn weights if not provided
        if self.weights is None:
            if T is not None:
                # Supervised: correlate features with time-to-event
                self.weights = self._learn_weights_supervised(X, T)
            else:
                # Unsupervised: use variance as proxy for importance
                self.weights = self._learn_weights_unsupervised(X_flat)

        self.is_fitted_ = True
        logger.info(
            f"Linear Threshold fitted: weights shape={self.weights.shape}, "
            f"threshold={self.threshold}, k_steps={self.k_steps}"
        )

        return self

    def _learn_weights_supervised(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Learn weights by correlating features with time-to-event."""
        # Aggregate each sequence to single vector (mean over time)
        if X.ndim == 3:
            X_agg = X.mean(axis=1)  # (n_samples, n_features)
        else:
            X_agg = X

        # Compute correlation with T
        correlations = np.array(
            [np.corrcoef(X_agg[:, i], T)[0, 1] for i in range(X_agg.shape[1])]
        )

        # Handle NaN (constant features)
        correlations = np.nan_to_num(correlations, 0.0)

        # Use absolute correlation as weights
        weights = np.abs(correlations)

        # Normalize to unit norm
        weights = weights / (np.linalg.norm(weights) + 1e-10)

        return weights

    def _learn_weights_unsupervised(self, X: np.ndarray) -> np.ndarray:
        """Learn weights from variance (more varying = more important)."""
        variances = np.var(X, axis=0)
        weights = variances / (variances.sum() + 1e-10)
        return weights

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute alert score for sequence.

        Args:
            X: Single sequence (seq_len, n_features) or batch (n_samples, seq_len, n_features)

        Returns:
            scores: Alert score at each timestep (higher = safer)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before scoring")

        # Handle both single and batch inputs
        is_single = X.ndim == 2
        if is_single:
            X = X[np.newaxis, :, :]  # Add batch dimension

        # Normalize if needed
        if self.normalize:
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_flat)
            X = X_scaled.reshape(original_shape)

        # Compute weighted score: score(t) = w^T * x(t)
        scores = np.einsum("ijk,k->ij", X, self.weights)  # (n_samples, seq_len)

        if is_single:
            scores = scores[0]  # Remove batch dimension

        return scores

    def predict_alert_time(self, X: np.ndarray) -> int:
        """
        Predict time of alert (first time score drops below threshold persistently).

        Args:
            X: Single sequence (seq_len, n_features)

        Returns:
            t_alert: Timestep of alert, or seq_len if no alert
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        scores = self.score(X)

        # Find first time where score < threshold for k_steps consecutive steps
        for t in range(len(scores) - self.k_steps + 1):
            if all(scores[t : t + self.k_steps] < self.threshold):
                return t

        # No alert triggered
        return len(scores)

    def predict_alert_time_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict alert times for batch of sequences.

        Args:
            X: Batch of sequences (n_samples, seq_len, n_features)

        Returns:
            t_alerts: Alert times (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        scores = self.score(X)  # (n_samples, seq_len)

        alert_times = []
        for i in range(len(scores)):
            t_alert = self.predict_alert_time_from_scores(scores[i])
            alert_times.append(t_alert)

        return np.array(alert_times)

    def predict_alert_time_from_scores(self, scores: np.ndarray) -> int:
        """Helper to predict alert from pre-computed scores."""
        for t in range(len(scores) - self.k_steps + 1):
            if all(scores[t : t + self.k_steps] < self.threshold):
                return t
        return len(scores)
