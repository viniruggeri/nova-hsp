"""
Early Warning Signals for critical transitions.

Detects critical slowing down (CSD) through:
- Increasing variance
- Increasing autocorrelation
- Flickering
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from scipy.stats import linregress
import logging
from typing import Any, Dict

from src.baseline.types import BaselineResult

logger = logging.getLogger(__name__)


class EarlyWarningSignals(BaseEstimator):
    """
    Early Warning Signals detector.

    Detects critical transitions through:
    - Rising variance (loss of resilience)
    - Rising autocorrelation (slower recovery)
    - Combined CSD score

    Usage:
        model = EarlyWarningSignals(window=10)
        model.fit(X_train, T_train)
        alert_times = model.predict_alert_time_batch(X_test)
    """

    def __init__(self, window: int = 10, lag: int = 1, threshold: float = 0.5):
        """
        Args:
            window: Rolling window size for statistics
            lag: Lag for autocorrelation
            threshold: CSD score threshold for alert
        """
        self.window = window
        self.lag = lag
        self.threshold = threshold
        self.is_fitted_ = False

    def variance(self, series: np.ndarray) -> np.ndarray:
        """
        Rolling variance.

        Args:
            series: Time series (T,)

        Returns:
            variances: Rolling variance (T-window+1,)
        """
        if len(series) < self.window:
            return np.array([np.var(series)])

        variances = pd.Series(series).rolling(self.window).var().values
        return variances[self.window - 1 :]  # Drop NaN

    def autocorrelation(self, series: np.ndarray, lag: int | None = None) -> np.ndarray:
        """
        Rolling autocorrelation at given lag.

        Args:
            series: Time series (T,)
            lag: Lag (default: self.lag)

        Returns:
            autocorrs: Rolling autocorrelation
        """
        if lag is None:
            lag = self.lag

        if len(series) < self.window:
            return np.array([self._compute_acf(series, lag)])

        autocorrs = []
        for i in range(self.window, len(series) + 1):
            window_data = series[i - self.window : i]
            acf = self._compute_acf(window_data, lag)
            autocorrs.append(acf)

        return np.array(autocorrs)

    def skewness(self, series: np.ndarray) -> np.ndarray:
        """
        Rolling skewness.

        Args:
            series: Time series (T,)

        Returns:
            skew values aligned with rolling windows.
        """
        if len(series) < self.window:
            std = np.std(series)
            if std < 1e-12:
                return np.array([0.0])
            z = (series - np.mean(series)) / std
            return np.array([float(np.mean(z**3))])

        values = pd.Series(series).rolling(self.window).skew().values
        values = values[self.window - 1 :]
        return np.nan_to_num(values, nan=0.0)

    def _compute_acf(self, series: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag."""
        if len(series) <= lag:
            return 0.0

        series = series - series.mean()
        c0 = np.dot(series, series) / len(series)

        if c0 == 0:
            return 0.0

        c_lag = np.dot(series[:-lag], series[lag:]) / len(series)
        return c_lag / c0

    def critical_slowing_down(self, series: np.ndarray) -> Dict[str, Any]:
        """
        Detect critical slowing down.

        Returns:
            dict with variance_trend, autocorr_trend, csd_score
        """
        var = self.variance(series)
        acf = self.autocorrelation(series)

        # Compute trends (positive slope = increasing)
        var_trend = self._compute_trend(var)
        acf_trend = self._compute_trend(acf)

        # CSD score: combination of both trends (normalized)
        csd_score = (var_trend + acf_trend) / 2.0

        return {
            "variance": var,
            "autocorrelation": acf,
            "variance_trend": var_trend,
            "autocorr_trend": acf_trend,
            "csd_score": csd_score,
        }

    def _compute_trend(self, series: np.ndarray) -> float:
        """
        Compute linear trend (slope).

        Positive = increasing, negative = decreasing
        """
        if len(series) < 2:
            return 0.0

        x = np.arange(len(series))
        try:
            slope, _, _, _, _ = linregress(x, series)
            return float(slope)
        except Exception:
            return 0.0

    def fit(self, X: np.ndarray, T: np.ndarray | None = None):
        """
        Fit EWS (calibrate threshold if needed).

        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            T: Time-to-event (optional, for calibration)
        """
        # EWS is primarily unsupervised, but we can calibrate threshold
        self.is_fitted_ = True
        logger.info(
            f"EWS fitted: window={self.window}, lag={self.lag}, threshold={self.threshold}"
        )
        return self

    def compute_ews_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute EWS score for each timestep.

        Args:
            X: Single sequence (seq_len, n_features)

        Returns:
            scores: CSD score at each timestep (seq_len,)
        """
        # Aggregate features (e.g., mean)
        if X.ndim == 2:
            series = X.mean(axis=1)  # (seq_len,)
        else:
            series = X

        # Compute CSD for rolling windows
        scores = np.zeros(len(series))

        for i in range(self.window, len(series) + 1):
            window_data = series[max(0, i - self.window) : i]
            csd = self.critical_slowing_down(window_data)
            scores[i - 1] = csd["csd_score"]

        return scores

    def predict_alert_time(self, X: np.ndarray) -> int:
        """
        Predict alert time when CSD score exceeds threshold.

        Args:
            X: Single sequence (seq_len, n_features)

        Returns:
            Alert timestep or seq_len if no alert
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        scores = self.compute_ews_score(X)

        # Find first time score exceeds threshold
        alert_indices = np.where(scores > self.threshold)[0]

        if len(alert_indices) > 0:
            return int(alert_indices[0])

        return len(scores)

    def predict_alert_time_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Predict alert times for batch.

        Args:
            X: Batch of sequences (n_samples, seq_len, n_features)

        Returns:
            alert_times: (n_samples,)
        """
        alert_times = []
        for i in range(len(X)):
            t_alert = self.predict_alert_time(X[i])
            alert_times.append(t_alert)
        return np.array(alert_times)

    def compute_indicator(
        self,
        trajectory: np.ndarray,
        indicator: str = "variance",
        step: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a full indicator time series for geometric benchmarking.

        Args:
            trajectory: Series (T,) or multivariate trajectory (T, d).
            indicator: One of {'variance', 'ac1', 'skewness'}.
            step: Optional downsampling step.

        Returns:
            (times, values) aligned arrays.
        """
        if trajectory.ndim == 2:
            series = trajectory.mean(axis=1)
        else:
            series = trajectory

        if indicator == "variance":
            values = self.variance(series)
            higher_means_risk = True
        elif indicator == "ac1":
            values = self.autocorrelation(series)
            higher_means_risk = True
        elif indicator == "skewness":
            values = self.skewness(series)
            higher_means_risk = False
        else:
            raise ValueError(
                f"Unknown indicator '{indicator}'. Use variance, ac1, or skewness."
            )

        start = self.window - 1 if len(series) >= self.window else 0
        times = np.arange(start, start + len(values), dtype=float)

        if step > 1:
            idx = np.arange(0, len(values), step)
            times = times[idx]
            values = values[idx]

        # keep variable to document risk orientation for caller behavior
        _ = higher_means_risk
        return times, values

    @staticmethod
    def _first_persistent_crossing(mask: np.ndarray, persistence: int) -> int | None:
        """Return first index with `persistence` consecutive True values."""
        if persistence <= 1:
            idx = np.where(mask)[0]
            return int(idx[0]) if len(idx) > 0 else None

        count = 0
        for i, flag in enumerate(mask):
            if flag:
                count += 1
                if count >= persistence:
                    return i - persistence + 1
            else:
                count = 0
        return None

    def compute_baseline_result(
        self,
        trajectory: np.ndarray,
        indicator: str = "variance",
        step: int = 1,
        persistence: int = 3,
        baseline_frac: float = 0.2,
        zscore_k: float = 2.0,
    ) -> BaselineResult:
        """
        Compute standardized BaselineResult from an EWS indicator.

        Alert uses a fixed z-score rule on the baseline segment to avoid
        opportunistic threshold tuning.
        """
        times, values = self.compute_indicator(trajectory, indicator=indicator, step=step)

        higher_means_risk = indicator in {"variance", "ac1"}
        baseline_n = max(3, int(len(values) * baseline_frac))
        base = values[:baseline_n]
        mu = float(np.mean(base))
        sigma = float(np.std(base))

        if higher_means_risk:
            threshold = mu + zscore_k * sigma
            mask = values > threshold
        else:
            threshold = mu - zscore_k * sigma
            mask = values < threshold

        alert_idx = self._first_persistent_crossing(mask, persistence=persistence)
        alert_time = float(times[alert_idx]) if alert_idx is not None else None

        return BaselineResult(
            name=f"Rolling {indicator}",
            times=times,
            values=values,
            alert_time=alert_time,
            higher_means_risk=higher_means_risk,
        )
