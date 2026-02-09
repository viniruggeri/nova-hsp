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
from scipy.signal import correlate
from scipy.stats import linregress
import logging

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
        return variances[self.window-1:]  # Drop NaN

    def autocorrelation(self, series: np.ndarray, lag: int = None) -> np.ndarray:
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
            window_data = series[i-self.window:i]
            acf = self._compute_acf(window_data, lag)
            autocorrs.append(acf)
        
        return np.array(autocorrs)

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
            'variance': var,
            'autocorrelation': acf,
            'variance_trend': var_trend,
            'autocorr_trend': acf_trend,
            'csd_score': csd_score,
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
        except:
            return 0.0

    def fit(self, X: np.ndarray, T: np.ndarray = None):
        """
        Fit EWS (calibrate threshold if needed).
        
        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            T: Time-to-event (optional, for calibration)
        """
        # EWS is primarily unsupervised, but we can calibrate threshold
        self.is_fitted_ = True
        logger.info(f"EWS fitted: window={self.window}, lag={self.lag}, threshold={self.threshold}")
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
            window_data = series[max(0, i-self.window):i]
            csd = self.critical_slowing_down(window_data)
            scores[i-1] = csd['csd_score']
        
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