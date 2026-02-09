"""
Accelerated Failure Time Model for survival analysis.

Parametric survival model: log(T) = β'X + σ*ε
where ε follows a distribution (Weibull, Log-normal, etc.)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from lifelines import WeibullAFTFitter, LogNormalAFTFitter
import logging

logger = logging.getLogger(__name__)


class AFTModel(BaseEstimator):
    """
    Accelerated Failure Time model.
    
    Models log(T) as linear function of covariates.
    
    Usage:
        model = AFTModel(distribution='weibull')
        model.fit(X_train, T_train, events_train)
        T_pred = model.predict_time(X_test)
    """

    def __init__(self, distribution: str = "weibull", penalizer: float = 0.1, normalize: bool = True):
        """
        Args:
            distribution: 'weibull' or 'lognormal'
            penalizer: L2 regularization strength
            normalize: Whether to standardize features
        """
        self.distribution = distribution
        self.penalizer = penalizer
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        
        if distribution == 'weibull':
            self.aft = WeibullAFTFitter(penalizer=penalizer)
        elif distribution == 'lognormal':
            self.aft = LogNormalAFTFitter(penalizer=penalizer)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        self.is_fitted_ = False
        self.feature_names_ = None

    def _aggregate_features(self, X: np.ndarray) -> np.ndarray:
        """Aggregate temporal sequences to static features."""
        if X.ndim == 3:
            last = X[:, -1, :]
            mean = X.mean(axis=1)
            std = X.std(axis=1)
            max_val = X.max(axis=1)
            X_agg = np.hstack([last, mean, std, max_val])
        elif X.ndim == 2:
            X_agg = X
        else:
            raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")
        return X_agg

    def fit(self, X: np.ndarray, durations: np.ndarray, events: np.ndarray = None):
        """
        Fit AFT model.
        
        Args:
            X: Covariates (n_samples, seq_len, n_features) or (n_samples, n_features)
            durations: Event/censoring times
            events: Event indicators (if None, assumes all events)
        """
        if events is None:
            events = np.ones_like(durations, dtype=int)
        
        X_agg = self._aggregate_features(X)
        
        if self.normalize:
            X_agg = self.scaler.fit_transform(X_agg)
        
        n_features = X_agg.shape[1]
        self.feature_names_ = [f'feature_{i}' for i in range(n_features)]
        
        df = pd.DataFrame(X_agg, columns=self.feature_names_)
        df['duration'] = durations
        df['event'] = events
        
        try:
            self.aft.fit(df, duration_col='duration', event_col='event', show_progress=False)
            self.is_fitted_ = True
            logger.info(f"AFT ({self.distribution}) fitted: {len(durations)} samples, "
                       f"concordance = {self.aft.concordance_index_:.3f}")
        except Exception as e:
            logger.error(f"AFT fitting failed: {e}")
            raise
        
        return self

    def predict_time(self, X: np.ndarray) -> np.ndarray:
        """
        Predict expected survival time.
        
        Returns:
            expected_times: (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_agg = self._aggregate_features(X)
        
        if self.normalize:
            X_agg = self.scaler.transform(X_agg)
        
        df = pd.DataFrame(X_agg, columns=self.feature_names_)
        
        # Predict expected time
        expected_times = self.aft.predict_expectation(df).values
        
        return expected_times

    def predict_median_time(self, X: np.ndarray) -> np.ndarray:
        """
        Predict median survival time.
        
        Returns:
            median_times: (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_agg = self._aggregate_features(X)
        
        if self.normalize:
            X_agg = self.scaler.transform(X_agg)
        
        df = pd.DataFrame(X_agg, columns=self.feature_names_)
        
        median_times = self.aft.predict_median(df).values
        
        return median_times