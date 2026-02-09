"""
Cox Proportional Hazards Model for survival analysis.

Parametric survival model with proportional hazards assumption:
h(t|X) = h₀(t) * exp(β'X)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
import logging

logger = logging.getLogger(__name__)


class CoxPHModel(BaseEstimator):
    """
    Cox Proportional Hazards model.
    
    Models hazard as: h(t|X) = h₀(t) * exp(β'X)
    where h₀(t) is baseline hazard and β are learned coefficients.
    
    Usage:
        model = CoxPHModel()
        model.fit(X_train, T_train, events_train)
        median_times = model.predict_median_time(X_test)
    """

    def __init__(self, penalizer: float = 0.1, normalize: bool = True):
        """
        Args:
            penalizer: L2 regularization strength
            normalize: Whether to standardize features
        """
        self.penalizer = penalizer
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.cph = CoxPHFitter(penalizer=penalizer)
        self.is_fitted_ = False
        self.feature_names_ = None

    def _aggregate_features(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregate temporal sequences to static features.
        
        Cox PH requires static covariates, so we extract:
        - Last observation
        - Mean over time
        - Std over time
        - Max over time
        """
        if X.ndim == 3:
            # (n_samples, seq_len, n_features)
            last = X[:, -1, :]  # (n_samples, n_features)
            mean = X.mean(axis=1)
            std = X.std(axis=1)
            max_val = X.max(axis=1)
            
            # Concatenate all aggregations
            X_agg = np.hstack([last, mean, std, max_val])
        elif X.ndim == 2:
            # Already aggregated
            X_agg = X
        else:
            raise ValueError(f"X must be 2D or 3D, got shape {X.shape}")
        
        return X_agg

    def fit(self, X: np.ndarray, durations: np.ndarray, events: np.ndarray = None):
        """
        Fit Cox PH model.
        
        Args:
            X: Covariates (n_samples, seq_len, n_features) or (n_samples, n_features)
            durations: Event/censoring times (n_samples,)
            events: Event indicators (1=event, 0=censored). If None, assumes all events.
        
        Returns:
            self
        """
        if events is None:
            events = np.ones_like(durations, dtype=int)
        
        # Aggregate features
        X_agg = self._aggregate_features(X)
        
        # Normalize
        if self.normalize:
            X_agg = self.scaler.fit_transform(X_agg)
        
        # Create DataFrame for lifelines
        n_features = X_agg.shape[1]
        self.feature_names_ = [f'feature_{i}' for i in range(n_features)]
        
        df = pd.DataFrame(X_agg, columns=self.feature_names_)
        df['duration'] = durations
        df['event'] = events
        
        # Fit Cox model
        try:
            self.cph.fit(df, duration_col='duration', event_col='event', show_progress=False)
            self.is_fitted_ = True
            
            logger.info(f"Cox PH fitted: {len(durations)} samples, "
                       f"concordance index = {self.cph.concordance_index_:.3f}")
        except Exception as e:
            logger.error(f"Cox PH fitting failed: {e}")
            raise
        
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk score (higher = higher risk).
        
        Risk = exp(β'X)
        
        Args:
            X: Test covariates
        
        Returns:
            risk_scores: (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_agg = self._aggregate_features(X)
        
        if self.normalize:
            X_agg = self.scaler.transform(X_agg)
        
        df = pd.DataFrame(X_agg, columns=self.feature_names_)
        
        # Partial hazard (risk score)
        risk_scores = self.cph.predict_partial_hazard(df).values
        
        return risk_scores

    def predict_survival_function(self, X: np.ndarray, times: np.ndarray = None) -> np.ndarray:
        """
        Predict survival function S(t|X) for each sample.
        
        Args:
            X: Test covariates
            times: Time points to evaluate (if None, uses training times)
        
        Returns:
            survival_probs: (n_samples, n_times)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_agg = self._aggregate_features(X)
        
        if self.normalize:
            X_agg = self.scaler.transform(X_agg)
        
        df = pd.DataFrame(X_agg, columns=self.feature_names_)
        
        # Predict survival function
        surv_func = self.cph.predict_survival_function(df, times=times)
        
        return surv_func.values.T  # (n_samples, n_times)

    def predict_median_time(self, X: np.ndarray) -> np.ndarray:
        """
        Predict median survival time for each sample.
        
        Args:
            X: Test covariates
        
        Returns:
            median_times: (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_agg = self._aggregate_features(X)
        
        if self.normalize:
            X_agg = self.scaler.transform(X_agg)
        
        df = pd.DataFrame(X_agg, columns=self.feature_names_)
        
        # Predict median survival time
        median_times = self.cph.predict_median(df).values
        
        # Handle cases where median not reached (return large value)
        median_times = np.where(np.isnan(median_times), 100.0, median_times)
        
        return median_times

    def get_concordance_index(self) -> float:
        """Get C-index from training."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted")
        return self.cph.concordance_index_