"""
Kaplan-Meier Estimator for Survival Analysis.

Non-parametric estimator of the survival function S(t).
Does not use covariates (baseline hazard only).
"""

import numpy as np
from sklearn.base import BaseEstimator
from lifelines import KaplanMeierFitter
import logging

logger = logging.getLogger(__name__)


class KaplanMeierModel(BaseEstimator):
    """
    Kaplan-Meier survival estimator.
    
    Estimates S(t) = P(T > t) non-parametrically from event times.
    
    Usage:
        model = KaplanMeierModel()
        model.fit(durations, events)
        survival_prob = model.predict_survival(times)
        median_time = model.predict_median_time()
    """

    def __init__(self):
        self.kmf = KaplanMeierFitter()
        self.is_fitted_ = False

    def fit(self, durations: np.ndarray, events: np.ndarray = None):
        """
        Fit Kaplan-Meier estimator.
        
        Args:
            durations: Array of event/censoring times (n_samples,)
            events: Array of event indicators (1=event, 0=censored).
                   If None, assumes all events observed (no censoring).
        
        Returns:
            self
        """
        if events is None:
            events = np.ones_like(durations, dtype=int)
        
        # Validate inputs
        durations = np.asarray(durations).flatten()
        events = np.asarray(events).flatten()
        
        if len(durations) != len(events):
            raise ValueError(f"durations and events must have same length: {len(durations)} vs {len(events)}")
        
        if len(durations) == 0:
            raise ValueError("Cannot fit with empty data")
        
        # Fit KM estimator
        self.kmf.fit(durations, events, label='KM_estimate')
        self.is_fitted_ = True
        
        logger.info(f"Kaplan-Meier fitted on {len(durations)} samples, "
                   f"{events.sum()} events, {(1-events).sum()} censored")
        
        return self

    def predict_survival(self, times: np.ndarray) -> np.ndarray:
        """
        Predict survival probability at given times.
        
        Args:
            times: Time points to evaluate S(t)
        
        Returns:
            survival_probs: S(t) for each time point
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        times = np.asarray(times).flatten()
        return self.kmf.survival_function_at_times(times).values

    def predict_median_time(self) -> float:
        """
        Predict median survival time (time at which S(t) = 0.5).
        
        Returns:
            median_time: Estimated median time to event
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        try:
            median = self.kmf.median_survival_time_
            if np.isnan(median):
                # If median not reached, use mean of observed times
                logger.warning("Median survival time not reached, using mean as fallback")
                return float(self.kmf.event_table.index.mean())
            return float(median)
        except Exception as e:
            logger.warning(f"Could not compute median: {e}, using mean")
            return float(self.kmf.event_table.index.mean())

    def predict_quantile(self, quantile: float = 0.5) -> float:
        """
        Predict time at given quantile of survival distribution.
        
        Args:
            quantile: Quantile to compute (0.5 = median, 0.25 = 25th percentile)
        
        Returns:
            time at quantile
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        
        try:
            return float(self.kmf.percentile(quantile))
        except:
            return self.predict_median_time()

    def get_survival_function(self):
        """
        Get the full survival function as DataFrame.
        
        Returns:
            DataFrame with index=times, values=S(t)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before getting survival function")
        
        return self.kmf.survival_function_