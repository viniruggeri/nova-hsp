"""
Advanced survival analysis metrics using scikit-survival.

Implements paper-ready metrics:
- Concordance Index (C-index)
- Integrated Brier Score (IBS)
- Time-dependent AUC/ROC
- Calibration metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    integrated_brier_score,
    cumulative_dynamic_auc,
    brier_score,
)
from sksurv.util import Surv
import warnings

logger = logging.getLogger(__name__)


class SurvivalMetrics:
    """
    Advanced survival analysis metrics.

    All metrics follow the convention:
    - Higher is better for: C-index, AUC
    - Lower is better for: Brier Score, IBS
    """

    @staticmethod
    def concordance_index(
        T_true: np.ndarray, T_pred: np.ndarray, E: np.ndarray, tied_tol: float = 1e-8
    ) -> Dict[str, float]:
        """
        Compute Concordance Index (C-index) for survival predictions.

        C-index measures the fraction of all pairs of subjects whose predicted
        survival times are correctly ordered. Range: [0, 1], 1 = perfect.

        Args:
            T_true: True event/censoring times (n_samples,)
            T_pred: Predicted survival times (n_samples,)
            E: Event indicators (1=event, 0=censored) (n_samples,)
            tied_tol: Tolerance for considering times as tied

        Returns:
            Dictionary with:
                - c_index: Concordance index [0, 1]
                - concordant: Number of concordant pairs
                - discordant: Number of discordant pairs
                - tied_risk: Number of pairs with tied risk
                - tied_time: Number of pairs with tied time
        """
        # Handle invalid predictions
        valid_mask = ~(np.isnan(T_pred) | np.isinf(T_pred))
        if not valid_mask.all():
            logger.warning(f"Found {(~valid_mask).sum()} invalid predictions")
            T_true = T_true[valid_mask]
            T_pred = T_pred[valid_mask]
            E = E[valid_mask]

        if len(T_pred) < 2:
            return {
                "c_index": np.nan,
                "concordant": 0,
                "discordant": 0,
                "tied_risk": 0,
                "tied_time": 0,
            }

        try:
            # scikit-survival expects lower predictions = higher risk
            # We predict time-to-event, so we need to negate
            c_index, concordant, discordant, tied_risk, tied_time = (
                concordance_index_censored(
                    event_indicator=E.astype(bool),
                    event_time=T_true,
                    estimate=-T_pred,  # Negate: lower time = higher risk
                    tied_tol=tied_tol,
                )
            )

            return {
                "c_index": float(c_index),
                "concordant": int(concordant),
                "discordant": int(discordant),
                "tied_risk": int(tied_risk),
                "tied_time": int(tied_time),
            }
        except Exception as e:
            logger.error(f"Error computing C-index: {e}")
            return {
                "c_index": np.nan,
                "concordant": 0,
                "discordant": 0,
                "tied_risk": 0,
                "tied_time": 0,
            }

    @staticmethod
    def concordance_index_ipcw(
        T_train: np.ndarray,
        E_train: np.ndarray,
        T_test: np.ndarray,
        E_test: np.ndarray,
        T_pred: np.ndarray,
        tau: Optional[float] = None,
    ) -> float:
        """
        Compute IPCW (Inverse Probability of Censoring Weighted) C-index.

        More robust to censoring than standard C-index by using inverse
        probability weighting.

        Args:
            T_train: Training event times (for censoring distribution)
            E_train: Training event indicators
            T_test: Test event times
            E_test: Test event indicators
            T_pred: Predicted survival times
            tau: Truncation time (default: None = 95th percentile of observed times)

        Returns:
            IPCW C-index [0, 1]
        """
        # Prepare structured arrays for sksurv
        try:
            # Create structured arrays
            y_train = Surv.from_arrays(event=E_train.astype(bool), time=T_train)
            y_test = Surv.from_arrays(event=E_test.astype(bool), time=T_test)

            # Handle invalid predictions
            valid_mask = ~(np.isnan(T_pred) | np.isinf(T_pred))
            if not valid_mask.all():
                logger.warning(f"Found {(~valid_mask).sum()} invalid predictions")
                y_test = y_test[valid_mask]
                T_pred = T_pred[valid_mask]

            if len(T_pred) < 2:
                return np.nan

            # Set tau to 95th percentile if not provided
            if tau is None:
                # Use 95th percentile of all observed times
                all_times = np.concatenate([T_train, T_test])
                tau = float(np.percentile(all_times, 95))

            # Ensure tau is less than max observed time
            max_train_time = float(np.max(T_train))
            max_test_time = float(np.max(T_test))
            tau = min(tau, max_train_time * 0.95, max_test_time * 0.95)

            # Estimate negated because we predict time (lower = higher risk)
            c_ipcw = concordance_index_ipcw(
                survival_train=y_train, survival_test=y_test, estimate=-T_pred, tau=tau
            )

            return float(c_ipcw)

        except Exception as e:
            logger.error(f"Error computing IPCW C-index: {e}")
            return np.nan

    @staticmethod
    def integrated_brier_score(
        T_train: np.ndarray,
        E_train: np.ndarray,
        T_test: np.ndarray,
        E_test: np.ndarray,
        survival_probs: np.ndarray,
        times: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute Integrated Brier Score (IBS).

        IBS measures prediction error over time, integrating Brier scores
        across time points. Range: [0, 1], 0 = perfect.

        Args:
            T_train: Training event times
            E_train: Training event indicators
            T_test: Test event times
            E_test: Test event indicators
            survival_probs: Survival probabilities at each time point
                           Shape: (n_samples, n_times)
            times: Time points corresponding to survival_probs (n_times,)

        Returns:
            Dictionary with:
                - ibs: Integrated Brier Score [0, 1]
                - brier_scores: Brier scores at each time point
                - times: Time points used
        """
        try:
            y_train = Surv.from_arrays(event=E_train.astype(bool), time=T_train)
            y_test = Surv.from_arrays(event=E_test.astype(bool), time=T_test)

            # Compute IBS
            ibs = integrated_brier_score(
                survival_train=y_train,
                survival_test=y_test,
                estimate=survival_probs,
                times=times,
            )

            # Also compute Brier score at each time point
            brier_scores = []
            for i, t in enumerate(times):
                try:
                    bs = brier_score(
                        survival_train=y_train,
                        survival_test=y_test,
                        estimate=survival_probs[:, i],
                        times=t,
                    )[
                        1
                    ]  # [1] extracts the score
                    brier_scores.append(bs)
                except:
                    brier_scores.append(np.nan)

            return {
                "ibs": float(ibs),
                "brier_scores": brier_scores,
                "times": times.tolist(),
            }

        except Exception as e:
            logger.error(f"Error computing IBS: {e}")
            return {"ibs": np.nan, "brier_scores": [], "times": []}

    @staticmethod
    def time_dependent_auc(
        T_train: np.ndarray,
        E_train: np.ndarray,
        T_test: np.ndarray,
        E_test: np.ndarray,
        risk_scores: np.ndarray,
        times: np.ndarray,
    ) -> Dict[str, any]:
        """
        Compute time-dependent AUC/ROC.

        Evaluates discrimination at specific time points.

        Args:
            T_train: Training event times
            E_train: Training event indicators
            T_test: Test event times
            E_test: Test event indicators
            risk_scores: Risk scores (higher = higher risk) (n_samples,)
            times: Time points to evaluate AUC (n_times,)

        Returns:
            Dictionary with:
                - auc_scores: AUC at each time point
                - mean_auc: Mean AUC across time points
                - times: Time points used
        """
        try:
            y_train = Surv.from_arrays(event=E_train.astype(bool), time=T_train)
            y_test = Surv.from_arrays(event=E_test.astype(bool), time=T_test)

            # Compute cumulative/dynamic AUC
            auc_scores, mean_auc = cumulative_dynamic_auc(
                survival_train=y_train,
                survival_test=y_test,
                estimate=risk_scores,
                times=times,
            )

            return {
                "auc_scores": auc_scores.tolist(),
                "mean_auc": float(mean_auc),
                "times": times.tolist(),
            }

        except Exception as e:
            logger.error(f"Error computing time-dependent AUC: {e}")
            return {"auc_scores": [], "mean_auc": np.nan, "times": []}

    @staticmethod
    def calibration_slope(
        T_true: np.ndarray, T_pred: np.ndarray, E: np.ndarray
    ) -> float:
        """
        Compute calibration slope.

        Slope of linear regression: log(observed_risk) ~ log(predicted_risk)
        Ideal slope = 1.0

        Args:
            T_true: True event times
            T_pred: Predicted event times
            E: Event indicators

        Returns:
            Calibration slope (ideally close to 1.0)
        """
        # Filter to observed events only
        event_mask = E.astype(bool)

        if event_mask.sum() < 5:  # Need minimum samples
            return np.nan

        T_true_events = T_true[event_mask]
        T_pred_events = T_pred[event_mask]

        # Remove invalid predictions
        valid_mask = ~(np.isnan(T_pred_events) | np.isinf(T_pred_events))
        T_true_events = T_true_events[valid_mask]
        T_pred_events = T_pred_events[valid_mask]

        if len(T_true_events) < 5:
            return np.nan

        # Compute log-log regression slope
        log_true = np.log(T_true_events + 1e-8)
        log_pred = np.log(T_pred_events + 1e-8)

        # Linear regression
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_pred, log_true
        )

        return float(slope)


class ExtendedEvaluator:
    """
    Extended evaluator combining standard and advanced survival metrics.
    """

    def __init__(self):
        self.metrics = SurvivalMetrics()

    def evaluate_survival_model(
        self,
        T_train: np.ndarray,
        E_train: np.ndarray,
        T_test: np.ndarray,
        E_test: np.ndarray,
        T_pred: np.ndarray,
        model_name: str = "model",
        compute_ipcw: bool = True,
        compute_calibration: bool = True,
    ) -> pd.DataFrame:
        """
        Comprehensive evaluation for survival models.

        Args:
            T_train: Training event times
            E_train: Training event indicators
            T_test: Test event times
            E_test: Test event indicators
            T_pred: Predicted event times
            model_name: Name of the model
            compute_ipcw: Whether to compute IPCW C-index (slower)
            compute_calibration: Whether to compute calibration slope

        Returns:
            DataFrame with all metrics
        """
        results = {"model_name": model_name}

        # Standard C-index
        c_results = self.metrics.concordance_index(T_test, T_pred, E_test)
        results["c_index"] = c_results["c_index"]
        results["concordant_pairs"] = c_results["concordant"]
        results["discordant_pairs"] = c_results["discordant"]

        # IPCW C-index (more robust to censoring)
        if compute_ipcw:
            c_ipcw = self.metrics.concordance_index_ipcw(
                T_train, E_train, T_test, E_test, T_pred
            )
            results["c_index_ipcw"] = c_ipcw

        # Calibration slope
        if compute_calibration:
            cal_slope = self.metrics.calibration_slope(T_test, T_pred, E_test)
            results["calibration_slope"] = cal_slope

        # Basic error metrics for comparison
        valid_mask = ~(np.isnan(T_pred) | np.isinf(T_pred))
        if valid_mask.any():
            mae = np.mean(np.abs(T_test[valid_mask] - T_pred[valid_mask]))
            rmse = np.sqrt(np.mean((T_test[valid_mask] - T_pred[valid_mask]) ** 2))
            results["mae"] = mae
            results["rmse"] = rmse

        return pd.DataFrame([results])

    def compare_models(
        self,
        results: pd.DataFrame,
        primary_metric: str = "c_index",
        alternative_metrics: List[str] = ["c_index_ipcw", "mae"],
    ) -> pd.DataFrame:
        """
        Rank and compare models by metrics.

        Args:
            results: DataFrame with evaluation results
            primary_metric: Main metric for ranking (higher is better for c_index)
            alternative_metrics: Additional metrics for comparison

        Returns:
            Ranked DataFrame
        """
        # Rank by primary metric
        if "c_index" in primary_metric.lower():
            # Higher is better
            results = results.sort_values(primary_metric, ascending=False)
        else:
            # Lower is better (MAE, RMSE, etc.)
            results = results.sort_values(primary_metric, ascending=True)

        # Add rank column
        results["rank"] = range(1, len(results) + 1)

        return results
