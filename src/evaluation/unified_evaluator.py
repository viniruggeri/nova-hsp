"""
Unified Baseline Evaluator.

Provides a single interface to evaluate all baseline models with
standardized metrics and statistical comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import KFold
from scipy import stats
import time
import logging

logger = logging.getLogger(__name__)


class UnifiedBaselineEvaluator:
    """
    Unified evaluator for all baseline models.
    
    Automatically detects model type and calls appropriate prediction methods.
    Computes standardized metrics across all models.
    
    Usage:
        models = {
            'KM': kaplan_meier_model,
            'LSTM': temporal_lstm_model,
            'HMM': hmm_model,
        }
        evaluator = UnifiedBaselineEvaluator(models)
        results = evaluator.evaluate_all(X_test, T_test, events_test)
    """
    
    def __init__(self, models: Dict[str, Any]):
        """
        Args:
            models: Dictionary mapping model names to fitted model instances
        """
        self.models = models
        self.model_types = {}
        self._detect_model_types()
    
    def _detect_model_types(self):
        """Automatically detect model type from class name and methods."""
        for name, model in self.models.items():
            class_name = model.__class__.__name__
            
            # Survival models
            if 'KaplanMeier' in class_name or 'CoxPH' in class_name or 'AFT' in class_name:
                self.model_types[name] = 'survival'
            # Heuristic models
            elif 'Threshold' in class_name or 'Warning' in class_name or 'EWS' in class_name:
                self.model_types[name] = 'heuristic'
            # State-based models
            elif 'Markov' in class_name or 'HMM' in class_name:
                self.model_types[name] = 'state'
            # Deep learning models
            elif 'Deep' in class_name or 'LSTM' in class_name or 'Temporal' in class_name or 'Hazard' in class_name:
                self.model_types[name] = 'deep'
            else:
                # Try to infer from available methods
                if hasattr(model, 'predict_median_time'):
                    self.model_types[name] = 'survival'
                elif hasattr(model, 'predict_time_to_collapse'):
                    self.model_types[name] = 'state'
                elif hasattr(model, 'predict_time'):
                    self.model_types[name] = 'deep'
                else:
                    logger.warning(f"Could not detect type for model {name}, defaulting to 'unknown'")
                    self.model_types[name] = 'unknown'
            
            logger.info(f"Detected model {name} as type: {self.model_types[name]}")
    
    def _predict_time(self, model_name: str, model: Any, X: np.ndarray, 
                      T_start: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Unified prediction interface across all model types.
        
        Args:
            model_name: Name of the model
            model: Model instance
            X: Input features (n_samples, seq_len, n_features) or (n_samples, n_features)
            T_start: Current time for each sample (for time-to-collapse models)
        
        Returns:
            predictions: Predicted time-to-event (n_samples,)
        """
        model_type = self.model_types[model_name]
        
        try:
            if model_type == 'survival':
                # Try different survival prediction methods
                if hasattr(model, 'predict_median_time'):
                    # Check if it needs X or not (Kaplan-Meier doesn't use X)
                    if 'KaplanMeier' in model.__class__.__name__:
                        pred = model.predict_median_time()
                        # Repeat for all samples
                        if isinstance(X, np.ndarray):
                            n_samples = len(X) if X.ndim >= 2 else 1
                            pred = np.array([pred] * n_samples)
                    else:
                        pred = model.predict_median_time(X)
                elif hasattr(model, 'predict_time'):
                    pred = model.predict_time(X)
                else:
                    raise AttributeError(f"Survival model {model_name} has no predict_time method")
                
                # Ensure array output
                if np.isscalar(pred):
                    pred = np.array([pred])
                elif isinstance(pred, list):
                    pred = np.array(pred)
                
                return pred
            
            elif model_type == 'heuristic':
                # Try different heuristic prediction methods
                if hasattr(model, 'predict_time_to_collapse'):
                    pred = model.predict_time_to_collapse(X)
                elif hasattr(model, 'predict_alert_time_batch'):
                    pred = model.predict_alert_time_batch(X)
                elif hasattr(model, 'predict_alert_time'):
                    # Single sample prediction - need to loop
                    if X.ndim == 3:  # Batch
                        pred = np.array([model.predict_alert_time(X[i]) for i in range(len(X))])
                    else:  # Single sample
                        pred = np.array([model.predict_alert_time(X)])
                else:
                    raise AttributeError(f"Heuristic model {model_name} has no predict method")
                
                if T_start is not None:
                    # Convert time-to-collapse to absolute time
                    pred = T_start + pred
                return pred
            
            elif model_type == 'state':
                pred = model.predict_time_to_collapse(X)
                if T_start is not None:
                    pred = T_start + pred
                return pred
            
            elif model_type == 'deep':
                pred = model.predict_time(X)
                return pred
            
            else:
                raise ValueError(f"Unknown model type: {model_type} for {model_name}")
        
        except Exception as e:
            logger.error(f"Error predicting with {model_name}: {e}")
            raise
    
    def compute_metrics(self, T_true: np.ndarray, T_pred: np.ndarray, 
                       T_start: Optional[np.ndarray] = None,
                       events: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute standardized metrics.
        
        Args:
            T_true: True event times (n_samples,)
            T_pred: Predicted event times (n_samples,)
            T_start: Start times for lead time computation (n_samples,)
            events: Event indicators (1=event, 0=censored)
        
        Returns:
            metrics: Dictionary of metric name -> value
        """
        metrics = {}
        
        # Handle NaN predictions
        valid_mask = ~(np.isnan(T_pred) | np.isinf(T_pred))
        if not valid_mask.all():
            logger.warning(f"Found {(~valid_mask).sum()} invalid predictions, excluding from metrics")
            T_true = T_true[valid_mask]
            T_pred = T_pred[valid_mask]
            if T_start is not None:
                T_start = T_start[valid_mask]
            if events is not None:
                events = events[valid_mask]
        
        if len(T_pred) == 0:
            return {'error': 'No valid predictions'}
        
        # Mean Absolute Error
        metrics['MAE'] = float(np.mean(np.abs(T_pred - T_true)))
        
        # Root Mean Squared Error
        metrics['RMSE'] = float(np.sqrt(np.mean((T_pred - T_true) ** 2)))
        
        # R² Score (coefficient of determination)
        ss_res = np.sum((T_true - T_pred) ** 2)
        ss_tot = np.sum((T_true - np.mean(T_true)) ** 2)
        metrics['R2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else -np.inf
        
        # Pearson correlation
        if len(T_pred) > 1:
            corr, p_value = stats.pearsonr(T_true, T_pred)
            metrics['Correlation'] = float(corr)
            metrics['Correlation_pvalue'] = float(p_value)
        
        # Lead Time (if start times provided)
        if T_start is not None:
            lead_times = []
            for i in range(len(T_true)):
                T_alert = T_pred[i]
                T_event = T_true[i]
                T_0 = T_start[i]
                
                if T_event > T_0:  # Valid event
                    lead_time = (T_event - T_alert) / (T_event - T_0)
                    lead_times.append(lead_time)
            
            if lead_times:
                metrics['LeadTime_mean'] = float(np.mean(lead_times))
                metrics['LeadTime_median'] = float(np.median(lead_times))
                metrics['LeadTime_std'] = float(np.std(lead_times))
        
        # Median/Mean Absolute Percentage Error
        mape = np.abs((T_true - T_pred) / T_true)
        mape = mape[np.isfinite(mape)]  # Remove inf/nan
        if len(mape) > 0:
            metrics['MAPE'] = float(np.mean(mape) * 100)
            metrics['MdAPE'] = float(np.median(mape) * 100)
        
        return metrics
    
    def evaluate_single(self, model_name: str, X: np.ndarray, T_true: np.ndarray,
                       T_start: Optional[np.ndarray] = None,
                       events: Optional[np.ndarray] = None,
                       return_predictions: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model_name: Name of the model to evaluate
            X: Input features
            T_true: True event times
            T_start: Start times (optional)
            events: Event indicators (optional)
            return_predictions: Whether to include predictions in output
        
        Returns:
            results: Dictionary with metrics and optionally predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Time the prediction
        start_time = time.time()
        T_pred = self._predict_time(model_name, model, X, T_start)
        pred_time = time.time() - start_time
        
        # Compute metrics
        metrics = self.compute_metrics(T_true, T_pred, T_start, events)
        metrics['prediction_time'] = pred_time
        metrics['model_name'] = model_name
        metrics['model_type'] = self.model_types[model_name]
        
        if return_predictions:
            metrics['predictions'] = T_pred
        
        return metrics
    
    def evaluate_all(self, X: np.ndarray, T_true: np.ndarray,
                    T_start: Optional[np.ndarray] = None,
                    events: Optional[np.ndarray] = None,
                    return_predictions: bool = False) -> pd.DataFrame:
        """
        Evaluate all models.
        
        Args:
            X: Input features
            T_true: True event times
            T_start: Start times (optional)
            events: Event indicators (optional)
            return_predictions: Whether to include predictions
        
        Returns:
            results_df: DataFrame with all results
        """
        results = []
        
        for model_name in self.models.keys():
            try:
                result = self.evaluate_single(
                    model_name, X, T_true, T_start, events, return_predictions
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                results.append({
                    'model_name': model_name,
                    'error': str(e)
                })
        
        df = pd.DataFrame(results)
        
        # Sort by primary metric (MAE)
        if 'MAE' in df.columns:
            df = df.sort_values('MAE')
        
        return df
    
    def cross_validate(self, X: np.ndarray, T: np.ndarray, 
                      events: Optional[np.ndarray] = None,
                      cv: int = 5, random_state: int = 42,
                      verbose: bool = False) -> pd.DataFrame:
        """
        Perform K-fold cross-validation for all models.
        
        Note: This refits the existing model instances. If you need to preserve
        original models, pass clones.
        
        Args:
            X: Full dataset features
            T: Full dataset event times
            events: Event indicators (optional)
            cv: Number of folds
            random_state: Random seed
            verbose: Print detailed error messages
        
        Returns:
            cv_results: DataFrame with cross-validation results
        """
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        cv_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            
            if events is not None:
                events_train = events[train_idx]
                events_test = events[test_idx]
            else:
                events_train = events_test = None
            
            # Refit all models on this fold
            for model_name, model in self.models.items():
                try:
                    # Refit model
                    if hasattr(model, 'fit'):
                        # Special case: Kaplan-Meier doesn't use features
                        if self.model_types[model_name] == 'survival' and 'KaplanMeier' in model.__class__.__name__:
                            model.fit(T_train, events_train if events_train is not None else np.ones_like(T_train))
                        elif self.model_types[model_name] == 'survival' and events_train is not None:
                            model.fit(X_train, T_train, events_train)
                        else:
                            model.fit(X_train, T_train)
                    else:
                        if verbose:
                            logger.warning(f"{model_name} has no fit method, using existing model")
                    
                    # Evaluate
                    result = self.evaluate_single(model_name, X_test, T_test, events=events_test)
                    result['fold'] = fold_idx
                    cv_results.append(result)
                
                except Exception as e:
                    error_msg = f"Fold {fold_idx}, model {model_name} failed: {e}"
                    logger.error(error_msg)
                    if verbose:
                        print(error_msg)
                        import traceback
                        traceback.print_exc()
        
        return pd.DataFrame(cv_results)
    
    def compare_models(self, results_df: pd.DataFrame, 
                      metric: str = 'MAE',
                      test: str = 'wilcoxon') -> pd.DataFrame:
        """
        Statistical comparison between models.
        
        Args:
            results_df: Results from cross_validate()
            metric: Metric to compare
            test: 'wilcoxon' (paired) or 'mannwhitneyu' (unpaired)
        
        Returns:
            comparison_df: Pairwise comparison matrix with p-values
        """
        if metric not in results_df.columns:
            raise ValueError(f"Metric {metric} not found in results")
        
        models = results_df['model_name'].unique()
        n_models = len(models)
        
        # Create comparison matrix
        p_values = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i == j:
                    p_values[i, j] = 1.0
                    continue
                
                scores1 = results_df[results_df['model_name'] == model1][metric].values
                scores2 = results_df[results_df['model_name'] == model2][metric].values
                
                if len(scores1) == 0 or len(scores2) == 0:
                    p_values[i, j] = np.nan
                    continue
                
                try:
                    if test == 'wilcoxon' and len(scores1) == len(scores2):
                        _, p_val = stats.wilcoxon(scores1, scores2)
                    elif test == 'mannwhitneyu':
                        _, p_val = stats.mannwhitneyu(scores1, scores2)
                    else:
                        _, p_val = stats.ttest_ind(scores1, scores2)
                    
                    p_values[i, j] = p_val
                except Exception as e:
                    logger.warning(f"Statistical test failed for {model1} vs {model2}: {e}")
                    p_values[i, j] = np.nan
        
        comparison_df = pd.DataFrame(p_values, index=models, columns=models)
        return comparison_df
    
    def summary_statistics(self, cv_results: pd.DataFrame, 
                          metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aggregate cross-validation results.
        
        Args:
            cv_results: Results from cross_validate()
            metrics: List of metrics to summarize (None = all)
        
        Returns:
            summary_df: Mean ± std for each metric
        """
        if metrics is None:
            # Auto-detect numeric metrics
            numeric_cols = cv_results.select_dtypes(include=[np.number]).columns
            metrics = [c for c in numeric_cols if c not in ['fold']]
        
        summary = []
        
        for model_name in cv_results['model_name'].unique():
            model_results = cv_results[cv_results['model_name'] == model_name]
            
            row = {'model_name': model_name}
            
            for metric in metrics:
                if metric not in model_results.columns:
                    continue
                
                values = model_results[metric].dropna()
                if len(values) > 0:
                    row[f'{metric}_mean'] = values.mean()
                    row[f'{metric}_std'] = values.std()
                    row[f'{metric}_median'] = values.median()
            
            summary.append(row)
        
        return pd.DataFrame(summary)
    
    def evaluate_with_advanced_metrics(
        self,
        X_train: np.ndarray,
        T_train: np.ndarray,
        E_train: np.ndarray,
        X_test: np.ndarray,
        T_test: np.ndarray,
        E_test: np.ndarray,
        survival_models_only: bool = True,
        compute_ipcw: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate models with advanced survival metrics (C-index, IPCW, calibration).
        
        Requires scikit-survival. Only applicable to survival models.
        
        Args:
            X_train: Training features
            T_train: Training event times
            E_train: Training event indicators
            X_test: Test features
            T_test: Test event times
            E_test: Test event indicators
            survival_models_only: Only evaluate survival models
            compute_ipcw: Compute IPCW C-index (slower but more robust)
            
        Returns:
            DataFrame with advanced metrics
        """
        try:
            from src.evaluation.metrics_advanced import ExtendedEvaluator
        except ImportError:
            logger.error("scikit-survival not available. Install with: pip install scikit-survival")
            return pd.DataFrame()
        
        extended_eval = ExtendedEvaluator()
        results = []
        
        for model_name, model in self.models.items():
            # Skip non-survival models if requested
            if survival_models_only and self.model_types[model_name] != 'survival':
                continue
            
            try:
                logger.info(f"Computing advanced metrics for {model_name}...")
                
                # Get predictions
                T_pred = self._predict_time(model_name, model, X_test)
                
                # Compute advanced survival metrics
                model_results = extended_eval.evaluate_survival_model(
                    T_train=T_train,
                    E_train=E_train,
                    T_test=T_test,
                    E_test=E_test,
                    T_pred=T_pred,
                    model_name=model_name,
                    compute_ipcw=compute_ipcw,
                    compute_calibration=True
                )
                
                results.append(model_results)
                
            except Exception as e:
                logger.error(f"Error computing advanced metrics for {model_name}: {e}")
                continue
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
