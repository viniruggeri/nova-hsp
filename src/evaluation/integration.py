"""
Integration layer between UnifiedBaselineEvaluator and RobustnessAnalyzer.

This module provides a unified interface for evaluating models with robustness analysis.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from src.evaluation.unified_evaluator import UnifiedBaselineEvaluator
from src.evaluation.robustness import RobustnessAnalyzer, PerturbationFactory

logger = logging.getLogger(__name__)


class IntegratedEvaluator:
    """
    Integrated evaluator combining standard metrics with robustness analysis.
    
    This class provides a single interface to:
    - Evaluate baseline models with standard metrics
    - Perform robustness analysis with various perturbations
    - Compare models across both performance and robustness
    - Generate comprehensive evaluation reports
    """
    
    def __init__(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        T: np.ndarray,
        E: Optional[np.ndarray] = None,
        perturbation_types: Optional[List[str]] = None,
        n_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the integrated evaluator.
        
        Args:
            models: Dictionary mapping model names to model instances
            X: Feature array (n_samples, n_timesteps, n_features)
            T: Time-to-event array (n_samples,)
            E: Event indicator array (n_samples,), optional for non-survival models
            perturbation_types: List of perturbation types to apply
                              Default: ['gaussian_noise', 'feature_dropout', 'temporal_delay']
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.models = models
        self.X = X
        self.T = T
        self.E = E
        self.n_folds = n_folds
        self.random_state = random_state
        self.perturbation_types = perturbation_types or [
            'gaussian_noise', 'feature_dropout', 'temporal_delay'
        ]
        
        # Initialize evaluators
        self.evaluator = UnifiedBaselineEvaluator(models=models)
        self.robustness_analyzer = RobustnessAnalyzer()
        
        # Evaluation results cache
        self._performance_results: Optional[pd.DataFrame] = None
        self._robustness_results: Optional[pd.DataFrame] = None
        self._integrated_results: Optional[pd.DataFrame] = None
        
    def evaluate_performance(self) -> pd.DataFrame:
        """
        Evaluate all models with standard metrics.
        
        Returns:
            DataFrame with performance metrics for all models
        """
        logger.info("Evaluating model performance...")
        self._performance_results = self.evaluator.evaluate_all(
            X=self.X,
            T_true=self.T,
            events=self.E
        )
        return self._performance_results
    
    def evaluate_robustness(self) -> pd.DataFrame:
        """
        Evaluate robustness of all models.
        
        Returns:
            DataFrame with robustness metrics for all models
        """
        logger.info("Evaluating model robustness...")
        
        robustness_results = []
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Analyzing robustness for {model_name}...")
                
                # Define prediction function
                def predict_fn(X_perturbed):
                    return self.evaluator._predict_time(model_name, model, X_perturbed)
                
                # Define metric function (MAE)
                def metric_fn(y_true, y_pred):
                    valid_mask = ~(np.isnan(y_pred) | np.isinf(y_pred))
                    if not valid_mask.any():
                        return np.inf
                    return np.mean(np.abs(y_true[valid_mask] - y_pred[valid_mask]))
                
                # Evaluate robustness
                results = self.robustness_analyzer.evaluate_model_robustness(
                    model_name=model_name,
                    X_test=self.X,
                    y_test=self.T,
                    predict_fn=predict_fn,
                    metric_fn=metric_fn,
                    metric_name='MAE'
                )
                
                # Convert results to DataFrame rows
                for pert_type, pert_data in results.items():
                    for intensity, degradation in zip(pert_data['intensities'], pert_data['degradations']):
                        robustness_results.append({
                            'model_name': model_name,
                            'perturbation_type': pert_type,
                            'intensity': intensity,
                            'degradation_percent': degradation,
                            'clean_error': pert_data['metric_clean'],
                            'perturbed_error': pert_data['metric_clean'] * (1 + degradation / 100)
                        })
                
            except Exception as e:
                logger.error(f"Error analyzing robustness for {model_name}: {e}")
                continue
        
        self._robustness_results = pd.DataFrame(robustness_results)
        return self._robustness_results
    
    def evaluate_advanced_survival_metrics(
        self,
        X_train: Optional[np.ndarray] = None,
        T_train: Optional[np.ndarray] = None,
        E_train: Optional[np.ndarray] = None,
        compute_ipcw: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate survival models with advanced metrics (C-index, IPCW, calibration).
        
        Args:
            X_train: Training features (default: use self.X for split)
            T_train: Training times (default: use self.T for split)
            E_train: Training events (default: use self.E for split)
            compute_ipcw: Compute IPCW C-index (slower)
            
        Returns:
            DataFrame with advanced survival metrics
        """
        logger.info("Evaluating advanced survival metrics...")
        
        # Use provided data or create train/test split
        if X_train is None or T_train is None or E_train is None:
            # Use 70/30 split for train/test
            n = len(self.X)
            n_train = int(0.7 * n)
            
            indices = np.arange(n)
            np.random.seed(self.random_state)
            np.random.shuffle(indices)
            
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            T_train, T_test = self.T[train_idx], self.T[test_idx]
            E_train, E_test = self.E[train_idx], self.E[test_idx]
        else:
            X_test, T_test, E_test = self.X, self.T, self.E
        
        # Compute advanced metrics
        advanced_results = self.evaluator.evaluate_with_advanced_metrics(
            X_train=X_train,
            T_train=T_train,
            E_train=E_train,
            X_test=X_test,
            T_test=T_test,
            E_test=E_test,
            survival_models_only=True,
            compute_ipcw=compute_ipcw
        )
        
        return advanced_results
    
    def evaluate_all(self, include_cv: bool = False, include_advanced: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform complete evaluation: performance + robustness + optional advanced metrics.
        
        Args:
            include_cv: Whether to include cross-validation results
            include_advanced: Whether to include advanced survival metrics (C-index, IPCW)
            
        Returns:
            Tuple of (performance_df, robustness_df, integrated_df)
        """
        # Evaluate performance
        performance_df = self.evaluate_performance()
        
        # Evaluate robustness
        robustness_df = self.evaluate_robustness()
        
        # Optionally include advanced survival metrics
        if include_advanced and self.E is not None:
            try:
                advanced_df = self.evaluate_advanced_survival_metrics()
                # Merge with performance results
                if not advanced_df.empty:
                    performance_df = performance_df.merge(
                        advanced_df[['model_name', 'c_index', 'c_index_ipcw', 'calibration_slope']],
                        on='model_name',
                        how='left'
                    )
            except Exception as e:
                logger.warning(f"Could not compute advanced metrics: {e}")
        
        # Create integrated summary
        integrated_df = self._create_integrated_summary(performance_df, robustness_df)
        
        # Optionally include cross-validation
        if include_cv:
            cv_results = self.evaluator.cross_validate(
                X=self.X,
                T=self.T,  # This is correct for cross_validate
                events=self.E,
                cv=self.n_folds,
                random_state=self.random_state
            )
            performance_df = pd.concat([performance_df, cv_results], ignore_index=True)
        
        self._integrated_results = integrated_df
        return performance_df, robustness_df, integrated_df
    
    def _create_integrated_summary(
        self,
        performance_df: pd.DataFrame,
        robustness_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create integrated summary combining performance and robustness.
        
        Args:
            performance_df: Performance evaluation results
            robustness_df: Robustness evaluation results
            
        Returns:
            DataFrame with integrated metrics
        """
        # Get summary statistics for robustness
        robustness_summary = robustness_df.groupby('model_name').agg({
            'degradation_percent': ['mean', 'std'],
            'perturbed_error': 'mean'
        }).reset_index()
        
        robustness_summary.columns = [
            'model_name',
            'mean_degradation',
            'std_degradation',
            'mean_perturbed_error'
        ]
        
        # Compute AUC-D for each model and perturbation type
        auc_d_results = []
        for model_name in robustness_df['model_name'].unique():
            model_data = robustness_df[robustness_df['model_name'] == model_name]
            
            for pert_type in model_data['perturbation_type'].unique():
                pert_data = model_data[model_data['perturbation_type'] == pert_type]
                
                # Compute AUC-D
                intensities = pert_data['intensity'].values
                degradations = pert_data['degradation_percent'].values
                
                # Sort by intensity
                sorted_idx = np.argsort(intensities)
                intensities = intensities[sorted_idx]
                degradations = degradations[sorted_idx]
                
                auc_d = float(np.trapezoid(degradations, intensities))
                
                auc_d_results.append({
                    'model_name': model_name,
                    'perturbation_type': pert_type,
                    'auc_d': auc_d
                })
        
        auc_d_df = pd.DataFrame(auc_d_results)
        
        # Pivot to get AUC-D by perturbation type
        auc_d_pivot = auc_d_df.pivot(
            index='model_name',
            columns='perturbation_type',
            values='auc_d'
        ).reset_index()
        
        # Rename columns
        auc_d_pivot.columns = ['model_name'] + [
            f'auc_d_{col}' for col in auc_d_pivot.columns[1:]
        ]
        
        # Merge with performance metrics
        integrated = performance_df.copy()
        integrated = integrated.merge(robustness_summary, on='model_name', how='left')
        integrated = integrated.merge(auc_d_pivot, on='model_name', how='left')
        
        # Compute composite score (lower is better)
        # Score = normalized_MAE + normalized_mean_degradation
        if 'MAE' in integrated.columns:
            mae_norm = (integrated['MAE'] - integrated['MAE'].min()) / (
                integrated['MAE'].max() - integrated['MAE'].min() + 1e-10
            )
            
            deg_norm = (integrated['mean_degradation'] - integrated['mean_degradation'].min()) / (
                integrated['mean_degradation'].max() - integrated['mean_degradation'].min() + 1e-10
            )
            
            integrated['composite_score'] = mae_norm + deg_norm
        
        return integrated
    
    def compare_models(self) -> pd.DataFrame:
        """
        Statistical comparison of models across performance and robustness.
        
        Returns:
            DataFrame with pairwise comparison results
        """
        if self._performance_results is None:
            self.evaluate_performance()
        
        # Get statistical comparison from evaluator
        comparison = self.evaluator.compare_models()
        
        # Add robustness comparison if available
        if self._robustness_results is not None:
            # Compute average degradation for each model
            rob_summary = self._robustness_results.groupby('model_name')['degradation_percent'].mean()
            
            # Rank by robustness (lower degradation = more robust)
            rob_ranking = rob_summary.rank(ascending=True).to_dict()
            
            # Add robustness rank to comparison
            comparison['robustness_rank'] = comparison['model_name'].map(rob_ranking)
        
        return comparison
    
    def summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics across all evaluations.
        
        Returns:
            DataFrame with summary statistics
        """
        if self._integrated_results is None:
            self.evaluate_all()
        
        return self.evaluator.summary_statistics(self._integrated_results)
    
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report as formatted string
        """
        # Run full evaluation
        perf_df, rob_df, int_df = self.evaluate_all()
        
        report = []
        report.append("=" * 80)
        report.append("INTEGRATED EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Performance summary
        report.append("PERFORMANCE METRICS")
        report.append("-" * 80)
        perf_summary = perf_df[['model_name', 'MAE', 'RMSE', 'R2', 'Lead_Time']].to_string(index=False)
        report.append(perf_summary)
        report.append("")
        
        # Robustness summary
        report.append("ROBUSTNESS METRICS")
        report.append("-" * 80)
        rob_summary = rob_df.groupby(['model_name', 'perturbation_type']).agg({
            'degradation_percent': ['mean', 'std']
        }).to_string()
        report.append(rob_summary)
        report.append("")
        
        # Integrated ranking
        report.append("INTEGRATED RANKING (by composite score)")
        report.append("-" * 80)
        ranking = int_df.sort_values('composite_score')[
            ['model_name', 'MAE', 'mean_degradation', 'composite_score']
        ].to_string(index=False)
        report.append(ranking)
        report.append("")
        
        report.append("=" * 80)
        
        report_str = "\n".join(report)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_str)
            logger.info(f"Report saved to {output_path}")
        
        return report_str
    
    def save_results(self, output_dir: Path):
        """
        Save all evaluation results to CSV files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self._performance_results is not None:
            self._performance_results.to_csv(
                output_dir / 'performance_results.csv',
                index=False
            )
            logger.info(f"Saved performance results to {output_dir / 'performance_results.csv'}")
        
        if self._robustness_results is not None:
            self._robustness_results.to_csv(
                output_dir / 'robustness_results.csv',
                index=False
            )
            logger.info(f"Saved robustness results to {output_dir / 'robustness_results.csv'}")
        
        if self._integrated_results is not None:
            self._integrated_results.to_csv(
                output_dir / 'integrated_results.csv',
                index=False
            )
            logger.info(f"Saved integrated results to {output_dir / 'integrated_results.csv'}")
