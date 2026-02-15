"""
Main evaluation orchestrator for baseline models.

Implements complete experimental protocol:
- Load/create temporal splits for multiple seeds
- Train/evaluate all baseline models
- Compute metrics per paradigm
- Evaluate robustness under perturbations
- Generate comprehensive visualizations
- Aggregate results across seeds
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, Tuple
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import json

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

from src.utils.device import get_device, set_reproducible
from src.utils.logging import setup_logging
from src.evaluation import (
    TemporalDataSplitter,
    create_splits_for_seeds,
    MetricsComputer,
    RobustnessAnalyzer,
    BaselineVisualizer,
)

logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """
    Complete evaluation pipeline for baseline models.

    Per experimental protocol:
    - Temporal data splits (70-15-15)
    - Multiple seeds (baseline + 4 additional)
    - Paradigm-specific metrics
    - Robustness analysis
    - Comprehensive visualizations
    """

    def __init__(
        self,
        data_dir: Path,
        results_dir: Path,
        world_name: str,
        seeds: List[int] = None,
    ):
        """Initialize evaluator."""
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.world_name = world_name
        self.seeds = seeds or [42, 123, 456, 789, 999]
        self.device = get_device()

        # Create output directories
        self.eval_dir = self.results_dir / "evaluation"
        self.splits_dir = self.eval_dir / "data_splits"
        self.models_dir = self.eval_dir / "models"
        self.metrics_dir = self.eval_dir / "metrics"
        self.plots_dir = self.eval_dir / "plots"

        for d in [
            self.eval_dir,
            self.splits_dir,
            self.models_dir,
            self.metrics_dir,
            self.plots_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.all_results: Dict[int, Dict[str, Any]] = {}

    def create_data_splits(self) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Create temporal data splits for all seeds.

        Returns:
            Dictionary: {seed: {split: {X, y, T}}}
        """
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: Creating Temporal Data Splits")
        logger.info("=" * 70)

        all_splits = create_splits_for_seeds(
            data_dir=self.data_dir,
            world_name=self.world_name,
            output_base_dir=self.splits_dir,
            seeds=self.seeds,
        )

        logger.info(f"\n[OK] Created and saved splits for seeds: {self.seeds}")
        return all_splits

    def evaluate_seed(
        self,
        seed: int,
        split_data: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Any]:
        """
        Evaluate all models for a single seed.

        Args:
            seed: Random seed
            split_data: {split: {X, y, T}}

        Returns:
            Dictionary with results for this seed
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Evaluating Seed {seed}")
        logger.info(f"{'='*70}")

        # Extract data
        X_train, y_train, T_train = (
            split_data["train"]["X"],
            split_data["train"]["y"],
            split_data["train"]["T"],
        )
        X_val, y_val, T_val = (
            split_data["val"]["X"],
            split_data["val"]["y"],
            split_data["val"]["T"],
        )
        X_test, y_test, T_test = (
            split_data["test"]["X"],
            split_data["test"]["y"],
            split_data["test"]["T"],
        )

        # Flatten for non-sequence models
        def flatten(X):
            if X.ndim == 3:
                return X.reshape(X.shape[0], -1)
            return X

        X_train_flat = flatten(X_train)
        X_val_flat = flatten(X_val)
        X_test_flat = flatten(X_test)

        seed_results = {
            "seed": seed,
            "models": {},
            "metrics": {},
            "robustness": {},
        }

        # =====================================================================
        # Train and Evaluate Models
        # =====================================================================

        # Random Forest
        logger.info("\nTraining Random Forest...")
        try:
            rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed
            )
            rf_model.fit(X_train_flat, y_train)

            y_pred_rf = rf_model.predict(X_test_flat)
            y_proba_rf = rf_model.predict_proba(X_test_flat)[:, 1]

            # Metrics
            class_metrics = MetricsComputer.classification_metrics(
                y_test, y_pred_rf, y_proba_rf
            )
            seed_results["metrics"]["random_forest"] = class_metrics
            seed_results["models"]["random_forest"] = rf_model

            logger.info(
                f"  F1={class_metrics['f1']:.4f}, AUC-ROC={class_metrics['auc_roc']:.4f}"
            )

            # Robustness (optional - may fail without impact)
            try:

                def predict_fn(X):
                    return rf_model.predict(flatten(X))

                def metric_fn(y_true, y_pred):
                    return float(np.mean(y_pred == y_true))  # Accuracy

                analyzer = RobustnessAnalyzer()
                robustness = analyzer.evaluate_model_robustness(
                    "random_forest", X_test, y_test, predict_fn, metric_fn, "accuracy"
                )
                seed_results["robustness"]["random_forest"] = robustness
                logger.info("  Robustness analysis skipped (optional)")
            except Exception as e2:
                logger.debug(f"  Robustness analysis skipped: {e2}")

        except Exception as e:
            logger.error(f"Random Forest failed: {e}")

        # MLP Classifier
        logger.info("\nTraining MLP Classifier...")
        try:
            mlp_model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=1000,
                random_state=seed,
            )
            mlp_model.fit(X_train_flat, y_train)

            y_pred_mlp = mlp_model.predict(X_test_flat)
            y_proba_mlp = mlp_model.predict_proba(X_test_flat)[:, 1]

            class_metrics = MetricsComputer.classification_metrics(
                y_test, y_pred_mlp, y_proba_mlp
            )
            seed_results["metrics"]["mlp"] = class_metrics
            seed_results["models"]["mlp"] = mlp_model

            logger.info(
                f"  F1={class_metrics['f1']:.4f}, AUC-ROC={class_metrics['auc_roc']:.4f}"
            )

        except Exception as e:
            logger.error(f"MLP failed: {e}")

        # Ridge Regression (for Cox-like survival)
        logger.info("\nTraining Ridge Regression...")
        try:
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_train_flat, y_train)

            y_pred_ridge = ridge_model.predict(X_test_flat)

            reg_metrics = MetricsComputer.regression_metrics(y_test, y_pred_ridge)
            seed_results["metrics"]["ridge"] = reg_metrics
            seed_results["models"]["ridge"] = ridge_model

            logger.info(
                f"  RMSE={reg_metrics['rmse']:.4f}, MAE={reg_metrics['mae']:.4f}"
            )

        except Exception as e:
            logger.error(f"Ridge failed: {e}")

        # =====================================================================
        # Regression Models for Time-to-Event Prediction
        # =====================================================================

        # Gradient Boosting Regressor
        logger.info("\nTraining Gradient Boosting Regressor (Time-to-Event)...")
        try:
            from src.baseline.regression import RegressionBaselines

            result = RegressionBaselines.train_gradient_boosting(
                X_train, y_train, X_test, y_test
            )
            if result:
                seed_results["metrics"]["gradient_boosting"] = {
                    "rmse": result["rmse"],
                    "mae": result["mae"],
                    "r2": result["r2"],
                }
                logger.info(
                    f"  RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, R²={result['r2']:.4f}"
                )
        except Exception as e:
            logger.debug(f"Gradient Boosting failed: {e}")

        # MLP Regressor
        logger.info("\nTraining MLP Regressor (Time-to-Event)...")
        try:
            from src.baseline.regression import RegressionBaselines

            result = RegressionBaselines.train_mlp_regression(
                X_train, y_train, X_test, y_test, device=device
            )
            if result:
                seed_results["metrics"]["mlp_regressor"] = {
                    "rmse": result["rmse"],
                    "mae": result["mae"],
                    "r2": result["r2"],
                }
                logger.info(
                    f"  RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, R²={result['r2']:.4f}"
                )
        except Exception as e:
            logger.debug(f"MLP Regressor failed: {e}")

        # Lead Time Analysis
        logger.info("\nComputing Lead Time...")
        try:
            # Simple heuristic: alert when prediction probability exceeds threshold
            alert_times = []

            try:
                # Try using RF if available, else use MLP
                model_to_use = seed_results["models"].get(
                    "random_forest"
                ) or seed_results["models"].get("mlp")

                if model_to_use:
                    for i in range(len(X_test)):
                        features = X_test_flat[i].reshape(1, -1)
                        try:
                            proba = model_to_use.predict_proba(features)
                            # Handle case with 1 or 2 classes
                            if proba.shape[1] >= 2:
                                prob_event = proba[0, 1]
                            else:
                                prob_event = proba[0, 0]
                        except:
                            prob_event = 0.5

                        # Simple alert time: proportional to probability and event time
                        threshold = 0.5
                        if prob_event > threshold:
                            alert_time = max(0, T_test[i] * (1 - prob_event))
                        else:
                            alert_time = T_test[i]

                        alert_times.append(alert_time)

                    alert_times = np.array(alert_times)
                    lt_norms = MetricsComputer.normalized_lead_time(T_test, alert_times)
                    lt_stats = MetricsComputer.compute_lead_time_stats(lt_norms)

                    seed_results["metrics"]["lead_time"] = lt_stats
                    seed_results["lead_time_norms"] = lt_norms

                    logger.info(
                        f"  Mean LT_norm={lt_stats['mean']:.4f}±{lt_stats['std']:.4f}"
                    )
            except Exception as e2:
                logger.warning(f"Could not compute lead times: {e2}")

        except Exception as e:
            logger.warning(f"Lead time analysis failed: {e}")

        logger.info(f"\n[OK] Completed evaluation for seed {seed}")
        return seed_results

    def run_evaluation(self) -> Dict[int, Dict[str, Any]]:
        """
        Run complete evaluation across all seeds.

        Returns:
            Dictionary with all results
        """
        logger.info("\n" + "=" * 70)
        logger.info("BASELINE EVALUATION - EXPERIMENTAL PROTOCOL")
        logger.info("=" * 70)
        logger.info(f"World: {self.world_name}")
        logger.info(f"Seeds: {self.seeds}")
        logger.info(f"Data Dir: {self.data_dir}")
        logger.info(f"Results Dir: {self.results_dir}")
        logger.info("=" * 70)

        # Create splits
        splits = self.create_data_splits()

        # Evaluate each seed
        for seed in self.seeds:
            set_reproducible(seed)

            seed_results = self.evaluate_seed(seed, splits[seed])
            self.all_results[seed] = seed_results

        # Aggregate results
        self.aggregate_results()

        # Save results
        self.save_results()

        # Generate visualizations
        self.generate_visualizations()

        logger.info("\n" + "=" * 70)
        logger.info("[OK] EVALUATION COMPLETE")
        logger.info("=" * 70)

        return self.all_results

    def aggregate_results(self) -> None:
        """Aggregate results across seeds and save summary."""
        logger.info("\n" + "=" * 70)
        logger.info("Aggregating Results Across Seeds")
        logger.info("=" * 70)

        # Aggregate metrics
        all_metrics = {model: [] for model in ["random_forest", "mlp", "ridge"]}
        all_lt = {model: [] for model in ["random_forest", "mlp", "ridge"]}

        for seed, results in self.all_results.items():
            for model_name, metrics in results.get("metrics", {}).items():
                if model_name in all_metrics:
                    all_metrics[model_name].append(metrics)

            if "lead_time_norms" in results:
                all_lt["random_forest"].append(results["lead_time_norms"])

        # Compute summary statistics
        summary = {
            "world": self.world_name,
            "seeds": self.seeds,
            "timestamp": datetime.now().isoformat(),
            "models": {},
        }

        for model_name in ["random_forest", "mlp", "ridge"]:
            if all_metrics[model_name]:
                metrics_list = all_metrics[model_name]

                # Average metrics across seeds
                model_summary = {}
                metric_names = list(metrics_list[0].keys())

                for metric_name in metric_names:
                    values = [m[metric_name] for m in metrics_list if metric_name in m]
                    if values:
                        model_summary[f"{metric_name}_mean"] = float(np.mean(values))
                        model_summary[f"{metric_name}_std"] = float(np.std(values))

                summary["models"][model_name] = model_summary

                logger.info(f"\n{model_name.upper()}")
                for key, val in model_summary.items():
                    logger.info(f"  {key}: {val:.4f}")

        # Save summary
        summary_path = self.metrics_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n[OK] Saved summary to {summary_path}")

    def save_results(self) -> None:
        """Save all results to disk."""
        logger.info("\n" + "=" * 70)
        logger.info("Saving Results")
        logger.info("=" * 70)

        # Save complete results as pickle
        results_path = self.eval_dir / "all_results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(self.all_results, f)

        logger.info(f"[OK] Saved complete results to {results_path}")

    def generate_visualizations(self) -> None:
        """Generate all plots and visualizations."""
        logger.info("\n" + "=" * 70)
        logger.info("Generating Visualizations")
        logger.info("=" * 70)

        visualizer = BaselineVisualizer(output_dir=self.plots_dir)

        # Extract data for visualization
        metrics_by_model = {}
        lead_times_by_seed = {}

        # Aggregate across all seeds
        for seed, seed_results in self.all_results.items():
            seed_key = f"seed_{seed}"

            # Metrics
            for model_name, metrics in seed_results.get("metrics", {}).items():
                if model_name not in metrics_by_model:
                    metrics_by_model[model_name] = []
                metrics_by_model[model_name].append(metrics)

            # Lead times
            if "lead_time_norms" in seed_results:
                lead_times_by_seed[seed_key] = seed_results["lead_time_norms"]

        logger.info(f"Data collected from {len(self.all_results)} seeds")
        logger.info(f"Models with metrics: {list(metrics_by_model.keys())}")
        logger.info(f"Seeds with lead times: {len(lead_times_by_seed)}")

        # 1. Metrics Comparison (average across seeds) - ALWAYS create this
        logger.info("Generating metrics comparison plot...")
        try:
            if metrics_by_model:
                # Average metrics across seeds
                avg_metrics = {}
                for model_name, metrics_list in metrics_by_model.items():
                    if metrics_list:
                        avg_metrics[model_name] = {}
                        # Get numeric metrics to average
                        first_metrics = metrics_list[0]
                        for key in first_metrics:
                            if isinstance(first_metrics[key], (int, float)):
                                values = [
                                    m[key]
                                    for m in metrics_list
                                    if key in m and isinstance(m[key], (int, float))
                                ]
                                if values:
                                    avg_metrics[model_name][key] = float(
                                        np.mean(values)
                                    )

                if avg_metrics:
                    # Select main metrics to plot
                    metric_names = ["f1", "auc_roc", "pr_auc", "rmse", "mae"]

                    visualizer.plot_metrics_comparison(
                        avg_metrics,
                        metric_names=metric_names,
                        title="Model Metrics Comparison (Averaged Across Seeds)",
                        save_name="03_metrics_comparison.png",
                    )
                    logger.info("[OK] Metrics comparison plot generated")
            else:
                logger.warning("No metrics data available for visualization")
        except Exception as e:
            logger.warning(f"Failed to generate metrics comparison plot: {e}")
            import traceback

            traceback.print_exc()

        # 2. Lead Time Distribution
        logger.info("Generating Lead Time distribution plot...")
        try:
            if lead_times_by_seed:
                visualizer.plot_lead_time_distribution(
                    lead_times_by_seed,
                    title="Normalized Lead Time Distribution (Test Set)",
                    save_name="01_lead_time_distribution.png",
                )
                logger.info("[OK] Lead time distribution plot generated")
            else:
                logger.warning("No lead time data available for visualization")
        except Exception as e:
            logger.warning(f"Failed to generate lead time plot: {e}")
            import traceback

            traceback.print_exc()

        # 3. Robustness Degradation Curves (optional, may be empty)
        logger.info("Checking for robustness data...")
        robustness_by_model = {}
        for seed, seed_results in self.all_results.items():
            # Robustness - collect from any seed with data
            for model_name, robustness in seed_results.get("robustness", {}).items():
                if robustness and model_name not in robustness_by_model:
                    robustness_by_model[model_name] = robustness

        if robustness_by_model:
            logger.info(
                f"Generating robustness plots for {len(robustness_by_model)} models..."
            )
            try:
                visualizer.plot_all_degradation_curves(
                    robustness_by_model,
                    title_prefix="Baseline Model Robustness Analysis",
                    save_prefix="02_robustness_degradation",
                )
                logger.info("[OK] Robustness degradation plots generated")
            except Exception as e:
                logger.warning(f"Failed to generate robustness plots: {e}")
                import traceback

                traceback.print_exc()
        else:
            logger.info(
                "No robustness data collected (robustness analysis may have been skipped)"
            )

        logger.info("\n[OK] Visualization generation complete!")
        logger.info(f"Plots saved to: {self.plots_dir}")
        logger.info("=" * 70)


def run_baseline_evaluation(
    world_name: str = "ant_colony",
    data_dir: Path = None,
    results_dir: Path = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Run complete baseline evaluation following experimental protocol.

    Args:
        world_name: Dataset name (ant_colony, sir_graph)
        data_dir: Path to data directory
        results_dir: Path to results directory

    Returns:
        Dictionary with all evaluation results
    """
    set_reproducible(seed=42)
    setup_logging(log_level=logging.INFO)

    # Determine paths
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results" / world_name

    data_dir = Path(data_dir)
    results_dir = Path(results_dir)

    # Create evaluator
    evaluator = BaselineEvaluator(
        data_dir=data_dir,
        results_dir=results_dir,
        world_name=world_name,
    )

    # Run evaluation
    results = evaluator.run_evaluation()
    evaluator.save_results()

    return results


if __name__ == "__main__":
    import sys

    world = sys.argv[1] if len(sys.argv) > 1 else "ant_colony"
    run_baseline_evaluation(world_name=world)
