"""
Visualization utilities for baseline evaluation results.

Creates plots per experimental protocol:
- Learning curves (train vs val vs test)
- ROC and Precision-Recall curves
- Lead Time vs Horizon
- Robustness degradation curves
- Optionality curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


class BaselineVisualizer:
    """Create visualizations for baseline evaluation."""

    def __init__(self, output_dir: Path):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_learning_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: List[float],
        val_metrics: List[float],
        title: str = "Learning Curves",
        save_name: str = "learning_curves.png",
    ) -> None:
        """
        Plot learning curves.
        
        Args:
            train_losses: Training loss per epoch
            val_losses: Validation loss per epoch
            train_metrics: Training metric per epoch
            val_metrics: Validation metric per epoch
            title: Plot title
            save_name: Filename to save
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss plot
        ax1.plot(epochs, train_losses, "b-", label="Train Loss", marker="o", markersize=4)
        ax1.plot(epochs, val_losses, "r-", label="Val Loss", marker="s", markersize=4)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Metric plot
        ax2.plot(epochs, train_metrics, "b-", label="Train Metric", marker="o", markersize=4)
        ax2.plot(epochs, val_metrics, "r-", label="Val Metric", marker="s", markersize=4)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metric")
        ax2.set_title("Metric Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved learning curves to {save_path}")
        plt.close()

    def plot_roc_curves(
        self,
        fpr_dict: Dict[str, np.ndarray],
        tpr_dict: Dict[str, np.ndarray],
        auc_dict: Dict[str, float],
        title: str = "ROC Curves - Test Set",
        save_name: str = "roc_curves.png",
    ) -> None:
        """
        Plot ROC curves for multiple models.
        
        Args:
            fpr_dict: False positive rates by model
            tpr_dict: True positive rates by model
            auc_dict: AUC scores by model
            title: Plot title
            save_name: Filename to save
        """
        plt.figure(figsize=(10, 8))
        
        # Diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier (AUC=0.5)")
        
        # Plot ROC curves
        colors = plt.cm.Set2(np.linspace(0, 1, len(fpr_dict)))
        for (model_name, fpr), color in zip(fpr_dict.items(), colors):
            tpr = tpr_dict[model_name]
            auc = auc_dict[model_name]
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", lw=2, color=color)
        
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved ROC curves to {save_path}")
        plt.close()

    def plot_precision_recall(
        self,
        precision_dict: Dict[str, np.ndarray],
        recall_dict: Dict[str, np.ndarray],
        pr_auc_dict: Dict[str, float],
        title: str = "Precision-Recall Curves - Test Set",
        save_name: str = "precision_recall.png",
    ) -> None:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            precision_dict: Precision values by model
            recall_dict: Recall values by model
            pr_auc_dict: PR-AUC scores by model
            title: Plot title
            save_name: Filename to save
        """
        plt.figure(figsize=(10, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(precision_dict)))
        for (model_name, precision), color in zip(precision_dict.items(), colors):
            recall = recall_dict[model_name]
            pr_auc = pr_auc_dict[model_name]
            plt.plot(recall, precision, label=f"{model_name} (PR-AUC={pr_auc:.3f})", 
                    lw=2, color=color)
        
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="best")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved precision-recall curves to {save_path}")
        plt.close()

    def plot_degradation_curves(
        self,
        robustness_results: Dict[str, Dict[str, Any]],
        perturbation_type: str = "gaussian_noise",
        title: str = "Robustness: Lead Time Degradation",
        save_name: str = "degradation_gaussian.png",
    ) -> None:
        """
        Plot degradation curves for robustness evaluation.
        
        x-axis: perturbation intensity
        y-axis: metric degradation %
        
        Args:
            robustness_results: Results from RobustnessAnalyzer
            perturbation_type: 'gaussian_noise', 'feature_dropout', or 'temporal_delay'
            title: Plot title
            save_name: Filename to save
        """
        plt.figure(figsize=(10, 7))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(robustness_results)))
        
        for (model_name, results), color in zip(robustness_results.items(), colors):
            if perturbation_type not in results:
                logger.warning(f"No {perturbation_type} results for {model_name}")
                continue
            
            data = results[perturbation_type]
            intensities = data["intensities"]
            degradations = data["degradations"]
            auc_d = data["auc_d"]
            
            plt.plot(intensities, degradations, marker="o", label=f"{model_name} (AUC-D={auc_d:.2f})",
                    lw=2, color=color, markersize=8)
        
        plt.xlabel("Perturbation Intensity")
        plt.ylabel("Metric Degradation (%)")
        plt.title(title)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="k", linestyle="--", lw=1, alpha=0.5)
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved degradation curve to {save_path}")
        plt.close()

    def plot_lead_time_distribution(
        self,
        lt_norms_dict: Dict[str, np.ndarray],
        title: str = "Normalized Lead Time Distribution",
        save_name: str = "lead_time_dist.png",
    ) -> None:
        """
        Plot distribution of normalized lead times.
        
        Args:
            lt_norms_dict: Normalized lead times by model
            title: Plot title
            save_name: Filename to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(lt_norms_dict)))
        
        # Boxplot
        ax = axes[0]
        data_for_box = [lt_norms[lt_norms >= 0] for lt_norms in lt_norms_dict.values()]
        bp = ax.boxplot(data_for_box, labels=list(lt_norms_dict.keys()), patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("Lead Time (Normalized)")
        ax.set_title("Lead Time Distribution")
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        # Histograms
        for idx, (model_name, lt_norms) in enumerate(lt_norms_dict.items()):
            if idx < 3:
                ax = axes[idx + 1]
                valid_lt = lt_norms[lt_norms >= 0]
                ax.hist(valid_lt, bins=20, color=colors[idx], alpha=0.7, edgecolor="black")
                ax.set_xlabel("Lead Time (Normalized)")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{model_name}")
                ax.grid(True, alpha=0.3, axis="y")
        
        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved lead time distribution to {save_path}")
        plt.close()

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metric_names: List[str],
        title: str = "Metrics Comparison",
        save_name: str = "metrics_comparison.png",
    ) -> None:
        """
        Plot comparison of metrics across models.
        
        Args:
            metrics_dict: Metrics by model by metric name
            metric_names: List of metrics to plot
            title: Plot title
            save_name: Filename to save
        """
        # Convert to DataFrame for easy plotting
        data = []
        for model_name, metrics in metrics_dict.items():
            row = {"Model": model_name}
            for metric_name in metric_names:
                row[metric_name] = metrics.get(metric_name, np.nan)
            data.append(row)
        
        df = pd.DataFrame(data)
        df_melted = df.melt(id_vars=["Model"], var_name="Metric", value_name="Value")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric", ax=ax)
        
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved metrics comparison to {save_path}")
        plt.close()

    def plot_all_degradation_curves(
        self,
        robustness_results: Dict[str, Dict[str, Any]],
        title_prefix: str = "Robustness Analysis",
        save_prefix: str = "degradation",
    ) -> None:
        """
        Plot degradation curves for all perturbation types.
        
        Args:
            robustness_results: Results from RobustnessAnalyzer
            title_prefix: Prefix for titles
            save_prefix: Prefix for filenames
        """
        perturbation_types = ["gaussian_noise", "feature_dropout", "temporal_delay"]
        titles = ["Gaussian Noise", "Feature Dropout", "Temporal Delay"]
        
        for perturb_type, title_suffix in zip(perturbation_types, titles):
            self.plot_degradation_curves(
                robustness_results,
                perturbation_type=perturb_type,
                title=f"{title_prefix}: {title_suffix}",
                save_name=f"{save_prefix}_{perturb_type}.png",
            )
