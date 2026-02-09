"""
Paper-ready visualization utilities.

Generates publication-quality plots for research papers:
- Model comparison plots
- Robustness degradation curves
- Calibration plots
- ROC/Precision-Recall curves
- Learning curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class PaperVisualizer:
    """Publication-quality visualization generator."""
    
    def __init__(self, style='seaborn-v0_8-paper', palette='colorblind'):
        """
        Initialize visualizer with publication settings.
        
        Args:
            style: Matplotlib style
            palette: Seaborn color palette
        """
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not found, using default")
        
        self.palette = sns.color_palette(palette)
        sns.set_palette(palette)
    
    def plot_model_comparison(
        self,
        results: pd.DataFrame,
        metrics: List[str],
        output_file: Optional[Path] = None,
        figsize: Tuple[float, float] = (10, 4),
        log_scale: bool = False
    ):
        """
        Bar plot comparing models across multiple metrics.
        
        Args:
            results: DataFrame with model results
            metrics: List of metrics to plot
            output_file: Optional output file path
            figsize: Figure size (width, height)
            log_scale: Use log scale for y-axis
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Get data
            if metric in results.columns:
                data = results[['model_name', metric]].dropna()
            elif f'{metric}_mean' in results.columns:
                data = results[['model_name', f'{metric}_mean']].dropna()
                data = data.rename(columns={f'{metric}_mean': metric})
            else:
                logger.warning(f"Metric {metric} not found")
                continue
            
            # Sort by metric value
            data = data.sort_values(metric, ascending=True)
            
            # Plot
            bars = ax.barh(data['model_name'], data[metric], color=self.palette[idx % len(self.palette)])
            
            # Highlight best
            best_idx = data[metric].argmin() if any(err in metric.lower() for err in ['mae', 'rmse', 'error']) else data[metric].argmax()
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
            
            ax.set_xlabel(metric.upper())
            ax.set_ylabel('')
            
            if log_scale:
                ax.set_xscale('log')
            
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved model comparison to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_robustness_degradation(
        self,
        robustness_results: pd.DataFrame,
        output_file: Optional[Path] = None,
        figsize: Tuple[float, float] = (12, 4)
    ):
        """
        Plot degradation curves for different perturbation types.
        
        Args:
            robustness_results: DataFrame with robustness analysis results
            output_file: Optional output file path
            figsize: Figure size
        """
        # Get perturbation types
        pert_types = robustness_results['perturbation_type'].unique()
        n_types = len(pert_types)
        
        fig, axes = plt.subplots(1, n_types, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        models = robustness_results['model_name'].unique()
        
        for idx, pert_type in enumerate(pert_types):
            ax = axes[idx]
            
            pert_data = robustness_results[robustness_results['perturbation_type'] == pert_type]
            
            for model_idx, model in enumerate(models):
                model_data = pert_data[pert_data['model_name'] == model].sort_values('intensity')
                
                if len(model_data) > 0:
                    ax.plot(
                        model_data['intensity'],
                        model_data['degradation_percent'],
                        marker='o',
                        label=model,
                        color=self.palette[model_idx % len(self.palette)],
                        linewidth=2,
                        markersize=6
                    )
            
            ax.set_xlabel('Perturbation Intensity')
            ax.set_ylabel('Degradation (%)')
            ax.set_title(pert_type.replace('_', ' ').title())
            ax.grid(alpha=0.3)
            ax.legend(loc='best', framealpha=0.9)
            
            # Add horizontal line at 0%
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved robustness plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_calibration(
        self,
        T_true: np.ndarray,
        T_pred: np.ndarray,
        model_name: str = 'Model',
        output_file: Optional[Path] = None,
        figsize: Tuple[float, float] = (6, 6),
        n_bins: int = 10
    ):
        """
        Calibration plot with binned predictions vs observations.
        
        Args:
            T_true: True event times
            T_pred: Predicted event times
            model_name: Model name for title
            output_file: Optional output file path
            figsize: Figure size
            n_bins: Number of bins for calibration
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Remove invalid predictions
        valid_mask = ~(np.isnan(T_pred) | np.isinf(T_pred))
        T_true = T_true[valid_mask]
        T_pred = T_pred[valid_mask]
        
        if len(T_true) < n_bins:
            logger.warning("Not enough samples for calibration plot")
            return
        
        # Bin predictions
        bins = np.percentile(T_pred, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(T_pred, bins[:-1]) - 1
        
        mean_pred = []
        mean_true = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                mean_pred.append(T_pred[mask].mean())
                mean_true.append(T_true[mask].mean())
        
        # Plot
        ax.scatter(mean_pred, mean_true, s=100, alpha=0.6, color=self.palette[0], edgecolors='black', linewidth=1.5)
        
        # Perfect calibration line
        min_val = min(min(mean_pred), min(mean_true))
        max_val = max(max(mean_pred), max(mean_true))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Calibration')
        
        ax.set_xlabel('Mean Predicted Time')
        ax.set_ylabel('Mean Observed Time')
        ax.set_title(f'Calibration Plot: {model_name}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved calibration plot to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_metric_heatmap(
        self,
        results: pd.DataFrame,
        metrics: List[str],
        output_file: Optional[Path] = None,
        figsize: Tuple[float, float] = (8, 6),
        normalize_cols: bool = True
    ):
        """
        Heatmap showing model performance across metrics.
        
        Args:
            results: DataFrame with model results
            metrics: List of metrics to include
            output_file: Optional output file path
            figsize: Figure size
            normalize_cols: Normalize each metric to [0, 1]
        """
        # Select data
        pivot_data = results.set_index('model_name')[metrics].copy()
        
        # Normalize if requested
        if normalize_cols:
            for metric in metrics:
                values = pivot_data[metric]
                min_val, max_val = values.min(), values.max()
                
                if max_val - min_val > 0:
                    # Check if lower is better
                    is_error = any(err in metric.lower() for err in ['mae', 'rmse', 'error'])
                    
                    if is_error:
                        # Lower is better: invert so higher values = better
                        pivot_data[metric] = 1 - (values - min_val) / (max_val - min_val)
                    else:
                        # Higher is better: normalize normally
                        pivot_data[metric] = (values - min_val) / (max_val - min_val)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Normalized Score' if normalize_cols else 'Value'},
            ax=ax,
            linewidths=0.5
        )
        
        ax.set_xlabel('Metric')
        ax.set_ylabel('Model')
        ax.set_title('Model Performance Heatmap')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved heatmap to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_pareto_front(
        self,
        results: pd.DataFrame,
        x_metric: str,
        y_metric: str,
        output_file: Optional[Path] = None,
        figsize: Tuple[float, float] = (8, 6),
        x_ascending: bool = True,
        y_ascending: bool = True
    ):
        """
        Pareto front showing trade-offs between two metrics.
        
        Args:
            results: DataFrame with model results
            x_metric: Metric for x-axis
            y_metric: Metric for y-axis
            output_file: Optional output file path
            figsize: Figure size
            x_ascending: Whether lower is better for x_metric
            y_ascending: Whether lower is better for y_metric
        """
        # Extract data
        data = results[['model_name', x_metric, y_metric]].dropna()
        
        if len(data) == 0:
            logger.warning("No data for Pareto plot")
            return
        
        # Normalize for Pareto dominance calculation
        x = data[x_metric].values
        y = data[y_metric].values
        
        # Flip if lower is better
        if x_ascending:
            x = -x
        if y_ascending:
            y = -y
        
        # Find Pareto front
        pareto_mask = np.ones(len(data), dtype=bool)
        
        for i in range(len(data)):
            for j in range(len(data)):
                if i != j:
                    # j dominates i if j is better in both objectives
                    if x[j] >= x[i] and y[j] >= y[i] and (x[j] > x[i] or y[j] > y[i]):
                        pareto_mask[i] = False
                        break
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Non-Pareto points
        ax.scatter(
            data.loc[~pareto_mask, x_metric],
            data.loc[~pareto_mask, y_metric],
            s=100,
            alpha=0.5,
            color='lightgray',
            label='Dominated'
        )
        
        # Pareto front
        pareto_data = data[pareto_mask].sort_values(x_metric, ascending=x_ascending)
        
        ax.scatter(
            pareto_data[x_metric],
            pareto_data[y_metric],
            s=150,
            alpha=0.9,
            color=self.palette[0],
            edgecolors='black',
            linewidth=2,
            label='Pareto Front',
            zorder=5
        )
        
        # Connect Pareto front
        ax.plot(
            pareto_data[x_metric],
            pareto_data[y_metric],
            'k--',
            alpha=0.5,
            linewidth=1.5,
            zorder=4
        )
        
        # Annotate points
        for _, row in data.iterrows():
            ax.annotate(
                row['model_name'],
                (row[x_metric], row[y_metric]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )
        
        ax.set_xlabel(x_metric.upper())
        ax.set_ylabel(y_metric.upper())
        ax.set_title(f'Pareto Front: {x_metric} vs {y_metric}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved Pareto front to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_all_plots(
        self,
        performance_results: pd.DataFrame,
        robustness_results: Optional[pd.DataFrame] = None,
        output_dir: Path = Path('results/figures'),
        metrics: List[str] = ['MAE', 'RMSE', 'c_index']
    ):
        """
        Generate complete set of publication-ready plots.
        
        Args:
            performance_results: Performance evaluation results
            robustness_results: Robustness evaluation results (optional)
            output_dir: Output directory for figures
            metrics: Metrics to visualize
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating plots in {output_dir}")
        
        # 1. Model comparison
        logger.info("1. Model comparison...")
        self.plot_model_comparison(
            performance_results,
            metrics=metrics[:3] if len(metrics) >= 3 else metrics,
            output_file=output_dir / 'model_comparison.pdf'
        )
        
        # 2. Robustness degradation
        if robustness_results is not None and not robustness_results.empty:
            logger.info("2. Robustness degradation...")
            self.plot_robustness_degradation(
                robustness_results,
                output_file=output_dir / 'robustness_degradation.pdf'
            )
        
        # 3. Performance heatmap
        logger.info("3. Performance heatmap...")
        self.plot_metric_heatmap(
            performance_results,
            metrics=metrics,
            output_file=output_dir / 'performance_heatmap.pdf'
        )
        
        # 4. Pareto front (error vs complexity proxy)
        if len(metrics) >= 2:
            logger.info("4. Pareto front...")
            self.plot_pareto_front(
                performance_results,
                x_metric=metrics[0],
                y_metric=metrics[1],
                output_file=output_dir / f'pareto_{metrics[0]}_{metrics[1]}.pdf',
                x_ascending=True,  # MAE/RMSE: lower is better
                y_ascending=False  # c_index: higher is better
            )
        
        logger.info(f"All plots saved to {output_dir}")
