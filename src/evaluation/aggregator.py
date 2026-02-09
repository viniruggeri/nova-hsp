"""
Results aggregation and cross-world comparison.

Aggregates evaluation results across different worlds, datasets, and experiments.
Provides statistical comparison and LaTeX export for paper-ready tables.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from scipy import stats
import logging
import json

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """
    Aggregate and compare results across multiple experiments.
    
    Supports:
    - Cross-world comparison (SIR graph, Ant Colony, Real Dataset)
    - Statistical significance testing
    - Multi-criteria ranking
    - LaTeX table generation
    """
    
    def __init__(self):
        self.results: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def add_results(
        self,
        name: str,
        results: pd.DataFrame,
        metadata: Optional[Dict] = None
    ):
        """
        Add evaluation results from a single experiment/world.
        
        Args:
            name: Unique identifier (e.g., 'sir_graph', 'ant_colony', 'real_data')
            results: DataFrame with model evaluation results
            metadata: Optional metadata (world params, config, etc.)
        """
        self.results[name] = results.copy()
        self.metadata[name] = metadata or {}
        logger.info(f"Added results for '{name}': {len(results)} models")
    
    def load_results_from_dir(
        self,
        results_dir: Path,
        pattern: str = '**/integrated_results.csv'
    ):
        """
        Load all result files from a directory structure.
        
        Expected structure:
        results/
            sir_graph/
                integrated_results.csv
                metadata.json
            ant_colony/
                integrated_results.csv
                metadata.json
        
        Args:
            results_dir: Root directory containing results
            pattern: Glob pattern to match result files
        """
        results_dir = Path(results_dir)
        
        for result_file in results_dir.glob(pattern):
            # Extract experiment name from parent directory
            exp_name = result_file.parent.name
            
            # Load results
            try:
                results = pd.read_csv(result_file)
                
                # Try to load metadata
                metadata_file = result_file.parent / 'metadata.json'
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                self.add_results(exp_name, results, metadata)
                
            except Exception as e:
                logger.warning(f"Failed to load {result_file}: {e}")
    
    def aggregate_by_model(
        self,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate results by model across all experiments.
        
        Args:
            metrics: List of metrics to aggregate (default: all numeric columns)
            
        Returns:
            DataFrame with aggregated statistics per model
        """
        if not self.results:
            logger.warning("No results to aggregate")
            return pd.DataFrame()
        
        all_results = []
        
        for exp_name, results in self.results.items():
            results_copy = results.copy()
            results_copy['experiment'] = exp_name
            all_results.append(results_copy)
        
        combined = pd.concat(all_results, ignore_index=True)
        
        # Determine metrics to aggregate
        if metrics is None:
            metrics = combined.select_dtypes(include=[np.number]).columns.tolist()
            # Remove unwanted columns
            metrics = [m for m in metrics if m not in ['rank', 'fold']]
        
        # Aggregate by model
        agg_funcs = {
            metric: ['mean', 'std', 'median', 'min', 'max']
            for metric in metrics
        }
        
        aggregated = combined.groupby('model_name').agg(agg_funcs)
        
        # Flatten column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        aggregated = aggregated.reset_index()
        
        return aggregated
    
    def compare_across_worlds(
        self,
        metric: str = 'MAE',
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Statistical comparison of models across different worlds.
        
        Uses Kruskal-Wallis test to determine if there are significant
        differences across worlds, followed by pairwise Mann-Whitney tests.
        
        Args:
            metric: Metric to compare (e.g., 'MAE', 'c_index')
            alpha: Significance level
            
        Returns:
            DataFrame with comparison results
        """
        if len(self.results) < 2:
            logger.warning("Need at least 2 experiments for comparison")
            return pd.DataFrame()
        
        comparison_results = []
        
        # Get unique models
        all_models = set()
        for results in self.results.values():
            if 'model_name' in results.columns:
                all_models.update(results['model_name'].unique())
        
        for model in all_models:
            # Collect metric values across worlds
            world_values = {}
            
            for exp_name, results in self.results.items():
                model_results = results[results['model_name'] == model]
                
                if len(model_results) > 0 and metric in model_results.columns:
                    values = model_results[metric].dropna().values
                    if len(values) > 0:
                        world_values[exp_name] = values
            
            if len(world_values) < 2:
                continue
            
            # Kruskal-Wallis test (non-parametric ANOVA)
            try:
                groups = list(world_values.values())
                h_stat, p_value = stats.kruskal(*groups)
                
                result = {
                    'model_name': model,
                    'metric': metric,
                    'n_worlds': len(world_values),
                    'kruskal_h': h_stat,
                    'kruskal_p': p_value,
                    'significant': p_value < alpha
                }
                
                # Add mean values per world
                for exp_name, values in world_values.items():
                    result[f'{exp_name}_mean'] = np.mean(values)
                    result[f'{exp_name}_std'] = np.std(values)
                
                comparison_results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed comparison for {model}: {e}")
        
        return pd.DataFrame(comparison_results)
    
    def pairwise_comparison(
        self,
        exp1: str,
        exp2: str,
        metric: str = 'MAE',
        test: str = 'mannwhitneyu'
    ) -> pd.DataFrame:
        """
        Pairwise statistical comparison between two experiments.
        
        Args:
            exp1: Name of first experiment
            exp2: Name of second experiment
            metric: Metric to compare
            test: Statistical test ('mannwhitneyu', 'wilcoxon', or 'ttest')
            
        Returns:
            DataFrame with pairwise comparison results
        """
        if exp1 not in self.results or exp2 not in self.results:
            logger.error(f"Experiments not found: {exp1}, {exp2}")
            return pd.DataFrame()
        
        results1 = self.results[exp1]
        results2 = self.results[exp2]
        
        # Get common models
        models1 = set(results1['model_name'].unique())
        models2 = set(results2['model_name'].unique())
        common_models = models1.intersection(models2)
        
        comparison_results = []
        
        for model in common_models:
            values1 = results1[results1['model_name'] == model][metric].dropna().values
            values2 = results2[results2['model_name'] == model][metric].dropna().values
            
            if len(values1) < 2 or len(values2) < 2:
                continue
            
            try:
                # Perform statistical test
                if test == 'mannwhitneyu':
                    statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                elif test == 'wilcoxon':
                    if len(values1) == len(values2):
                        statistic, p_value = stats.wilcoxon(values1, values2)
                    else:
                        logger.warning(f"Wilcoxon requires equal sample sizes, skipping {model}")
                        continue
                elif test == 'ttest':
                    statistic, p_value = stats.ttest_ind(values1, values2)
                else:
                    logger.error(f"Unknown test: {test}")
                    continue
                
                # Effect size (Cohen's d)
                mean_diff = np.mean(values1) - np.mean(values2)
                pooled_std = np.sqrt((np.std(values1)**2 + np.std(values2)**2) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                comparison_results.append({
                    'model_name': model,
                    f'{exp1}_mean': np.mean(values1),
                    f'{exp1}_std': np.std(values1),
                    f'{exp2}_mean': np.mean(values2),
                    f'{exp2}_std': np.std(values2),
                    'statistic': statistic,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant_005': p_value < 0.05,
                    'significant_001': p_value < 0.01
                })
                
            except Exception as e:
                logger.warning(f"Failed pairwise test for {model}: {e}")
        
        return pd.DataFrame(comparison_results)
    
    def rank_models(
        self,
        metrics: List[str],
        weights: Optional[List[float]] = None,
        ascending: Optional[List[bool]] = None
    ) -> pd.DataFrame:
        """
        Multi-criteria ranking of models.
        
        Args:
            metrics: List of metrics to consider
            weights: Weight for each metric (default: equal weights)
            ascending: Whether lower is better for each metric
                      (default: True for MAE/RMSE, False for R2/c_index)
            
        Returns:
            DataFrame with model rankings
        """
        aggregated = self.aggregate_by_model(metrics)
        
        if aggregated.empty:
            return pd.DataFrame()
        
        # Set default weights and directions
        if weights is None:
            weights = [1.0] * len(metrics)
        
        if ascending is None:
            ascending = []
            for metric in metrics:
                # Lower is better for error metrics
                is_error = any(err in metric.lower() for err in ['mae', 'rmse', 'mse', 'error', 'brier', 'ibs'])
                ascending.append(is_error)
        
        # Normalize metrics to [0, 1] range
        normalized = aggregated.copy()
        
        for metric, asc in zip(metrics, ascending):
            mean_col = f'{metric}_mean'
            if mean_col not in normalized.columns:
                logger.warning(f"Metric {mean_col} not found")
                continue
            
            values = normalized[mean_col]
            min_val, max_val = values.min(), values.max()
            
            if max_val - min_val > 0:
                if asc:
                    # Lower is better: normalize so 0 = best, 1 = worst
                    normalized[f'{metric}_norm'] = (values - min_val) / (max_val - min_val)
                else:
                    # Higher is better: normalize so 1 = best, 0 = worst
                    normalized[f'{metric}_norm'] = 1 - (values - min_val) / (max_val - min_val)
            else:
                normalized[f'{metric}_norm'] = 0.5
        
        # Compute weighted score
        score = np.zeros(len(normalized))
        total_weight = sum(weights)
        
        for metric, weight in zip(metrics, weights):
            norm_col = f'{metric}_norm'
            if norm_col in normalized.columns:
                score += normalized[norm_col].values * (weight / total_weight)
        
        normalized['composite_score'] = score
        normalized['rank'] = normalized['composite_score'].rank(method='min')
        
        # Sort by rank
        normalized = normalized.sort_values('rank')
        
        return normalized
    
    def export_latex_table(
        self,
        output_file: Path,
        metrics: Optional[List[str]] = None,
        caption: str = 'Model Comparison',
        label: str = 'tab:model_comparison',
        format_spec: str = '.3f',
        bold_best: bool = True
    ):
        """
        Export aggregated results as LaTeX table.
        
        Args:
            output_file: Output .tex file path
            metrics: Metrics to include (default: all)
            caption: Table caption
            label: LaTeX label
            format_spec: Number format specification
            bold_best: Bold the best value in each column
        """
        aggregated = self.aggregate_by_model(metrics)
        
        if aggregated.empty:
            logger.error("No results to export")
            return
        
        # Select columns to export (mean values)
        mean_cols = [col for col in aggregated.columns if col.endswith('_mean')]
        export_cols = ['model_name'] + mean_cols
        export_df = aggregated[export_cols].copy()
        
        # Rename columns (remove _mean suffix)
        rename_map = {col: col.replace('_mean', '').upper() for col in mean_cols}
        rename_map['model_name'] = 'Model'
        export_df = export_df.rename(columns=rename_map)
        
        # Format numbers
        for col in export_df.columns:
            if col != 'Model':
                export_df[col] = export_df[col].apply(lambda x: f"{x:{format_spec}}")
        
        # Bold best values
        if bold_best:
            for col in export_df.columns:
                if col == 'Model':
                    continue
                
                # Determine if lower or higher is better
                is_error = any(err in col.lower() for err in ['mae', 'rmse', 'error'])
                
                values = aggregated[[c for c in aggregated.columns if col.lower() in c.lower() and '_mean' in c]]
                if not values.empty:
                    if is_error:
                        best_idx = values.values.argmin()
                    else:
                        best_idx = values.values.argmax()
                    
                    # Bold the best value
                    current_val = export_df.iloc[best_idx][col]
                    export_df.at[best_idx, col] = f"\\textbf{{{current_val}}}"
        
        # Generate LaTeX
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")
        
        # Column specification
        n_cols = len(export_df.columns)
        col_spec = "l" + "r" * (n_cols - 1)
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append("\\toprule")
        
        # Header
        header = " & ".join(export_df.columns) + " \\\\"
        latex.append(header)
        latex.append("\\midrule")
        
        # Data rows
        for _, row in export_df.iterrows():
            row_str = " & ".join(str(v) for v in row.values) + " \\\\"
            latex.append(row_str)
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        # Write to file
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex))
        
        logger.info(f"Exported LaTeX table to {output_file}")
    
    def export_summary_markdown(self, output_file: Path):
        """
        Export human-readable summary in Markdown format.
        
        Args:
            output_file: Output .md file path
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("# Evaluation Results Summary\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            f.write(f"- Total experiments: {len(self.results)}\n")
            
            for exp_name, results in self.results.items():
                n_models = len(results['model_name'].unique())
                f.write(f"- {exp_name}: {n_models} models, {len(results)} runs\n")
            
            f.write("\n## Aggregated Results\n\n")
            
            # Aggregated results
            aggregated = self.aggregate_by_model()
            if not aggregated.empty:
                f.write(aggregated.to_markdown(index=False))
            
            f.write("\n\n## Cross-World Comparison\n\n")
            
            # Statistical comparison
            if len(self.results) >= 2:
                comparison = self.compare_across_worlds()
                if not comparison.empty:
                    f.write(comparison.to_markdown(index=False))
        
        logger.info(f"Exported summary to {output_file}")
