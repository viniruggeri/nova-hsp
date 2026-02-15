"""
Generate all paper-ready figures.

Usage:
    python scripts/generate_figures.py --results_dir results/aggregated --output_dir results/figures
"""

import argparse
from pathlib import Path
import logging
import pandas as pd

from src.visualization.paper_plots import PaperVisualizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready figures")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/aggregated"),
        help="Directory containing aggregated results",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["basin_contraction_corr", "lead_time_norm", "monotonicity_frac"],
        help="Metrics to visualize",
    )
    parser.add_argument(
        "--format", choices=["pdf", "png", "svg"], default="pdf", help="Output format"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading aggregated results...")

    # Load performance results
    perf_file = args.results_dir / "aggregated_results.csv"
    if not perf_file.exists():
        logger.error(f"Performance results not found: {perf_file}")
        logger.info("Run aggregate_results.py first to generate aggregated data")
        return

    performance_results = pd.read_csv(perf_file)
    logger.info(f"Loaded performance results: {len(performance_results)} models")

    # Try to load robustness results
    rob_results = None
    rob_files = list(args.results_dir.glob("*robustness*.csv"))

    if rob_files:
        try:
            rob_results = pd.read_csv(rob_files[0])
            logger.info(f"Loaded robustness results: {len(rob_results)} entries")
        except Exception as e:
            logger.warning(f"Could not load robustness results: {e}")

    # Initialize visualizer
    logger.info("Initializing visualizer...")
    viz = PaperVisualizer(style="seaborn-v0_8-paper", palette="colorblind")

    # Adjust metrics for mean columns
    available_metrics = []
    for metric in args.metrics:
        if metric in performance_results.columns:
            available_metrics.append(metric)
        elif f"{metric}_mean" in performance_results.columns:
            available_metrics.append(metric)
        else:
            logger.warning(f"Metric {metric} not found in results")

    if not available_metrics:
        logger.error("No valid metrics found")
        return

    logger.info(f"Using metrics: {available_metrics}")

    # Generate plots
    logger.info("\nGenerating figures...")

    try:
        viz.generate_all_plots(
            performance_results=performance_results,
            robustness_results=rob_results,
            output_dir=args.output_dir,
            metrics=available_metrics,
        )
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        import traceback

        traceback.print_exc()
        return

    logger.info("\n" + "=" * 60)
    logger.info("FIGURE GENERATION COMPLETE!")
    logger.info(f"Figures saved to {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
