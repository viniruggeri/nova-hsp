"""
Generate paper-ready results aggregation and LaTeX tables.

Usage:
    python scripts/aggregate_results.py --results_dir results/simulated --output_dir results/aggregated
"""

import argparse
from pathlib import Path
import logging

from src.evaluation.aggregator import ResultsAggregator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/simulated"),
        help="Directory containing result subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/aggregated"),
        help="Output directory for aggregated results",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "basin_contraction_corr",
            "lead_time_norm",
            "monotonicity_frac",
            "separability_auc",
        ],
        help="Metrics to include in analysis",
    )
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    parser.add_argument(
        "--markdown",
        action="store_true",
        default=True,
        help="Generate Markdown summary",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading results from {args.results_dir}")

    # Initialize aggregator
    aggregator = ResultsAggregator()

    # Load all results
    aggregator.load_results_from_dir(
        results_dir=args.results_dir, pattern="**/integrated_results.csv"
    )

    if not aggregator.results:
        logger.error("No results found. Make sure integrated_results.csv files exist.")
        return

    logger.info(f"Loaded {len(aggregator.results)} experiments")
    for exp_name, results in aggregator.results.items():
        n_models = len(results["model_name"].unique())
        logger.info(f"  - {exp_name}: {n_models} models, {len(results)} runs")

    # Aggregate results
    logger.info("\nAggregating results...")
    aggregated = aggregator.aggregate_by_model(metrics=args.metrics)

    if not aggregated.empty:
        output_csv = args.output_dir / "aggregated_results.csv"
        aggregated.to_csv(output_csv, index=False)
        logger.info(f"Saved aggregated results to {output_csv}")

    # Cross-world comparison
    if len(aggregator.results) >= 2:
        logger.info("\nPerforming cross-world comparison...")

        for metric in args.metrics:
            comparison = aggregator.compare_across_worlds(metric=metric)

            if not comparison.empty:
                output_csv = args.output_dir / f"comparison_{metric}.csv"
                comparison.to_csv(output_csv, index=False)
                logger.info(f"Saved comparison for {metric} to {output_csv}")

    # Pairwise comparisons (all pairs)
    if len(aggregator.results) >= 2:
        logger.info("\nPerforming pairwise comparisons...")
        exp_names = list(aggregator.results.keys())

        for i, exp1 in enumerate(exp_names):
            for exp2 in exp_names[i + 1 :]:
                logger.info(f"  Comparing {exp1} vs {exp2}...")

                pairwise = aggregator.pairwise_comparison(
                    exp1, exp2, metric=args.metrics[0]
                )

                if not pairwise.empty:
                    output_csv = args.output_dir / f"pairwise_{exp1}_vs_{exp2}.csv"
                    pairwise.to_csv(output_csv, index=False)

    # Multi-criteria ranking
    logger.info("\nComputing multi-criteria ranking...")
    ranked = aggregator.rank_models(
        metrics=args.metrics[:2] if len(args.metrics) >= 2 else args.metrics,
        weights=None,  # Equal weights
        ascending=None,  # Auto-detect
    )

    if not ranked.empty:
        output_csv = args.output_dir / "model_ranking.csv"
        ranked.to_csv(output_csv, index=False)
        logger.info(f"Saved model ranking to {output_csv}")

        # Print top 5
        logger.info("\nTop 5 Models:")
        top5 = ranked.head(5)
        for idx, row in top5.iterrows():
            logger.info(
                f"  {int(row['rank'])}. {row['model_name']} (score: {row['composite_score']:.4f})"
            )

    # LaTeX export
    if args.latex:
        logger.info("\nGenerating LaTeX tables...")

        latex_file = args.output_dir / "table_comparison.tex"
        aggregator.export_latex_table(
            output_file=latex_file,
            metrics=args.metrics,
            caption="Model Performance Comparison Across Worlds",
            label="tab:model_comparison",
            bold_best=True,
        )

        logger.info(f"LaTeX table saved to {latex_file}")

    # Markdown export
    if args.markdown:
        logger.info("\nGenerating Markdown summary...")

        md_file = args.output_dir / "summary.md"
        aggregator.export_summary_markdown(md_file)

        logger.info(f"Markdown summary saved to {md_file}")

    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATION COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
