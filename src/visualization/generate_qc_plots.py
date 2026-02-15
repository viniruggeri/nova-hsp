"""
Generate sanity check plots for generated datasets.

Creates visualizations to verify dataset quality:
- Histogram of T_event (collapse times)
- Proportion of collapsed runs per split
- Sample trajectory of observables
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def plot_dataset_qc(world_name: str, output_dir: Path = None):
    """
    Generate quality control plots for a dataset.

    Args:
        world_name: Name of world (sir_graph or ant_colony)
        output_dir: Output directory (defaults to results/{world_name}/)
    """
    data_root = Path(f"data/processed/{world_name}")

    if not data_root.exists():
        print(f"Dataset not found: {data_root}")
        return

    if output_dir is None:
        output_dir = Path(f"results/{world_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load summary data
    train_df = pd.read_csv(data_root / "train.csv")
    val_df = pd.read_csv(data_root / "val.csv")
    test_df = pd.read_csv(data_root / "test.csv")

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Histogram of T_event across all splits
    ax1 = fig.add_subplot(gs[0, :])
    for df, split, color in [
        (train_df, "Train", "blue"),
        (val_df, "Val", "orange"),
        (test_df, "Test", "green"),
    ]:
        collapsed_df = df[df["collapsed"] & df["T_event"].notna()]
        if len(collapsed_df) > 0:
            ax1.hist(
                collapsed_df["T_event"],
                bins=30,
                alpha=0.5,
                label=f"{split} (n={len(collapsed_df)})",
                color=color,
            )

    ax1.set_xlabel("Collapse Time (T_event)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(
        f"{world_name}: Distribution of Collapse Times", fontsize=14, fontweight="bold"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Collapse rate per split
    ax2 = fig.add_subplot(gs[1, 0])
    splits = ["Train", "Val", "Test"]
    collapse_rates = [
        train_df["collapsed"].mean(),
        val_df["collapsed"].mean(),
        test_df["collapsed"].mean(),
    ]
    colors_bar = ["blue", "orange", "green"]

    bars = ax2.bar(splits, collapse_rates, color=colors_bar, alpha=0.7)
    ax2.set_ylabel("Collapse Rate", fontsize=12)
    ax2.set_title("Collapse Rate by Split", fontsize=12, fontweight="bold")
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, axis="y")

    # Add percentage labels on bars
    for bar, rate in zip(bars, collapse_rates):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 3. Statistics table
    ax3 = fig.add_subplot(gs[1, 1:])
    ax3.axis("off")

    stats_data = []
    for df, split in [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]:
        collapsed_df = df[df["collapsed"] & df["T_event"].notna()]
        if len(collapsed_df) > 0:
            stats_data.append(
                [
                    split,
                    len(df),
                    len(collapsed_df),
                    f"{collapsed_df['T_event'].mean():.2f}",
                    f"{collapsed_df['T_event'].std():.2f}",
                    f"{collapsed_df['T_event'].min():.0f}",
                    f"{collapsed_df['T_event'].max():.0f}",
                ]
            )

    table = ax3.table(
        cellText=stats_data,
        colLabels=[
            "Split",
            "N Runs",
            "N Collapsed",
            "Mean T",
            "Std T",
            "Min T",
            "Max T",
        ],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax3.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=20)

    # 4. Sample trajectories (plot 3 random runs from train)
    ax4 = fig.add_subplot(gs[2, :])

    # Get 3 random seeds from train
    sample_seeds = train_df["seed"].sample(min(3, len(train_df))).tolist()

    for i, seed in enumerate(sample_seeds):
        run_dir = data_root / "train" / f"run_{seed}"
        obs_file = run_dir / "obs.npy"
        events_file = run_dir / "events.json"

        if obs_file.exists() and events_file.exists():
            obs = np.load(obs_file)
            with open(events_file, "r") as f:
                events = json.load(f)

            T_event = events.get("T_event")

            # Plot first feature
            ax4.plot(
                obs[:, 0],
                label=(
                    f"Seed {seed} (T_event={T_event})"
                    if T_event
                    else f"Seed {seed} (censored)"
                ),
                alpha=0.7,
                linewidth=2,
            )

            # Mark collapse time
            if T_event is not None and T_event < len(obs):
                ax4.axvline(x=T_event, color=f"C{i}", linestyle="--", alpha=0.5)

    ax4.set_xlabel("Timestep", fontsize=12)
    ax4.set_ylabel("Feature 0 Value", fontsize=12)
    ax4.set_title("Sample Trajectories (First Feature)", fontsize=12, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Save figure
    output_file = output_dir / "dataset_qc.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Saved QC plot: {output_file}")
    plt.close()

    # Generate text report
    report_file = output_dir / "dataset_qc_report.txt"
    with open(report_file, "w") as f:
        f.write(f"Dataset Quality Control Report: {world_name}\n")
        f.write("=" * 80 + "\n\n")

        all_df = pd.concat([train_df, val_df, test_df], keys=["train", "val", "test"])

        f.write(f"Total runs: {len(all_df)}\n")
        f.write(
            f"Total collapsed: {all_df['collapsed'].sum()} ({all_df['collapsed'].mean():.1%})\n\n"
        )

        for split, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            f.write(f"{split}:\n")
            f.write(f"  Runs: {len(df)}\n")
            f.write(
                f"  Collapsed: {df['collapsed'].sum()} ({df['collapsed'].mean():.1%})\n"
            )

            collapsed_df = df[df["collapsed"] & df["T_event"].notna()]
            if len(collapsed_df) > 0:
                f.write(f"  T_event mean: {collapsed_df['T_event'].mean():.2f}\n")
                f.write(f"  T_event std: {collapsed_df['T_event'].std():.2f}\n")
                f.write(
                    f"  T_event range: [{collapsed_df['T_event'].min():.0f}, {collapsed_df['T_event'].max():.0f}]\n"
                )
            f.write("\n")

    print(f"✓ Saved QC report: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate sanity check plots for datasets"
    )
    parser.add_argument(
        "--world",
        type=str,
        required=True,
        choices=["sir_graph", "ant_colony", "all"],
        help="World to generate plots for",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/{world}/)",
    )

    args = parser.parse_args()

    worlds = ["sir_graph", "ant_colony"] if args.world == "all" else [args.world]

    for world in worlds:
        print(f"\n{'='*80}")
        print(f"Generating QC plots for: {world}")
        print(f"{'='*80}")

        output_dir = Path(args.output_dir) if args.output_dir else None
        plot_dataset_qc(world, output_dir)

    print(f"\n✓ QC plots generated successfully!")


if __name__ == "__main__":
    main()
