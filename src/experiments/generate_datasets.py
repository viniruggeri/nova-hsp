"""
Generate simulation datasets for training/validating/testing baselines.

This script generates structured datasets from world simulations (SirGraph, AntColony)
with proper train/val/test splits for use in baseline experiments.

Usage:
    python src/experiments/generate_datasets.py --world sir_graph
    python src/experiments/generate_datasets.py --world ant_colony
    python src/experiments/generate_datasets.py --world all
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

# Import world classes
from src.worlds.sir_graph import SirGraph
from src.worlds.ant_colony import AntColony

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# World registry
WORLD_REGISTRY = {"sir_graph": SirGraph, "ant_colony": AntColony}

# Default seed ranges for splits
SEED_RANGES = {"train": (0, 49), "val": (50, 69), "test": (70, 99)}

# Default horizon for classification labels
DEFAULT_HORIZON = 20


def dict_to_vector(obs: dict) -> np.ndarray:
    """Convert observation dict to numpy vector (sorted by key for consistency)."""
    return np.array([obs[k] for k in sorted(obs.keys())], dtype=np.float32)


def generate_single_run(
    world_class,
    cfg,
    seed: int,
    output_dir: Path,
    sample_frequency: int = 1,
) -> Dict:
    """
    Generate data for a single simulation run.

    Args:
        world_class: World class constructor
        cfg: World configuration
        seed: Random seed
        output_dir: Directory to save outputs
        sample_frequency: Sampling frequency (1 = every timestep)

    Returns:
        Dictionary with run metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize world
    world = world_class(cfg)
    world.reset(seed)

    # Collect observations
    obs_list = []
    timesteps = []
    max_steps = cfg.max_steps

    for t in range(max_steps):
        if t % sample_frequency == 0:
            obs = world.observe()

            # Convert dict to vector if needed
            if isinstance(obs, dict):
                obs_vec = dict_to_vector(obs)
            else:
                obs_vec = np.array(obs, dtype=np.float32)

            obs_list.append(obs_vec)
            timesteps.append(t)

        # Check for collapse before stepping
        collapsed, T_event = world.is_collapsed()
        if collapsed:
            break

        world.step()

    # Get final collapse status
    collapsed, T_event = world.is_collapsed()

    # Build arrays
    if len(obs_list) == 0:
        raise ValueError(f"No observations collected for seed {seed}")

    obs_arr = np.stack(obs_list)  # shape (T_obs, F)
    T_obs = len(obs_list)
    n_features = obs_arr.shape[1]

    # Build time-to-event array
    if collapsed and T_event is not None:
        event_indicator = 1
        # Time to event at each timestep
        time_to_event = np.maximum(0, T_event - np.array(timesteps))
    else:
        event_indicator = 0
        T_event = None
        # Censored: time to max_steps
        time_to_event = np.array([max_steps - t for t in timesteps])

    # Save obs.npy
    np.save(output_dir / "obs.npy", obs_arr)

    # Save time_to_event.npy
    np.save(output_dir / "time_to_event.npy", time_to_event)

    # Save events.json
    with open(output_dir / "events.json", "w") as f:
        json.dump(
            {"T_event": T_event, "event_indicator": event_indicator},
            f,
            indent=2,
        )

    # Save survival_record.csv
    records = []
    for i, t in enumerate(timesteps):
        record = {
            "t": t,
            **{f"feature_{j}": obs_arr[i, j] for j in range(n_features)},
            "event_indicator": event_indicator,
            "time_to_event": time_to_event[i],
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "survival_record.csv", index=False)

    # Save meta.json
    meta = {
        "seed": seed,
        "run_id": f"{cfg.get('name', 'world')}_seed_{seed}",
        "n_timesteps": T_obs,
        "n_features": n_features,
        "collapsed": bool(collapsed),
        "T_event": T_event,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def generate_split(
    world_name: str,
    world_class,
    cfg,
    split: str,
    seed_range: Tuple[int, int],
    output_root: Path,
    sample_frequency: int = 1,
) -> pd.DataFrame:
    """
    Generate data for an entire split (train/val/test).

    Args:
        world_name: Name of world
        world_class: World class constructor
        cfg: World configuration
        split: Split name (train/val/test)
        seed_range: (start_seed, end_seed) inclusive
        output_root: Root output directory
        sample_frequency: Sampling frequency

    Returns:
        DataFrame with run summaries
    """
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    start_seed, end_seed = seed_range
    seeds = range(start_seed, end_seed + 1)

    logger.info(
        f"Generating {split} split for {world_name}: seeds {start_seed}-{end_seed}"
    )

    run_summaries = []
    failed_seeds = []
    error_log_path = output_root / "errors.log"

    for seed in tqdm(seeds, desc=f"{world_name} {split}"):
        run_dir = split_dir / f"run_{seed}"

        try:
            meta = generate_single_run(
                world_class, cfg, seed, run_dir, sample_frequency
            )

            run_summaries.append(
                {
                    "run_id": meta["run_id"],
                    "seed": seed,
                    "T_event": meta["T_event"],
                    "collapsed": meta["collapsed"],
                    "n_timesteps": meta["n_timesteps"],
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate run for seed {seed}: {e}")
            failed_seeds.append(seed)

            # Log error
            with open(error_log_path, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Seed {seed} - {split}\n")
                f.write(traceback.format_exc())

            # Add failed run to summary
            run_summaries.append(
                {
                    "run_id": f"{world_name}_seed_{seed}",
                    "seed": seed,
                    "T_event": None,
                    "collapsed": False,
                    "n_timesteps": 0,
                    "failed": True,
                }
            )

    # Create summary DataFrame
    df_summary = pd.DataFrame(run_summaries)

    # Save split summary
    df_summary.to_csv(split_dir.parent / f"{split}.csv", index=False)

    # Check collapse rate
    n_collapsed = df_summary["collapsed"].sum()
    collapse_rate = n_collapsed / len(df_summary)

    logger.info(
        f"{split} split: {n_collapsed}/{len(df_summary)} collapsed ({collapse_rate:.1%})"
    )

    if collapse_rate < 0.5:
        logger.warning(
            f"⚠️  Less than 50% of runs collapsed in {split}. "
            f"Consider adjusting collapse thresholds in config."
        )

    if failed_seeds:
        logger.warning(f"Failed seeds in {split}: {failed_seeds}")

    return df_summary


def create_aggregated_csvs(output_root: Path, splits: List[str]):
    """
    Create aggregated CSVs across all runs.

    Args:
        output_root: Root output directory
        splits: List of split names
    """
    logger.info("Creating aggregated CSVs...")

    all_runs_records = []

    for split in splits:
        split_dir = output_root / split

        if not split_dir.exists():
            continue

        # Find all run directories
        run_dirs = sorted(split_dir.glob("run_*"))

        for run_dir in run_dirs:
            survival_csv = run_dir / "survival_record.csv"
            meta_json = run_dir / "meta.json"

            if not survival_csv.exists() or not meta_json.exists():
                continue

            # Load data
            df_survival = pd.read_csv(survival_csv)
            with open(meta_json, "r") as f:
                meta = json.load(f)

            # Add run_id and split columns
            df_survival["run_id"] = meta["run_id"]
            df_survival["split"] = split
            df_survival["seed"] = meta["seed"]

            all_runs_records.append(df_survival)

    if all_runs_records:
        df_all = pd.concat(all_runs_records, ignore_index=True)
        df_all.to_csv(output_root / "all_runs.csv", index=False)
        logger.info(f"Saved all_runs.csv with {len(df_all)} records")


def create_classification_labels(
    output_root: Path, splits: List[str], horizon: int = DEFAULT_HORIZON
):
    """
    Create classification labels for horizon-based prediction.

    Args:
        output_root: Root output directory
        splits: List of split names
        horizon: Time horizon for classification
    """
    for split in splits:
        summary_file = output_root / f"{split}.csv"

        if not summary_file.exists():
            continue

        df = pd.read_csv(summary_file)

        # Create label: 1 if collapsed within horizon, 0 otherwise
        df[f"label_horizon_{horizon}"] = (
            (df["collapsed"]) & (df["T_event"] <= horizon)
        ).astype(int)

        # Save labels
        labels_file = output_root / f"labels_horizon_{horizon}_{split}.csv"
        df[["run_id", "seed", f"label_horizon_{horizon}"]].to_csv(
            labels_file, index=False
        )

        logger.info(
            f"Created classification labels (horizon={horizon}) for {split}: "
            f"{labels_file.name}"
        )


def create_generation_report(
    output_root: Path, world_name: str, split_summaries: Dict[str, pd.DataFrame]
):
    """
    Create generation report with summary statistics.

    Args:
        output_root: Root output directory
        world_name: Name of world
        split_summaries: Dictionary mapping split names to summary DataFrames
    """
    report_path = output_root / "generation_report.txt"

    with open(report_path, "w") as f:
        f.write(f"Dataset Generation Report: {world_name}\n")
        f.write("=" * 80 + "\n\n")

        for split, df in split_summaries.items():
            f.write(f"{split.upper()} Split\n")
            f.write("-" * 40 + "\n")

            n_runs = len(df)
            n_failed = df.get("failed", pd.Series([False] * n_runs)).sum()
            n_collapsed = df["collapsed"].sum()
            collapse_rate = n_collapsed / n_runs

            f.write(f"Total runs: {n_runs}\n")
            f.write(f"Failed runs: {n_failed}\n")
            f.write(f"Collapsed: {n_collapsed} ({collapse_rate:.1%})\n")

            # Stats for collapsed runs
            collapsed_df = df[df["collapsed"] & df["T_event"].notna()]
            if len(collapsed_df) > 0:
                mean_T = collapsed_df["T_event"].mean()
                std_T = collapsed_df["T_event"].std()
                min_T = collapsed_df["T_event"].min()
                max_T = collapsed_df["T_event"].max()

                f.write(f"\nCollapse Time Statistics:\n")
                f.write(f"  Mean: {mean_T:.2f}\n")
                f.write(f"  Std:  {std_T:.2f}\n")
                f.write(f"  Min:  {min_T}\n")
                f.write(f"  Max:  {max_T}\n")

            # Seeds that didn't collapse
            non_collapsed_seeds = df[~df["collapsed"]]["seed"].tolist()
            if non_collapsed_seeds:
                f.write(f"\nSeeds that did not collapse: {non_collapsed_seeds}\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Data Format:\n")
        f.write("  - obs.npy: (T, F) array of observations\n")
        f.write("  - time_to_event.npy: (T,) array of remaining time\n")
        f.write("  - events.json: T_event and event_indicator\n")
        f.write("  - survival_record.csv: per-timestep features and labels\n")
        f.write("  - meta.json: run metadata\n")
        f.write("\nAggregate files:\n")
        f.write("  - {split}.csv: run-level summaries\n")
        f.write("  - all_runs.csv: concatenated survival records\n")
        f.write("  - labels_horizon_{H}_{split}.csv: classification labels\n")

    logger.info(f"Generation report saved to {report_path}")


def validate_dataset(output_root: Path, splits: List[str]) -> bool:
    """
    Validate generated dataset.

    Args:
        output_root: Root output directory
        splits: List of split names

    Returns:
        True if validation passes
    """
    logger.info("Validating generated dataset...")

    all_valid = True
    feature_dims = set()

    for split in splits:
        split_dir = output_root / split
        if not split_dir.exists():
            continue

        run_dirs = list(split_dir.glob("run_*"))

        for run_dir in run_dirs:
            obs_file = run_dir / "obs.npy"

            if not obs_file.exists():
                logger.error(f"Missing obs.npy in {run_dir}")
                all_valid = False
                continue

            # Load and check shape
            try:
                obs = np.load(obs_file)
                if obs.ndim != 2:
                    logger.error(f"Invalid shape in {obs_file}: {obs.shape}")
                    all_valid = False
                    continue

                T, F = obs.shape
                feature_dims.add(F)

            except Exception as e:
                logger.error(f"Failed to load {obs_file}: {e}")
                all_valid = False

    # Check feature dimensionality consistency
    if len(feature_dims) > 1:
        logger.error(f"Inconsistent feature dimensions: {feature_dims}")
        all_valid = False
    elif len(feature_dims) == 1:
        logger.info(f"✓ Feature dimension consistent: F={feature_dims.pop()}")

    # Check CSVs
    for split in splits:
        csv_file = output_root / f"{split}.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                expected_cols = ["run_id", "seed", "T_event", "collapsed"]
                if not all(col in df.columns for col in expected_cols):
                    logger.error(f"Missing columns in {csv_file}")
                    all_valid = False
                else:
                    logger.info(f"✓ {csv_file.name} valid ({len(df)} runs)")
            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")
                all_valid = False

    return all_valid


def generate_world_datasets(
    world_name: str,
    seed_ranges: Dict[str, Tuple[int, int]] = None,
    sample_frequency: int = 1,
    horizon: int = DEFAULT_HORIZON,
) -> bool:
    """
    Generate complete dataset for a world.

    Args:
        world_name: Name of world (sir_graph or ant_colony)
        seed_ranges: Custom seed ranges (uses defaults if None)
        sample_frequency: Sampling frequency
        horizon: Horizon for classification labels

    Returns:
        True if generation successful
    """
    if world_name not in WORLD_REGISTRY:
        logger.error(f"Unknown world: {world_name}")
        return False

    if seed_ranges is None:
        seed_ranges = SEED_RANGES

    # Load configuration
    config_path = Path(f"configs/worlds/{world_name}.yaml")
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return False

    cfg = OmegaConf.load(config_path)
    cfg.name = world_name  # Add name to config

    world_class = WORLD_REGISTRY[world_name]

    # Setup output directory
    output_root = Path(f"data/processed/{world_name}")
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"Generating datasets for: {world_name}")
    logger.info(f"Output directory: {output_root}")
    logger.info(f"{'='*80}\n")

    # Generate each split
    split_summaries = {}

    for split, seed_range in seed_ranges.items():
        try:
            df_summary = generate_split(
                world_name,
                world_class,
                cfg,
                split,
                seed_range,
                output_root,
                sample_frequency,
            )
            split_summaries[split] = df_summary

        except Exception as e:
            logger.error(f"Failed to generate {split} split: {e}")
            logger.error(traceback.format_exc())
            return False

    # Create aggregated files
    try:
        create_aggregated_csvs(output_root, list(seed_ranges.keys()))
        create_classification_labels(output_root, list(seed_ranges.keys()), horizon)
        create_generation_report(output_root, world_name, split_summaries)

    except Exception as e:
        logger.error(f"Failed to create aggregated files: {e}")
        logger.error(traceback.format_exc())
        return False

    # Validate
    if not validate_dataset(output_root, list(seed_ranges.keys())):
        logger.error("Dataset validation failed")
        return False

    logger.info(f"\n✓ Successfully generated datasets for {world_name}")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate simulation datasets for baseline experiments"
    )
    parser.add_argument(
        "--world",
        type=str,
        required=True,
        choices=["sir_graph", "ant_colony", "all"],
        help="World to generate datasets for",
    )
    parser.add_argument(
        "--sample-frequency",
        type=int,
        default=1,
        help="Sampling frequency (1 = every timestep)",
    )
    parser.add_argument(
        "--horizon", type=int, default=DEFAULT_HORIZON, help="Classification horizon"
    )

    args = parser.parse_args()

    # Determine which worlds to process
    if args.world == "all":
        worlds = ["sir_graph", "ant_colony"]
    else:
        worlds = [args.world]

    # Generate datasets
    success_count = 0
    for world_name in worlds:
        try:
            success = generate_world_datasets(
                world_name, sample_frequency=args.sample_frequency, horizon=args.horizon
            )
            if success:
                success_count += 1
        except Exception as e:
            logger.error(f"Failed to generate {world_name}: {e}")
            logger.error(traceback.format_exc()[-2000:])  # Last 2000 chars

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Dataset Generation Complete")
    logger.info(f"{'='*80}")
    logger.info(f"Successfully generated: {success_count}/{len(worlds)} worlds")
    logger.info(f"\nData location: data/processed/{{world_name}}/")
    logger.info(f"\nTo use in baseline training:")
    logger.info(
        f"  - Load obs: np.load('data/processed/{{world}}/{{split}}/run_{{seed}}/obs.npy')"
    )
    logger.info(
        f"  - Load events: json.load('data/processed/{{world}}/{{split}}/run_{{seed}}/events.json')"
    )
    logger.info(
        f"  - Load aggregated: pd.read_csv('data/processed/{{world}}/all_runs.csv')"
    )

    return success_count == len(worlds)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
