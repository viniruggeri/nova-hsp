"""
Temporal data splitting for reproducible baseline evaluation.

Implements temporal split (70% train, 15% val, 15% test) without shuffling
to avoid temporal leakage. Supports multiple seeds for statistical robustness.

Protocol:
  - Split temporal (no shuffle)
  - Fixed seed = 42 for baseline split
  - Support 5 distinct seeds for robust statistical analysis
  - Deterministic and reproducible across runs
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Any
import logging
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


class TemporalDataSplitter:
    """
    Temporal data splitter following experimental protocol.
    
    No shuffling - maintains temporal order. Supports multiple seeds
    for statistical robustness analysis.
    """

    def __init__(
        self,
        data_dir: Path,
        world_name: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        horizon: int = None,
    ):
        """Initialize temporal splitter.
        
        Args:
            horizon: Label horizon to use (e.g., 15 or 20). If None, auto-detect.
        """
        self.data_dir = Path(data_dir)
        self.world_name = world_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.horizon = horizon
        
        # Validate ratios
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6
        
        # Storage for split indices and data
        self.split_indices: Dict[str, np.ndarray] = {}
        self.split_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.metadata: Dict[str, pd.DataFrame] = {}

    def _load_world_data(self) -> Tuple[List[str], List[int], List[float], pd.DataFrame]:
        """Load world data and extract run IDs, event times, and labels for ALL 100 runs.
        
        Auto-detects best available horizon:
        - sir_graph: prefers horizon_15 (33% balance), fallback to horizon_20
        - ant_colony: uses horizon_20 (only available)
        """
        data_path = self.data_dir / "processed" / self.world_name
        
        # Auto-detect horizon if not specified
        if self.horizon is None:
            if self.world_name == "sir_graph":
                # Check if horizon_15 labels exist (better balance)
                if (data_path / "labels_horizon_15_train.csv").exists():
                    self.horizon = 15
                    logger.info("Auto-detected: Using horizon_15 (better class balance)")
                else:
                    self.horizon = 20
                    logger.info("Auto-detected: Using horizon_20 (horizon_15 not available)")
            else:
                self.horizon = 20  # Default
        
        # Try to load all_runs.csv first (has all data)
        all_runs_file = data_path / "all_runs.csv"
        if all_runs_file.exists():
            all_runs = pd.read_csv(all_runs_file)
            # Group by run to get metadata
            run_metadata = all_runs.groupby("seed").agg({
                "time_to_event": "first",
                "run_id": "first"
            }).reset_index()
            run_metadata = run_metadata.rename(columns={"time_to_event": "T_event"})
        else:
            # Fallback: load from split files
            all_meta = []
            for split in ["train", "val", "test"]:
                meta_file = data_path / f"{split}.csv"
                if meta_file.exists():
                    meta = pd.read_csv(meta_file)
                    all_meta.append(meta)
            
            if not all_meta:
                raise FileNotFoundError(f"No data found for {self.world_name}")
            
            run_metadata = pd.concat(all_meta, ignore_index=True).drop_duplicates(subset=["seed"])
        
        # Load labels from horizon files
        labels_data = []
        for split in ["train", "val", "test"]:
            labels_file = data_path / f"labels_horizon_{self.horizon}_{split}.csv"
            if labels_file.exists():
                lbl = pd.read_csv(labels_file)[["seed", f"label_horizon_{self.horizon}"]]
                lbl = lbl.rename(columns={f"label_horizon_{self.horizon}": "label_horizon"})
                labels_data.append(lbl)
        
        if labels_data:
            all_labels = pd.concat(labels_data, ignore_index=True).drop_duplicates(subset=["seed"])
            run_metadata = run_metadata.merge(all_labels, on="seed", how="left")
            run_metadata["label_horizon"] = run_metadata["label_horizon"].fillna(0).astype(int)
        else:
            run_metadata["label_horizon"] = 0
        
        # Sort by seed
        run_metadata = run_metadata.sort_values("seed").reset_index(drop=True)
        
        seeds = run_metadata["seed"].values.tolist()
        T_events = run_metadata["T_event"].values.tolist()
        
        logger.info(f"Found {len(seeds)} unique runs for {self.world_name}")
        logger.info(f"Using horizon_{self.horizon} | Label distribution: {run_metadata['label_horizon'].value_counts().to_dict()}")
        
        return seeds, T_events, run_metadata

    def _temporal_split(self, n_samples: int, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split indices with stratification when possible.
        
        Uses stratified split to maintain class balance ONLY if each class
        has at least 2 samples. Otherwise, falls back to temporal split.
        
        Returns:
            (train_idx, val_idx, test_idx) - indices for each split
        """
        # Check if stratification is possible
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_samples = counts.min()
        
        if min_samples >= 2:
            # Use stratified split to ensure representation of all classes
            logger.info(f"Using stratified split (all classes have >=2 samples)")
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=self.train_ratio,
                test_size=self.val_ratio + self.test_ratio,
                random_state=self.random_seed
            )
            
            train_idx, temp_idx = next(sss.split(np.arange(n_samples), labels))
            
            # Split remaining into val and test
            temp_labels = labels[temp_idx]
            n_temp = len(temp_idx)
            val_size = int(n_temp * (self.val_ratio / (self.val_ratio + self.test_ratio)))
            
            if temp_labels.sum() >= 2 or (len(temp_labels) - temp_labels.sum()) >= 2:
                sss2 = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=val_size,
                    test_size=n_temp - val_size,
                    random_state=self.random_seed + 1
                )
                val_local_idx, test_local_idx = next(sss2.split(np.arange(n_temp), temp_labels))
            else:
                # Fallback if temp set is too small
                val_local_idx = np.arange(val_size)
                test_local_idx = np.arange(val_size, n_temp)
            
            val_idx = temp_idx[val_local_idx]
            test_idx = temp_idx[test_local_idx]
        else:
            # Fallback to temporal split when classes are too imbalanced
            logger.warning(f"Class imbalance detected (min samples={min_samples}). Using temporal split.")
            logger.warning(f"Class distribution: {dict(zip(unique_labels, counts))}")
            
            train_cutoff = int(n_samples * self.train_ratio)
            val_cutoff = train_cutoff + int(n_samples * self.val_ratio)
            
            train_idx = np.arange(0, train_cutoff)
            val_idx = np.arange(train_cutoff, val_cutoff)
            test_idx = np.arange(val_cutoff, n_samples)
        
        logger.info(
            f"Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
        )
        logger.info(f"  Train label dist: {dict(zip(*np.unique(labels[train_idx], return_counts=True)))}")
        logger.info(f"  Val label dist:   {dict(zip(*np.unique(labels[val_idx], return_counts=True)))}")  
        logger.info(f"  Test label dist:  {dict(zip(*np.unique(labels[test_idx], return_counts=True)))}")
        
        return train_idx, val_idx, test_idx

    def _load_observations(
        self,
        seeds: List[int],
        indices: np.ndarray,
        metadata: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load observation sequences from disk.
        
        Loads from all split directories (train/val/test) and combines.
        
        Returns:
            X: (n_samples, max_time, n_features) - padded sequences
            y: (n_samples,) - labels (collapse event)
            T: (n_samples,) - event times
        """
        data_path = self.data_dir / "processed" / self.world_name
        
        # Build mapping of seed -> (split, label, T_event)
        seed_info = {}
        for _, row in metadata.iterrows():
            seed = int(row["seed"])
            seed_info[seed] = {
                "label": int(row.get("label_horizon", 0)),
                "T_event": float(row.get("T_event", 0.0))
            }
        
        features_list = []
        labels_list = []
        times_list = []
        
        for idx in indices:
            seed = seeds[idx] if idx < len(seeds) else None
            if seed is None:
                logger.warning(f"Could not map index {idx} to seed")
                continue
            
            # Try to find obs.npy in any of the split directories
            obs_file = None
            for split in ["train", "val", "test"]:
                candidate = data_path / split / f"run_{seed}" / "obs.npy"
                if candidate.exists():
                    obs_file = candidate
                    break
            
            if obs_file and obs_file.exists():
                try:
                    obs = np.load(obs_file)
                    info = seed_info.get(seed, {"label": 0, "T_event": 0.0})
                    
                    features_list.append(obs)
                    labels_list.append(info["label"])
                    times_list.append(info["T_event"])
                except Exception as e:
                    logger.warning(f"Failed to load run {seed}: {e}")
            else:
                logger.warning(f"No obs.npy found for seed {seed}")
        
        if not features_list:
            raise ValueError("No observations could be loaded")
        
        # Pad sequences to same length
        max_len = max(f.shape[0] for f in features_list)
        padded_X = []
        for f in features_list:
            if f.shape[0] < max_len:
                f = np.pad(f, ((0, max_len - f.shape[0]), (0, 0)), mode="edge")
            padded_X.append(f)
        
        X = np.array(padded_X)
        y = np.array(labels_list)
        T = np.array(times_list)
        
        logger.info(f"Loaded {len(features_list)} observations: X.shape={X.shape}, y.shape={y.shape}, T.shape={T.shape}")
        if len(features_list) > 0:
            logger.info(f"  Label distribution: {np.bincount(y)}")
        
        return X, y, T
        y = np.array(labels_list)
        T = np.array(times_list)
        
        logger.info(f"Loaded observations: X.shape={X.shape}, y.shape={y.shape}, T.shape={T.shape}")
        
        return X, y, T

    def split(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Perform stratified split and load all data.
        
        Uses stratified split to ensure representation of all classes
        in train/val/test splits.
        
        Returns:
            Dictionary with structure:
            {
                'train': {'X': array, 'y': array, 'T': array},
                'val': {'X': array, 'y': array, 'T': array},
                'test': {'X': array, 'y': array, 'T': array},
            }
        """
        np.random.seed(self.random_seed)
        
        # Load and analyze world data (now gets all 100 runs + labels)
        seeds, T_events, metadata = self._load_world_data()
        n_samples = len(seeds)
        
        # Get labels for stratification (uses auto-detected horizon)
        labels = metadata["label_horizon"].values
        
        # Perform stratified split to maintain class balance
        train_idx, val_idx, test_idx = self._temporal_split(n_samples, labels)
        
        # Load observations for each split
        logger.info("Loading train data...")
        X_train, y_train, T_train = self._load_observations(seeds, train_idx, metadata)
        
        logger.info("Loading val data...")
        X_val, y_val, T_val = self._load_observations(seeds, val_idx, metadata)
        
        logger.info("Loading test data...")
        X_test, y_test, T_test = self._load_observations(seeds, test_idx, metadata)
        
        # Ensure all splits have same sequence length
        # Use maximum across ALL splits for consistency
        all_sequences = [X_train, X_val, X_test]
        max_len = max(x.shape[1] for x in all_sequences)
        
        # Pad all to same length
        def pad_to_length(X, target_len):
            if X.shape[1] < target_len:
                padding = ((0, 0), (0, target_len - X.shape[1]), (0, 0))
                X = np.pad(X, padding, mode="edge")
            return X
        
        X_train = pad_to_length(X_train, max_len)
        X_val = pad_to_length(X_val, max_len)
        X_test = pad_to_length(X_test, max_len)
        
        # Store results
        self.split_data = {
            "train": {"X": X_train, "y": y_train, "T": T_train},
            "val": {"X": X_val, "y": y_val, "T": T_val},
            "test": {"X": X_test, "y": y_test, "T": T_test},
        }
        
        logger.info("\n" + "="*60)
        logger.info("Data Split Summary")
        logger.info("="*60)
        for split_name, data in self.split_data.items():
            logger.info(
                f"{split_name:6s}: X={data['X'].shape} y={data['y'].shape} T={data['T'].shape}"
            )
        logger.info("="*60 + "\n")
        
        return self.split_data

    def save_split(self, output_dir: Path) -> None:
        """Save split data to disk for reproducibility."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, data in self.split_data.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            np.save(split_dir / "X.npy", data["X"])
            np.save(split_dir / "y.npy", data["y"])
            np.save(split_dir / "T.npy", data["T"])
        
        logger.info(f"Saved split data to {output_dir}")

    @staticmethod
    def load_split(split_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
        """Load previously saved split data."""
        split_data = {}
        for split_name in ["train", "val", "test"]:
            split_path = split_dir / split_name
            split_data[split_name] = {
                "X": np.load(split_path / "X.npy"),
                "y": np.load(split_path / "y.npy"),
                "T": np.load(split_path / "T.npy"),
            }
        return split_data


def create_splits_for_seeds(
    data_dir: Path,
    world_name: str,
    output_base_dir: Path,
    seeds: List[int] = None,
) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    """
    Create and save temporal splits for multiple seeds.
    
    Per protocol: baseline seed=42, plus 4 additional seeds for robustness.
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 999]  # Baseline + 4 additional
    
    all_splits = {}
    
    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Creating split for seed {seed}")
        logger.info(f"{'='*60}")
        
        splitter = TemporalDataSplitter(
            data_dir=data_dir,
            world_name=world_name,
            random_seed=seed,
        )
        
        split_data = splitter.split()
        
        # Save to disk
        seed_output_dir = output_base_dir / f"seed_{seed}"
        splitter.save_split(seed_output_dir)
        
        all_splits[seed] = split_data
    
    logger.info(f"\n\nCreated splits for seeds: {seeds}")
    
    return all_splits
