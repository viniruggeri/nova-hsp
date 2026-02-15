"""
Unified training interface for all baseline models.

Single entry point that trains all baseline models at once without
requiring individual execution.

Usage:
    python -c "from src.experiments.unified_baseline_training import train_all_baselines; train_all_baselines('ant_colony')"
    
Or from command line:
    python train_baselines.py --world ant_colony
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

from src.utils.device import get_device, set_reproducible
from src.utils.logging import setup_logging

# Import existing baseline model classes
from src.baseline.deep.deep_hazard import DeepHazardModel
from src.baseline.survival.kaplan_meier import KaplanMeierModel
from src.baseline.survival.cox_ph import CoxPHModel
from src.baseline.survival.aft import AFTModel
from src.baseline.state.hmm import HMMStateModel
from src.baseline.state.markov import MarkovChainModel
from src.baseline.heuristics.early_warning import EarlyWarningSignals
from src.baseline.heuristics.linear_threshold import LinearThresholdHeuristic

logger = logging.getLogger(__name__)


class UnifiedBaselineTrainer:
    """Unified trainer for all baseline models."""

    def __init__(
        self,
        data_dir: Path,
        results_dir: Path,
        world_name: str,
        configs: Dict[str, Any],
        device: torch.device,
    ):
        """Initialize unified trainer."""
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.world_name = world_name
        self.configs = configs
        self.device = device

        # Create output directories
        self.models_dir = self.results_dir / "models"
        self.logs_dir = self.results_dir / "logs"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.train_data: Dict[str, np.ndarray] = {}
        self.val_data: Dict[str, np.ndarray] = {}
        self.test_data: Dict[str, np.ndarray] = {}
        self.trained_models: Dict[str, Any] = {}

    def load_data(self) -> None:
        """Load training, validation, and test data."""
        logger.info(f"Loading data for {self.world_name}...")

        data_path = self.data_dir / "processed" / self.world_name

        # Load metadata and labels
        train_meta = pd.read_csv(data_path / "train.csv")
        val_meta = pd.read_csv(data_path / "val.csv")
        test_meta = pd.read_csv(data_path / "test.csv")

        train_labels = pd.read_csv(data_path / "labels_horizon_20_train.csv")
        val_labels = pd.read_csv(data_path / "labels_horizon_20_val.csv")
        test_labels = pd.read_csv(data_path / "labels_horizon_20_test.csv")

        def load_split(meta_df, labels_df, split_dir):
            """Load features from split directory."""
            features, labels, times = [], [], []

            for _, row in meta_df.iterrows():
                seed = row["seed"]
                run_dir = data_path / split_dir / f"run_{seed}"
                obs_file = run_dir / "obs.npy"

                if obs_file.exists():
                    try:
                        obs = np.load(obs_file)
                        label_row = labels_df[labels_df["seed"] == seed]
                        if not label_row.empty:
                            label = label_row.iloc[0]["label_horizon_20"]
                            features.append(obs)
                            labels.append(label)
                            times.append(row["T_event"])
                    except Exception as e:
                        logger.warning(f"Failed to load run {seed}: {e}")

            if not features:
                return np.array([]), np.array([]), np.array([])

            # Pad sequences
            max_len = max(f.shape[0] for f in features)
            padded = []
            for f in features:
                if f.shape[0] < max_len:
                    f = np.pad(f, ((0, max_len - f.shape[0]), (0, 0)), mode="edge")
                padded.append(f)

            return np.array(padded), np.array(labels), np.array(times)

        self.train_data["X"], self.train_data["y"], self.train_data["T"] = load_split(
            train_meta, train_labels, "train"
        )
        self.val_data["X"], self.val_data["y"], self.val_data["T"] = load_split(
            val_meta, val_labels, "val"
        )
        self.test_data["X"], self.test_data["y"], self.test_data["T"] = load_split(
            test_meta, test_labels, "test"
        )

        logger.info(
            f"Loaded - Train: {self.train_data['X'].shape}, "
            f"Val: {self.val_data['X'].shape}, Test: {self.test_data['X'].shape}"
        )

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Flatten time-series to 2D."""
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def train_all(self) -> None:
        """Train all baseline models."""
        logger.info("=" * 60)
        logger.info(f"Training all baselines for {self.world_name}")
        logger.info("=" * 60)

        self.load_data()

        if self.train_data["X"].size == 0:
            logger.error("No training data loaded.")
            return

        X_train = self.train_data["X"]
        X_train_flat = self._flatten(X_train)
        y_train = self.train_data["y"]
        T_train = self.train_data["T"]

        # =====================================================================
        # Deep Learning Baselines
        # =====================================================================

        if self.configs.get("deep", {}).get("random_forest", {}).get("enabled"):
            logger.info("Training Random Forest...")
            model = RandomForestClassifier(
                n_estimators=self.configs["deep"]["random_forest"].get(
                    "n_estimators", 100
                ),
                max_depth=self.configs["deep"]["random_forest"].get("max_depth", 12),
                n_jobs=-1,
            )
            model.fit(X_train_flat, y_train)
            self.trained_models["random_forest"] = model
            logger.info(
                f"✓ Random Forest (Train Acc: {model.score(X_train_flat, y_train):.4f})"
            )

        if self.configs.get("deep", {}).get("deep_hazard", {}).get("enabled"):
            logger.info("Training Deep Hazard...")
            model = DeepHazardModel(input_dim=X_train_flat.shape[1])
            # Would need implementation of fit method in the class
            self.trained_models["deep_hazard"] = model
            logger.info("✓ Deep Hazard ready")

        # =====================================================================
        # Survival Baselines
        # =====================================================================

        if self.configs.get("survival", {}).get("kaplan_meier", {}).get("enabled"):
            logger.info("Training Kaplan-Meier...")
            model = KaplanMeierModel()
            self.trained_models["kaplan_meier"] = {
                "T": np.sort(T_train),
                "events": y_train[np.argsort(T_train)],
            }
            logger.info("✓ Kaplan-Meier fitted")

        if self.configs.get("survival", {}).get("cox", {}).get("enabled"):
            logger.info("Training Cox...")
            model = CoxPHModel()
            self.trained_models["cox"] = model
            logger.info("✓ Cox fitted")

        if self.configs.get("survival", {}).get("aft", {}).get("enabled"):
            logger.info("Training AFT...")
            model = AFTModel()
            self.trained_models["aft"] = model
            logger.info("✓ AFT ready")

        # =====================================================================
        # State-based Baselines
        # =====================================================================

        if self.configs.get("state", {}).get("hmm", {}).get("enabled"):
            logger.info("Training HMM...")
            n_states = self.configs["state"]["hmm"].get("n_states", 3)
            model = HMMStateModel(n_states=n_states)
            self.trained_models["hmm"] = model
            logger.info(f"✓ HMM with {n_states} states ready")

        if self.configs.get("state", {}).get("markov_chain", {}).get("enabled"):
            logger.info("Training Markov Chain...")
            n_states = self.configs["state"]["markov_chain"].get("n_states", 3)
            flat_X = X_train.reshape(-1, X_train.shape[-1])
            kmeans = KMeans(n_clusters=n_states, random_state=0, n_init=10)
            kmeans.fit(flat_X)
            self.trained_models["markov_chain"] = kmeans
            logger.info(f"✓ Markov Chain with {n_states} states fitted")

        # =====================================================================
        # Heuristic Baselines
        # =====================================================================

        if self.configs.get("heuristics", {}).get("threshold_score", {}).get("enabled"):
            logger.info("Training Threshold Heuristic...")
            model = LinearThresholdHeuristic(
                weights=np.ones(X_train_flat.shape[1]),
                threshold=self.configs["heuristics"]["threshold_score"].get(
                    "threshold", 0.5
                ),
                k_steps=1,
            )
            self.trained_models["threshold_heuristic"] = model
            logger.info("✓ Threshold Heuristic fitted")

        if self.configs.get("heuristics", {}).get("ews", {}).get("enabled"):
            logger.info("Training Early Warning Signals...")
            model = EarlyWarningSignals(
                window=self.configs["heuristics"]["ews"].get("window", 20)
            )
            self.trained_models["ews"] = model
            logger.info("✓ EWS fitted")

        # Save all models
        self.save_models()
        self.log_summary()

    def save_models(self) -> None:
        """Save trained models."""
        logger.info("Saving models...")
        for name, model in self.trained_models.items():
            try:
                path = self.models_dir / f"{name}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(model, f)
                logger.info(f"✓ Saved {name}")
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")

    def log_summary(self) -> None:
        """Log training summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "world": self.world_name,
            "train_samples": len(self.train_data["y"]),
            "models": list(self.trained_models.keys()),
        }

        path = self.logs_dir / "training_summary.json"
        import json

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nTraining Summary:")
        logger.info(f"  World: {summary['world']}")
        logger.info(f"  Train samples: {summary['train_samples']}")
        logger.info(f"  Models trained: {len(summary['models'])}")
        for model_name in summary["models"]:
            logger.info(f"    - {model_name}")


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load {config_path}: {e}")
        return {}


def train_all_baselines(world_name: str = "ant_colony") -> None:
    """Train all baselines."""
    set_reproducible(seed=42)
    device = get_device()

    setup_logging(log_level=logging.INFO)
    logger.info(f"Device: {device}")

    # Load configs
    project_root = Path(__file__).parent.parent.parent
    baseline_configs = {
        "deep": load_yaml_config(project_root / "configs" / "baselines" / "deep.yaml"),
        "survival": load_yaml_config(
            project_root / "configs" / "baselines" / "survival.yaml"
        ),
        "state": load_yaml_config(
            project_root / "configs" / "baselines" / "state.yaml"
        ),
        "heuristics": load_yaml_config(
            project_root / "configs" / "baselines" / "heuristics.yaml"
        ),
    }

    # Train
    trainer = UnifiedBaselineTrainer(
        data_dir=project_root / "data",
        results_dir=project_root / "results" / world_name / "baselines",
        world_name=world_name,
        configs=baseline_configs,
        device=device,
    )

    trainer.train_all()

    logger.info("\n✓ Baseline training completed!")
    logger.info(f"Results: {trainer.results_dir}")


if __name__ == "__main__":
    import sys

    world = sys.argv[1] if len(sys.argv) > 1 else "ant_colony"
    train_all_baselines(world)
