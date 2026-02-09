"""
Test baselines end-to-end on sir_graph dataset - Sprint 1.

Tests: Kaplan-Meier, Linear Threshold, Markov Chain
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging

from src.baseline.survival.kaplan_meier import KaplanMeierModel
from src.baseline.heuristics.linear_threshold import LinearThresholdHeuristic
from src.baseline.state.markov import MarkovChainModel
from src.evaluation.metrics_simple import BaselineEvaluator
from src.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def load_world_data(world_name: str, split: str, max_samples: int = None):
    """
    Load data for a specific world and split.
    
    Returns:
        X: (n_samples, seq_len, n_features)
        T: (n_samples,) time-to-event
    """
    data_path = Path(f"data/processed/{world_name}")
    meta_df = pd.read_csv(data_path / f"{split}.csv")
    
    if max_samples:
        meta_df = meta_df.head(max_samples)
    
    X_list = []
    T_list = []
    
    for _, row in meta_df.iterrows():
        seed = row['seed']
        run_dir = data_path / split / f"run_{seed}"
        
        # Load observations
        obs_file = run_dir / "obs.npy"
        if not obs_file.exists():
            logger.warning(f"Missing {obs_file}")
            continue
        
        obs = np.load(obs_file)
        
        # Load T_event
        T_event = row['T_event']
        
        X_list.append(obs)
        T_list.append(T_event)
    
    # Pad sequences to same length
    max_len = max(x.shape[0] for x in X_list)
    X_padded = []
    for x in X_list:
        if x.shape[0] < max_len:
            pad = np.repeat(x[-1:], max_len - x.shape[0], axis=0)
            x = np.vstack([x, pad])
        X_padded.append(x)
    
    X = np.array(X_padded)
    T = np.array(T_list)
    
    logger.info(f"Loaded {split}: X={X.shape}, T={T.shape}")
    return X, T


def test_kaplan_meier(T_train, T_test):
    """Test Kaplan-Meier estimator."""
    logger.info("\n" + "="*70)
    logger.info("Testing Kaplan-Meier")
    logger.info("="*70)
    
    # KM doesn't use X (non-parametric baseline hazard)
    model = KaplanMeierModel()
    model.fit(T_train)
    
    median_time = model.predict_median_time()
    logger.info(f"Predicted median survival time: {median_time:.2f}")
    
    # For evaluation, KM predicts constant time for all samples
    evaluator = BaselineEvaluator()
    results = evaluator.evaluate_model(model, None, T_test, "Kaplan-Meier")
    
    return results


def test_linear_threshold(X_train, T_train, X_test, T_test):
    """Test Linear Threshold Heuristic."""
    logger.info("\n" + "="*70)
    logger.info("Testing Linear Threshold Heuristic")
    logger.info("="*70)
    
    model = LinearThresholdHeuristic(
        threshold=0.5,
        k_steps=3,
        normalize=True
    )
    model.fit(X_train, T_train)
    
    evaluator = BaselineEvaluator()
    results = evaluator.evaluate_model(model, X_test, T_test, "Linear Threshold")
    
    return results


def test_markov_chain(X_train, T_train, X_test, T_test):
    """Test Markov Chain model."""
    logger.info("\n" + "="*70)
    logger.info("Testing Markov Chain")
    logger.info("="*70)
    
    model = MarkovChainModel(
        n_states=5,
        collapse_quantile=0.2,
        normalize=True,
        random_state=42
    )
    model.fit(X_train, T_train)
    
    evaluator = BaselineEvaluator()
    results = evaluator.evaluate_model(model, X_test, T_test, "Markov Chain")
    
    return results


def main():
    """Run all tests."""
    logger.info("="*70)
    logger.info("BASELINE SPRINT 1 - END-TO-END TEST")
    logger.info("="*70)
    
    # Load data (use small sample for quick test)
    world = "sir_graph"
    logger.info(f"\nLoading {world} dataset...")
    
    X_train, T_train = load_world_data(world, "train", max_samples=30)
    X_test, T_test = load_world_data(world, "test", max_samples=15)
    
    logger.info(f"\nTrain: {len(T_train)} samples, T_event range [{T_train.min():.1f}, {T_train.max():.1f}]")
    logger.info(f"Test: {len(T_test)} samples, T_event range [{T_test.min():.1f}, {T_test.max():.1f}]")
    
    # Test each baseline
    results = {}
    
    try:
        results['kaplan_meier'] = test_kaplan_meier(T_train, T_test)
    except Exception as e:
        logger.error(f"Kaplan-Meier failed: {e}", exc_info=True)
    
    try:
        results['linear_threshold'] = test_linear_threshold(X_train, T_train, X_test, T_test)
    except Exception as e:
        logger.error(f"Linear Threshold failed: {e}", exc_info=True)
    
    try:
        results['markov_chain'] = test_markov_chain(X_train, T_train, X_test, T_test)
    except Exception as e:
        logger.error(f"Markov Chain failed: {e}", exc_info=True)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)
    
    for model_name, result in results.items():
        if 'error' not in result:
            lt = result['lead_time']
            logger.info(f"\n{model_name}:")
            logger.info(f"  Lead Time: {lt['mean']:.3f} ± {lt['std']:.3f}")
            logger.info(f"  Positive Rate: {lt['positive_rate']:.1%}")
            logger.info(f"  MAE: {result['mae']:.3f}")
            logger.info(f"  RMSE: {result['rmse']:.3f}")
        else:
            logger.info(f"\n{model_name}: FAILED - {result['error']}")
    
    logger.info("\n" + "="*70)
    logger.info("TEST COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
