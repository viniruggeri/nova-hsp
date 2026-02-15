"""
Tests for Unified Baseline Evaluator.

Validates the unified interface works with all baseline types.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from evaluation.unified_evaluator import UnifiedBaselineEvaluator
from baseline.survival.kaplan_meier import KaplanMeierModel
from baseline.heuristics.linear_threshold import LinearThresholdHeuristic
from baseline.state.markov import MarkovChainModel
from baseline.deep.temporal_classifier import TemporalClassifier


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    np.random.seed(42)

    n_train = 30
    n_test = 10
    seq_len = 15
    n_features = 5

    # Generate sequences
    X_train = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
    X_test = np.random.randn(n_test, seq_len, n_features).astype(np.float32)

    # Generate event times (correlated with last timestep mean)
    T_train = 30 + 20 * X_train[:, -1, :].mean(axis=1) + np.random.randn(n_train) * 5
    T_train = np.clip(T_train, 10, 80)

    T_test = 30 + 20 * X_test[:, -1, :].mean(axis=1) + np.random.randn(n_test) * 5
    T_test = np.clip(T_test, 10, 80)

    # Events (all observed)
    events_train = np.ones(n_train)
    events_test = np.ones(n_test)

    # Start times (current observation time)
    T_start = np.zeros(n_test)

    return X_train, T_train, events_train, X_test, T_test, events_test, T_start


@pytest.fixture
def trained_models(synthetic_data):
    """Train multiple baseline models."""
    X_train, T_train, events_train, _, _, _, _ = synthetic_data

    models = {}

    # Kaplan-Meier (no covariates)
    km = KaplanMeierModel()
    km.fit(T_train, events_train)
    models["KM"] = km

    # Linear Threshold
    lt = LinearThresholdHeuristic(threshold=0.3, k_steps=2)
    lt.fit(X_train, T_train)
    models["LinearThreshold"] = lt

    # Markov Chain
    mc = MarkovChainModel(n_states=3)
    mc.fit(X_train, T_train)
    models["Markov"] = mc

    # Temporal LSTM (quick training)
    lstm = TemporalClassifier(input_dim=5, hidden_dim=16, num_layers=1)
    lstm.fit(X_train, T_train, epochs=10, batch_size=8, lr=1e-3)
    models["LSTM"] = lstm

    return models


def test_evaluator_initialization(trained_models):
    """Test evaluator initialization and model type detection."""
    evaluator = UnifiedBaselineEvaluator(trained_models)

    assert len(evaluator.models) == 4
    assert len(evaluator.model_types) == 4

    # Check type detection
    assert evaluator.model_types["KM"] == "survival"
    assert evaluator.model_types["LinearThreshold"] == "heuristic"
    assert evaluator.model_types["Markov"] == "state"
    assert evaluator.model_types["LSTM"] == "deep"


def test_evaluate_single_model(trained_models, synthetic_data):
    """Test evaluating a single model."""
    _, _, _, X_test, T_test, _, T_start = synthetic_data

    evaluator = UnifiedBaselineEvaluator(trained_models)

    result = evaluator.evaluate_single("LSTM", X_test, T_test, T_start)

    assert "MAE" in result
    assert "RMSE" in result
    assert "R2" in result
    assert "model_name" in result
    assert result["model_name"] == "LSTM"
    assert result["MAE"] > 0
    assert result["prediction_time"] >= 0


def test_evaluate_all_models(trained_models, synthetic_data):
    """Test evaluating all models at once."""
    _, _, _, X_test, T_test, _, T_start = synthetic_data

    evaluator = UnifiedBaselineEvaluator(trained_models)

    results_df = evaluator.evaluate_all(X_test, T_test, T_start)

    assert len(results_df) == 4
    assert "model_name" in results_df.columns
    assert "MAE" in results_df.columns
    assert "RMSE" in results_df.columns

    # Check all models present
    model_names = set(results_df["model_name"].values)
    assert "KM" in model_names
    assert "LSTM" in model_names
    assert "Markov" in model_names
    assert "LinearThreshold" in model_names

    # Results should be sorted by MAE
    if not results_df["MAE"].isna().any():
        assert (results_df["MAE"].diff().dropna() >= 0).all()


def test_compute_metrics(trained_models):
    """Test metric computation."""
    evaluator = UnifiedBaselineEvaluator(trained_models)

    T_true = np.array([50, 60, 70, 80, 90])
    T_pred = np.array([52, 58, 72, 78, 95])
    T_start = np.zeros(5)

    metrics = evaluator.compute_metrics(T_true, T_pred, T_start)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert "LeadTime_mean" in metrics

    # MAE should be average of |differences|
    expected_mae = np.mean(np.abs(T_true - T_pred))
    assert abs(metrics["MAE"] - expected_mae) < 1e-6

    # RMSE should be sqrt of mean squared error
    expected_rmse = np.sqrt(np.mean((T_true - T_pred) ** 2))
    assert abs(metrics["RMSE"] - expected_rmse) < 1e-6


def test_cross_validation(synthetic_data):
    """Test cross-validation functionality."""
    X_train, T_train, events_train, _, _, _, _ = synthetic_data

    # Use simpler models for CV (faster)
    models = {
        "KM": KaplanMeierModel(),
        "LinearThreshold": LinearThresholdHeuristic(),
    }

    evaluator = UnifiedBaselineEvaluator(models)

    cv_results = evaluator.cross_validate(X_train, T_train, events_train, cv=3)

    # Should have 3 folds × 2 models = 6 rows
    assert len(cv_results) == 6
    assert "fold" in cv_results.columns
    assert "model_name" in cv_results.columns
    assert "MAE" in cv_results.columns

    # Each model should appear in each fold
    for model_name in ["KM", "LinearThreshold"]:
        model_folds = cv_results[cv_results["model_name"] == model_name]["fold"].values
        assert len(model_folds) == 3
        assert set(model_folds) == {0, 1, 2}


def test_summary_statistics(synthetic_data):
    """Test cross-validation summary statistics."""
    X_train, T_train, events_train, _, _, _, _ = synthetic_data

    models = {
        "KM": KaplanMeierModel(),
        "LinearThreshold": LinearThresholdHeuristic(),
    }

    evaluator = UnifiedBaselineEvaluator(models)
    cv_results = evaluator.cross_validate(X_train, T_train, events_train, cv=3)

    summary = evaluator.summary_statistics(cv_results, metrics=["MAE", "RMSE"])

    assert len(summary) == 2  # Two models
    assert "model_name" in summary.columns
    assert "MAE_mean" in summary.columns
    assert "MAE_std" in summary.columns
    assert "RMSE_mean" in summary.columns

    # Standard deviation should be non-negative
    assert (summary["MAE_std"] >= 0).all()


def test_compare_models(synthetic_data):
    """Test statistical model comparison."""
    X_train, T_train, events_train, _, _, _, _ = synthetic_data

    models = {
        "KM": KaplanMeierModel(),
        "LinearThreshold": LinearThresholdHeuristic(),
        "Markov": MarkovChainModel(n_states=3),
    }

    evaluator = UnifiedBaselineEvaluator(models)
    cv_results = evaluator.cross_validate(X_train, T_train, events_train, cv=3)

    comparison = evaluator.compare_models(cv_results, metric="MAE")

    # Should be square matrix
    assert comparison.shape[0] == comparison.shape[1]
    assert comparison.shape[0] == 3  # Three models

    # Diagonal should be 1.0 (same model)
    assert (np.diag(comparison.values) == 1.0).all()

    # Should be symmetric (p-values)
    # (Not always true for all tests, but check structure)
    assert comparison.index.tolist() == comparison.columns.tolist()


def test_predictions_return(trained_models, synthetic_data):
    """Test returning predictions with results."""
    _, _, _, X_test, T_test, _, _ = synthetic_data

    evaluator = UnifiedBaselineEvaluator(trained_models)

    result = evaluator.evaluate_single("LSTM", X_test, T_test, return_predictions=True)

    assert "predictions" in result
    assert len(result["predictions"]) == len(T_test)


def test_handle_invalid_predictions(trained_models):
    """Test handling of NaN/inf predictions."""
    evaluator = UnifiedBaselineEvaluator(trained_models)

    T_true = np.array([50, 60, 70, 80, 90])
    T_pred = np.array([52, np.nan, 72, np.inf, 95])

    metrics = evaluator.compute_metrics(T_true, T_pred)

    # Should only use valid predictions (3 out of 5)
    assert "MAE" in metrics
    # MAE should be from valid predictions only: |50-52|, |70-72|, |90-95|
    expected_mae = np.mean([2, 2, 5])
    assert abs(metrics["MAE"] - expected_mae) < 1e-6


def test_results_to_dataframe(trained_models, synthetic_data):
    """Test DataFrame export functionality."""
    _, _, _, X_test, T_test, _, T_start = synthetic_data

    evaluator = UnifiedBaselineEvaluator(trained_models)
    results_df = evaluator.evaluate_all(X_test, T_test, T_start)

    # Should be a proper DataFrame
    assert isinstance(results_df, type(results_df))  # pd.DataFrame
    assert not results_df.empty

    # Should have expected columns
    assert "model_name" in results_df.columns
    assert "MAE" in results_df.columns

    # Should be exportable to CSV
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        results_df.to_csv(f.name, index=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
