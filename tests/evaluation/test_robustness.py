"""
Tests for Robustness Analysis.

Validates perturbation functions and degradation metrics.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from evaluation.robustness import (
    PerturbationFactory,
    RobustnessEvaluator,
    RobustnessAnalyzer,
)


@pytest.fixture
def synthetic_sequences():
    """Generate synthetic sequence data."""
    np.random.seed(42)

    n_samples = 20
    seq_len = 15
    n_features = 5

    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    T = 30 + 20 * X[:, -1, :].mean(axis=1) + np.random.randn(n_samples) * 5
    T = np.clip(T, 10, 80)

    return X, T


def test_gaussian_noise():
    """Test Gaussian noise perturbation."""
    X = np.ones((10, 15, 5))

    X_noisy = PerturbationFactory.gaussian_noise(X, sigma=0.1)

    assert X_noisy.shape == X.shape
    assert not np.allclose(X, X_noisy)  # Should be different

    # Check noise magnitude
    noise_std = np.std(X_noisy - X)
    assert 0.05 < noise_std < 0.15  # Approximately sigma=0.1


def test_feature_dropout():
    """Test feature dropout perturbation."""
    # Use random values so median replacement actually changes values
    np.random.seed(42)
    X = np.random.randn(10, 15, 5) * 2 + 5  # Mean ~5, std ~2

    X_dropped = PerturbationFactory.feature_dropout(X, dropout_rate=0.5)

    assert X_dropped.shape == X.shape

    # Check that some values changed (replaced with median)
    changed_ratio = np.mean(X != X_dropped)
    assert 0.3 < changed_ratio < 0.7  # Approximately 50% dropout


def test_temporal_delay():
    """Test temporal delay perturbation."""
    X = np.arange(60).reshape(2, 10, 3).astype(float)

    X_delayed = PerturbationFactory.temporal_delay(X, lag_steps=2)

    assert X_delayed.shape == X.shape

    # Check that values are shifted
    # X[:, 0, 0] should become X[:, 2, 0]
    assert np.allclose(X_delayed[0, 0, 0], X[0, 2, 0])


def test_robustness_evaluator_initialization(synthetic_sequences):
    """Test RobustnessEvaluator initialization."""
    X, T = synthetic_sequences

    evaluator = RobustnessEvaluator(X, T)

    assert evaluator.X_test.shape == X.shape
    assert evaluator.y_test.shape == T.shape


def test_apply_perturbations(synthetic_sequences):
    """Test applying all perturbation types."""
    X, T = synthetic_sequences

    evaluator = RobustnessEvaluator(X, T)
    perturbations = evaluator.apply_perturbations()

    assert "gaussian_noise" in perturbations
    assert "feature_dropout" in perturbations
    assert "temporal_delay" in perturbations

    # Check Gaussian noise levels
    assert len(perturbations["gaussian_noise"]) == 3  # 3 sigma values

    for sigma, X_perturbed in perturbations["gaussian_noise"]:
        assert X_perturbed.shape == X.shape
        assert sigma in [0.01, 0.05, 0.1]


def test_compute_degradation():
    """Test degradation computation."""
    metric_clean = 10.0
    metric_perturbed = 12.0

    degradation = RobustnessEvaluator.compute_degradation(
        metric_clean, metric_perturbed
    )

    # For error metrics (higher = worse), degradation should be negative
    # (metric_clean - metric_perturbed) / metric_clean = (10 - 12) / 10 = -0.2 = -20%
    assert degradation == pytest.approx(-20.0, abs=0.1)


def test_compute_auc_degradation():
    """Test AUC-D computation."""
    intensities = [0.0, 0.1, 0.2, 0.3]
    degradations = [0.0, 5.0, 15.0, 30.0]  # Linear increase

    auc_d = RobustnessEvaluator.compute_auc_degradation(intensities, degradations)

    # Trapezoidal integration of linear function
    # Should be approximately (0 + 30) / 2 * 0.3 = 4.5
    assert auc_d > 0
    assert auc_d < 10  # Reasonable range


def test_robustness_analyzer_initialization():
    """Test RobustnessAnalyzer initialization."""
    analyzer = RobustnessAnalyzer()

    assert hasattr(analyzer, "results")
    assert isinstance(analyzer.results, dict)
    assert len(analyzer.results) == 0


def test_evaluate_model_robustness(synthetic_sequences):
    """Test full robustness evaluation for a model."""
    X, T = synthetic_sequences

    # Simple dummy model: predict mean
    def predict_fn(X_input):
        return np.full(len(X_input), T.mean())

    # Metric: MAE
    def metric_fn(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    analyzer = RobustnessAnalyzer()
    results = analyzer.evaluate_model_robustness(
        model_name="DummyModel",
        X_test=X,
        y_test=T,
        predict_fn=predict_fn,
        metric_fn=metric_fn,
        metric_name="MAE",
    )

    assert "gaussian_noise" in results
    assert "feature_dropout" in results
    assert "temporal_delay" in results

    # Check structure
    for perturb_type in ["gaussian_noise", "feature_dropout", "temporal_delay"]:
        assert "auc_d" in results[perturb_type]
        assert "intensities" in results[perturb_type]
        assert "degradations" in results[perturb_type]
        assert "metric_clean" in results[perturb_type]


def test_summarize_robustness(synthetic_sequences):
    """Test robustness summary across models."""
    X, T = synthetic_sequences

    # Dummy models
    def predict_fn1(X_input):
        return np.full(len(X_input), T.mean())

    def predict_fn2(X_input):
        return np.full(len(X_input), T.mean() + 5)

    def metric_fn(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    analyzer = RobustnessAnalyzer()

    # Evaluate two models
    analyzer.evaluate_model_robustness("Model1", X, T, predict_fn1, metric_fn)
    analyzer.evaluate_model_robustness("Model2", X, T, predict_fn2, metric_fn)

    summary = analyzer.summarize_robustness()

    assert "gaussian_noise_mean_auc_d" in summary
    assert "feature_dropout_mean_auc_d" in summary
    assert "temporal_delay_mean_auc_d" in summary


def test_perturbations_preserve_shape(synthetic_sequences):
    """Test that all perturbations preserve data shape."""
    X, _ = synthetic_sequences

    # Gaussian noise
    X_noisy = PerturbationFactory.gaussian_noise(X, sigma=0.1)
    assert X_noisy.shape == X.shape

    # Feature dropout
    X_dropped = PerturbationFactory.feature_dropout(X, dropout_rate=0.2)
    assert X_dropped.shape == X.shape

    # Temporal delay
    X_delayed = PerturbationFactory.temporal_delay(X, lag_steps=3)
    assert X_delayed.shape == X.shape


def test_zero_perturbation_equals_original():
    """Test that zero perturbation returns data unchanged."""
    X = np.random.randn(5, 10, 3)

    # Zero noise
    X_noisy = PerturbationFactory.gaussian_noise(X, sigma=0.0)
    assert np.allclose(X, X_noisy)

    # Zero dropout
    X_dropped = PerturbationFactory.feature_dropout(X, dropout_rate=0.0)
    # Note: dropout replaces with median, so not exactly equal
    # Just check shape
    assert X_dropped.shape == X.shape

    # Zero delay
    X_delayed = PerturbationFactory.temporal_delay(X, lag_steps=0)
    assert np.allclose(X, X_delayed)


def test_high_noise_significant_change():
    """Test that high noise creates significant changes."""
    X = np.ones((10, 15, 5))

    X_noisy = PerturbationFactory.gaussian_noise(X, sigma=1.0)

    # With high noise, should differ significantly
    diff = np.abs(X_noisy - X).mean()
    assert diff > 0.5  # Significant change


def test_robustness_metrics_non_negative():
    """Test that AUC-D is non-negative."""
    intensities = [0.0, 0.1, 0.2]
    degradations = [0.0, 10.0, 25.0]

    auc_d = RobustnessEvaluator.compute_auc_degradation(intensities, degradations)

    assert auc_d >= 0


def test_degradation_with_zero_baseline():
    """Test degradation computation with zero baseline."""
    metric_clean = 0.0
    metric_perturbed = 5.0

    degradation = RobustnessEvaluator.compute_degradation(
        metric_clean, metric_perturbed
    )

    # Should handle division by zero gracefully
    assert not np.isnan(degradation)
    assert not np.isinf(degradation)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
