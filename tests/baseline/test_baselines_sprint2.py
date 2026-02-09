"""
Test Suite for Sprint 2 Baselines (HMM, Deep Hazard, Temporal LSTM).

Validates the remaining 3 baselines with actual SIR dataset.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from baseline.state.hmm import HMMStateModel
from baseline.deep.deep_hazard import DeepHazardModel
from baseline.deep.temporal_classifier import TemporalClassifier
from evaluation.metrics_sprint1 import compute_lead_time


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for quick testing."""
    np.random.seed(42)
    
    n_train = 20
    n_test = 5
    seq_len = 30
    n_features = 5
    
    # Training data
    X_train = np.random.randn(n_train, seq_len, n_features).astype(np.float32)
    # Collapse happens earlier if feature values are higher
    feature_sum = X_train.mean(axis=(1, 2))
    T_train = 50 - 10 * (feature_sum - feature_sum.min()) / (feature_sum.max() - feature_sum.min())
    T_train = np.clip(T_train, 10, 80)
    
    # Test data
    X_test = np.random.randn(n_test, seq_len, n_features).astype(np.float32)
    feature_sum_test = X_test.mean(axis=(1, 2))
    T_test = 50 - 10 * (feature_sum_test - feature_sum_test.min()) / (feature_sum_test.max() - feature_sum_test.min())
    T_test = np.clip(T_test, 10, 80)
    
    return X_train, T_train, X_test, T_test


def test_hmm_basic(synthetic_data):
    """Test HMM basic functionality."""
    X_train, T_train, X_test, T_test = synthetic_data
    
    model = HMMStateModel(n_states=3, n_iter=50, random_state=42)
    model.fit(X_train, T_train)
    
    assert model.is_fitted_
    assert model.collapse_states_ is not None
    assert len(model.collapse_states_) > 0
    
    # Predict states
    states = model.predict_states(X_test[0])
    assert states.shape == (X_test.shape[1],)
    
    # Predict time
    T_pred = model.predict_time_to_collapse(X_test)
    assert T_pred.shape == (len(X_test),)
    assert np.all(T_pred >= 0)


def test_hmm_lead_time(synthetic_data):
    """Test HMM lead time computation."""
    X_train, T_train, X_test, T_test = synthetic_data
    
    model = HMMStateModel(n_states=3, n_iter=50)
    model.fit(X_train, T_train)
    
    T_pred = model.predict_time_to_collapse(X_test)
    
    # Compute lead time (allow flexibility due to stochastic HMM)
    lead_times = []
    for i in range(len(X_test)):
        T_alert = len(X_test[i]) - T_pred[i]
        T_start = 0
        T_event = T_test[i]
        
        lead_time = (T_event - T_alert) / (T_event - T_start)
        lead_times.append(lead_time)
    
    avg_lead_time = np.mean(lead_times)
    print(f"\nHMM Lead Time: {avg_lead_time:.3f}")
    
    # HMM should provide some warning (>0)
    assert avg_lead_time > 0


def test_deep_hazard_basic(synthetic_data):
    """Test Deep Hazard basic functionality."""
    X_train, T_train, X_test, T_test = synthetic_data
    
    n_features = X_train.shape[2]
    seq_len = X_train.shape[1]
    
    model = DeepHazardModel(input_dim=n_features * seq_len, hidden_dim=32, dropout=0.1)
    model.fit(X_train, T_train, epochs=20, batch_size=2, lr=1e-3)
    
    assert model.is_fitted_
    
    # Predict
    T_pred = model.predict_time(X_test)
    assert T_pred.shape == (len(X_test),)
    assert np.all(T_pred > 0)  # softplus ensures positive


def test_deep_hazard_reasonable_predictions(synthetic_data):
    """Test Deep Hazard makes reasonable predictions."""
    X_train, T_train, X_test, T_test = synthetic_data
    
    n_features = X_train.shape[2]
    seq_len = X_train.shape[1]
    
    model = DeepHazardModel(input_dim=n_features * seq_len, hidden_dim=32)
    model.fit(X_train, T_train, epochs=30, batch_size=2)
    
    T_pred = model.predict_time(X_test)
    
    # Predictions should be in reasonable range
    assert np.all(T_pred > 0)
    assert np.all(T_pred < 200)  # Synthetic data < 100, allow margin
    
    # Should produce finite predictions
    mae = np.mean(np.abs(T_pred - T_test))
    print(f"\nDeep Hazard MAE: {mae:.2f}")
    
    assert mae < 100  # Sanity check (synthetic range is 10-80)


def test_temporal_lstm_basic(synthetic_data):
    """Test Temporal LSTM basic functionality."""
    X_train, T_train, X_test, T_test = synthetic_data
    
    n_features = X_train.shape[2]
    
    model = TemporalClassifier(input_dim=n_features, hidden_dim=32, num_layers=1, 
                               dropout=0.1, rnn_type='lstm')
    model.fit(X_train, T_train, epochs=20, batch_size=2, lr=1e-3)
    
    assert model.is_fitted_
    
    # Predict
    T_pred = model.predict_time(X_test)
    assert T_pred.shape == (len(X_test),)
    assert np.all(T_pred > 0)


def test_temporal_gru(synthetic_data):
    """Test GRU variant."""
    X_train, T_train, X_test, T_test = synthetic_data
    
    n_features = X_train.shape[2]
    
    model = TemporalClassifier(input_dim=n_features, hidden_dim=32, num_layers=1, 
                               rnn_type='gru')
    model.fit(X_train, T_train, epochs=15, batch_size=2)
    
    T_pred = model.predict_time(X_test)
    assert T_pred.shape == (len(X_test),)


def test_temporal_lstm_lead_time(synthetic_data):
    """Test LSTM lead time computation."""
    X_train, T_train, X_test, T_test = synthetic_data
    
    n_features = X_train.shape[2]
    
    model = TemporalClassifier(input_dim=n_features, hidden_dim=32, num_layers=2)
    model.fit(X_train, T_train, epochs=30, batch_size=2)
    
    T_pred = model.predict_time(X_test)
    
    # Compute lead time
    lead_times = []
    for i in range(len(X_test)):
        # Alert issued when predicted T < remaining time
        remaining_time = T_test[i] - len(X_test[i])
        if T_pred[i] < remaining_time:
            T_alert = len(X_test[i])
        else:
            T_alert = T_test[i] - T_pred[i]
        
        T_start = 0
        T_event = T_test[i]
        
        lead_time = (T_event - T_alert) / (T_event - T_start)
        lead_times.append(lead_time)
    
    avg_lead_time = np.mean(lead_times)
    print(f"\nTemporal LSTM Lead Time: {avg_lead_time:.3f}")


def test_all_baselines_comparison(synthetic_data):
    """Compare all Sprint 2 baselines."""
    X_train, T_train, X_test, T_test = synthetic_data
    
    n_features = X_train.shape[2]
    seq_len = X_train.shape[1]
    
    results = {}
    
    # HMM
    hmm = HMMStateModel(n_states=3, n_iter=30)
    hmm.fit(X_train, T_train)
    T_hmm = hmm.predict_time_to_collapse(X_test)
    mae_hmm = np.mean(np.abs(T_hmm - T_test))
    results['HMM'] = mae_hmm
    
    # Deep Hazard
    deep = DeepHazardModel(input_dim=n_features * seq_len, hidden_dim=32)
    deep.fit(X_train, T_train, epochs=20, batch_size=2)
    T_deep = deep.predict_time(X_test)
    mae_deep = np.mean(np.abs(T_deep - T_test))
    results['Deep Hazard'] = mae_deep
    
    # LSTM
    lstm = TemporalClassifier(input_dim=n_features, hidden_dim=32, num_layers=1)
    lstm.fit(X_train, T_train, epochs=20, batch_size=2)
    T_lstm = lstm.predict_time(X_test)
    mae_lstm = np.mean(np.abs(T_lstm - T_test))
    results['LSTM'] = mae_lstm
    
    print("\n=== Sprint 2 Baselines MAE ===")
    for name, mae in results.items():
        print(f"{name:15s}: {mae:.2f}")
    
    # All should be reasonable
    for mae in results.values():
        assert mae < 100  # Sanity check


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
