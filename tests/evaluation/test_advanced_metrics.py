"""Quick test for advanced survival metrics."""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from src.evaluation.metrics_advanced import SurvivalMetrics, ExtendedEvaluator

print("=" * 60)
print("TESTING ADVANCED SURVIVAL METRICS")
print("=" * 60)

# Generate synthetic survival data
np.random.seed(42)
n_train, n_test = 60, 40

# Training data
T_train = np.random.exponential(10, n_train)
E_train = np.random.binomial(1, 0.7, n_train)

# Test data
T_test = np.random.exponential(10, n_test)
E_test = np.random.binomial(1, 0.7, n_test)

# Predictions (with some noise)
T_pred = T_test + np.random.normal(0, 2, n_test)
T_pred = np.maximum(T_pred, 0.1)  # Ensure positive

print(f"\nData: Train={n_train}, Test={n_test}")
print(f"Censoring: Train={1-E_train.mean():.1%}, Test={1-E_test.mean():.1%}")

# Test 1: C-index
print("\n1. Testing C-index...")
c_results = SurvivalMetrics.concordance_index(T_test, T_pred, E_test)
print(f"   C-index: {c_results['c_index']:.4f}")
print(f"   Concordant pairs: {c_results['concordant']}")
print(f"   Discordant pairs: {c_results['discordant']}")

# Test 2: IPCW C-index
print("\n2. Testing IPCW C-index...")
c_ipcw = SurvivalMetrics.concordance_index_ipcw(
    T_train, E_train, T_test, E_test, T_pred
)
print(f"   C-index (IPCW): {c_ipcw:.4f}")

# Test 3: Calibration slope
print("\n3. Testing Calibration slope...")
cal_slope = SurvivalMetrics.calibration_slope(T_test, T_pred, E_test)
print(f"   Calibration slope: {cal_slope:.4f} (ideal=1.0)")

# Test 4: Extended Evaluator
print("\n4. Testing ExtendedEvaluator...")
evaluator = ExtendedEvaluator()
results = evaluator.evaluate_survival_model(
    T_train, E_train, T_test, E_test, T_pred, model_name="TestModel"
)
print(f"   Results shape: {results.shape}")
print(
    "\n"
    + results[["model_name", "c_index", "c_index_ipcw", "mae"]].to_string(index=False)
)

# Test 5: Multiple models comparison
print("\n5. Testing model comparison...")
# Create slightly different predictions
T_pred2 = T_test + np.random.normal(0, 1.5, n_test)
T_pred2 = np.maximum(T_pred2, 0.1)

results2 = evaluator.evaluate_survival_model(
    T_train, E_train, T_test, E_test, T_pred2, model_name="TestModel2"
)

import pandas as pd

all_results = pd.concat([results, results2], ignore_index=True)
ranked = evaluator.compare_models(all_results, primary_metric="c_index")
print("\n" + ranked[["rank", "model_name", "c_index", "mae"]].to_string(index=False))

print("\n" + "=" * 60)
print("[OK] ALL ADVANCED METRICS TESTS PASSED!")
print("=" * 60)
