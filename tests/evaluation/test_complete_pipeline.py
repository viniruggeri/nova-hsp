"""Test complete evaluation pipeline with advanced metrics."""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from src.evaluation.integration import IntegratedEvaluator
from src.baseline.survival.kaplan_meier import KaplanMeierModel
from src.baseline.survival.cox_ph import CoxPHModel

print("=" * 60)
print("TESTING COMPLETE EVALUATION PIPELINE")
print("=" * 60)

# Generate survival data
np.random.seed(42)
n = 80
X = np.random.randn(n, 10, 3)
T = np.random.exponential(10, n)
E = np.random.binomial(1, 0.7, n)

print(f"\nData: n={n}, censoring={1-E.mean():.1%}")

# Fit models
km = KaplanMeierModel()
km.fit(T, E)

cox = CoxPHModel()
cox.fit(X[:, -1, :], T, E)  # Use last timestep features

models = {"KM": km, "CoxPH": cox}
print(f"Models: {list(models.keys())}")

# Create evaluator
evaluator = IntegratedEvaluator(
    models=models, X=X, T=T, E=E, perturbation_types=["gaussian_noise"], n_folds=3
)
print("[OK] IntegratedEvaluator initialized")

# Test 1: Standard evaluation
print("\n1. Standard Evaluation (performance + robustness)...")
perf, rob, integrated = evaluator.evaluate_all(include_cv=False, include_advanced=False)
print(f"   Performance: {perf.shape}")
print(f"   Robustness: {rob.shape}")
print(f"   Integrated: {integrated.shape}")

# Test 2: With advanced metrics
print("\n2. Evaluation with Advanced Metrics...")
try:
    perf_adv, rob_adv, int_adv = evaluator.evaluate_all(
        include_cv=False, include_advanced=True
    )
    print(f"   Performance (with C-index): {perf_adv.shape}")

    if "c_index" in perf_adv.columns:
        print("\n   Advanced Metrics:")
        print(
            perf_adv[["model_name", "MAE", "c_index", "c_index_ipcw"]].to_string(
                index=False
            )
        )
    else:
        print("   [WARN] C-index not computed")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 3: Advanced metrics standalone
print("\n3. Standalone Advanced Metrics...")
try:
    advanced = evaluator.evaluate_advanced_survival_metrics(compute_ipcw=True)
    print(f"   Shape: {advanced.shape}")
    if not advanced.empty:
        print(
            "\n"
            + advanced[
                ["model_name", "c_index", "c_index_ipcw", "calibration_slope"]
            ].to_string(index=False)
        )
except Exception as e:
    print(f"   [ERROR] {e}")

print("\n" + "=" * 60)
print("[OK] COMPLETE PIPELINE TEST FINISHED!")
print("=" * 60)
