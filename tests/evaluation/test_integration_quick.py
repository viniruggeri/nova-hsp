"""
Quick test for IntegratedEvaluator.
"""

import numpy as np
from src.evaluation.integration import IntegratedEvaluator
from src.baseline.survival.kaplan_meier import KaplanMeierModel
from src.baseline.heuristics.linear_threshold import LinearThresholdHeuristic

print("=" * 60)
print("TESTING INTEGRATED EVALUATOR")
print("=" * 60)

# Generate synthetic data
np.random.seed(42)
n_samples, n_timesteps, n_features = 50, 10, 3

X = np.random.randn(n_samples, n_timesteps, n_features)
T = np.random.uniform(5, 20, n_samples)
E = np.random.binomial(1, 0.7, n_samples)

print(f"Data shape: X={X.shape}, T={T.shape}, E={E.shape}")

# Initialize models
models = {
    "KM": KaplanMeierModel(),
    "LinearThreshold": LinearThresholdHeuristic(threshold=0.5, k_steps=2),
}

# Fit models
print("Fitting models...")
models["KM"].fit(T, E)
models["LinearThreshold"].fit(X, T)
print(f"Models fitted: {list(models.keys())}")

# Initialize integrated evaluator
print("\n1. Initializing IntegratedEvaluator...")
evaluator = IntegratedEvaluator(
    models=models,
    X=X,
    T=T,
    E=E,
    perturbation_types=["gaussian_noise", "feature_dropout"],
    n_folds=3,
)
print("   [OK] Initialized")

# Evaluate performance
print("\n2. Evaluating performance...")
perf_df = evaluator.evaluate_performance()
print(f"   [OK] Performance results: {perf_df.shape}")
print(f"   Columns: {list(perf_df.columns)}")
print("\n" + perf_df[["model_name", "MAE", "RMSE"]].to_string(index=False))

# Evaluate robustness
print("\n3. Evaluating robustness...")
rob_df = evaluator.evaluate_robustness()
print(f"   [OK] Robustness results: {rob_df.shape}")
print(
    "\n"
    + rob_df.groupby(["model_name", "perturbation_type"])["degradation_percent"]
    .mean()
    .to_string()
)

# Full evaluation
print("\n4. Running full evaluation...")
perf, rob, integrated = evaluator.evaluate_all()
print(f"   [OK] Performance: {perf.shape}")
print(f"   [OK] Robustness: {rob.shape}")
print(f"   [OK] Integrated: {integrated.shape}")

print("\n5. Integrated summary:")
print(
    "\n"
    + integrated[
        ["model_name", "MAE", "mean_degradation", "composite_score"]
    ].to_string(index=False)
)

# Generate report
print("\n6. Generating report...")
report = evaluator.generate_report()
print("\n--- REPORT PREVIEW (first 500 chars) ---")
print(report[:500])
print("...")

print("\n" + "=" * 60)
print("[OK] ALL INTEGRATION TESTS PASSED!")
print("=" * 60)
