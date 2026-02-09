"""Very simple integration test."""
import sys
import numpy as np

# Workaround for Windows console encoding  
sys.stdout.reconfigure(encoding='utf-8')

from src.evaluation.integration import IntegratedEvaluator
from src.baseline.survival.kaplan_meier import KaplanMeierModel
from src.baseline.heuristics.linear_threshold import LinearThresholdHeuristic

np.random.seed(42)
X = np.random.randn(30, 8, 3)
T = np.random.uniform(5, 20, 30)
E = np.random.binomial(1, 0.7, 30)

# Fit models
km = KaplanMeierModel()
km.fit(T, E)

lt = LinearThresholdHeuristic(threshold=0.5, k_steps=2)
lt.fit(X, T)

models = {'KM': km, 'LinearThreshold': lt}

# Create evaluator
evaluator = IntegratedEvaluator(models=models, X=X, T=T, E=E, perturbation_types=['gaussian_noise'])

# Test 1: Performance
print("Test 1: Performance")
perf = evaluator.evaluate_performance()
print(f"Shape: {perf.shape}, Columns: {list(perf.columns)[:5]}")
print(perf[['model_name']].to_string(index=False))

# Test 2: Robustness  
print("\nTest 2: Robustness")
rob = evaluator.evaluate_robustness()
print(f"Shape: {rob.shape}")
print(rob.head(3).to_string(index=False))

# Test 3: Integration
print("\nTest 3: Integration")
p, r, i = evaluator.evaluate_all()
print(f"Integrated shape: {i.shape}, cols: {list(i.columns)[:6]}")

print("\n[OK] All tests passed!")
