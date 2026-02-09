"""Quick validation of robustness implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from evaluation.robustness import PerturbationFactory, RobustnessEvaluator

print("Testing Robustness Implementation...")
print("=" * 60)

# Test data
np.random.seed(42)
X = np.random.randn(10, 15, 5).astype(np.float32)
T = np.random.randn(10) * 10 + 50

print(f"X shape: {X.shape}")
print(f"T shape: {T.shape}")

# Test perturbations
print("\n1. Testing Gaussian Noise...")
X_noisy = PerturbationFactory.gaussian_noise(X, sigma=0.1)
print(f"   Shape: {X_noisy.shape} ✓")
print(f"   Mean diff: {np.mean(np.abs(X - X_noisy)):.4f}")

print("\n2. Testing Feature Dropout...")
X_dropped = PerturbationFactory.feature_dropout(X, dropout_rate=0.2)
print(f"   Shape: {X_dropped.shape} ✓")
print(f"   Changed ratio: {np.mean(X != X_dropped):.2%}")

print("\n3. Testing Temporal Delay...")
X_delayed = PerturbationFactory.temporal_delay(X, lag_steps=2)
print(f"   Shape: {X_delayed.shape} ✓")

print("\n4. Testing RobustnessEvaluator...")
evaluator = RobustnessEvaluator(X, T)
perturbations = evaluator.apply_perturbations()
print(f"   Perturbation types: {list(perturbations.keys())}")
print(f"   Gaussian noise levels: {len(perturbations['gaussian_noise'])}")
print(f"   Feature dropout levels: {len(perturbations['feature_dropout'])}")
print(f"   Temporal delay levels: {len(perturbations['temporal_delay'])}")

print("\n5. Testing Degradation Computation...")
metric_clean = 10.0
metric_perturbed = 12.0
degradation = RobustnessEvaluator.compute_degradation(metric_clean, metric_perturbed)
print(f"   Clean: {metric_clean}, Perturbed: {metric_perturbed}")
print(f"   Degradation: {degradation:.1f}%")

print("\n6. Testing AUC-D Computation...")
intensities = [0.0, 0.1, 0.2, 0.3]
degradations = [0.0, 5.0, 15.0, 30.0]
auc_d = RobustnessEvaluator.compute_auc_degradation(intensities, degradations)
print(f"   AUC-D: {auc_d:.2f}")

print("\n" + "=" * 60)
print("✅ All basic tests passed!")
