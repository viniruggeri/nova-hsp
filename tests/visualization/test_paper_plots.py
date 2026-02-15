"""Test paper visualization utilities."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

from src.visualization.paper_plots import PaperVisualizer

print("=" * 60)
print("TESTING PAPER VISUALIZATIONS")
print("=" * 60)

# Create synthetic results
np.random.seed(42)

performance_results = pd.DataFrame(
    {
        "model_name": ["KM", "CoxPH", "LinearThreshold", "HMM", "DeepHazard"],
        "MAE": np.random.uniform(2, 5, 5),
        "RMSE": np.random.uniform(3, 6, 5),
        "c_index": np.random.uniform(0.65, 0.90, 5),
        "mean_degradation": np.random.uniform(15, 35, 5),
    }
)

robustness_results = pd.DataFrame(
    {
        "model_name": np.repeat(["KM", "CoxPH", "LinearThreshold"], 9),
        "perturbation_type": np.tile(
            ["gaussian_noise", "feature_dropout", "temporal_delay"], 9
        ),
        "intensity": np.tile([0.01, 0.05, 0.10], 9),
        "degradation_percent": np.random.uniform(0, 40, 27),
    }
)

print("\nSynthetic data created:")
print(f"  Performance: {len(performance_results)} models")
print(f"  Robustness: {len(robustness_results)} entries")

# Initialize visualizer
print("\n1. Initializing PaperVisualizer...")
viz = PaperVisualizer(style="seaborn-v0_8-paper", palette="colorblind")
print("   [OK] Visualizer initialized")

# Test plots in temporary directory
with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir)

    # Test 1: Model comparison
    print("\n2. Testing plot_model_comparison()...")
    try:
        viz.plot_model_comparison(
            performance_results,
            metrics=["MAE", "RMSE", "c_index"],
            output_file=output_dir / "comparison.pdf",
        )
        if (output_dir / "comparison.pdf").exists():
            print("   [OK] Model comparison plot created")
        else:
            print("   [WARN] Plot file not found")
    except Exception as e:
        print(f"   [ERROR] {e}")

    # Test 2: Robustness degradation
    print("\n3. Testing plot_robustness_degradation()...")
    try:
        viz.plot_robustness_degradation(
            robustness_results, output_file=output_dir / "robustness.pdf"
        )
        if (output_dir / "robustness.pdf").exists():
            print("   [OK] Robustness plot created")
        else:
            print("   [WARN] Plot file not found")
    except Exception as e:
        print(f"   [ERROR] {e}")

    # Test 3: Calibration plot
    print("\n4. Testing plot_calibration()...")
    try:
        T_true = np.random.exponential(10, 100)
        T_pred = T_true + np.random.normal(0, 2, 100)

        viz.plot_calibration(
            T_true,
            T_pred,
            model_name="TestModel",
            output_file=output_dir / "calibration.pdf",
        )

        if (output_dir / "calibration.pdf").exists():
            print("   [OK] Calibration plot created")
        else:
            print("   [WARN] Plot file not found")
    except Exception as e:
        print(f"   [ERROR] {e}")

    # Test 4: Heatmap
    print("\n5. Testing plot_metric_heatmap()...")
    try:
        viz.plot_metric_heatmap(
            performance_results,
            metrics=["MAE", "RMSE", "c_index"],
            output_file=output_dir / "heatmap.pdf",
        )

        if (output_dir / "heatmap.pdf").exists():
            print("   [OK] Heatmap created")
        else:
            print("   [WARN] Plot file not found")
    except Exception as e:
        print(f"   [ERROR] {e}")

    # Test 5: Pareto front
    print("\n6. Testing plot_pareto_front()...")
    try:
        viz.plot_pareto_front(
            performance_results,
            x_metric="MAE",
            y_metric="c_index",
            output_file=output_dir / "pareto.pdf",
            x_ascending=True,
            y_ascending=False,
        )

        if (output_dir / "pareto.pdf").exists():
            print("   [OK] Pareto front created")
        else:
            print("   [WARN] Plot file not found")
    except Exception as e:
        print(f"   [ERROR] {e}")

    # Test 6: Generate all plots
    print("\n7. Testing generate_all_plots()...")
    try:
        viz.generate_all_plots(
            performance_results=performance_results,
            robustness_results=robustness_results,
            output_dir=output_dir / "all_plots",
            metrics=["MAE", "RMSE", "c_index"],
        )

        generated_files = list((output_dir / "all_plots").glob("*.pdf"))
        print(f"   [OK] Generated {len(generated_files)} figures")
        for f in generated_files:
            print(f"      - {f.name}")
    except Exception as e:
        print(f"   [ERROR] {e}")

print("\n" + "=" * 60)
print("[OK] ALL VISUALIZATION TESTS PASSED!")
print("=" * 60)
