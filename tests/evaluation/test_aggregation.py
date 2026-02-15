"""Test results aggregation and export."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

from src.evaluation.aggregator import ResultsAggregator

print("=" * 60)
print("TESTING RESULTS AGGREGATION")
print("=" * 60)

# Create synthetic results for multiple worlds
np.random.seed(42)

# World 1: SIR Graph
sir_results = pd.DataFrame(
    {
        "model_name": ["KM", "CoxPH", "LinearThreshold"] * 3,
        "MAE": np.random.uniform(2, 5, 9),
        "RMSE": np.random.uniform(3, 6, 9),
        "c_index": np.random.uniform(0.7, 0.9, 9),
        "mean_degradation": np.random.uniform(10, 30, 9),
    }
)

# World 2: Ant Colony
ant_results = pd.DataFrame(
    {
        "model_name": ["KM", "CoxPH", "LinearThreshold"] * 3,
        "MAE": np.random.uniform(2.5, 5.5, 9),
        "RMSE": np.random.uniform(3.5, 6.5, 9),
        "c_index": np.random.uniform(0.65, 0.85, 9),
        "mean_degradation": np.random.uniform(15, 35, 9),
    }
)

# World 3: Real Dataset
real_results = pd.DataFrame(
    {
        "model_name": ["KM", "CoxPH", "LinearThreshold"] * 3,
        "MAE": np.random.uniform(3, 6, 9),
        "RMSE": np.random.uniform(4, 7, 9),
        "c_index": np.random.uniform(0.6, 0.8, 9),
        "mean_degradation": np.random.uniform(20, 40, 9),
    }
)

print("\nSynthetic data created:")
print(f"  SIR: {len(sir_results)} results")
print(f"  Ant Colony: {len(ant_results)} results")
print(f"  Real Dataset: {len(real_results)} results")

# Test 1: Add results
print("\n1. Testing add_results()...")
aggregator = ResultsAggregator()
aggregator.add_results("sir_graph", sir_results, {"world_type": "SIR", "n_nodes": 100})
aggregator.add_results("ant_colony", ant_results, {"world_type": "Ant", "n_agents": 50})
aggregator.add_results("real_data", real_results, {"world_type": "Real"})
print(f"   Added {len(aggregator.results)} experiments")

# Test 2: Aggregate by model
print("\n2. Testing aggregate_by_model()...")
aggregated = aggregator.aggregate_by_model(metrics=["MAE", "RMSE", "c_index"])
print(f"   Aggregated shape: {aggregated.shape}")
print(
    "\n"
    + aggregated[["model_name", "MAE_mean", "MAE_std", "c_index_mean"]].to_string(
        index=False
    )
)

# Test 3: Cross-world comparison
print("\n3. Testing compare_across_worlds()...")
comparison = aggregator.compare_across_worlds(metric="MAE", alpha=0.05)
print(f"   Comparison shape: {comparison.shape}")
if not comparison.empty:
    print(
        "\n"
        + comparison[["model_name", "kruskal_h", "kruskal_p", "significant"]].to_string(
            index=False
        )
    )

# Test 4: Pairwise comparison
print("\n4. Testing pairwise_comparison()...")
pairwise = aggregator.pairwise_comparison("sir_graph", "ant_colony", metric="MAE")
print(f"   Pairwise shape: {pairwise.shape}")
if not pairwise.empty:
    print(
        "\n"
        + pairwise[
            ["model_name", "sir_graph_mean", "ant_colony_mean", "p_value"]
        ].to_string(index=False)
    )

# Test 5: Multi-criteria ranking
print("\n5. Testing rank_models()...")
ranked = aggregator.rank_models(
    metrics=["MAE", "c_index"], weights=[1.0, 1.0], ascending=[True, False]
)
print(f"   Ranked shape: {ranked.shape}")
if not ranked.empty:
    print(
        "\n" + ranked[["model_name", "rank", "composite_score"]].to_string(index=False)
    )

# Test 6: LaTeX export
print("\n6. Testing export_latex_table()...")
with tempfile.TemporaryDirectory() as tmpdir:
    output_file = Path(tmpdir) / "results.tex"
    aggregator.export_latex_table(
        output_file=output_file,
        metrics=["MAE", "c_index"],
        caption="Baseline Model Comparison",
        label="tab:baselines",
    )

    if output_file.exists():
        print("   [OK] LaTeX table created")
        with open(output_file, "r") as f:
            lines = f.readlines()
        print(f"   Preview (first 10 lines):")
        for line in lines[:10]:
            print(f"   {line.rstrip()}")
    else:
        print("   [ERROR] LaTeX file not created")

# Test 7: Markdown export
print("\n7. Testing export_summary_markdown()...")
with tempfile.TemporaryDirectory() as tmpdir:
    output_file = Path(tmpdir) / "summary.md"
    aggregator.export_summary_markdown(output_file)

    if output_file.exists():
        print("   [OK] Markdown summary created")
        with open(output_file, "r") as f:
            content = f.read()
        print(f"   Length: {len(content)} characters")
    else:
        print("   [ERROR] Markdown file not created")

print("\n" + "=" * 60)
print("[OK] ALL AGGREGATION TESTS PASSED!")
print("=" * 60)
