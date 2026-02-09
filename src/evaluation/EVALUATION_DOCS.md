# Evaluation Infrastructure - Complete Documentation

## Overview

Complete evaluation system for HSP and baseline models with paper-ready outputs:
- **Unified evaluation** across 9 baseline models
- **Robustness analysis** with 3 perturbation types
- **Advanced survival metrics** (C-index, IPCW, IBS)
- **Cross-world aggregation** with statistical tests
- **Publication-quality visualizations**

---

## Architecture

```
src/evaluation/
├── unified_evaluator.py      # Core evaluator (530 lines)
├── robustness.py             # Perturbation analysis (337 lines)
├── metrics_advanced.py       # Survival metrics (380 lines)
├── integration.py            # Combined evaluator (442 lines)
└── aggregator.py             # Results aggregation (520 lines)

src/visualization/
└── paper_plots.py            # Publication plots (410 lines)

scripts/
├── test_*.py                 # Validation scripts
├── aggregate_results.py      # CLI aggregation tool
└── generate_figures.py       # CLI visualization tool
```

**Total**: ~2,600 lines of production code

---

## Quick Start

### 1. Evaluate Models

```python
from src.evaluation.integration import IntegratedEvaluator

# Initialize
evaluator = IntegratedEvaluator(
    models={'KM': kaplan_meier, 'CoxPH': cox_model},
    X=X_features,
    T=event_times,
    E=event_indicators
)

# Full evaluation: performance + robustness + advanced metrics
perf, rob, integrated = evaluator.evaluate_all(
    include_cv=True,           # K-fold cross-validation
    include_advanced=True      # C-index, IPCW, calibration
)

# Save results
evaluator.save_results(output_dir='results/experiment_name/')
```

### 2. Aggregate Cross-World Results

```bash
python scripts/aggregate_results.py \
    --results_dir results/simulated \
    --output_dir results/aggregated \
    --metrics MAE RMSE c_index \
    --latex  # Generate LaTeX tables
```

### 3. Generate Paper Figures

```bash
python scripts/generate_figures.py \
    --results_dir results/aggregated \
    --output_dir results/figures \
    --metrics MAE RMSE c_index \
    --format pdf
```

---

## Component Details

### 1. Unified Evaluator (`unified_evaluator.py`)

**Features:**
- Auto-detects model types (survival, heuristic, state, deep)
- Standard metrics: MAE, RMSE, R², Correlation, MAPE
- K-fold cross-validation
- Statistical comparison (Wilcoxon, t-test)

**Usage:**
```python
from src.evaluation.unified_evaluator import UnifiedBaselineEvaluator

evaluator = UnifiedBaselineEvaluator(models)
results = evaluator.evaluate_all(X_test, T_test, events=E_test)
cv_results = evaluator.cross_validate(X, T, events=E, cv=5)
comparison = evaluator.compare_models()
```

---

### 2. Robustness Analysis (`robustness.py`)

**Perturbations:**
- **Gaussian noise**: σ ∈ {0.01, 0.05, 0.10}
- **Feature dropout**: rate ∈ {0.1, 0.2, 0.3}
- **Temporal delay**: steps ∈ {1, 2, 5}

**Metrics:**
- Degradation % relative to clean data
- AUC-D (Area Under Degradation curve)

**Usage:**
```python
from src.evaluation.robustness import RobustnessAnalyzer

analyzer = RobustnessAnalyzer()
robustness = analyzer.evaluate_model_robustness(
    model_name='CoxPH',
    X_test=X,
    y_test=T,
    predict_fn=model.predict,
    metric_fn=lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
)
```

---

### 3. Advanced Survival Metrics (`metrics_advanced.py`)

**Metrics:**
- **C-index**: Concordance index [0, 1] (higher = better)
- **C-index IPCW**: Inverse probability weighted (censoring-robust)
- **IBS**: Integrated Brier Score [0, 1] (lower = better)
- **Time-dependent AUC**: Discrimination at specific time points
- **Calibration slope**: Ideal = 1.0

**Requires:** `scikit-survival`

**Usage:**
```python
from src.evaluation.metrics_advanced import SurvivalMetrics

# C-index
c_results = SurvivalMetrics.concordance_index(T_test, T_pred, E_test)
print(f"C-index: {c_results['c_index']:.4f}")

# IPCW C-index (more robust)
c_ipcw = SurvivalMetrics.concordance_index_ipcw(
    T_train, E_train, T_test, E_test, T_pred
)
```

---

### 4. Integration Layer (`integration.py`)

**Combines:**
- Unified evaluator (performance metrics)
- Robustness analyzer (perturbation testing)
- Advanced metrics (C-index, IPCW)

**Output:**
- `performance_results.csv`: MAE, RMSE, R², etc.
- `robustness_results.csv`: Degradation by perturbation
- `integrated_results.csv`: Combined with composite score

**Usage:**
```python
from src.evaluation.integration import IntegratedEvaluator

evaluator = IntegratedEvaluator(models, X, T, E)

# Full pipeline
perf, rob, integrated = evaluator.evaluate_all(
    include_cv=True,
    include_advanced=True
)

# Generate report
report = evaluator.generate_report(output_path='report.txt')
```

---

### 5. Results Aggregation (`aggregator.py`)

**Features:**
- Cross-world comparison (SIR vs Ant Colony vs Real Dataset)
- Statistical tests: Kruskal-Wallis, Mann-Whitney, Wilcoxon
- Multi-criteria ranking with Pareto fronts
- LaTeX table export (paper-ready)
- Markdown summary generation

**Usage:**
```python
from src.evaluation.aggregator import ResultsAggregator

agg = ResultsAggregator()

# Load results from directory structure
agg.load_results_from_dir('results/simulated/')

# Aggregate by model
aggregated = agg.aggregate_by_model(metrics=['MAE', 'c_index'])

# Statistical comparison
comparison = agg.compare_across_worlds(metric='MAE')
pairwise = agg.pairwise_comparison('sir_graph', 'ant_colony')

# Export LaTeX
agg.export_latex_table(
    output_file='table.tex',
    caption='Model Comparison',
    bold_best=True
)
```

---

### 6. Paper Visualizations (`paper_plots.py`)

**Plots:**
- **Model comparison**: Bar charts with highlighted best
- **Robustness degradation**: Curves by perturbation type
- **Calibration**: Binned predictions vs observations
- **Heatmap**: Normalized performance across metrics
- **Pareto front**: Trade-off visualization

**Publication settings:**
- DPI: 300
- Format: PDF (vector graphics)
- Style: Seaborn paper style
- Colors: Colorblind-friendly palette

**Usage:**
```python
from src.visualization.paper_plots import PaperVisualizer

viz = PaperVisualizer()

# Generate all plots
viz.generate_all_plots(
    performance_results=perf_df,
    robustness_results=rob_df,
    output_dir='figures/',
    metrics=['MAE', 'RMSE', 'c_index']
)
```

---

## Workflow Example

### End-to-End Evaluation

```python
# 1. Fit models
km = KaplanMeierModel()
km.fit(T_train, E_train)

cox = CoxPHModel()
cox.fit(X_train, T_train, E_train)

models = {'KM': km, 'CoxPH': cox}

# 2. Integrated evaluation
evaluator = IntegratedEvaluator(models, X_test, T_test, E_test)
perf, rob, integrated = evaluator.evaluate_all(include_advanced=True)

# 3. Save per-experiment results
evaluator.save_results('results/sir_graph/')

# 4. Aggregate across experiments (after running for multiple worlds)
from src.evaluation.aggregator import ResultsAggregator
agg = ResultsAggregator()
agg.load_results_from_dir('results/')
aggregated = agg.aggregate_by_model()
agg.export_latex_table('table.tex')

# 5. Generate figures
from src.visualization.paper_plots import PaperVisualizer
viz = PaperVisualizer()
viz.generate_all_plots(aggregated, output_dir='figures/')
```

---

## Testing

All components have validation scripts:

```bash
# Test individual components
python scripts/test_advanced_metrics.py
python scripts/test_aggregation.py
python scripts/test_visualizations.py

# Test complete pipeline
python scripts/test_complete_pipeline.py
```

---

## Output Structure

```
results/
├── sir_graph/
│   ├── performance_results.csv
│   ├── robustness_results.csv
│   ├── integrated_results.csv
│   └── metadata.json
├── ant_colony/
│   └── ...
├── aggregated/
│   ├── aggregated_results.csv
│   ├── comparison_MAE.csv
│   ├── model_ranking.csv
│   ├── table_comparison.tex      # LaTeX table
│   └── summary.md                # Human-readable summary
└── figures/
    ├── model_comparison.pdf
    ├── robustness_degradation.pdf
    ├── performance_heatmap.pdf
    └── pareto_MAE_c_index.pdf
```

---

## Metrics Reference

### Standard Metrics (Unified Evaluator)

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| MAE | [0, ∞] | Lower | Mean Absolute Error |
| RMSE | [0, ∞] | Lower | Root Mean Squared Error |
| R² | (-∞, 1] | Higher | Coefficient of determination |
| Correlation | [-1, 1] | Higher | Pearson correlation |
| MAPE | [0, ∞] | Lower | Mean Absolute Percentage Error |

### Advanced Survival Metrics

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| C-index | [0, 1] | Higher | Concordance index (discrimination) |
| C-index IPCW | [0, 1] | Higher | Censoring-robust C-index |
| IBS | [0, 1] | Lower | Integrated Brier Score |
| Calibration slope | ℝ | ~1.0 | Ideal = 1.0 (perfect calibration) |

### Robustness Metrics

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| Degradation % | [0, ∞] | Lower | Performance loss under noise |
| AUC-D | [0, ∞] | Lower | Area under degradation curve |

---

## Dependencies

Core:
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

Survival analysis:
- `lifelines`
- `scikit-survival`

Visualization:
- `matplotlib`
- `seaborn`

---

## Paper Integration

### Typical Paper Sections

**Methods → Evaluation:**
- "We evaluate models using MAE, RMSE, and C-index"
- "Robustness tested with Gaussian noise (σ=0.01-0.10), dropout (10-30%), temporal delay (1-5 steps)"
- "Statistical significance assessed via Kruskal-Wallis and Mann-Whitney U tests"

**Results → Tables:**
- Use `table_comparison.tex` for main results table
- Bold best values automatically highlighted

**Results → Figures:**
- Model comparison: `model_comparison.pdf`
- Robustness: `robustness_degradation.pdf`
- Trade-offs: `pareto_MAE_c_index.pdf`

### Citation Template

```
We evaluate model performance using a comprehensive suite of metrics
including mean absolute error (MAE), concordance index (C-index), and
robustness under perturbations. Statistical significance was assessed
using non-parametric Kruskal-Wallis tests (α=0.05).
```

---

## Status

✅ **All 6 components complete and tested**

| Component | Lines | Status |
|-----------|-------|--------|
| Unified Evaluator | 530 | ✅ |
| Robustness | 337 | ✅ |
| Integration | 442 | ✅ |
| Advanced Metrics | 380 | ✅ |
| Aggregation | 520 | ✅ |
| Visualizations | 410 | ✅ |
| **Total** | **2,619** | **✅** |
