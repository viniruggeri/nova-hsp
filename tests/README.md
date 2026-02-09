# Test Suite

This directory contains all test files for the Nova-HSP project.

## Directory Structure

```
tests/
├── evaluation/          # Evaluation pipeline tests
│   ├── test_advanced_metrics.py       # Advanced metrics validation
│   ├── test_aggregation.py            # Cross-world aggregation & export
│   ├── test_complete_pipeline.py      # End-to-end pipeline
│   ├── test_integration_minimal.py    # Minimal integration tests
│   ├── test_robustness.py            # Robustness analysis
│   └── test_unified_evaluator.py     # Unified evaluator tests
│
├── visualization/       # Visualization tests
│   └── test_paper_plots.py           # Paper-ready plot generation
│
└── baseline/           # Baseline model tests
    ├── test_deep_hazard.py
    ├── test_heuristics.py
    ├── test_state.py
    └── test_survival.py
```

## Running Tests

### Individual Component Tests

```bash
# Evaluation components
python tests/evaluation/test_advanced_metrics.py
python tests/evaluation/test_aggregation.py
python tests/visualization/test_paper_plots.py

# Complete pipeline
python tests/evaluation/test_complete_pipeline.py
```

### Baseline Model Tests

```bash
python tests/baseline/test_survival.py
python tests/baseline/test_state.py
python tests/baseline/test_heuristics.py
python tests/baseline/test_deep_hazard.py
```

### All Tests (using pytest)

```bash
pytest tests/
```

## Test Coverage

- **Evaluation Tests**: Advanced metrics (C-index IPCW, Brier, lead time), aggregation, visualization, robustness
- **Visualization Tests**: Model comparison, calibration plots, Pareto fronts, heatmaps
- **Baseline Tests**: Survival models (Cox, KM, AFT), state models (HMM, Markov), heuristics, deep learning

## Notes

- All evaluation tests use synthetic data for validation
- Visualization tests generate PDF output in current directory
- Tests are standalone and can run without full dataset
