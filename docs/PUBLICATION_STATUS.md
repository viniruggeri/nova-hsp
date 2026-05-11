# Publication Status (2026-04-12)

This file tracks what is already publication-ready and what is still required.

## Completed Today

- Sprint 1 synthetic pipeline is reproducible by script.
- Unified benchmark (S_t vs B1-B5 + Cox PH) is reproducible by script.
- Effect size reporting (Cliff's delta + magnitude) is integrated.
- Final synthetic CSV outputs for Table 1 and Table 3 are generated in `results/synthetic/`.

## Ready Artifacts

- `results/synthetic/table1_summary.csv`
- `results/synthetic/table3_main_comparison.csv`
- `results/synthetic/benchmark_summary.csv`
- `results/synthetic/benchmark_run_level.csv`
- `results/synthetic/run_level_metrics.csv`
- `results/synthetic/sigma_sweep_summary.csv`

## Remaining To Be Fully Publishable

- Sprint 2 neural validation (learned S_t) with 3 structural criteria:
  - Spearman agreement vs analytical S_t
  - monotonicity threshold
  - lead-time concordance
- Table 2, Table 4, and Table 5 generation in final paper format.
- Figures 1-5 in publication quality (SVG/PDF) and manuscript text finalization.
- Final reproducibility pass (fresh environment rerun + lockfile/frozen env capture).

## Suggested Immediate Next Steps

1. Run neural Sprint 2 and write outputs to `results/neural/`.
2. Add one script to generate Tables 2/4/5 from those outputs.
3. Export all final tables and figures to a single `results/paper/` bundle.
4. Draft `paper/main.tex` skeleton with sections and table imports.
