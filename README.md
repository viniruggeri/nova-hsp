# Nova-HSP: Basin Access Probability

Geometric early warning for critical transitions via basin accessibility estimation.

## Overview

Nova-HSP estimates Basin Access Probability
$S_t = P[\Phi^H(x_t + \varepsilon) \in B(p_{t+H})]$.

At each time step, the method perturbs the current state, rolls trajectories forward,
and computes the fraction that remain in the desired basin. Persistent drops in $S_t$
signal an approaching critical transition.

Why this is useful:

- Geometric signal (tracks basin contraction directly)
- Dynamic signal (time-resolved, not only a static basin snapshot)
- Unsupervised alerting (z-score threshold + persistence rule)

## Current Reproducible Results

Sprint 1 synthetic and Table 3 benchmark are fully scriptable from CLI.

### 1. Run synthetic S_t benchmark (Table 1 source)

```bash
python -m src.experiments.run_hsp_synthetic \
  --seeds 10 \
  --sigmas 0.03 0.05 0.10 \
  --n-rollouts 300 \
  --horizon 80 \
  --step 5 \
  --out-dir results/synthetic
```

Generates:

- `results/synthetic/run_level_metrics.csv`
- `results/synthetic/sigma_sweep_summary.csv`
- `results/synthetic/table1_summary.csv`

### 2. Run unified benchmark S_t vs baselines (Table 3 source)

```bash
python -m src.experiments.run_table3_benchmark \
  --seeds 10 \
  --sigmas 0.03 0.05 0.10 \
  --n-rollouts 300 \
  --horizon 80 \
  --step 5 \
  --out-dir results/synthetic
```

Generates:

- `results/synthetic/benchmark_run_level.csv`
- `results/synthetic/benchmark_summary.csv`
- `results/synthetic/table3_main_comparison.csv`

### 3. Export paper-ready table artifacts (Markdown and LaTeX)

```bash
python -m src.experiments.export_paper_tables \
  --synthetic-dir results/synthetic \
  --out-dir results/paper
```

Generates:

- `results/paper/table1.md`
- `results/paper/table1.tex`
- `results/paper/table3.md`
- `results/paper/table3.tex`
- `results/paper/README.md`

## Repository Structure (Current)

```text
nova-hsp/
  src/
    baseline/
      deep/
      heuristics/
      state/
      structural/
      survival/
    experiments/
      run_hsp_synthetic.py
      run_table3_benchmark.py
      export_paper_tables.py
    hsp/
      collapse.py
      encoding.py
      explanation.py
      graph.py
      metrics.py
      optionality.py
      sampling.py
      viability.py
    visualization/
    worlds/
  docs/
    HSP_BASIN_ACCESS.md
    PATRICK_HANDOFF.md
    ROADMAP.md
    PUBLICATION_STATUS.md
  results/
    synthetic/
    paper/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional dependency for Neural ODE experiments:

```bash
pip install torchdiffeq
```

## Documentation

- [docs/HSP_BASIN_ACCESS.md](./docs/HSP_BASIN_ACCESS.md): formal method spec and theory
- [docs/PATRICK_HANDOFF.md](./docs/PATRICK_HANDOFF.md): baselines and dataset handoff
- [docs/ROADMAP.md](./docs/ROADMAP.md): sprint roadmap to arXiv
- [docs/PUBLICATION_STATUS.md](./docs/PUBLICATION_STATUS.md): up-to-date publication readiness

## Citation

```bibtex
@misc{nova-hsp-2026,
  title={Basin Access Probability: A Geometric Early Warning Signal for Critical Transitions},
  author={Ruggeri, Vinicius and Mansour, Patrick},
  year={2026},
  note={In preparation}
}
```

## License

See [LICENSE](./LICENSE).
