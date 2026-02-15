# Nova-HSP: Basin Access Probability

**Geometric early warning for critical transitions via basin accessibility estimation**

> "Classical EWS detect *statistical* signatures of approaching bifurcation.  
> $S_t$ measures whether the basin is still *geometrically accessible*."

---

## Overview

HSP estimates **Basin Access Probability** $S_t = P[\Phi^H(x_t + \varepsilon) \in B(p_{t+H})]$ — the fraction of perturbations around the current state that remain within the basin of attraction after $H$ rollout steps.

When $S_t$ drops persistently below a threshold $\delta$ for $K$ consecutive steps, the system is approaching a critical transition.

**What makes it different:**
- **Geometric** (like Basin Stability) — measures actual basin contraction, not statistical proxies
- **Dynamic** (like classical EWS) — tracks evolution over time, not a single snapshot
- **No labels needed** — unsupervised detection via persistence rule

---

## Architecture v3

```
Encoder  →  Perturbation  →  Rollout  →  Survival Check  →  S_t  →  Detector
```

| Component | Synthetic | Real Data |
|-----------|-----------|-----------|
| **Encoder** | Identity (pass-through) | LSTM / Transformer (learned) |
| **Dynamics** | Analytical ODE (known) | MLP / ResidualNet / NeuralODE (learned) |
| **Perturbation** | Gaussian / Uniform Ball | Same |
| **Detector** | K-consecutive rule | Same |

**Key:** Only 1 learned component for synthetic (none), 2 for real data (encoder + dynamics).

---

## Metrics

| Metric | What it measures | Expected |
|--------|-----------------|----------|
| Basin Contraction Correlation $\rho(S_t, W_t)$ | Does $S_t$ track basin width? | $> 0.80$ |
| Monotonicity Fraction | $P[S_{t+1} \leq S_t]$ pre-collapse | $> 0.80$ (SN/ECO) |
| Lead Time | How early does $S_t < \delta$? | $> 0.15$ normalized |
| Separability AUC | Pre- vs post-collapse classification | $> 0.85$ |
| Partial Correlation $\rho(S_t, \tau \mid \text{EWS})$ | Info beyond classical EWS | $> 0.10$ |

---

## Project Structure

```
nova-hsp/
├── src/
│   ├── hsp/                # HSP v3 framework
│   │   ├── basin_access.py  # Core S_t computation (analytical + learned)
│   │   ├── encoder.py       # Identity / LSTM / Transformer
│   │   ├── dynamics.py      # Identity / MLP / Residual / NeuralODE
│   │   ├── perturbation.py  # Gaussian / Uniform Ball
│   │   ├── detector.py      # K-consecutive collapse detection
│   │   └── metrics.py       # Geometric evaluation metrics
│   ├── baseline/            # Comparison models (9 baselines)
│   ├── evaluation/          # Evaluation infrastructure
│   ├── experiments/         # Dataset generation + training
│   ├── utils/               # Device, logging
│   ├── visualization/       # Paper-ready plots
│   └── worlds/              # SIR graph, ant colony
├── configs/                 # Hydra configs (hsp/, baselines/, worlds/, metrics/)
├── notebooks/               # NB 01–11 (validation chain)
├── docs/                    # HSP_BASIN_ACCESS.md, PATRICK_HANDOFF.md
├── data/                    # Raw + processed
├── results/                 # Experiment outputs
└── tests/                   # Test suite
```

---

## Quick Start

```bash
git clone https://github.com/viniruggeri/nova-hsp.git
cd nova-hsp

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Optional: Neural ODE support
pip install torchdiffeq

# Run tests
pytest tests/ -v
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [HSP Basin Access (v3)](./docs/HSP_BASIN_ACCESS.md) | Formal spec — $S_t$ definition, Proposition 1, empirical results |
| [Patrick Handoff](./docs/PATRICK_HANDOFF.md) | Onboarding — what changed, new baselines/datasets, sprint plan |
| [Roadmap](./docs/ROADMAP.md) | 6 sprints Feb–May 2026 → arXiv submission |
| [HSP Original (v1)](./docs/HSP%20—%20Hidden%20Survival%20Paths.md) | Founding idea — optionality futura, núcleo matemático |
| [Docs Index](./docs/README.md) | Navigation guide |

---

## Team

| Role | Scope |
|------|-------|
| **Vini** | HSP framework — $S_t$ computation, encoder/dynamics, experiments, benchmarks |
| **Patrick** | Baselines (variance, AC1, skewness, DFA, Basin Stability) + datasets (Scheffer, Dakos, Stommel, May) |

---

## Field

**Early Warning Signals for Critical Transitions**

$S_t$ sits at the intersection of two research lineages:
- **Statistical EWS** (Scheffer 2009, Dakos 2008) — detect approaching bifurcation via variance/AC1
- **Basin Stability** (Menck 2013) — measure basin volume via Monte Carlo sampling

$S_t$ is the first metric that is **both geometric and dynamic**.

---

## Citation

```bibtex
@misc{nova-hsp-2026,
  title={Basin Access Probability: A Geometric Early Warning Signal
         for Critical Transitions},
  author={Ruggeri, Vinicius and Mansour, Patrick},
  year={2026},
  note={In preparation}
}
```

---

## License

See [LICENSE](./LICENSE) file.
