# HSP v3 — Documentation

Basin Access Probability ($S_t$) as a geometric early warning signal for critical transitions.

---

## Documents

### 1. [HSP — Hidden Survival Paths](./HSP%20—%20Hidden%20Survival%20Paths.md)
**The origin.** Original idea document — optionality futura, conceito fundacional, outline de paper, núcleo matemático.
Mantido como registro histórico. A formulação evoluiu (v1 → v2 → v3), mas a intuição central permanece.

### 2. [HSP Basin Access](./HSP_BASIN_ACCESS.md)
**The current spec (v3).** Formal definition of $S_t = P[\Phi^H(x_t + \varepsilon) \in B(p_{t+H})]$, what died (optionality, entropy, GNN), what survived, Proposition 1 (monotonicity), all empirical results from NB 09–11, architecture v3, limitations, publication path.

### 3. [Patrick Handoff](./PATRICK_HANDOFF.md)
**Onboarding doc.** What changed (TTF → geometric EWS), new baselines (variance, AC1, skewness, DFA, Basin Stability), new datasets (Scheffer, Dakos, Stommel, May), sprint plan, math context.

### 4. [Roadmap](./ROADMAP.md)
**Plano até o arXiv.** 6 sprints (Feb–May 2026), tables/figures do paper, divisão Vini/Patrick, timeline visual.

---

## Architecture v3

```
Encoder → Perturbation → Rollout → Survival Check → S_t → Detector
```

- **Synthetic systems:** Encoder = identity, dynamics = analytical (known ODE)
- **Real data:** Encoder = LSTM/Transformer (learned), dynamics = MLP/NeuralODE (learned)
- **Only 1 learned component** for synthetic (none needed), **2 for real data**

---

## Field

**Early Warning Signals for Critical Transitions**
- Scheffer et al. (2009) Nature — ~4000 citations
- Dakos et al. (2008) PNAS — ~1500 citations
- Menck et al. (2013) Nature Physics — ~800 citations

$S_t$ fills a gap: first metric that is **both geometric** (like Basin Stability) **and dynamic** (like classical EWS).

---

## Deleted Documents (v2, superseded)

| Document | Reason |
|----------|--------|
| `EXECUTIVE_SUMMARY.md` | Pure v2 — GAT, RUL, C-MAPSS, optionality pipeline |
| `PLANO_IMPLEMENTACAO_HSP.md` | Pure v2 — 1537 lines of dead architecture |
| `BASELINES_STATUS.md` | Old baselines (LSTM/Transformer TTF) no longer relevant |

---

**Version:** 3.0 | **Last updated:** February 2026
