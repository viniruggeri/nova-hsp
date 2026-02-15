# Roadmap: HSP → arXiv

**Objetivo:** Publicar preprint no arXiv demonstrando Basin Access Probability ($S_t$)
como early warning signal geométrico para transições críticas.

**Início:** 14 Feb 2026
**Submissão arXiv:** ~May 2026
**Categoria:** `nlin.CD` (Nonlinear Sciences — Chaotic Dynamics) ou `cs.LG` (Machine Learning)

---

## Sprint 1 — Fundação Experimental
**Feb 15 – Mar 1 (2 semanas)**

O que temos: $S_t$ validado em notebooks (NB 09-11). O que falta: experimentos
reproduzíveis com seeds, salvando resultados estruturados.

### Definições Operacionais (fixar ANTES dos 90 runs)

**Alert threshold.** Alerta em $t^*$ quando $S_t$ cruza um limiar adaptativo:
$$t^* = \min\{t : S_t < \mu_0 - k\,\sigma_0\}$$
onde $\mu_0, \sigma_0$ são média e desvio de $S_t$ nos primeiros 20% da série (baseline),
e $k = 2$ (z-score). Justificativa: não requer tuning de threshold absoluto (como $\delta = 0.5$),
é calibrado pela própria série, e é imune a diferenças de escala entre sistemas.

**Lead time.** Distância temporal entre o alerta e a bifurcação:
$$L = t_{\text{bif}} - t^*$$
normalizado pela duração total da série: $L_{\text{norm}} = L / T$.
Se o alerta nunca dispara → $L = 0$ (miss). Se dispara após a bifurcação → $L = 0$ (late).
Isso garante métrica imune a tuning oportunista: o threshold é fixo ($k = 2$),
a normalização remove dependência da escala temporal.

**Persistência.** Para evitar alertas espúrios por ruído MC, exigir $S_t < \mu_0 - k\sigma_0$
por $K = 3$ passos consecutivos antes de declarar alerta.

### Vini
- [ ] `src/experiments/run_hsp_synthetic.py` — roda $S_t$ analítico nos 3 sistemas (SN, ECO, DW)
- [ ] 10 seeds × 3 sistemas × 3 valores de σ = 90 runs
- [ ] Salvar resultados em `results/synthetic/` (CSV: seed, sistema, σ, ρ, mono%, lead_time)
- [ ] Gerar **Table 1** do paper: ρ(S_t, W_t), monotonicity %, lead time por sistema

### Patrick
- [ ] Implementar 5 baselines EWS: Rolling Variance, AC1, Skewness, DFA, Basin Stability
- [ ] Interface: `BaselineResult(name, alert_times, scores)` (definida no handoff)
- [ ] Rodar nos mesmos 3 sistemas, mesmos seeds

### Checkpoint
- [ ] Table 1 completa
- [ ] 5 baselines rodando e salvando resultados

---

## Sprint 2 — Neural Basin Access
**Mar 1 – Mar 22 (3 semanas)**

O ponto central do paper: $S_t$ funciona quando a dinâmica é **aprendida**, não apenas analítica.

### Vini
- [ ] Treinar `LSTMEncoder` + `LatentDynamicsMLP` no SIR Graph
  - Input: observações parciais (só infected count, não o grafo completo)
  - Output: $z_t$ latente → rollout → $S_t^{\text{learned}}$
- [ ] **Testes de validação (3 critérios):**
  1. Spearman ρ($S_t^{\text{learned}}$, $S_t^{\text{analytical}}$) > 0.85
  2. Monotonicity % do learned ≥ 75% (preserva ordering temporal)
  3. Concordância de lead time: $|L_{\text{learned}} - L_{\text{analytical}}| < 0.05 T$ (alerta no mesmo momento)
- [ ] Repetir no Ant Colony (2º sistema)
- [ ] 3 ablations:
  - (a) sem encoder (identity) — observações cruas funcionam?
  - (b) sem dynamics (identity) — encoding puro basta?
  - (c) sem perturbação (1 rollout) — MC é necessário?
- [ ] Gerar **Table 2:** S_t learned vs analytical vs baselines

### Patrick
- [ ] Integrar 1 dataset clássico de EWS (Scheffer lake ou Dakos paleoclimate)
- [ ] Formatar como `BaseWorld` interface
- [ ] Rodar baselines EWS nesse dataset

### Checkpoint
- [ ] Neural $S_t$ funciona em ≥ 2 sistemas
- [ ] 3 critérios atendidos: ρ > 0.85, mono% ≥ 75%, concordância de lead time < 0.05T

### ⚠️ Risco
Se Neural $S_t$ não atingir os 3 critérios:
- Relaxar primeiro: mono% ≥ 60% + ρ > 0.75 ainda é publicável com caveats
- Se nem isso: paper funciona como contribuição teórica + empírica (sem DL)
- Foco muda: "aqui está a métrica + prova + experimentos analíticos"
- Ainda é publicável — Basin Stability (Menck 2013, Nature Physics) não tinha DL

---

## Sprint 3 — Benchmark Completo
**Mar 22 – Apr 5 (2 semanas)**

### Vini
- [ ] Experimento final: $S_t$ (analítico + learned) vs 5 baselines vs Cox PH
- [ ] 10 seeds por configuração
- [ ] Wilcoxon signed-rank test para lead time (p < 0.05)
- [ ] **Effect size:** Cliff's delta para cada par ($S_t$ vs baseline)
  - $|\delta| > 0.474$: large effect → diferença prática, não só estatística
  - Reportar na Table 3 junto com p-value (reviewer vai perguntar)
- [ ] Sensitivity analysis com configs prontas:
  - σ sweep: [0.01, 0.03, 0.05, 0.10, 0.20]
  - H sweep: [20, 40, 60, 80, 100]
  - N sweep: [50, 100, 300, 500, 1000]
- [ ] Gerar **Table 3** (main comparison), **Table 4** (ablation), **Table 5** (sensitivity)

### Patrick
- [ ] Rodar baselines nos datasets externos
- [ ] Contribuir dados pra Table 3

### Checkpoint
- [ ] Todas as tables prontas
- [ ] Significância estatística confirmada (p < 0.05)
- [ ] Effect size confirmado (Cliff's δ large em ≥ 3/5 baselines)

---

## Sprint 4 — Figures + Escrita
**Apr 5 – Apr 26 (3 semanas)**

### Figures
- [ ] **Fig 1:** Conceito — basin contraction visual (estado antes vs depois da bifurcação, perturbações caindo fora da basin)
- [ ] **Fig 2:** $S_t$ vs $W_t$ vs baselines ao longo do tempo (3 painéis, 1 por sistema)
- [ ] **Fig 3:** Neural $S_t$ vs analytical $S_t$ (scatter plot + correlação)
- [ ] **Fig 4:** Ablation heatmap (encoder × dynamics × perturbation)
- [ ] **Fig 5:** Sensitivity ($S_t$ vs σ, $S_t$ vs H, $S_t$ vs N)

### Escrita
- [ ] **Section 1 — Introduction:** Limitações de EWS estatísticos, gap geométrico, contribuição
- [ ] **Section 2 — Related Work:** Scheffer/Dakos (EWS), Menck (Basin Stability), Kuehn (math), neural EWS recentes
- [ ] **Section 3 — Method:** Definição formal de $S_t$, Proposição 1, arquitetura (analítica + neural)
- [ ] **Section 4 — Experimental Setup:** Sistemas, baselines, métricas, protocolo

### Checkpoint
- [ ] 5 figures prontas (PDF/SVG)
- [ ] Sections 1-4 em draft

---

## Sprint 5 — Paper Final
**Apr 26 – May 10 (2 semanas)**

- [ ] **Section 5 — Results:** Tables 1-5, interpretação
- [ ] **Section 6 — Discussion:** Limitações, quando $S_t$ falha (DW/A3 violada), implicações
- [ ] **Section 7 — Conclusion:** Contribuição, trabalho futuro (real-time, high-dimensional)
- [ ] **Abstract** (escrever por último)
- [ ] **Appendix:** Prova completa da Proposição 1, hyperparameters, detalhes dos datasets
- [ ] 3 passes de revisão interna
- [ ] Formatar em LaTeX (template arXiv padrão, ou NeurIPS se quiser submeter depois)

### Checkpoint
- [ ] Paper completo, revisado internamente

---

## Sprint 6 — Submissão
**May 10 – May 15**

- [ ] Conseguir endorsement arXiv (`nlin.CD` ou `cs.LG`)
- [ ] Upload do paper + código no arXiv
- [ ] Tornar repo GitHub público (ou link no paper)
- [ ] Postar no Twitter/LinkedIn

---

## Estrutura do Paper

```
Title: Basin Access Probability: A Geometric Early Warning Signal
       for Critical Transitions

Abstract                           (~200 palavras)
1. Introduction                    (~1.5 páginas)
2. Related Work                    (~1 página)
3. Method                          (~2 páginas)
   3.1 Basin Access Probability
   3.2 Proposition 1 (Monotonicity)
   3.3 Neural Basin Access
4. Experimental Setup              (~1 página)
   4.1 Dynamical Systems
   4.2 Baselines
   4.3 Metrics
5. Results                         (~2 páginas)
   5.1 Analytical S_t (Table 1)
   5.2 Neural S_t (Table 2)
   5.3 Comparison (Table 3)
   5.4 Ablation (Table 4)
   5.5 Sensitivity (Table 5)
6. Discussion                      (~1 página)
7. Conclusion                      (~0.5 página)
Appendix A: Proof of Proposition 1
Appendix B: Hyperparameters
Appendix C: Dataset Details

Total: ~10 páginas + appendix
```

---

## Tables do Paper

| Table | Conteúdo | Sprint |
|-------|----------|--------|
| **1** | $S_t$ analítico: ρ(S,W), mono%, lead time × 3 sistemas | 1 |
| **2** | $S_t$ learned vs analytical: ρ, mono%, concordância de lead time | 2 |
| **3** | Main comparison: $S_t$ vs 5 EWS vs Cox PH (lead time + separability + Cliff's δ) | 3 |
| **4** | Ablation: encoder × dynamics × perturbation | 3 |
| **5** | Sensitivity: σ, H, N sweeps | 3 |

## Figures do Paper

| Figure | Conteúdo | Sprint |
|--------|----------|--------|
| **1** | Conceito visual: basin contraction + perturbações | 4 |
| **2** | Time series: $S_t$ vs $W_t$ vs baselines (3 sistemas) | 4 |
| **3** | Neural vs analytical $S_t$ (scatter) | 4 |
| **4** | Ablation heatmap | 4 |
| **5** | Sensitivity analysis | 4 |

---

## Divisão de Trabalho

| Quem | Responsabilidade |
|------|-----------------|
| **Vini** | $S_t$ (analítico + neural), experimentos, ablation, figures, escrita |
| **Patrick** | 5 baselines EWS, 1 dataset externo, contribuição pra Table 3, revisão math |

---

## Timeline Visual

```
        Feb              Mar              Apr              May
  14----28  1-----------22  22----------5  5-----------26  10--15
  |  S1  |  |    S2     |  |    S3     |  |    S4     |  |S5|S6|
  |Found.|  |Neural S_t |  | Benchmark |  |Fig+Write |  |Final|
  |Table1|  |  Table 2  |  |Tables 3-5 |  |Figs 1-5  |  |arXiv|
  |Basel.|  |  Ablation  |  |Stat tests |  |Secs 1-4  |  |  ↑  |
```

---

*Última atualização: 14 Feb 2026 (rev 2 — definições operacionais, critérios estruturais, effect size)*
