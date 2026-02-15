# HSP — Handoff: Baselines & Datasets

**Para:** Patrick  
**De:** Vini  
**Data:** Fevereiro 2026  
**Repo:** `viniruggeri/nova-hsp` → branch `feature/baseline-sprint1`

---

## TL;DR

O HSP mudou. Não é mais um pipeline de DL pra prever time-to-failure.  
É um **estimador geométrico de contração da bacia de atração**.

Isso muda tudo: as métricas, os baselines, e os datasets.  
Esse documento explica o que mudou, por que mudou, e o que você precisa fazer.

---

## 1. O que aconteceu (resumo rápido)

Nos últimos notebooks (NB 09–11), testamos a idéia original do HSP e três coisas morreram:

| Conceito | Veredicto | Por quê |
|:---------|:----------|:--------|
| Opcionalidade binária $\hat{O}_t$ | **MORTO** | Permutation test $p = 1.0$ — zero poder estatístico |
| Entropia dos futuros $E_t$ | **MORTO** | Direção errada pra fold bifurcations (entropia sobe quando deveria cair) |
| Pipeline GNN + Set Transformer | **MORTO** | Desnecessário — a informação é uma fração, não precisa de grafo |

O que **sobreviveu** — e foi formalizado — é a **probabilidade de acesso à bacia**:

$$S_t = \mathbb{P}\!\left[\Phi^H(x_t + \epsilon) \in \mathcal{B}(p_{t+H})\right]$$

onde:
- $\Phi^H$ é o flow map (integra a dinâmica por $H$ passos)
- $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ é uma perturbação limitada
- $\mathcal{B}(p)$ é a bacia de atração do equilíbrio desejado sob parâmetro $p$

**Interpretação:** perturba o estado atual $N$ vezes, roda $H$ passos cada, conta quantos sobrevivem. Quando $S_t \to 0$, qualquer perturbação finita expulsa o sistema da bacia — colapso é geometricamente inevitável.

Provamos (Proposição 1, NB 11): sob contração progressiva da bacia, perturbações limitadas, e evolução quasi-estática, $S_t$ é **monotonicamente não-crescente**.

A doc formal completa tá em [`docs/HSP_BASIN_ACCESS.md`](./HSP_BASIN_ACCESS.md).

---

## 2. Consequência: o framing do paper mudou

### Antes (Framing A — PHM/Prognostics)
> "HSP prevê colapso melhor que LSTM"
- Comparar via RMSE, C-index, NASA Score  
- Datasets: C-MAPSS (degradação monotônica de turbina)
- Baselines: LSTM regressor, Transformer RUL, DeepSurv

### Agora (Framing C — Geometric Early Warning)
> "$S_t$ é um estimador geométrico de estabilidade que funciona onde EWS temporais falham"  
- Comparar via correlação com bacia, lead time to basin collapse, discriminação de regime  
- Datasets: sistemas com **regime shift real** (bifurcação, tipping point, bistabilidade)  
- Baselines: EWS clássicos (variance, AC1), basin stability, hazard models

**Por que a mudança:** C-MAPSS é degradação monotônica inevitável — não tem bacia alternativa, não tem recuperação possível. $S_t$ ali vira só outra parametrização de risco. Seria como usar um microscópio e avaliá-lo como régua.

**Target venues:** Physical Review E, Chaos, J. Nonlinear Science (não PHM journals).

---

## 3. Sua parte: o que precisa ser feito

Você fica responsável por **2 entregas**: novos baselines e novos datasets.

### 3.1 Baselines

Os 9 baselines antigos (LSTM, Cox PH, etc.) não são mais os comparadores certos. A tabela de comparação agora é **estrutura contra estrutura** — métodos que estimam estabilidade, não que prevêem TTF.

#### Baselines necessários

| # | Baseline | O que faz | Referência | Prioridade |
|:--|:---------|:----------|:-----------|:-----------|
| B1 | **Rolling Variance** | $\text{Var}(x)$ em janela deslizante — sobe perto de tipping point (critical slowing down) | Held & Kleinen 2004 | **ALTA** |
| B2 | **Rolling AC1** | Autocorrelação lag-1 em janela deslizante — sobe com CSD | Dakos et al. 2008 | **ALTA** |
| B3 | **Rolling Skewness** | Assimetria em janela — muda de sinal perto de bifurcação | Guttal & Jayaprakash 2008 | **ALTA** |
| B4 | **DFA** | Detrended Fluctuation Analysis — expoente $\alpha$ mede correlação de longo alcance | Peng et al. 1994 | MÉDIA |
| B5 | **Basin Stability** | $\text{BS}(x^*) = \text{Vol}(\mathcal{B}(x^*)) / \text{Vol}(\Omega)$ — fração do espaço de fases que converge pro equilíbrio | Menck et al. 2013 | **ALTA** |
| B6 | **Cox PH como hazard** | Já implementado — manter como baseline de "risco sem geometria" | — | MÉDIA |

> **Nota:** B1-B3 já existem parcialmente no `src/baseline/heuristics/early_warning.py`. Mas precisam ser refatorados pra: (a) retornar a série temporal completa do indicador (não só um alerta binário), e (b) rodar nos mesmos sistemas e schedules que o $S_t$.

#### O que cada baseline precisa entregar

Para cada baseline $B$, nos mesmos sistemas e mesmos parâmetros que o $S_t$:

```python
class Baseline:
    def compute_indicator(self, trajectory, p_schedule) -> (times, indicator):
        """Retorna série temporal do indicador."""
        ...
```

Output padronizado: `(times: np.ndarray, values: np.ndarray)` — mesma resolução temporal do $S_t$.

#### Interface unificada

```python
@dataclass
class BaselineResult:
    name: str                    # ex: "Rolling Variance"
    times: np.ndarray            # timestamps
    values: np.ndarray           # indicator values
    alert_time: float | None     # primeiro instante de alerta (se aplicável)
    higher_means_risk: bool      # True se valor alto = mais risco
```

Isso permite comparação direta: $\rho(B_t, \tau)$ vs $\rho(S_t, \tau)$ onde $\tau = t_\text{bif} - t$.

---

### 3.2 Datasets

#### O que NÃO serve mais

| Dataset | Por que não | Status |
|:--------|:-----------|:-------|
| C-MAPSS | Degradação monotônica, sem bacia alternativa, sem recuperação | **DROPPED** |
| SWaT | Anomaly detection, sem regime shift genuíno | **DROPPED** |
| FEMTO | Bearing degradation, monotônico | **DROPPED** |

#### O que precisamos

Sistemas com **pelo menos uma** destas propriedades:
- Bifurcação controlada (parâmetro que cruza valor crítico)
- Bistabilidade (dois atratores, transição possível entre eles)
- Tipping point com possibilidade de recuperação
- Regime shift documentado na literatura

#### Candidatos (em ordem de prioridade)

| # | Sistema | Tipo | Por que serve | Dados | Prioridade |
|:--|:--------|:-----|:-------------|:------|:-----------|
| D1 | **Modelo de lago** (Scheffer) | Eutrofização | Bistável: clear ↔ turbid. Bifurcação fold clássica. Referência canônica em regime shifts. | Modelo ODE (implementar) | **ALTA** |
| D2 | **Paleoclimate** (Dakos 2008) | Transições glaciais | 8 transições documentadas. Dados reais. Dakos mostrou que AC1/variance falham em alguns. | Público (PANGAEA) | **ALTA** |
| D3 | **Stommel thermohaline** | Circulação oceânica | Bistável: on ↔ off. Modelo 2-box clássico. | Modelo ODE (implementar) | MÉDIA |
| D4 | **May's harvesting model** | Ecologia | Fold com zona de histerese. Multistável. | Modelo ODE (implementar) | MÉDIA |
| D5 | **Power grid sync** | Eng. elétrica | Stable sync ↔ cascading failure. Bacia real. | Modelo (Kuramoto) | BAIXA |

#### Para modelos ODE (D1, D3, D4): o que entregar

Para cada modelo, precisamos de uma implementação que siga o padrão dos sistemas existentes:

```python
def lake_eutrophication(T: int = 500, seed: int = 42) -> tuple:
    """
    Modelo de Scheffer para eutrofização.
    
    dx/dt = a - bx + r * x^p / (x^p + 1)   (loading - decay + recycling)
    
    Parâmetro de stress: 'a' (nutrient loading) cresce monotonicamente.
    Bifurcação fold em a_crit.
    
    Returns:
        trajectory: np.ndarray (T,) — série temporal de x
        p_schedule: np.ndarray (T,) — valores de 'a' ao longo do tempo
        bif_time: int — instante da bifurcação
    """
    ...
```

**Requisitos:**
1. Mesma assinatura: `(T, seed) -> (trajectory, p_schedule, bif_time)`
2. Parâmetro de stress varia monotonicamente
3. Bifurcação é documentada (valor crítico analítico ou numérico)
4. Ruído de processo incluído ($\sigma_\text{process}$ pequeno)
5. Função `basin_width(p)` que retorna a largura analítica/numérica da bacia pra cada $p$

Exemplo de referência: olhar no NB 11 (`notebooks/11_monotonicity_proof.ipynb`), cell 3 — tem `saddle_node()`, `double_well()`, `ecosystem()` já implementados.

#### Para dados reais (D2): o que entregar

1. Download + parsing dos dados de Dakos et al. 2008 (paleoclimate transitions)
2. Identificação das transições (timestamps)
3. Série temporal pré-processada: `(times, x_values, transition_times)`
4. Referência bibliográfica

Os dados estão em: [PANGAEA](https://doi.pangaea.de/) — buscar Dakos 2008 supplementary.  
Paper: Dakos, V. et al. (2008). "Slowing down as an early warning signal for abrupt climate change." *PNAS*, 105(38), 14308-14312.

---

## 4. Métricas novas (pra você saber o que vamos medir)

Quando eu rodar os benchmarks HSP vs teus baselines, vou usar estas métricas:

| Métrica | Definição | O que testa |
|:--------|:----------|:-----------|
| **Basin Contraction Correlation** | $\rho(B_t, W_t)$ — Spearman entre indicador e largura real da bacia | O indicador rastreia geometria? |
| **Lead Time to Basin Collapse** | $t_\text{alert} - t_\text{collapse}$ normalizado | Quão cedo alerta? |
| **Recovery Detectability** | AUC para classificar estados recuperáveis vs irreversíveis | Distingue "ainda dá" de "acabou"? |
| **Separability Score** | AUC para classificar estados pré vs pós perda de bacia | Separação clean? |
| **Partial Correlation** | $\rho_\text{partial}(B_t, \tau \mid \text{outros indicadores})$ | Informação incremental? |

Pra todas essas, preciso que teus baselines entreguem a série temporal completa — não só um alerta binário.

---

## 5. Math corner (pra te dar contexto do que provamos)

Como você gosta de matemática, aqui vai o core formal:

### Proposição 1 (Monotonicidade de $S_t$)

Seja $\dot{x} = f(x, p(t))$ com $p$ monotônico. Defina:

$$A(t) = \left\{\epsilon \in \mathbb{R}^n : \|\epsilon\| \leq \bar{\epsilon}, \; \Phi^H(x_t + \epsilon) \in \mathcal{B}(p_{t+H})\right\}$$

o conjunto de perturbações cujo flow cai na bacia futura. Então $S_t = \mu(A(t))$ onde $\mu$ é a medida induzida pela distribuição de perturbação.

**Sob:**
1. $\mathcal{B}(p_1) \supseteq \mathcal{B}(p_2)$ para $p_1 < p_2$ *(nested basins)*
2. $\|\epsilon\| \leq \bar{\epsilon}$ *(bounded)*
3. $\dot{p} / \lambda \ll 1$ onde $\lambda$ é a taxa de relaxação *(quasi-static)*
4. $x_t \in \mathcal{B}(p(t))$ *(trajectory in basin)*  
5. $\Phi^H$ contínua *(smooth flow)*

**Temos:** $t_1 < t_2 \implies S_{t_1} \geq S_{t_2}$

A prova usa que bacias nested + bacia-alvo menor → o conjunto de perturbações viáveis encolhe → a medida é não-crescente.

**O que verificamos numericamente (NB 11):**
- SN: monotonicidade 80.1%, $\rho(S, W) = +0.917$
- ECO: monotonicidade 84.2%, $\rho(S, W) = +0.890$
- DW: monotonicidade 54.2% (A4 falha — trajetória sai da bacia 27% do tempo)

As violações de monotonicidade são:  
(a) ruído MC (diminuem com $N$),  
(b) concentradas perto de $\tau \to 0$ (onde quasi-static quebra),  
(c) maiores perto da bifurcação ($\rho(\tau, |\text{viol}|) = -0.65$).

O DW serve como **controle negativo**: quando as premissas falham, $S_t$ falha — o que valida a estrutura lógica da proposição.

### O resultado matador (NB 10.1)

EWS clássicos (variance, AC1, skewness) têm **$\rho(\tau) \approx 0$** nesses sistemas.  
$S_t$ tem $\rho_\text{partial}(S, \tau \mid \text{todos EWS}) > +0.4$.

Isso quer dizer: EWS simplesmente **não funcionam** como early warning aqui. $S_t$ não é "um EWS melhor" — é uma **classe diferente** de métrica (geométrica vs temporal).

Se você conseguir implementar os baselines e nós replicarmos isso em dados reais (Dakos), é o paper.

---

## 6. Estrutura de arquivos

```
src/baseline/
├── heuristics/
│   ├── early_warning.py      ← REFATORAR (B1-B3: var, AC1, skew como séries)
│   └── linear_threshold.py   ← manter como está
├── structural/                ← NOVO diretório
│   ├── __init__.py
│   ├── dfa.py                ← B4: Detrended Fluctuation Analysis
│   └── basin_stability.py    ← B5: Basin Stability (Menck et al.)
├── survival/
│   └── cox_ph.py             ← B6: manter como hazard baseline
└── ...

src/worlds/                    ← NOVOS sistemas
├── base.py
├── sir_graph.py
├── ant_colony.py
├── lake.py                   ← D1: Scheffer eutrophication
├── thermohaline.py           ← D3: Stommel 2-box
└── may_harvest.py            ← D4: May's model

data/
├── raw/
│   └── dakos2008/            ← D2: paleoclimate transitions
└── processed/
```

---

## 7. Checklist de entrega

### Sprint 1 (1-2 semanas): Baselines

- [ ] Refatorar `early_warning.py` → retornar séries temporais (var, AC1, skew) não só alerta
- [ ] Implementar `dfa.py` — expoente $\alpha$ em janela deslizante
- [ ] Implementar `basin_stability.py` — BS via Monte Carlo (amostra $N$ condições iniciais, conta convergência)
- [ ] Interface `BaselineResult` unificada
- [ ] Testar todos nos 3 sistemas existentes (SN, DW, ECO)

### Sprint 2 (2-3 semanas): Datasets

- [ ] `lake.py` — modelo de Scheffer com bifurcação fold
- [ ] `lake.py` → `basin_width(p)` analítica/numérica
- [ ] Download + parsing Dakos 2008 (paleoclimate)
- [ ] Pelo menos 1 dos opcionais (Stommel ou May)
- [ ] Testes unitários pra cada sistema novo

### Validação

Quando terminar, eu rodo o benchmark: $S_t$ vs todos os teus baselines nos teus datasets, com as métricas novas. Se $\rho(S, W) \gg \rho(B, W)$ e lead time positivo → paper.

---

## 8. Referências-chave

Lê pelo menos os abstracts (os com ★ lê o paper inteiro):

1. ★ **Scheffer et al. (2009)** — "Early-warning signals for critical transitions." *Nature* 461, 53-59. → O paper canônico de EWS. Nosso benchmark teórico.

2. ★ **Dakos et al. (2008)** — "Slowing down as an early warning signal for abrupt climate change." *PNAS* 105(38). → Dados paleoclimate. Mostra limites de AC1/variance.

3. **Menck et al. (2013)** — "How basin stability complements the linear-stability paradigm." *Nature Physics* 9, 89-92. → Define Basin Stability. Nosso baseline B5.

4. **Held & Kleinen (2004)** — "Detection of climate system bifurcations by degenerate fingerprinting." *GRL* 31. → Variance como EWS.

5. **Scheffer et al. (2001)** — "Catastrophic shifts in ecosystems." *Nature* 413. → Modelo de lago. Nosso dataset D1.

6. ★ **Nosso doc formal:** [`docs/HSP_BASIN_ACCESS.md`](./HSP_BASIN_ACCESS.md) — Toda a matemática, resultados, e justificativas.

---

## 9. Como rodar o que já existe

```bash
# Setup
git clone https://github.com/viniruggeri/nova-hsp.git
cd nova-hsp
git checkout feature/baseline-sprint1
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Testes existentes
pytest tests/ -v  # 14/14 devem passar

# Ver os notebooks (precisa de Jupyter)
# NB 11 tem toda a prova + gráficos
jupyter notebook notebooks/11_monotonicity_proof.ipynb
```

---

## Dúvidas

Me chama. A doc [`HSP_BASIN_ACCESS.md`](./HSP_BASIN_ACCESS.md) tem tudo que precisar de contexto matemático/conceitual.

A parte mais importante: os baselines precisam retornar **séries temporais**, não alertas binários. Sem isso não consigo calcular as correlações parciais que são o core do benchmark.

Bora. 🚀
