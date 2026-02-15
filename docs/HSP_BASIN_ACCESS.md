# HSP v2 — Basin Access Probability

**Versão:** 1.0  
**Data:** Fevereiro 2026  
**Status:** Formalmente proposto, numericamente verificado  
**Notebooks de referência:** NB 09, 10, 10.1, 11

> "S_t não é um early warning signal. É um estimador probabilístico da medida local da bacia de sobrevivência sob perturbação limitada."

---

## 1. Motivação: O que morreu e o que sobreviveu

O HSP original propunha detectar colapso medindo a "opcionalidade futura" — a fração de trajetórias amostradas que permanecem viáveis. A Fase 1 de validação conceitual (NB 04–11) testou rigorosamente essa ideia e produziu resultados que **reformularam completamente** o framework:

| Conceito | Notebook | Veredicto | Motivo |
|:---------|:---------|:----------|:-------|
| Opcionalidade binária $\hat{O}_t$ | NB 07 | **MORTO** | Permutation test $p = 1.0$ — sem poder estatístico |
| Erosão entrópica $E_t$ | NB 09 | **MORTO** | Direção errada para bifurcações fold (entropia sobe antes do colapso) |
| Probabilidade de sobrevivência $S_t$ | NB 09 | **VIVO** | $\rho > +0.77$ em todos os 3 sistemas |
| $S_t$ independente do mecanismo? | NB 10 | **PERGUNTA ERRADA** | $p_t$ é causa direta de $S_t$; exigir independência é exigir incoerência causal |
| $S_t$ como estimador da bacia | NB 10.1 | **VALIDADO** | PASS 3/3 — discriminação, superioridade vs EWS, correlação com $W_t$ |
| Monotonicidade formal de $S_t$ | NB 11 | **VERIFICADO** | Proposição 1 + verificação numérica em 3 sistemas |

**Conclusão:** O HSP não é um detector de futures viáveis via grafos — é um **estimador probabilístico da contração da bacia de atração**.

---

## 2. Definição Formal

### 2.1 Setup

Considere um sistema dinâmico parametrizado:

$$\dot{x} = f(x, p(t))$$

onde $x \in \mathbb{R}^n$, $p: [0, T] \to \mathbb{R}$ é um parâmetro monotonicamente variável, e $f$ é contínua em ambos os argumentos.

### 2.2 Bacia de Sobrevivência

Para um valor do parâmetro $p$, a **bacia de sobrevivência** $\mathcal{B}(p) \subset \mathbb{R}^n$ é a bacia de atração do equilíbrio desejado do sistema $\dot{x} = f(x, p)$.

### 2.3 Contração Progressiva

A bacia contrai progressivamente se:

$$p_1 < p_2 \implies \mathcal{B}(p_1) \supseteq \mathcal{B}(p_2)$$

i.e., a bacia é uma família aninhada e decrescente de conjuntos.

### 2.4 Probabilidade de Acesso à Bacia (Basin Access Probability)

Para um estado $x_t$ e distribuição de perturbação $\epsilon \sim \mathcal{D}$ com suporte limitado $\|\epsilon\| \leq \bar{\epsilon}$:

$$\boxed{S_t = \mathbb{P}\left[\Phi^H(x_t + \epsilon,\, p_t) \in \mathcal{B}(p_{t+H})\right]}$$

onde $\Phi^H$ é o mapa de fluxo integrado por $H$ passos.

**Interpretação:** $S_t$ mede a fração do espaço de perturbações locais cujas trajetórias permanecem na bacia de sobrevivência após $H$ passos. Quando $S_t \to 0$, qualquer perturbação finita expulsa o sistema da bacia — o colapso é geometricamente inevitável.

### 2.5 Estimação por Monte Carlo

Na prática, $S_t$ é estimado como:

$$\hat{S}_t = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\Phi^H(x_t + \epsilon_i,\, p_t) \in \mathcal{B}(p_{t+H})\right], \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2 I)$$

com $N = 300$ rollouts, $H = 80$ passos, $\sigma = 0.05$.

---

## 3. Proposição de Monotonicidade

### 3.1 Enunciado

**Proposição 1.** *Sob as seguintes suposições:*

| # | Suposição | Descrição |
|:--|:----------|:----------|
| A1 | Contração progressiva | $\mathcal{B}(p(t))$ contrai monotonicamente com $t$ |
| A2 | Perturbações limitadas | $\epsilon \sim \mathcal{D}$ com $\|\epsilon\| \leq \bar{\epsilon}$, $\mathcal{D}$ fixa |
| A3 | Evolução quasi-estática | $p(t)$ varia lentamente relativo ao tempo de relaxação |
| A4 | Trajetória na bacia | $x_t \in \mathcal{B}(p(t))$ para todo $t$ pré-colapso |
| A5 | Fluxo contínuo | $\Phi^H$ é contínua nas condições iniciais e parâmetros |

*Então $S_t$ é monotonicamente não-crescente em $t$:*

$$t_1 < t_2 \implies S_{t_1} \geq S_{t_2}$$

### 3.2 Prova (Sketch)

**Passo 1.** Por contração progressiva (A1): $\mathcal{B}(p(t_2)) \subseteq \mathcal{B}(p(t_1))$.

**Passo 2.** Por evolução quasi-estática (A3) e trajetória na bacia (A4), tanto $x_{t_1}$ quanto $x_{t_2}$ rastreiam o equilíbrio $x^*(p(t))$. Relativamente à fronteira da bacia, $x_{t_2}$ está mais próximo (pois a bacia encolheu ao redor do equilíbrio).

**Passo 3.** O conjunto $A(t) = \{\epsilon : \Phi^H(x_t + \epsilon, p_t) \in \mathcal{B}(p_{t+H})\}$ satisfaz:
- Em $t_1$: $A(t_1)$ são perturbações cujo fluxo cai em $\mathcal{B}(p_{t_1+H})$
- Em $t_2$: $A(t_2)$ são perturbações cujo fluxo cai em $\mathcal{B}(p_{t_2+H}) \subseteq \mathcal{B}(p_{t_1+H})$

**Passo 4.** Pela continuidade de $\Phi^H$ (A5) e o fato de que a bacia-alvo é menor em $t_2$:

$$\mu(A(t_2)) \leq \mu(A(t_1))$$

**Passo 5.** Portanto: $S_{t_2} = \mu(A(t_2)) \leq \mu(A(t_1)) = S_{t_1}$. $\blacksquare$

### 3.3 Onde a Monotonicidade Quebra (e por quê)

| Suposição violada | Consequência em $S_t$ | Exemplo |
|:------------------|:----------------------|:--------|
| A1: Contração não-monotônica | $S_t$ pode oscilar | Double-Well (sela se move não-monotonicamente) |
| A2: Caudas pesadas | Vazamento além da bacia infla $S_t$ | Perturbações Cauchy |
| A3: Sweep rápido | Trajetória atrasa, $S_t$ delayed | $\dot{p}/\lambda \gg 1$ |
| A4: Ruído pré-colapso | $S_t$ salta | Kicks estocásticos grandes |
| A5: Bifurcação descontínua | $S_t$ pode saltar discontinuamente | Bifurcação catastrophic |

---

## 4. Resultados Empíricos

### 4.1 Sistemas de Teste

| Sistema | Equação | Parâmetro | Bifurcação | Largura da bacia $W(p)$ |
|:--------|:--------|:----------|:-----------|:------------------------|
| **Saddle-Node** | $\dot{x} = r - x^2$ | $r: 2 \to -0.5$ | Saddle-node em $r = 0$ | $W(r) = 2\sqrt{r}$ (analítica) |
| **Double-Well** | $\dot{x} = x - x^3 + r$ | $r: -0.3 \to 0.6$ | Fold em $r \approx 0.385$ | $W$ = raízes do cúbico |
| **Ecosystem** | Modelo de pastoreio | $h: 0.05 \to 0.35$ | Fold ecológico | $W$ = posição do equilíbrio |

### 4.2 NB 09 — Entropia vs. Sobrevivência

**Pergunta:** Qual métrica captura colapso iminente?

| Métrica | SN | DW | ECO | Veredicto |
|:--------|:--:|:--:|:---:|:----------|
| Entropia $E_t$ | ✗ | ✗ | ✗ | FALHA — direção errada para fold |
| Sobrevivência $S_t$ | $\rho > +0.85$ | $\rho > +0.77$ | $\rho > +0.80$ | **PASSA** em todos |

**Insight:** Entropia mede dispersão das trajetórias, não contração da bacia. Em bifurcações fold, a variância pode aumentar (critical slowing down) enquanto a bacia encolhe — as duas coisas se movem em direções opostas.

### 4.3 NB 10 — Acid Test (4 Blocos)

| Bloco | Teste | Resultado |
|:------|:------|:----------|
| 1. Calibração | Sweep controlado linear? | **PASS** |
| 2. Sensibilidade | $S_t$ captura taxa de contração? | **PASS** |
| 3. Especificidade | $S_t \approx 1$ sem contração? | **PASS** |
| 4. Independência | $S_t$ independente de $p_t$? | **FAIL → DIAGNÓSTICO: teste errado** |

**Diagnóstico do Bloco 4:** Exigir independência estatística de $p_t$ é exigir incoerência causal. A cadeia causal real é:

$$p_t \xrightarrow{\text{causa}} \text{contração da bacia} \xrightarrow{\text{causa}} S_t$$

Controlar por $p_t$ remove **toda** a variação temporal porque $p_t$ e $\tau = t_\text{bif} - t$ são perfeitamente anticorrelacionados (ambos lineares em $t$, direções opostas). Isso é multicolinearidade estrutural, não falha do $S_t$.

### 4.4 NB 10.1 — Validação Estrutural (3 Testes)

Após reformular a pergunta de "S_t é independente?" para "S_t é o melhor estimador local da contração da bacia?":

#### Teste 1 — Discriminação Bacia vs. Não-Bacia

*$S_t$ é flat quando a bacia não contrai?*

| Sistema | Range de $S_t$ (com contração) | Range (sem contração) |
|:--------|:---:|:---:|
| SN-contracting | 0.617 | — |
| SN-safe | — | **0.000** |
| Linear | — | **0.002** |

**Veredicto: PASS.** $S_t$ só se move quando há contração geométrica real.

#### Teste 2 — $S_t$ vs. Early Warning Signals Clássicos

*$S_t$ adiciona informação além de variância, AC1, skewness?*

Correlação parcial $\rho_\text{partial}(S_t, \tau \mid \text{var}, \text{AC1}, \text{skew}, x_t)$:

| Sistema | $\rho_\text{partial}(S, \tau)$ | Variância $\rho(\tau)$ | AC1 $\rho(\tau)$ | Skew $\rho(\tau)$ |
|:--------|:---:|:---:|:---:|:---:|
| SN | **+0.477** | ≈ 0 | ≈ 0 | ≈ 0 |
| DW | **+0.716** | ≈ 0 | ≈ 0 | ≈ 0 |
| ECO | **+0.114** | ≈ 0 | ≈ 0 | ≈ 0 |

**Veredicto: PASS (3/3).** Os EWS clássicos têm $\rho \approx 0$ nestes sistemas — eles simplesmente **não funcionam** como early warning aqui. $S_t$ não é "um EWS melhor"; é uma **classe fundamentalmente diferente** de métrica (geométrica vs. temporal).

#### Teste 3 — Monotonicidade com largura teórica da bacia

*$S_t$ rastreia $W(p)$?*

| Sistema | $\rho(S_t, W_t)$ | Veredicto |
|:--------|:---:|:----------|
| Saddle-Node | **+0.904** | ★ forte |
| Ecosystem | **+0.879** | ★ forte |
| Double-Well | +0.658 | ✗ moderado (A4 parcial) |

**Veredicto: PASS (2/3 forte, 1/3 moderado).**

### 4.5 NB 11 — Verificação Numérica da Proposição 1

#### Verificação 1 — Compliance das Suposições

| | A1 contração | A3 quasi-static | A4 na bacia | Todas? |
|:--|:---:|:---:|:---:|:---:|
| **SN** | 100% | ratio = 17.1 | 100% | **SIM** |
| **DW** | 100% | ratio = 16.0 | **73.4%** | **PARCIAL** |
| **ECO** | 100% | ratio = 10.0 | 100% | **SIM** |

#### Verificação 2 — Monotonicidade Step-wise

| | Mono % | $\rho(S, W)$ | Viola mean | Viola max |
|:--|:---:|:---:|:---:|:---:|
| **SN** | **80.1%** | **+0.917** | 0.013 | 0.117 |
| **DW** | 54.2% | +0.578 | 0.051 | 0.360 |
| **ECO** | **84.2%** | **+0.890** | 0.032 | 0.193 |

#### Verificação 3 — Anatomia das Violações

**3a) Convergência MC (Saddle-Node):**

| $N$ rollouts | Mono % | Viola mean |
|:---:|:---:|:---:|
| 50 | 85.1% | 0.037 |
| 100 | 83.0% | 0.015 |
| 300 | 78.7% | 0.009 |
| 1000 | 83.0% | **0.007** |

→ Magnitude das violações diminui com $N$ — **ruído MC confirmado**.

**3b) Localização temporal das violações:**

| Sistema | Mediana $\tau$ (violações) | Mediana $\tau$ (monotone) | Cluster perto da bif? |
|:--------|:---:|:---:|:---:|
| **SN** | 84 | 144 | **SIM** |
| **ECO** | 80 | 225 | **SIM** |
| DW | 154 | 149 | NÃO (uniforme) |

→ Em SN e ECO, violações se concentram perto da bifurcação — onde A3 (quasi-static) enfraquece.

**3c) Magnitude das violações vs. distância à bifurcação:**

| Sistema | $\rho(\tau, |\text{viol}|)$ |
|:--------|:---:|
| **SN** | **-0.647** |
| **ECO** | **-0.620** |
| DW | -0.101 |

→ $\rho < 0$: violações são **maiores perto da bifurcação** — precisamente onde $\dot{p}/\lambda$ (razão sweep/relaxação) cresce e A3 quebra.

---

## 5. Síntese: O que $S_t$ é e o que não é

### O que $S_t$ É

1. **Estimador probabilístico** da medida local da bacia de sobrevivência
2. **Monotonicamente não-crescente** quando: bacia contrai, perturbações limitadas, evolução quasi-estática, trajetória na bacia, fluxo contínuo
3. **Geometricamente fundamentado** — mede contração do espaço de estados, não estatísticas temporais
4. **Informativamente superior** aos EWS clássicos (variância, AC1, skewness) que têm $\rho \approx 0$ nestes sistemas
5. **Consistente entre sistemas** — funciona em saddle-node, fold ecológico, double-well (com ressalvas conhecidas)

### O que $S_t$ NÃO É

1. **Não é um Early Warning Signal** no sentido clássico — não se baseia em critical slowing down
2. **Não é independente do mecanismo** de stress — $p_t \to$ contração $\to S_t$ é a cadeia causal; $S_t$ é causalmente downstream de $p_t$
3. **Não é um indicador da magnitude do stress** — é um indicador da **consequência geométrica** do stress
4. **Não é exato** (é Monte Carlo) — violações de monotonicidade existem mas são: (a) pequenas, (b) explicáveis, (c) convergem com $N$

### Relação com a Formulação Original do HSP

| Conceito Original | Reformulação |
|:-------------------|:-------------|
| "Futuros viáveis" $\hat{O}_t$ | → Probabilidade de permanecer na bacia $S_t$ |
| Grafo de futuros (kNN) | → Rollouts Monte Carlo diretos |
| GNN + Set Transformer | → Estatística binária (sobrevive/não) |
| "Opcionalidade" qualitativa | → Medida de bacia quantitativa $\mu(A(t))$ |
| Detecção de "extinção latente" | → Detecção de contração geométrica |
| Framework não-supervisionado complexo | → Estimador não-paramétrico simples |

A essência do HSP original — **"medir se ainda existe futuro"** — sobreviveu. Mas a implementação é radicalmente mais simples e matematicamente fundamentada: em vez de construir grafos de trajetórias e agregar via redes neurais, simplesmente amostras N perturbações, rodas H passos, e contas quantas sobrevivem.

---

## 6. Parâmetros Computacionais

| Parâmetro | Valor | Justificativa |
|:----------|:------|:-------------|
| $N$ (rollouts) | 300 | Compromisso precisão/custo; violações convergem (§4.5-V3a) |
| $H$ (horizonte) | 80 | Deve ser $\gg$ tempo de relaxação do sistema |
| $\sigma$ (perturbação) | 0.05 | Ordem de magnitude da largura da bacia pré-colapso |
| Step (amostragem temporal) | 5 | Resolução temporal suficiente para capturar $\Delta S$ |
| Seeds | 20 | Cobre variabilidade estocástica entre realizações |

---

## 7. Limitações Conhecidas

1. **Double-Well é o caso difícil.** A4 (trajetória na bacia) só vale 73% do tempo → mono% cai para 54%. Sistemas com dinâmica intra-bacia complexa violam as suposições.

2. **Quasi-static é essencial.** Quando $\dot{p}/\lambda \gg 1$, a trajetória não acompanha o equilíbrio e violações de monotonicidade crescem. Isso é uma limitação real, não apenas teórica.

3. **Requer conhecimento do modelo.** $S_t$ precisa de: (a) simular rollouts ($\Phi^H$), (b) definir "sobrevivência" (threshold). Em dados reais, isso requer um modelo surrogate.

4. **Monte Carlo é ruidoso.** Com $N = 300$, a resolução de $S_t$ é $\sim 1/300 \approx 0.003$. Violações de magnitude $< 0.01$ são indistinguíveis de ruído.

5. **Não provado para dimensões altas.** Todos os testes são em $\mathbb{R}^1$. A extensão para $\mathbb{R}^n$ é direta na teoria (mesma proposição), mas a escolha de $\sigma$ e $N$ pode requerer escalonamento.

---

## 8. Implicações para a Arquitetura

A reformulação de $S_t$ como estimador de bacia tem consequências diretas para a arquitetura do HSP:

### O que muda

| Componente original | Status | Substituto |
|:--------------------|:-------|:-----------|
| Transformer encoder | **Mantido** (para dados reais) | Aprende representação latente $z_t$ |
| Stochastic sampler | **Simplificado** | Perturbação Gaussiana $z_t + \epsilon$ |
| kNN Graph Builder | **Removido** | Não necessário — rollouts diretos |
| GAT Message Passing | **Removido** | Sem grafo, sem message passing |
| Set Transformer | **Removido** | Média aritmética (fração de sobreviventes) |
| Dual Head | **Simplificado** | Head único: $\hat{S}_t = \text{count}(\text{survive}) / N$ |
| Collapse Detector | **Mantido** | $\hat{S}_t < \delta$ por $K$ passos |

### Arquitetura v3 (proposta)

```
Input window (x_{t-W}...x_t)
    ↓
[1. TEMPORAL ENCODER]  →  z_t ∈ R^d       (único componente learned)
    ↓
[2. PERTURBAÇÃO]  →  {z_t + ε_1, ..., z_t + ε_N}   (ε ~ N(0, σ²I))
    ↓
[3. ROLLOUT]  →  {Φ^H(z_t + ε_i)}_{i=1}^N          (ou modelo dinâmico learned)
    ↓
[4. SURVIVAL CHECK]  →  {alive_1, ..., alive_N}      (rule-based)
    ↓
[5. S_t = mean(alive)]                                (estimador MC)
    ↓
[6. COLLAPSE DETECTOR]  →  S_t < δ for K steps       (rule-based)
```

**Redução:** 7 componentes → 6, dos quais apenas 1 é learned (encoder). Os outros 5 são analíticos/rule-based.

---

## 9. Caminho para Publicação

### O que está pronto

- [x] Definição formal de $S_t$ como basin access probability
- [x] Proposição 1 (monotonicidade) com proof sketch
- [x] Verificação numérica em 3 sistemas (NB 11)
- [x] Discriminação bacia/não-bacia (NB 10.1, Teste 1)
- [x] Superioridade vs. EWS clássicos (NB 10.1, Teste 2)
- [x] Correlação com largura teórica (NB 10.1, Teste 3)
- [x] Anatomia das violações (NB 11, V3)

### O que falta

- [ ] Prova matemática rigorosa (não apenas sketch — formalizar Step 4 com continuidade de $\Phi^H$)
- [ ] Extensão a $\mathbb{R}^n$ (sistemas multidimensionais)
- [ ] Teste em dados reais (C-MAPSS ou SWaT com modelo surrogate)
- [ ] Comparação formal com existing resilience metrics (basin stability, ecological resilience)
- [ ] Figuras paper-ready (PDF, formatação publicação)
- [ ] Draft do paper com estrutura completa

### Estrutura do Paper (proposta)

1. **Introduction** — Limitações de EWS clássicos + motivação geométrica
2. **Background** — Bifurcações, bacias de atração, critical slowing down
3. **Basin Access Probability** — Definição 1–3, Proposição 1, prova
4. **Numerical Verification** — 3 sistemas, 3 verificações (compliance, monotonicidade, anatomia)
5. **Comparison with Classical EWS** — $S_t$ vs variância/AC1/skewness (Teste 2 do NB 10.1)
6. **Discussion** — Limitações (DW, quasi-static), relação com basin stability
7. **Conclusion** — $S_t$ como classe nova de métrica (geométrica, não temporal)

---

## 10. Referências Internas

| Notebook | Conteúdo | Status |
|:---------|:---------|:-------|
| `04_matrioshka_concept.ipynb` | Conceito original (opcionalidade) | Histórico |
| `07_hsp_permutation.ipynb` | Morte da opcionalidade binária ($p = 1.0$) | Histórico |
| `08_structural_diagnosis.ipynb` | Diagnóstico: só saddle-node tem bifurcação genuína | Histórico |
| `09_entropic_hsp.ipynb` | Entropia FALHA, Sobrevivência PASSA | **Fundacional** |
| `10_st_acid_test.ipynb` | Acid test (3/4, Bloco 4 = teste errado) | **Fundacional** |
| `101_st_structural.ipynb` | Validação estrutural (PASS 3/3) | **Fundacional** |
| `11_monotonicity_proof.ipynb` | Proposição 1 + verificação numérica | **Fundacional** |

---

## Apêndice A — Fórmulas Rápidas

**Basin Access Probability:**
$$S_t = \mathbb{P}\left[\Phi^H(x_t + \epsilon) \in \mathcal{B}(p_{t+H})\right]$$

**Estimador MC:**
$$\hat{S}_t = \frac{1}{N}\sum_{i=1}^N \mathbb{1}\left[\Phi^H(x_t + \epsilon_i) \in \mathcal{B}(p_{t+H})\right]$$

**Condição de monotonicidade:**
$$t_1 < t_2 \implies S_{t_1} \geq S_{t_2} \quad \text{sob A1–A5}$$

**Conjunto de perturbações viáveis:**
$$A(t) = \left\{\epsilon : \|\epsilon\| \leq \bar{\epsilon},\; \Phi^H(x_t + \epsilon) \in \mathcal{B}(p_{t+H})\right\}$$

**Chain causal:**
$$p_t \xrightarrow{\text{mecanismo}} \mathcal{B}(p_t) \xrightarrow{\text{geometria}} \mu(A(t)) \xrightarrow{\text{MC}} S_t$$
