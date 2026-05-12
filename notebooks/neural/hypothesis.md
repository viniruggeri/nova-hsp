# Neural HSP: Hypothesis Space (Sprint 2)

Este documento centraliza as hipóteses teóricas e empíricas que norteiam o desenvolvimento do **Neural HSP** (Hidden Survival Probability via Neural Operators). O objetivo é garantir que a transição do estimador de Monte Carlo (Sprint 1) para a aproximação neural não perca a interpretabilidade geométrica que faz do HSP uma métrica estrutural única, e não apenas um "forecasting" tradicional.

---

## Hipóteses Centrais

### H1. O Operador Contínuo de Viabilidade (NPE)
**Hipótese:** A probabilidade de sobrevivência geométrica $S_t = \mathbb{P}[\Phi^H(x_t + \epsilon) \in \mathcal{B}(p_{t+H})]$ pode ser aprendida nativamente como um operador contínuo $\mathcal{S}_\theta: C([0,T], \mathbb{R}^n) \to [0,1]$ através de Neural Posterior Estimation (NPE).
- **Justificativa:** Em alta dimensão, integrações de Monte Carlo sofrem degradação exponencial (maldição da dimensionalidade). Redes neurais podem interiorizar a topologia da bacia como um mapeamento direto do espaço de trajetórias contínuas para a métrica de sobrevivência (habilitando backbones como FNO ou DeepONet).
- **Como validar:** Comparação pontual entre o $S_t$ analítico (Sprint 1) e o $S_t$ neural nos sistemas base (Saddle-Node, Double-Well, Sir Graph). O erro $L_2$ deve ser irrelevante, mas o custo computacional neural deve ser $\mathcal{O}(1)$.

### H2. Regularização Geométrica Implícita (OT-CFM)
**Hipótese:** Modelos baseados puramente em NPE estimam as margens, mas degeneram perto das bifurcações onde as bacias se estreitam. O uso de **Optimal Transport Conditional Flow Matching (OT-CFM)** funciona como um regularizador estrutural limitador.
- **Justificativa:** OT-CFM induz caminhos mais retos no espaço latente, reduzindo a variância do estimador neural perto da bifurcação onde a bacia é estreita. Além disso, garante que as *streamlines* não cruzem suas rotas perto de tipping points (o que seria uma violação do teorema de unicidade de ODEs).
- **Como validar:** Testes de ablação no notebook avaliando mapas de bifurcação onde OT-CFM vs Treinamento Autoregressivo padrão é testado. 

### H3. Invariância à Escala Temporal e Monotonicidade
**Hipótese:** O Neural HSP herda a **Proposição 1 (Monotonicidade em contratação)** definida na formulação teórica original. 
- **Justificativa:** Se o framework neural não reflete o achatamento topológico da bacia de atração em fase crítica, ele falha como métrica de EWS estrutural. No caso de sistemas mais inconstantes como o *Double-Well*, eventuais violações de monotonicidade neural devem ser totalmente explicáveis pelas mesmas condições estritas que violam as premissas originais (A3/A4) no estimador analítico.
- **Como validar:** Plotar a derivada discreta de $\Delta S_t$ da aproximação Neural no dataset _Double-Well_ e _Saddle-Node_ garantindo perturbações muito mais controladas e justificáveis perante EWS tradicionais.

### H4. Eficiência de Pipeline no Mundo Real (RW Scalability)
**Hipótese:** Integradores diferenciais de alta precisão (como Diffrax: `Dopri5`) sobre campos vetoriais de Redes Neurais (Equinox) conseguem processar as trajetórias críticas do dataset real massivamente mais rápido que as simulações baseadas num forward-solver de física em JAX puro do *Sprint 1*.
- **Como validar:** Benchmark com *NCMPASS* ou *FEMTO*, rodando wall-clock da inferência neural contra o custo insustentável do MC HSP tradicional em alta carga.

### H5. Transferência entre Sistemas (Out-Of-Distribution)
**Hipótese:** O Neural HSP treinado na topologia de um sistema (ex: Saddle-Node) generaliza para outro sistema da mesma classe estrutural (ex: Double-Well) sem precisar de retraining completo da ODE.
$$\mathcal{S}_\theta^{\text{SN}} \xrightarrow{?} \text{funciona em DW}$$
- **Justificativa:** Se ele generaliza, construímos um operador universal de viabilidade estocástica. Se falhar, precisaremos de *fine-tuning* local, e saberemos suas limitações. Este passo é o "litmus test" que atesta que o modelo aprendeu a geometria orgânica das bacias em vez de criar overfitting preditivo de trajectórias de um sistema em particular.
- **Como validar:** Treinar $\mathcal{S}_\theta$ estritamente no Saddle-Node e forçar inferência de $\Delta S_t$ em validações do Double-Well ou vice-versa.

---

## Roadmap e Verificações em Notebooks

Para fechar adequadamente as validações propostas:
1. `01_neural_hsp_hypothesis.ipynb`: (Atual) Setar o foundation funcional JAX, instanciar a classe Operador e provar no Toy dataset de "Moons".
2. `02_geometric_regularization.ipynb`: Treinar usando OT-CFM vs Baseline MLP apenas com regressão no _Saddle-Node_ para testar H2.
3. `03_monotonicity_scale.ipynb`: Inspecionar H3 no Dataset _Double-Well_, extraindo as métricas comparadas e rastreando justificativas do Sprint 1.
4. `04_transfer_systems.ipynb`: Inspecionar H5, aplicando transferência sem re-treino entre *Saddle-Node* e *Double-Well*.
5. `05_rw_inference.ipynb`: Por fim, o escalonamento em dados massivos de turbinas/máquinas verificando H4.
--

## Referencias Bibliograficas

- [flow map matching](https://arxiv.org/html/2406.07507)
- [DeepONet]
- [???]