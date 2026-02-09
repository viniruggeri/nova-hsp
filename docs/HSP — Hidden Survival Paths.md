HSP — Hidden Survival Paths

1\. Ideia central

O HSP é um framework para detectar colapso sistêmico antes de sinais observáveis, medindo não o estado atual do sistema, mas a existência de futuros viáveis.

Sobrevivência ≠ ainda não falhou  
Sobrevivência \= existem trajetórias futuras possíveis a partir do estado atual.

Quando esse conjunto de trajetórias colapsa, o sistema já está extinto — mesmo que métricas tradicionais ainda pareçam estáveis.

\---

2\. Formulação conceitual

Considere um sistema dinâmico com estado latente .

A partir de , amostramos múltiplas trajetórias futuras .

Definimos uma função de viabilidade:

trajetórias que respeitam restrições físicas / ecológicas / estruturais sobrevivem

trajetórias inviáveis são descartadas

O HSP estima:

\> 

A queda persistente de  indica colapso iminente.

\---

3\. Arquitetura geral

3.1 Pipeline

1\. Observação do estado atual

2\. Amostragem de futuros possíveis (simulação / modelo generativo)

3\. Construção de um grafo de futuros:

nós \= estados futuros

arestas \= transições viáveis

4\. Agregação estrutural via GNN

5\. Cálculo do score HSP (densidade, conectividade, entropia)

6\. Detecção de colapso por regra simples (não supervisionada)

\---

4\. Modelos envolvidos

Simulação / Mundo artificial

ABM (ex: formigueiro, epidemia em grafo)

Representação

PyTorch Geometric (GNNs leves)

Graph-level pooling

Modelos auxiliares

LSTM / NSSM simples para gerar trajetórias

Nenhum classificador supervisionado

\---

5\. Métricas principais

Lead Time (principal)

quanto antes o HSP alerta em relação ao evento real

RMSE / erro de reconstrução (auxiliar)

Estabilidade temporal do score

Baselines de comparação

LSTM de previsão direta

Transformer temporal

Detecção de drift estatístico

\---

6\. Mundos de teste (PoC)

6.1 Epidemia em grafo

Casos aparentam estabilidade

Conectividade futura colapsa

6.2 ABM de colônia (formigas)

Refúgios locais

Extinção global inevitável

Objetivo: mostrar que o HSP alerta antes de qualquer métrica clássica.

\---

7\. Stack técnica (enxuta)

Python

PyTorch \+ PyTorch Geometric

NumPy / SciPy / math

Pandas \+ Great Expectations

Matplotlib / Seaborn

MLOps

MLflow

Hydra

DVC

pre-commit

Arize AI

\---

8\. Restrições práticas

Hardware limitado (8GB RAM)

Prioridade total em provar a ideia, não escalar

Amostras pequenas, mundos controlados

\---

9\. Papel do Patrick

Construção dos mundos artificiais

Definição das regras dinâmicas

Garantir que o colapso é estrutural (não bug)

Sanity checks matemáticos

O HSP observa. O mundo precisa ser correto.

\---

10\. Frase-resumo

\> Baselines olham o presente.  
O HSP mede se ainda existe futuro.

\---

Outline de Paper (Draft)

Título (provisório)

Hidden Survival Paths: Early Detection of Latent Extinction via Future Reachability Collapse

Abstract

Propomos o Hidden Survival Paths (HSP), um framework não supervisionado para detecção precoce de extinção latente em sistemas dinâmicos. Diferentemente de abordagens supervisionadas que dependem de colapso observável, o HSP define extinção como perda de alcançabilidade futura no espaço de estados. Demonstramos, em mundos simulados com epidemias em grafos e agentes com refúgios, que o HSP detecta colapso estrutural com maior lead time sob observações parciais, superando LSTM, Transformer temporal e detectores de drift.

1\. Introdução

Limitações de métodos supervisionados baseados em observáveis

Extinção como propriedade dinâmica/topológica

Contribuições principais

2\. Definição do Problema

Sistema dinâmico desconhecido

Observações parciais

Objetivo: detectar perda de futuros viáveis antes do colapso observável

3\. Hidden Survival Paths

Amostragem de futuros

Construção de grafo implícito de transições

Métrica de opcionalidade futura

Critério de alerta

4\. Mundos Simulados

Epidemia em grafos (BA, modular)

ABM com refúgios (formigas ou Fallout-lite)

5\. Baselines

LSTM forecasting \+ threshold

Temporal Transformer (light)

Detector de drift (CUSUM / ADWIN)

6\. Métricas

Lead Time (principal)

RMSE (secundária, contextual)

FPR e robustez sob ruído

7\. Resultados

Curvas de opcionalidade vs observáveis

Tabela de lead time

Análise contrafactual

8\. Discussão

Falhas estruturais dos baselines

Implicações para detecção precoce

9\. Conclusão

Extinção latente é detectável sem rótulos

HSP como ferramenta científica

**NÚCLEO MATEMÁTICO**  
\\\\section{Núcleo Matemático}

\\paragraph{Sistema.}  
Seja $(\\mathcal{Z},\\mathcal{B})$ um espaço de estados mensurável.  
O sistema evolui como um processo estocástico possivelmente não-Markoviano  
\\\[  
z\_{t+1}\\sim\\mathcal{P}(\\cdot\\mid z\_t,\\Theta),  
\\\]  
onde \\(\\Theta\\) denota parâmetros latentes, potencialmente não  
identificáveis, e \\(\\mathcal{P}\\) pode estar mal-especificado.

\\paragraph{Domínio de Sobrevivência.}  
Denotemos por \\(\\mathcal{S}\\subset\\mathcal{Z}\\) o conjunto de estados  
viáveis. Não assumimos que \\(\\mathcal{S}\\) seja totalmente observável.

\\paragraph{Trajetórias Futuras.}  
Para um horizonte \\(T\\in\\mathbb{N}\\) e estado atual \\(z\_t\\in\\mathcal{Z}\\),  
seja \\(\\mathbb{P}\_{\\Gamma}(\\cdot\\mid z\_t)\\) a medida de probabilidade  
induzida sobre trajetórias de comprimento \\(T\\)  
\\\[  
\\gamma=(z\_t,z\_{t+1},\\dots,z\_{t+T}),  
\\\]  
obtida ao amostrar parâmetros latentes \\(\\Theta\\) e ruído do sistema.  
Não assumimos que \\(\\mathbb{P}\_{\\Gamma}\\) seja o processo gerador  
verdadeiro, apenas que gera futuros localmente plausíveis.

\\paragraph{Hidden Survival Path (HSP).}  
Dado \\(\\tau\\in(0,1)\\), uma trajetória  
\\(\\gamma\\sim\\mathbb{P}\_{\\Gamma}(\\cdot\\mid z\_t)\\) é uma  
\\emph{Hidden Survival Path (HSP)} se  
\\\[  
\\mathbb{P}\\\!\\left(z\_{t+k}\\in\\mathcal{S},\\;\\forall k\\le T \\mid z\_t\\right)  
\\ge\\tau.  
\\\]  
Esta definição trata sobrevivência como propriedade de trajetórias,  
não de estados instantâneos.

\\paragraph{Optionalidade Futura.}  
Definimos a optionalidade futura em \\(t\\) como  
\\\[  
\\mathcal{O}\_T(z\_t)  
:=  
\\mathbb{P}\_{\\gamma\\sim\\mathbb{P}\_{\\Gamma}(\\cdot\\mid z\_t)}  
\\big(\\gamma\\ \\text{é uma HSP}\\big).  
\\\]  
Assim, \\(\\mathcal{O}\_T(z\_t)\\) mede a \\emph{massa} de futuros viáveis,  
em oposição à mera existência de uma única trajetória.

\\paragraph{Aproximação por Grafos (epistêmica).}  
Seja \\(V\_t\\) uma amostra finita de trajetórias extraídas de  
\\(\\mathbb{P}\_{\\Gamma}(\\cdot\\mid z\_t)\\). Construa-se o grafo epistemico  
\\(G\_t=(V\_t,E\_t)\\) onde as arestas são amostradas por seed de acordo com  
uma similaridade dependente da tarefa:  
\\\[  
\\mathbb{P}\\big\[(v\_i,v\_j)\\in E\_t\\big\] \= \\exp\\\!\\big(-d(v\_i,v\_j)\\big).  
\\\]  
Tratamos \\(G\_t\\) como aproximação epistêmica da medida contínua  
\\(\\mathbb{P}\_{\\Gamma}\\); componentes conectadas em grafos amostrados são  
interpretadas como hipóteses sobre regiões mutuamente alcançáveis no  
espaço de futuros.

\\paragraph{Estimador de Optionalidade baseado em Grafo.}  
Para índice de amostra \\(m\\), seja  
\\(\\widehat{\\mathcal{O}}\_t^{(m)}\\) a optionalidade empírica calculada sobre  
componentes conectadas e scores de viabilidade por nó  
\\(s\_i\\in\[0,1\]\\) (ver seção de engenharia). A estimativa por ensemble é  
\\\[  
\\widehat{\\mathcal{O}}\_t \= \\mathbb{E}\_m\\big\[\\widehat{\\mathcal{O}}\_t^{(m)}\\big\],  
\\\]  
usada como estimador de Monte Carlo para \\(\\mathcal{O}\_T(z\_t)\\).  
\\textbf{Nota:} \\(s\_i\\) é uma estimativa epistêmica de viabilidade  
(baseada no modelo gerador ou em regras), não uma probabilidade de  
verdadeiro ground-truth.

\\paragraph{Extinção Latente (persistência estocástica).}  
Dados \\(\\delta,\\eta\\in(0,1)\\) e \\(K\\in\\mathbb{N}\\), declaramos uma  
extinção latente persistente em \\(t^\\ast\\) se  
\\\[  
\\mathbb{P}\\\!\\big(\\mathcal{O}\_T(z\_t)\<\\delta\\big) \> \\eta  
\\quad\\text{para todo } t\\in\[t^\\ast,t^\\ast+K\].  
\\\]  
Na prática, \\(\\mathbb{P}(\\mathcal{O}\_T(z\_t)\<\\delta)\\) é estimada pela  
fração de seeds com \\(\\widehat{\\mathcal{O}}\_t^{(m)}\<\\delta\\), e a  
persistência é avaliada sobre essa estimativa empírica.

\\paragraph{Lacuna de Observação.}  
Sejam \\(y\_t=h(z\_t)\\) os sinais observáveis. Em geral,  
\\\[  
\\mathcal{O}\_T(z\_t)\\downarrow  
\\;\\;\\nRightarrow\\;\\;  
y\_t\\downarrow,  
\\\]  
i.e., observáveis podem permanecer estáveis enquanto a massa de futuros  
viáveis colapsa.

\\paragraph{Tempo de Detecção.}  
Definimos o tempo de detecção do HSP como  
\\\[  
T\_{\\mathrm{HSP}} := \\inf\\big\\{t:\\; \\mathbb{P}(\\mathcal{O}\_T(z\_t)\<\\delta)\>\\eta\\big\\}.  
\\\]

\\paragraph{Avaliação.}  
Para um baseline supervisionado com tempo de detecção  
\\(T\_{\\mathrm{base}}\\), definimos o lead time  
\\(\\Delta T := T\_{\\mathrm{base}} \- T\_{\\mathrm{HSP}}\\).

\\paragraph{Lema de Convergência (esboço).}  
\\textbf{Lema.} Sob condições regulares (boundedness de scores por  
trajetória, consistência do estimador de distância \\(d\\), e  
\\(|V\_t|\\to\\infty\\)), o estimador de optionalidade baseado em grafos  
converge em probabilidade:  
\\\[  
\\widehat{\\mathcal{O}}\_t \\xrightarrow{p} \\mathcal{O}\_T(z\_t).  
\\\]

\\textbf{Esboço da prova.} À medida que \\(|V\_t|\\to\\infty\\), a medida  
empírica sobre trajetórias converge fraca\\-mente para  
\\(\\mathbb{P}\_{\\Gamma}(\\cdot\\mid z\_t)\\). Dada a consistência de \\(d\\),  
a amostragem de arestas induz uma aproximação local consistente da  
conectividade sob a métrica \\(d\\). A soma por componente de variáveis  
limitadas \\(s\_i\\) converge, via lei dos grandes números, para a integral  
da massa viável, levando ao resultado.

\\paragraph{Condições de Validade.}  
Um estimador HSP é admissível se satisfaz:  
\\begin{enumerate}  
    \\item \\emph{Estabilidade:} \\(\\mathcal{O}\_T\\) é Lipschitz-continua  
    sob pequenas perturbações de \\(\\mathbb{P}\_{\\Gamma}\\).  
    \\item \\emph{Contrafactualidade:} Intervenções que aumentam  
    estritamente a alcançabilidade (reachability) aumentam  
    \\(\\mathcal{O}\_T\\).  
    \\item \\emph{Robustez:} \\(\\mathcal{O}\_T\\) é invariante sob ruído de  
    observação limitado.  
\\end{enumerate}

\\paragraph{Ligação com a construção em grafos.}  
A seção seguinte desenvolve a construção baseada em grafos apresentada  
acima, explicitando o caráter epistêmico de amostragem por seed e a  
relação operativa entre \\(\\widehat{\\mathcal{O}}\_t\\) e  
\\(\\mathcal{O}\_T(z\_t)\\).

MAPEAMENTO FORMAL

Mapeamento Formal → Implementação (HSP)

A regra é simples:

\> Nenhum símbolo abstrato sem um representante computável claro.

Vou listar em três camadas:

1\. objetos matemáticos

2\. proxy computacional

3\. implementação concreta (PoC-level)

\---

1\. Estado 

Definição teórica

z\_t \\in \\mathcal{Z}

Estado completo (latente ou parcialmente observável) do sistema.

Proxy computacional

Embedding vetorial latente:

z\_t \\approx h\_\\phi(x\_t)

onde  são observações parciais.

Implementação

GNN (PyTorch Geometric)

Cada nó \= agente / região / célula

Edge \= contato / mobilidade / troca

Output:

z\_t: Tensor \[num\_nodes, d\]

Sem decoder. Sem autoencoder completo. Só embedding funcional.

\---

2\. Espaço de estados 

Teoria

Conjunto implícito de estados possíveis.

Proxy

Conjunto de embeddings já visitados \+ rollouts futuros.

Implementação

Buffer leve (NumPy .npy)

Amostragem online

Nada de armazenar tudo (OOM é o inimigo)

\---

3\. Dinâmica 

Teoria

z\_{t+1} \\sim \\mathcal{P}(\\cdot \\mid z\_t)

Desconhecida.

Proxy

Modelo de transição aproximado ou simulador direto.

Implementação (PoC)

Mundo 1 (SIR): simulador explícito

Mundo 2 (ABM): regras determinísticas \+ ruído

Opcional:

pequeno MLP residual em cima do simulador

só se precisar

Nada de treinar dynamics model pesado.

\---

4\. Trajetórias 

Teoria

Conjunto de futuros possíveis.

Proxy

Ensemble de rollouts estocásticos.

Implementação

for k in range(N\_ensembles):  
    z\_rollout\[k\] \= simulate(z\_t, noise\_k, T)

– já basta

Paraleliza com joblib

Armazena só scores, não estados completos

\---

5\. Conjunto de sobrevivência 

Teoria

Estados viáveis.

Proxy

Score latente de viabilidade:

s(z) \\in \[0,1\]

Implementação

Função simples, explícita, auditável:

conectividade mínima

população mínima

recursos \> 0

grau médio \> limiar

def survival\_score(z):  
    return float(score \> epsilon)

Sem classificador treinado. Reviewer agradece.

\---

6\. Hidden Survival Path (HSP)

Teoria

Trajetória que permanece em  com alta probabilidade.

Proxy

Rollout cuja sequência de scores nunca zera.

Implementação

is\_hsp \= all(survival\_score(z\_tk) for t\_k in rollout)

Ou versão soft:

mean\_score \> tau

\---

7\. Opcionalidade futura 

Teoria

Probabilidade de existir HSP.

Estimador empírico

\\hat{\\mathcal{O}}\_T(z\_t)  
\=  
\\frac{1}{N}  
\\sum\_{i=1}^N  
\\mathbb{1}\[\\gamma\_i \\text{ é HSP}\]

Implementação

O\_t \= np.mean(hsp\_flags)

Barato. Estável. Interpretável.

\---

8\. Detecção de colapso

Teoria

\\mathcal{O}\_T(z\_t) \< \\delta \\text{ por } K \\text{ passos}

Implementação

janela deslizante

threshold fixo

if all(O\_t\_window \< delta):  
    alert \= True

Nada de CUSUM aqui. Isso é o sinal HSP, não baseline.

\---

9\. Baselines supervisionados

Teoria

Detectam colapso via observáveis .

Implementação mínima

LSTM: forecast \+ threshold

Transformer pequeno

ADWIN / CUSUM

Eles só veem:

y\_t \= observable\_metrics(world)

Não veem grafos latentes. Esse é o ponto.

\---

10\. Lead time 

Teoria

\\Delta T \= T\_{\\text{baseline}} \- T\_{\\text{HSP}}

Implementação

lead\_time \= t\_baseline \- t\_hsp

Isso é a métrica-mãe. O resto é decorativo.

\---

11\. Limites de hardware (importante)

Não guardar rollouts completos

Não treinar GNN profundo

Não backprop em tempo longo

Não usar batch gigante

Tudo:

online

streaming

estatístico.

ARQUITETURA HSP

Arquitetura HSP, em três níveis:

1\. arquitetura canônica da PoC (o que entra agora, sem OOM, sem loucura)

2\. ideias que entram como hooks opcionais (plugáveis, não centrais)

3\. ideias explicitamente fora da PoC (guardadas para v2/paper futuro)

Assim você mantém rigor \+ ambição sem se sabotar.

\---

3/4 — Arquitetura do HSP (destilada)

Visão macro (fluxo)

Observações / Simulador  
        ↓  
GNN Encoder (estado latente z\_t)  
        ↓  
Sampler de futuros (ensembles)  
        ↓  
Avaliação de sobrevivência (𝒮)  
        ↓  
Estimador de opcionalidade 𝒪\_t  
        ↓  
Detector de colapso \+ lead time

Nada aqui é supérfluo. Cada bloco existe por necessidade teórica.

\---

1\. Encoder de estado 

Escolha canônica (PoC)

GraphSAGE ou GATConv

2–3 camadas

hidden dim pequeno (32–64)

skip connections ✔️

Por quê

GraphSAGE: estável, barato, inductive

GAT: bom se heterogeneidade/importância de hubs importa (BA graphs)

Skip connection é importante por um motivo conceitual:

\> nem toda perturbação é colapso (ex.: COVID ≠ extinção)

Ela preserva informação prévia quando o sinal novo é ruidoso.

z\_t \= GNN(x\_t, A\_t) \+ z\_{t-1}

\---

Ativação

SiLU ou GELU ✔️  
Estáveis, suaves, melhores pra gradientes em sinais fracos.

\---

❌ Fora da PoC

UNet (overkill)

TGN (temporal graph é lindo, mas pesado)

NSSM completo (guarda pra paper 2\)

\---

2\. Dinâmica temporal (opcional, leve)

Aqui você tem duas opções, ambas válidas.

Opção A — sem dinâmica aprendida (default)

usa simulador explícito

o GNN só embeda o estado atual

✔️ Mais fiel à teoria  
✔️ Mais barato  
✔️ Menos risco de leakage

\---

Opção B — dinâmica mínima aprendida

GRU temporal sobre embeddings

hidden pequeno

sem rollout longo

z\_t' \= GRU(z\_t, z\_{t-1})

Usar só se:

observação parcial for severa

dinâmica não for totalmente conhecida

\---

❌ Fora da PoC

NSSM variacional

modelos markovianos profundos

backprop through time longo

\---

3\. Sampling de futuros (núcleo do HSP)

Isso é o coração, não o GNN.

Implementação

Monte Carlo rollouts

ruído paramétrico \+ estrutural

sem backprop aqui

for i in range(N):  
    future \= simulate(z\_t, noise\_i, T)  
    score\_i \= survival\_score(future)

Otimizações reais

early stopping de rollout se morrer cedo

salvar só scores, nunca estados completos

paralelizar CPU (joblib)

\---

4\. Sobrevivência 

Forma canônica

Função rule-based \+ contínua:

conectividade do grafo

população mínima

diversidade de caminhos

recursos \> 0

Nada supervisionado aqui.

𝒮(z) \= sigmoid( w₁·conn \+ w₂·pop \+ w₃·paths − c )

Isso deixa o gradiente existir sem treinar um classificador.

\---

5\. Opcionalidade futura 

Estimador empírico, ponto final.

𝒪\_t \= (\# rollouts sobreviventes) / N

Suavização

média móvel

ReduceLROnPlateau ❌ aqui não faz sentido  
(não tem loss sendo otimizada)

Mas:

histerese no threshold ✔️

janela K ✔️

\---

6\. Detector de colapso

Nada fancy. Elegância \> complexidade.

if 𝒪\_t \< δ for K steps:  
    alert

Isso é o sinal científico. O resto são baselines.

\---

7\. Baselines (onde entram suas ideias)

Aqui sim entram algumas coisas.

Forecasting

LSTM

Transformer pequeno

TCN (ok, barato)

Drift

ADWIN

CUSUM

ECOD / Isolation Forest ✔️

❌ SMOTE

Não faz sentido aqui:

não é classificação supervisionada

vai confundir reviewer

\---

8\. RL? Sim, mas não agora

RL é válido conceitualmente, mas:

RL entra quando você age para preservar futuros

a PoC só detecta

Então:

deixa um stub

uma frase no paper

zero código agora

\---

9\. Atenção, ResNet, MLP interno

✔️ MLP residual pequeno dentro do GNN  
✔️ atenção só se estruturalmente justificada (GAT já cobre)

Nada de attention genérica “porque sim”.

\---

10\. O que fica explícito no paper

Você não esconde que isso é modular.

Frase-chave:

\> “The HSP framework is architecture-agnostic; we adopt a minimal GNN encoder to isolate the effect of future reachability.”

REPO FINAL

Esqueleto do Repositório (nova-hsp)

A regra de ouro do repo:

\> rodar inteiro em CPU, GPU só acelera ensembles ou GNN se existir

Nada pode quebrar sem CUDA.

\---

Estrutura de pastas (final)

nova-hsp/  
│  
├── README.md  
├── pyproject.toml  
├── requirements.txt  
├── .pre-commit-config.yaml  
│  
├── configs/  
│   ├── config.yaml              \# entrypoint Hydra  
│   │  
│   ├── hsp/  
│   │   ├── base.yaml             \# determinístico mínimo  
│   │   ├── stochastic.yaml       \# epistemic seeds, graph sampling  
│   │   ├── ablation\_no\_gnn.yaml  
│   │   └── ablation\_no\_sampling.yaml  
│   │  
│   ├── worlds/  
│   │   ├── sir\_graph.yaml  
│   │   ├── ant\_colony.yaml  
│   │   └── real\_dataset.yaml  
│   │  
│   ├── baselines/  
│   │   ├── survival.yaml  
│   │   ├── state.yaml  
│   │   ├── heuristics.yaml  
│   │   └── deep.yaml  
│   │  
│   ├── experiments/  
│   │   ├── simulation.yaml  
│   │   ├── real\_data.yaml  
│   │   └── counterfactual.yaml  
│   │  
│   └── metrics/  
│       ├── lead\_time.yaml  
│       ├── robustness.yaml  
│       └── collapse\_prob.yaml  
│  
├── data/  
│   ├── raw/  
│   ├── processed/  
│   └── metrics/  
│  
├── src/  
│   ├── hsp/  
│   │   ├── \_\_init\_\_.py  
│   │   ├── encoding.py        \# Phase 0  
│   │   ├── sampling.py        \# Phase 1  
│   │   ├── graph.py           \# Phase 2  
│   │   ├── viability.py       \# Phase 3  
│   │   ├── optionality.py     \# Phase 4  
│   │   ├── collapse.py        \# Phase 5 (persistence)  
│   │   ├── explanation.py     \# Phase 6 (optional)  
│   │   └── metrics.py  
│   │  
│   ├── worlds/  
│   │   ├── base.py            \# BaseWorld interface  
│   │   ├── sir\_graph.py  
│   │   ├── ant\_colony.py  
│   │   └── utils.py  
│   │  
│   ├── baselines/  
│   │   ├── survival/  
│   │   ├── state/  
│   │   ├── heuristics/  
│   │   └── deep/  
│   │  
│   ├── experiments/  
│   │   ├── run\_hsp.py  
│   │   ├── run\_baselines.py  
│   │   ├── run\_counterfactuals.py  
│   │   └── aggregate.py  
│   │  
│   ├── utils/  
│   │   ├── logging.py  
│   │   ├── seeds.py  
│   │   └── device.py  
│   │  
│   └── visualization/  
│       ├── plots.py  
│       └── figures.py  
│  
├── notebooks/  
│   ├── 01\_sanity.ipynb  
│   ├── 02\_hsp\_vs\_baselines.ipynb  
│   └── 03\_paper\_figures.ipynb  
│  
└── results/  
    ├── simulated/  
    │   ├── sir\_graph/  
    │   └── ant\_colony/  
    └── real/

\---

Decisões arquiteturais importantes (explícitas)

1\. CPU-first design

torch.device("cuda" if available else "cpu")

batch pequeno

nenhum tensor gigante persistente

Patrick pode ligar GPU sem mudar código.

\---

2\. Hydra como espinha dorsal

Tudo configurável, nada hardcoded:

hsp:  
  ensemble\_size: 300  
  horizon: 50  
  delta: 0.3  
  persistence\_k: 3

Isso evita:

tuning manual

scripts Frankenstein

\---

3\. MLflow (leve, local)

Só pra:

registrar runs

salvar curvas 𝒪\_t

logar lead time

Nada de servidor remoto.

\---

4\. DVC (opcional, mas limpo)

Usar só para:

seeds dos mundos

métricas finais

Não versionar gigabytes inúteis.

\---

Onde cada conceito mora (mapa mental)

Conceito teórico	Código

	encoder.py  
	dynamics.py  
	sampler.py  
	survival.py  
	optionality.py  
Colapso	detector.py  
Lead time	metrics.py

Se alguém perguntar “onde isso vive?”, você aponta.

\---

Como roda a PoC (1 comando)

python src/experiments/run\_hsp.py \\  
  \+world=sir\_graph \\  
  \+baseline=lstm

Depois:

python src/experiments/aggregate\_results.py

Resultado:

tabela de lead time

figuras prontas

Sem drama.

\---

Política clara de extensões (pra não virar caos)

✔️ entram como módulos opcionais

GRU

atenção

Optuna

RL

❌ não entram na PoC

NSSM variacional

UNet

TGN completo

SMOTE

Isso fica documentado no README. Ordem salva projeto.

Isso é vini. isso é sota. isso é n.o.v.a.