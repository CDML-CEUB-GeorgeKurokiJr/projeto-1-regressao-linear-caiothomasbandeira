[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/gIUcegNI)

---

#  Regressão Linear Simples — Precificação de Risco em Empréstimos (Risk-Based Pricing)

**Autor:** Caio Thomas Silva Bandeira  
**Disciplina:** Ciência de Dados para Engenheiros — Deep Learning  
**Instituição:** CEUB — Centro Universitário de Brasília  
**Professor:** George Kuroki Jr.

---

##  Sumário

1. [Descrição do Projeto](#1--descrição-do-projeto)
2. [O Problema de Negócio — Risk-Based Pricing](#2--o-problema-de-negócio--risk-based-pricing)
3. [Dataset Utilizado](#3--dataset-utilizado)
4. [Variáveis do Estudo](#4--variáveis-do-estudo)
5. [Metodologia e Pipeline](#5--metodologia-e-pipeline)
6. [Tecnologias e Bibliotecas](#6--tecnologias-e-bibliotecas)
7. [Estrutura do Repositório](#7--estrutura-do-repositório)
8. [Como Executar o Projeto](#8--como-executar-o-projeto)
9. [Métricas e Resultados Esperados](#9--métricas-e-resultados-esperados)
10. [Referências](#10--referências)

---

## 1 · Descrição do Projeto

Este projeto constrói e avalia **3 modelos de Regressão Linear Simples**, cada um utilizando uma variável independente diferente para prever a mesma variável alvo: a **taxa de juros (`int_rate`)** de empréstimos concedidos pela plataforma Lending Club.

Além das regressões clássicas via `statsmodels`, cada modelo possui uma versão equivalente implementada como **rede neural simples (single-neuron)** em `TensorFlow/Keras`, demonstrando que uma rede com um único neurônio linear é matematicamente equivalente à regressão linear.

O foco do trabalho está no **rigor estatístico** (verificação das premissas de regressão) e na **clareza didática** (código ricamente comentado em português), de modo que cada decisão possa ser explicada e defendida em apresentação.

---

## 2 · O Problema de Negócio — Risk-Based Pricing

### O que é?

**Risk-Based Pricing** (Precificação Baseada em Risco) é a prática na qual instituições financeiras ajustam a taxa de juros cobrada de um cliente de acordo com o **risco de crédito** que ele representa.

### Por que modelar?

| Perfil do Cliente | Risco Estimado | Taxa de Juros Esperada |
|---|---|---|
| Alta renda, pouca dívida | Baixo | ↓ Menor |
| Baixa renda, muita dívida | Alto | ↑ Maior |

Se conseguimos estimar a taxa de juros a partir de indicadores financeiros do tomador, o banco pode:

- **Automatizar** decisões de crédito.
- **Reduzir inadimplência**, cobrando juros proporcionais ao risco.
- **Oferecer taxas competitivas** para bons pagadores, aumentando a carteira de clientes.

A Regressão Linear Simples é o ponto de partida ideal: permite isolar o efeito de **uma única variável** sobre a taxa de juros e verificar se essa relação é estatisticamente significativa antes de partir para modelos mais complexos.

---

## 3 · Dataset Utilizado

| Atributo | Detalhe |
|---|---|
| **Nome** | Lending Club Loan Data |
| **Fonte** | [Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| **Volume original** | ~2,2 milhões de linhas × 150+ colunas |
| **Amostragem usada** | Amostra estratificada de **300.000 registros** (para viabilizar execução local) |
| **Variável-alvo** | `int_rate` — Taxa de juros anual do empréstimo (%) |

> **Nota:** O script de preparação realiza a amostragem estratificada por faixa de `grade` (classificação de risco do Lending Club, de A a G), garantindo representatividade da distribuição original.

---

## 4 · Variáveis do Estudo

Foram selecionadas **3 variáveis independentes** com forte justificativa de negócio e correlação esperada com a taxa de juros:

| Modelo | Variável Independente (X) | Justificativa de Negócio |
|---|---|---|
| **Modelo 1** | `annual_inc` — Renda anual declarada (USD) | Clientes com maior renda tendem a representar menor risco de crédito → juros menores. |
| **Modelo 2** | `dti` — Debt-to-Income Ratio (%) | Razão dívida/renda: quanto maior, mais comprometida a renda → maior risco → juros maiores. |
| **Modelo 3** | `fico_score` — Score FICO médio | Pontuação de crédito: quanto maior o score, menor o risco → juros menores. |

> O `fico_score` é calculado como a média de `fico_range_low` e `fico_range_high` presentes no dataset original.

---

## 5 · Metodologia e Pipeline

O notebook segue o pipeline abaixo, aplicado a cada um dos 3 modelos:

```
┌─────────────────────────────────────────────────────┐
│  1. CARREGAMENTO E AMOSTRAGEM DO DATASET            │
│     • Leitura do CSV                                │
│     • Amostra estratificada (300k registros)        │
│     • Seleção de colunas relevantes                 │
├─────────────────────────────────────────────────────┤
│  2. ANÁLISE EXPLORATÓRIA (EDA)                      │
│     • Estatísticas descritivas                      │
│     • Distribuição da variável-alvo (histograma)    │
│     • Matriz de correlação (heatmap)                │
│     • Scatter plots (X vs int_rate)                 │
├─────────────────────────────────────────────────────┤
│  3. PRÉ-PROCESSAMENTO / ENGENHARIA DE FEATURES      │
│     • Tratamento de valores ausentes                │
│     • Detecção e remoção de outliers (IQR)          │
│     • Padronização / Normalização (se necessário)   │
├─────────────────────────────────────────────────────┤
│  4. DIVISÃO DOS DADOS                               │
│     • Train / Test split (80/20)                    │
│     • Reprodutibilidade (random_state fixo)         │
├─────────────────────────────────────────────────────┤
│  5. MODELAGEM                                       │
│     Para cada variável X:                           │
│     a) Regressão via statsmodels (OLS summary)      │
│     b) Rede Neural simples via TensorFlow/Keras     │
├─────────────────────────────────────────────────────┤
│  6. VALIDAÇÃO DAS PREMISSAS (ASSUMPTIONS)           │
│     • Linearidade (scatter + fitted line)           │
│     • Homocedasticidade (resíduos vs ŷ)             │
│     • Normalidade dos Resíduos (Q-Q plot + teste)   │
│     • Independência (Durbin-Watson)                 │
├─────────────────────────────────────────────────────┤
│  7. AVALIAÇÃO E COMPARAÇÃO                          │
│     • R², R² ajustado, RMSE, MAE                    │
│     • Tabela comparativa dos 3 modelos              │
│     • (Bônus) VIF se regressão múltipla final       │
├─────────────────────────────────────────────────────┤
│  8. GUIA DE DEFESA / APRESENTAÇÃO                   │
│     • Interpretação das métricas                    │
│     • Como defender o modelo para o professor       │
└─────────────────────────────────────────────────────┘
```

---

## 6 · Tecnologias e Bibliotecas

| Biblioteca | Versão | Finalidade |
|---|---|---|
| `python` | 3.10+ | Linguagem base |
| `pandas` | ≥ 2.0 | Manipulação de dados |
| `numpy` | ≥ 1.24 | Operações numéricas |
| `matplotlib` | ≥ 3.7 | Visualizações estáticas |
| `seaborn` | ≥ 0.12 | Visualizações estatísticas |
| `statsmodels` | ≥ 0.14 | Sumário estatístico OLS |
| `tensorflow` | ≥ 2.15 | Rede neural simples |
| `scipy` | ≥ 1.11 | Testes estatísticos |

---

## 7 · Estrutura do Repositório

```
projeto-1-regressao-linear/
│
├── README.md                 ← Este documento
├── requirements.txt          ← Dependências do projeto
├── LinearRegressions.ipynb   ← Notebook principal (código + análise)
│
└── data/                     ← (Opcional) Diretório para o CSV local
    └── .gitkeep
```

> **Sobre o dataset:** O arquivo CSV do Lending Club é muito grande para ser versionado no GitHub. O notebook inclui instruções para download direto do Kaggle ou via `kagglehub`.

---

## 8 · Como Executar o Projeto

### Pré-requisitos

- **Python 3.10 ou superior** instalado na máquina.
- **Git** instalado (para clonar o repositório).
- Conta no **Kaggle** (para download do dataset, se necessário).

### Passo a passo

#### 1️ Clone o repositório

```bash
git clone https://github.com/CDML-CEUB-GeorgeKurokiJr/projeto-1-regressao-linear-caiothomasbandeira.git
cd projeto-1-regressao-linear-caiothomasbandeira
```

#### 2️ Crie e ative um ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> Após a ativação, você verá `(venv)` no início da linha do terminal, indicando que o ambiente virtual está ativo.

#### 3️ Instale as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4️ Obtenha o dataset

O notebook utiliza a biblioteca `kagglehub` para baixar o dataset automaticamente. Na primeira execução, será solicitada sua chave de API do Kaggle. Alternativamente:

1. Acesse [Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
2. Faça o download do arquivo `accepted_2007_to_2018Q4.csv.gz`.
3. Coloque-o no diretório `data/` do projeto.

#### 5️ Execute o notebook

```bash
jupyter notebook LinearRegressions.ipynb
```

Ou, se preferir o VS Code, abra o arquivo `LinearRegressions.ipynb` diretamente — o VS Code detectará o kernel do ambiente virtual automaticamente.

---

## 9 · Métricas e Resultados Esperados

Para cada modelo de Regressão Linear Simples, serão reportadas:

| Métrica | O que mede | Interpretação |
|---|---|---|
| **R² (Coeficiente de Determinação)** | Proporção da variância de `int_rate` explicada pela variável X | Quanto mais próximo de 1, melhor o ajuste. Em regressão simples, valores entre 0,10–0,50 já podem ser considerados informativos. |
| **RMSE (Root Mean Squared Error)** | Erro médio em unidades da variável-alvo (%) | Indica, em pontos percentuais, o erro típico da previsão. |
| **MAE (Mean Absolute Error)** | Erro médio absoluto (%) | Similar ao RMSE, porém menos sensível a outliers. |
| **p-value do coeficiente** | Significância estatística da relação X → Y | Se p < 0,05, a relação é estatisticamente significativa. |

> **Nota:** Em Regressão Linear **Simples** (uma única variável), não esperamos R² extremamente altos, pois a taxa de juros depende de múltiplos fatores. O objetivo é demonstrar se cada variável, **isoladamente**, possui relação estatisticamente significativa com `int_rate`.

---

## 10 · Referências

1. **JAMES, G. et al.** *An Introduction to Statistical Learning (ISLR)*. 2ª ed. Springer, 2021. — Capítulos 3 (Linear Regression).
2. **MONTGOMERY, D. C.; PECK, E. A.; VINING, G. G.** *Introduction to Linear Regression Analysis*. 5ª ed. Wiley, 2012.
3. **Lending Club Dataset (Kaggle).** Disponível em: https://www.kaggle.com/datasets/wordsforthewise/lending-club
4. **Documentação statsmodels — OLS.** Disponível em: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
5. **Documentação TensorFlow/Keras.** Disponível em: https://www.tensorflow.org/api_docs/python/tf/keras

---

<p align="center">
  <em>Projeto desenvolvido para fins acadêmicos — CEUB 2026.</em>
</p>
