# Tech Challenge Fase 1 — Diagnóstico de Câncer de Mama com ML

Projeto da Fase 1 do Tech Challenge da **FIAP PosTech IA para Devs**.

Peguei um problema da área da saúde — classificar tumores de mama em **maligno** ou
**benigno** — e construí uma solução de Machine Learning de ponta a ponta: análise dos
dados, pré-processamento, treino e comparação de modelos, explicabilidade, e uma API
que serve as predições. Tudo roda em Docker.

> **Aviso:** é um projeto acadêmico. Um modelo desses **não substitui um médico** — no
> máximo serve de apoio pra priorizar exames. A API devolve um disclaimer nesse sentido
> em toda resposta.

---

## Resumo rápido (pra quem só quer o essencial)

| | |
|---|---|
| **Problema** | Classificar tumor de mama: maligno (1) ou benigno (0) |
| **Dataset** | Breast Cancer Wisconsin (Diagnostic) — 569 exames, 30 features numéricas, vem do `scikit-learn` |
| **Métrica que otimizei** | **Recall** da classe maligna (deixar passar um caso maligno é o pior erro) |
| **Modelos comparados** | Logistic Regression, KNN, SVM, Random Forest, Decision Tree |
| **Melhor modelo** | **Regressão Logística** — recall 92,9% / acurácia 96,5% / ROC AUC 99,6% |
| **Entregáveis** | 4 notebooks + módulos em `src/` + API FastAPI + testes pytest + Docker |
| **Como rodar** | `docker compose up -d api` → API em `http://localhost:8000/docs` |

---

## Como executar

O modelo treinado já vem versionado em `models/`, então a API sobe direto. Mas se quiser
retreinar do zero, é só rodar o `scripts/train.py` antes (instruções abaixo).

### Com Docker (recomendado)

```bash
docker compose up -d api          # sobe a API em background
docker compose up -d jupyter      # (opcional) sobe o Jupyter
```

- API: `http://localhost:8000` — documentação interativa em `/docs`
- Jupyter: `http://localhost:8888`

Rodar os testes (são 14: pré-processamento, modelos e API):

```bash
docker compose run --rm api pytest tests -v
```

Retreinar o modelo:

```bash
docker compose run --rm api python scripts/train.py
```

> Dockerizei o projeto porque no meu Windows o Controle de Aplicativo às vezes bloqueia o
> import do `pydantic` e do `shap`. Dentro do container Linux isso não acontece e os 14
> testes passam.

### Sem Docker (ambiente local)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows  (ou: source .venv/bin/activate)
pip install -r requirements.txt

python scripts/train.py             # gera models/best_model.pkl
uvicorn api.main:app --reload       # sobe a API
```

Variações úteis do treino:

```bash
python scripts/train.py --model "Random Forest"   # treina só um modelo
python scripts/train.py --metric f1_score         # seleciona pelo F1 em vez de recall
python scripts/train.py --save-all                # salva todos, não só o melhor
```

---

## Os notebooks

Ordem que faz sentido seguir:

1. **`01_eda.ipynb`** — análise exploratória. Distribuição das classes (~63% benigno /
   ~37% maligno), estatísticas, histogramas, boxplots e violinplots por classe, heatmap de
   correlação, e quais features mais se correlacionam com o diagnóstico.
2. **`02_preprocessing.ipynb`** — split estratificado em treino/validação/teste, pipeline de
   pré-processamento (`SimpleImputer` + `StandardScaler`), efeito da padronização, e uma
   exploração com PCA (que **não** entrou no modelo final — explico o porquê lá).
3. **`03_models.ipynb`** — treino dos 5 modelos, validação cruzada por recall, métricas no
   teste, matrizes de confusão, curvas ROC, comparação visual, e a escolha do melhor.
4. **`04_evaluation.ipynb`** — avaliação final do modelo escolhido + explicabilidade com
   feature importance e SHAP (summary, bar, e waterfall de 3 pacientes — um maligno claro, um
   benigno claro e um caso "em cima do muro").

---

## Resultados

Depois de `python scripts/train.py`, o melhor modelo foi a **Regressão Logística**. Métricas
no conjunto de teste:

| Métrica | Valor |
|---|---|
| Accuracy | 96,49% |
| Precision | 97,50% |
| Recall | 92,86% |
| F1-score | 95,12% |
| ROC AUC | 99,60% |

Os resultados completos de todos os modelos ficam salvos em `models/training_results.json`
a cada execução do treino. Os números variam um pouco a cada rodada por causa das divisões
aleatórias.

---

## A API

Com a API no ar, a documentação interativa (Swagger) fica em `http://localhost:8000/docs`.

| Endpoint | O que faz |
|---|---|
| `GET /` | Health check — confirma que a API está no ar |
| `GET /model/info` | Nome do modelo em produção, as 30 features esperadas e as métricas do treino |
| `POST /predict` | Recebe os 30 valores do exame e devolve o diagnóstico estimado |

Exemplo de resposta do `/predict`:

```json
{
  "diagnosis": "Maligno",
  "diagnosis_code": 1,
  "confidence": 99.87,
  "risk_level": "Alto",
  "disclaimer": "Este resultado é uma estimativa estatística...",
  "model_used": "Logistic Regression"
}
```

Detalhes que vale destacar:

- O cliente manda os valores **crus** do exame — a padronização (`StandardScaler`) acontece
  dentro da API, embutida no pipeline. Não tem como esquecer de aplicar.
- A entrada é validada pelo Pydantic — faltou uma feature ou mandou texto onde era número,
  volta `422` com mensagem clara.
- O disclaimer vai em **toda** resposta, inclusive nos erros. Foi proposital: em saúde, o
  resultado não pode passar a impressão de ser um laudo.

Tem 5 exemplos prontos pra testar em `exemplos_teste_api.txt` (na raiz) — 3 malignos e 2
benignos, tirados de linhas reais do dataset, então dá pra conferir se o modelo acertou.

---

## Estrutura do projeto

```
tech-challenge-fase1/
├── data/                # CSV gerado a partir do sklearn
├── models/              # best_model.pkl + training_results.json
├── notebooks/           # 01_eda, 02_preprocessing, 03_models, 04_evaluation
├── src/                 # código reutilizável
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluate.py
│   └── explainability.py
├── api/                 # API FastAPI
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
├── scripts/
│   ├── train.py         # treina e salva o melhor modelo
│   └── evaluate.py      # avalia um modelo já salvo
├── tests/               # pytest (14 testes)
├── Dockerfile + docker-compose.yml
├── requirements.txt
└── README.md
```

A ideia foi separar **o código que faz as coisas** (em `src/`) **da forma como ele é usado**
(notebooks, API, script CLI). Assim não repito código entre os notebooks e o mesmo módulo é
usado em produção.

---

## Algumas decisões que tomei (e por quê)

- **Otimizei recall, não acurácia.** O erro grave aqui é o falso negativo — dizer "benigno"
  quando era "maligno". Recall mede exatamente quantos dos casos malignos o modelo pegou. Mas
  não fui só por recall: se eu marcasse "maligno" pra todo mundo o recall seria 1.0 e o modelo
  seria inútil. Então olhei também precisão, F1 e ROC AUC — recall é o critério de desempate.
- **Inverti o label pra `1 = maligno`.** O sklearn entrega `0 = maligno` por padrão, o que é
  contraintuitivo. Invertendo, a classe positiva passa a ser a maligna — aí recall, precisão
  etc. ficam naturais de ler.
- **Pipeline do scikit-learn (`SimpleImputer` + `StandardScaler` + modelo).** Junta tudo num
  objeto só; quando salvo o `.pkl`, o scaler vai junto. Na API não tem risco de esquecer de
  escalar os dados.
- **Estratifiquei o split.** Com 569 amostras, sem estratificação dava pra ter azar e acabar
  com validação/teste em proporção bem diferente do treino.
- **Não usei PCA no modelo final.** Testei só pra visualizar (as classes já separam bem em 2D).
  Mas PCA destrói a interpretabilidade — não dá pra explicar pro médico o que é "componente
  principal 3". Num contexto médico isso não é opcional.

---

## Limitações que reconheço

- Dataset pequeno (569 amostras) e de uma única origem. Pra uso real, teria que validar em
  dados de outros hospitais.
- As 30 features já vêm prontas; num sistema real alguém precisaria extrair essas medidas a
  partir da imagem do exame.
- Não tem dados demográficos (idade, histórico familiar etc.), que na prática pesam.
- Os resultados parecem bons demais — esse dataset é considerado "fácil" pra ML. Em produção,
  com dados mais bagunçados, o desempenho provavelmente cairia.
- O modelo só diz "maligno ou benigno" — não diferencia tipos de tumor maligno.

---

## O que eu aprendi no projeto

- **Pipeline do scikit-learn** — colocar scaler + modelo num único objeto resolve o problema de
  esquecer de escalar os dados em produção.
- **Estratificação no split** — pequeno detalhe que faz diferença com dataset pequeno.
- **Por que acurácia engana** num problema de saúde com classes desbalanceadas (um modelo que
  prevê "tudo benigno" teria ~63% de acurácia e seria inútil).
- **SHAP** — primeira vez que usei. Permite explicar a decisão do modelo pra cada paciente
  específico. Achei a parte mais interessante do projeto.

---

## O que daria pra melhorar com mais tempo

- Testar um ensemble combinando os top-3 modelos pra ganhar um pouco mais de recall.
- Um endpoint `/explain` na API devolvendo a explicação SHAP junto com a predição.
- Validar o modelo em outro dataset público pra ver se generaliza.
- Monitoramento do modelo em produção ao longo do tempo.

---

## Tecnologias

Python 3.11 · pandas · numpy · scikit-learn · matplotlib · seaborn · plotly · shap ·
FastAPI · uvicorn · joblib · jupyter · pytest · Docker / docker-compose

---

Tech Challenge Fase 1 — FIAP PosTech IA para Devs.
