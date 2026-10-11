# Tech Challenge — Fase 2 | POSTECH Data Analytics

**Predição de aprovação de pedidos de cartão de crédito com aprendizado de máquina**

Modelo de classificação que estima o risco de um solicitante de cartão se tornar mau
pagador, com pipeline reprodutível do dado bruto às métricas de validação.

---

## 1. Identificação

| Campo | Valor |
|---|---|
| Turma | 12DTAT |
| Grupo | Grupo 47 |
| Data de entrega | 10/10/2026 |

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| Eduardo Henrique Neves | RM377826 | eduardohn@bb.com.br |
| Elizangela Moreira de Souza | RM377829 | lizaloiacone@bb.com.br |

---

## 2. Links da entrega

| Item | Link |
|---|---|
| Repositório | https://github.com/eddgrung-debug/tech_challenge_fiap.git |
| Vídeo executivo (≤ 5 min) | https://youtu.be/lJSZhKOQ2FM |
| Apresentação | https://github.com/eddgrung-debug/tech_challenge_fiap/blob/64390d4ea982d03ed5a3a6242b17462d44f88750/docs/Apresentacao_TechChallenge_Fase2.pdf |

---

## 3. O problema

Conceder crédito é a atividade central de uma instituição financeira e também a sua
principal fonte de risco: cada cartão aprovado para quem não paga vira perda. Analisar
pedidos um a um, só com regras fixas, é lento e deixa passar padrões que os dados
históricos revelam. O objetivo deste projeto é usar aprendizado de máquina para classificar
solicitantes de cartão de crédito como **bons** ou **maus pagadores**, apoiando a decisão de
aprovar ou recusar o pedido.

O trabalho responde a três perguntas de negócio:

1. As informações da ficha cadastral (renda, ocupação, estado civil, tempo de emprego)
   conseguem prever quem vai atrasar?
2. O que mais ajuda a prever?
3. Como transformar o modelo em uma política de aprovação?

### Variável alvo

A base não traz o rótulo pronto; ele foi construído a partir do histórico mensal de
pagamento (`credit_record.csv`).

**`MAU_PAGADOR = 1` se a conta atrasou 60 dias ou mais em algum dos seus 12 primeiros meses
de vida; `0` caso contrário.** Só entram contas com os 12 meses observados.

As duas escolhas foram tiradas dos dados (notebook 02):

- **Limiar de 60 dias, pela taxa de rolagem.** Das contas que chegam a 30–59 dias de atraso,
  só 14% pioram; 86% se regularizam. Das que chegam a 60–89 dias, 49% seguem piorando, e
  acima de 90 dias, 75% ou mais. Os 60 dias são o ponto em que o atraso deixa de ser
  esquecimento e passa a ser risco real de perda.
- **Janela fixa de 12 meses, pelo efeito de exposição.** A proporção de contas com atraso de
  60+ dias "alguma vez" vai de 0,3% (contas com até 6 meses de histórico) a 5,5% (contas com
  mais de 4 anos). Essa diferença é tempo de observação, não risco. Avaliar todas as contas
  no mesmo período de vida elimina o viés. Aos 12 meses a curva de safra já acumulou 62% dos
  atrasos que aparecem em 3 anos.

**Distribuição das classes:** 24.566 contas, das quais 370 más (**1,51%**), ou 65 bons para
cada mau. Esse desbalanceamento define a escolha das métricas e o tratamento no treino.

### Dataset

| Campo | Valor |
|---|---|
| Fonte | [Google Drive indicado no enunciado](https://drive.google.com/file/d/1z4yEyiCE_CGCWbvAAZQZSz-5-E5T5eYd/view?usp=sharing). |
| Linhas × colunas | `application_record.csv`: 438.557 × 18 · `credit_record.csv`: 1.048.575 × 3 |
| Após o tratamento | 24.566 contas × 32 colunas (26 atributos, alvo, identificadores e colunas de auditoria) |
| Período / versão | 61 meses de histórico, em meses relativos à extração (0 = mês atual, −60 = cinco anos antes) |
| Licença de uso | dataset público de uso educacional; consultar os termos na [página do dataset no Kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) |

Descrição das variáveis originais:

| Variável | Tipo | Descrição |
|---|---|---|
| `ID` | inteiro | identificador da conta (chave entre os dois arquivos) |
| `CODE_GENDER` | categórica | gênero (F/M) |
| `FLAG_OWN_CAR` | binária | possui carro |
| `FLAG_OWN_REALTY` | binária | possui imóvel |
| `CNT_CHILDREN` | inteiro | número de filhos |
| `AMT_INCOME_TOTAL` | numérica | renda anual |
| `NAME_INCOME_TYPE` | categórica | tipo de renda (assalariado, autônomo/comercial, servidor, aposentado, estudante) |
| `NAME_EDUCATION_TYPE` | categórica | escolaridade |
| `NAME_FAMILY_STATUS` | categórica | estado civil |
| `NAME_HOUSING_TYPE` | categórica | tipo de moradia |
| `DAYS_BIRTH` | inteiro | data de nascimento, em dias contados para trás |
| `DAYS_EMPLOYED` | inteiro | início do emprego atual, em dias para trás (365243 = sem vínculo) |
| `FLAG_MOBIL`, `FLAG_WORK_PHONE`, `FLAG_PHONE`, `FLAG_EMAIL` | binárias | possui celular, telefone comercial, telefone fixo, e-mail |
| `OCCUPATION_TYPE` | categórica | ocupação (31% nulos) |
| `CNT_FAM_MEMBERS` | numérica | tamanho da família |
| `MONTHS_BALANCE` | inteiro | mês do registro, relativo à extração (0, −1, −2 …) |
| `STATUS` | categórica | situação no mês: `X` sem dívida, `C` quitado, `0` 1–29 dias de atraso, `1` 30–59, `2` 60–89, `3` 90–119, `4` 120–149, `5` 150+ dias |

Atributos criados (notebook 02):

| Variável | Descrição |
|---|---|
| `AGE_YEARS`, `YEARS_EMPLOYED` | idade e tempo no emprego atual, em anos |
| `FLAG_NO_JOB` | 1 se não tem vínculo de emprego (código 365243) |
| `INCOME_PER_MEMBER` | renda dividida pelo tamanho da família |
| `EMPLOYED_SHARE` | fração da vida adulta passada no emprego atual |
| `HIST_HAS` | 1 se o cliente já tinha outra conta antes deste pedido |
| `HIST_N_ACCOUNTS`, `HIST_MONTHS` | quantas contas anteriores e quantos meses de histórico |
| `HIST_WORST_STATUS`, `HIST_LAST_STATUS` | pior atraso já registrado e situação no mês anterior ao pedido |
| `HIST_MONTHS_30PLUS`, `HIST_SHARE_LATE` | meses com atraso de 30+ dias e fração dos meses com algum atraso |
| `ID_CLIENTE` | agrupador de contas da mesma pessoa (não é atributo do modelo) |

Os atributos `HIST_*` usam apenas meses **anteriores** à abertura da conta e apenas
**outras** contas do mesmo cliente: é a informação que o banco teria no dia do pedido.

---

## 4. Como reproduzir

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Baixe os dois arquivos do dataset e coloque em `data/raw/` com os nomes
`application_record.csv` e `credit_record.csv` (os dados **não** são versionados — veja
`data/README.md`).

**Opção A — notebooks** (`jupyter notebook`), nesta ordem:

| # | Notebook | O que faz |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Análise exploratória: qualidade, distribuições, correlações, outliers, balanceamento, atributos × alvo |
| 2 | `notebooks/02_preprocessamento.ipynb` | Nulos, duplicatas, definição do alvo, limpeza, feature engineering; grava `data/processed/dataset_tratado.csv` |
| 3 | `notebooks/03_modelagem.ipynb` | Split agrupado por cliente, validação cruzada de 6 modelos × 2 conjuntos de atributos, ajuste de hiperparâmetros, escolha e treino final (cerca de 12 minutos) |
| 4 | `notebooks/04_avaliacao.ipynb` | Métricas no teste, matriz de confusão, política de corte, importância de variáveis, conclusões |

**Opção B — linha de comando**, do dado bruto às métricas do modelo final:

```bash
python run_pipeline.py            # usa os hiperparâmetros já ajustados
python run_pipeline.py --tune     # refaz a busca de hiperparâmetros
```

**Semente fixa:** `RANDOM_STATE = 42`, declarada na primeira célula de cada notebook e em
`src/config.py`. Rodar na ordem acima, em ambiente limpo, reproduz os números da seção 5.

---

## 5. Resultados

Métricas no **conjunto de teste** (4.914 contas de 1.576 clientes que nenhum modelo viu; 74
maus). Classe positiva = mau pagador. Limiar de cada modelo definido antes, com predições
out-of-fold do treino (máximo F1).

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC | PR-AUC | KS | AUC-ROC na validação cruzada |
|---|---|---|---|---|---|---|---|---|
| **Regressão logística** (ficha + histórico) — escolhido | 0,972 | 0,140 | 0,162 | 0,150 | **0,734** | 0,084 | 0,372 | 0,751 ± 0,035 |
| Random Forest (ficha + histórico) | 0,972 | 0,156 | 0,203 | 0,176 | 0,724 | 0,096 | 0,389 | 0,761 ± 0,036 |
| Gradient Boosting (ficha + histórico) | 0,974 | 0,165 | 0,176 | 0,170 | 0,712 | 0,082 | 0,338 | 0,755 ± 0,033 |
| KNN (ficha + histórico) | 0,976 | 0,111 | 0,081 | 0,094 | 0,709 | 0,047 | 0,358 | 0,710 ± 0,024 |
| Árvore de decisão (ficha + histórico) | 0,974 | 0,165 | 0,176 | 0,170 | 0,698 | 0,061 | 0,318 | 0,742 ± 0,039 |
| Random Forest (só ficha cadastral) | 0,965 | 0,079 | 0,122 | 0,096 | 0,634 | 0,060 | 0,274 | 0,603 ± 0,045 |
| Linha de base (taxa média para todos) | 0,985 | 0 | 0 | 0 | 0,500 | 0,015 | 0 | 0,500 |

Intervalo de confiança de 95% da AUC-ROC do modelo escolhido no teste: **0,66 a 0,80**
(bootstrap por cliente).

**Modelo escolhido:** regressão logística com ficha cadastral + histórico anterior do
cliente. A Random Forest teve a maior AUC-ROC média na validação cruzada repetida (0,761,
contra 0,755 do Gradient Boosting e 0,751 da logística), mas a diferença para a logística
(+0,010) é menor que a variação normal entre rodadas (desvio de 0,017). Nesse empate, a regra
de escolha (`choose_model`, em `src/models.py`) fica com o modelo mais simples: em crédito,
poder explicar cada recusa ao cliente e ao regulador tem valor próprio. A regra usa só a
validação cruzada do treino. No teste os dois seguem empatados: a logística tem AUC-ROC
maior, a Random Forest tem PR-AUC, KS e F1 maiores, e ambas barram os mesmos 22 maus no
corte de 5%. O custo da escolha é declarado: no treino, a Random Forest barrou um pouco mais
de maus no topo do ranking (30% contra 25% no corte de 5%).

**Métricas priorizadas:** AUC-ROC (critério de escolha), KS e PR-AUC, complementadas por
precisão e recall da classe "mau pagador". **Acurácia não foi usada para decidir nada**: com
1,5% de maus, aprovar todo mundo dá 98,5% de acurácia sem barrar um inadimplente (é a linha
de base da tabela). Quanto ao custo dos erros, aprovar um mau pagador (falso negativo) custa
o saldo não pago; recusar um bom (falso positivo) custa a receita daquele cliente. O
primeiro é mais caro por caso, mas os bons são 65 vezes mais numerosos, então o ponto de
corte é uma decisão de negócio. Por isso o resultado também é apresentado como **política
de corte**:

| Reprovando os … de maior risco | Maus barrados | Inadimplência entre os aprovados | Bons recusados por mau barrado |
|---|---|---|---|
| 0% (sem modelo) | 0% | 1,51% | — |
| 2% | 18% | 1,27% | 7 |
| 5% | 30% | 1,11% | 10 |
| 10% | 39% | 1,02% | 16 |
| 20% | 55% | 0,84% | 23 |
| 30% | 64% | 0,78% | 30 |

**Sobre o protocolo de validação.** As contas com histórico pertencem a muito menos
pessoas do que parece (3,7 contas por cliente, com ficha idêntica). Um split aleatório põe a
mesma pessoa no treino e no teste: nesse cenário, o mesmo Random Forest só com a ficha
cadastral marca AUC de 0,76 a 0,82 na validação; com folds agrupados por cliente, 0,60. Todo
split e toda validação cruzada deste projeto são agrupados por cliente
(`StratifiedGroupKFold`), e o desbalanceamento é tratado com pesos de classe dentro do
pipeline, sem oversampling antes do split.

Tabelas completas em `results/metrics/` e gráficos em `results/figures/`.

---

## 6. Principais conclusões

1. **O melhor indicador de quem vai atrasar é quem já atrasou.** Clientes com atraso
   anterior de 60+ dias em outra conta voltam a atrasar em 17,7% dos casos; clientes com
   histórico limpo, em 0,6%; clientes novos, em 2,4%. Cinco das seis variáveis mais
   importantes do modelo são de histórico anterior, com *tempo de relacionamento* e *pior
   atraso anterior* na frente.
2. **A ficha cadastral, sozinha, prevê pouco.** Só com renda, ocupação, estado civil, tempo
   de emprego e demais dados cadastrais, o melhor modelo chega a AUC 0,63 no teste, com
   intervalo que encosta no acaso. Renda, idade e escolaridade praticamente não separam bons
   de maus nesta base; o que ajuda um pouco é estabilidade (tempo de emprego, estado civil,
   tipo de vínculo). Acrescentar o histórico leva a AUC a 0,73.
3. **O modelo ordena risco; não adivinha inadimplentes.** A metade de menor risco tem 0,61%
   de maus e os 5% de maior risco têm 8,9%. Uso recomendado em três faixas: aprovação
   automática na metade segura, limite inicial conservador na faixa intermediária e análise
   manual ou recusa no topo.
4. **O ponto de corte é decisão de negócio.** Recusar os 5% de maior risco evita 30% da
   inadimplência, ao custo de 10 bons clientes recusados por mau barrado. O corte se paga se
   a perda com um inadimplente superar a margem de 10 bons clientes.
5. **Cliente novo é o ponto cego.** Quem pede a primeira conta tem mais risco (2,1% contra
   1,2%) e é onde o modelo menos discrimina (AUC 0,66 contra 0,74). Para esse público,
   informação externa (bureau, cadastro positivo) vale mais que algoritmos mais complexos.
   E gênero, idade e estado civil podem ser retirados do modelo sem perda mensurável.

### Limitações e próximos passos

- **Poder de discriminação moderado** (AUC 0,73; IC 95% de 0,66 a 0,80) e **poucos eventos**
  (370 maus na base, 74 no teste): as estimativas têm margem de erro larga e a diferença
  entre os melhores algoritmos não é mensurável.
- **Identidade do cliente inferida** por ficha cadastral idêntica, na falta de um
  identificador de pessoa.
- **Histórico truncado:** a base cobre 60 meses e `credit_record.csv` tem exatamente o
  limite de linhas do Excel, indício de corte na origem.
- **Só contas com histórico têm rótulo** (8% das solicitações); não há inferência de
  rejeitados. **Não há valores** (limite, saldo), então não se estima perda em dinheiro.
- **Escore não calibrado:** ordena risco, mas não é probabilidade de inadimplência.
- **Próximos passos:** enriquecer com bureau de crédito; validar em safra futura
  (*out-of-time*); calibrar o escore e definir o corte por valor esperado; avaliar equidade
  por grupo; monitorar estabilidade após a implantação.

---

## 7. Estrutura do repositório

```
.
├── data/              dados brutos (raw) e tratados (processed) — não versionados
├── notebooks/         análise em ordem numerada (01 a 04), com saídas salvas
├── src/               funções reutilizadas pelos notebooks e pelo pipeline
│   ├── config.py          caminhos, semente, definição do alvo, listas de atributos
│   ├── data.py            leitura e gravação
│   ├── preprocessing.py   limpeza, alvo, feature engineering
│   ├── models.py          split agrupado, pipelines, validação cruzada, busca
│   ├── evaluation.py      métricas, limiar, política de corte, importância
│   ├── plots.py           estilo e gráficos
│   └── predict.py         uso do modelo treinado em novos pedidos
├── results/
│   ├── figures/       gráficos exportados
│   ├── metrics/       tabelas comparativas (.csv / .json)
│   └── models/        modelos serializados — não versionados
├── run_pipeline.py    pipeline completo em linha de comando
├── submissao/         gerador do PDF de submissão
└── docs/              apresentação executiva
```

Detalhes e convenções em [`ESTRUTURA.md`](ESTRUTURA.md).
Antes de enviar, percorra o [`CHECKLIST.md`](CHECKLIST.md).

---

## 8. Tecnologias

Python 3.13 · pandas 3.0 · NumPy 2.5 · SciPy 1.18 · scikit-learn 1.9 (`Pipeline`,
`ColumnTransformer`, `StratifiedGroupKFold`, `RandomForestClassifier`,
`HistGradientBoostingClassifier`, `LogisticRegression`, `RandomizedSearchCV`,
`permutation_importance`) · Matplotlib 3.11 · Seaborn 0.13 · joblib · Jupyter
