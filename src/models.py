"""Definição, validação e treino dos modelos.

Todo modelo é um Pipeline (pré-processamento + classificador). Escala, log,
agrupamento de categorias raras e one-hot são ajustados apenas no fold de
treino, o que evita vazamento na validação cruzada.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (CAT_CADASTRO, CV_FOLDS, CV_REPEATS, FEATURE_SETS, GROUP,
                        LOG_FEATURES, RANDOM_STATE, TARGET)


# -------------------------------------------------------------------- split --
def split_train_test(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    """Hold-out de 20% estratificado pelo alvo e AGRUPADO por cliente.

    Um split aleatório comum colocaria contas da mesma pessoa (ficha idêntica)
    nos dois lados e o modelo seria premiado por decorar fichas. Aqui todas as
    contas de um cliente ficam do mesmo lado.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    idx_train, idx_test = next(sgkf.split(df, df[TARGET], groups=df[GROUP]))
    return df.iloc[idx_train].reset_index(drop=True), df.iloc[idx_test].reset_index(drop=True)


# ------------------------------------------------------- pré-processamento --
def build_preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    """One-hot nas categóricas; log + padronização nas numéricas se `scale`.

    Árvores não dependem de escala, então recebem as numéricas como estão.
    Regressão logística e KNN dependem: log nas variáveis de cauda longa e
    StandardScaler em todas (média 0, desvio 1).
    """
    cat = [c for c in features if c in CAT_CADASTRO]
    num = [c for c in features if c not in cat]
    log = [c for c in num if c in LOG_FEATURES]
    lin = [c for c in num if c not in log]

    # Categorias com menos de 1% do treino viram uma só ("infrequent").
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                           sparse_output=False)
    if scale:
        log_scale = Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ])
        blocos = [("log", log_scale, log), ("num", StandardScaler(), lin), ("cat", onehot, cat)]
    else:
        blocos = [("num", "passthrough", num), ("cat", onehot, cat)]
    return ColumnTransformer(blocos, verbose_feature_names_out=False)


# ------------------------------------------------------------------ modelos --
def get_models(features: list[str]) -> dict[str, Pipeline]:
    """Modelos candidatos: cinco famílias distintas mais uma linha de base.

    O desbalanceamento (1,5% de maus) é tratado com `class_weight="balanced"`:
    cada mau pesa cerca de 65 vezes mais que um bom na função de perda. Tem o
    mesmo efeito de um oversampling, sem criar registros sintéticos e sem
    risco de replicar linhas antes do split.
    """
    lin = lambda: build_preprocessor(features, scale=True)    # noqa: E731
    arv = lambda: build_preprocessor(features, scale=False)   # noqa: E731
    return {
        "baseline": Pipeline([
            ("prep", arv()),
            ("clf", DummyClassifier(strategy="prior")),
        ]),
        "logistic_regression": Pipeline([
            ("prep", lin()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000,
                                       random_state=RANDOM_STATE)),
        ]),
        "knn": Pipeline([
            ("prep", lin()),
            ("clf", KNeighborsClassifier(n_neighbors=200, weights="distance")),
        ]),
        "decision_tree": Pipeline([
            ("prep", arv()),
            ("clf", DecisionTreeClassifier(max_depth=4, min_samples_leaf=100,
                                           class_weight="balanced",
                                           random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("prep", arv()),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=20,
                                           max_features="sqrt",
                                           class_weight="balanced_subsample",
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "gradient_boosting": Pipeline([
            ("prep", arv()),
            ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150,
                                                   max_depth=3, min_samples_leaf=50,
                                                   l2_regularization=1.0,
                                                   class_weight="balanced",
                                                   random_state=RANDOM_STATE)),
        ]),
    }


PARAM_GRIDS = {
    "logistic_regression": {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "random_forest": {
        "clf__min_samples_leaf": [10, 20, 50, 100],
        "clf__max_features": ["sqrt", 0.5],
        "clf__max_depth": [None, 6, 10],
    },
    "gradient_boosting": {
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__max_iter": [100, 150, 300],
        "clf__max_depth": [2, 3, 4],
        "clf__min_samples_leaf": [20, 50, 100],
        "clf__l2_regularization": [0.0, 1.0, 10.0],
    },
}


# ------------------------------------------------------- validação cruzada --
def cv_splits(df: pd.DataFrame, repeats: int = CV_REPEATS, folds: int = CV_FOLDS,
              random_state: int = RANDOM_STATE):
    """Folds estratificados e agrupados por cliente, repetidos com sementes diferentes.

    Com ~60 maus por fold a métrica de um fold isolado oscila bastante; repetir
    a partição dá uma média mais estável e um desvio que mede o ruído.
    """
    for r in range(repeats):
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state + r)
        for k, (tr, va) in enumerate(sgkf.split(df, df[TARGET], groups=df[GROUP])):
            yield r, k, tr, va


def cross_validate_model(model: Pipeline, df: pd.DataFrame, features: list[str],
                         repeats: int = CV_REPEATS, folds: int = CV_FOLDS,
                         random_state: int = RANDOM_STATE):
    """Devolve (probabilidades out-of-fold médias, tabela com AUC/PR-AUC/KS por fold)."""
    from src.evaluation import ks_statistic
    from sklearn.metrics import average_precision_score, roc_auc_score

    X, y = df[features], df[TARGET].to_numpy()
    oof = np.zeros((repeats, len(df)))
    linhas = []
    for r, k, tr, va in cv_splits(df, repeats, folds, random_state):
        m = clone(model).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[r, va] = p
        linhas.append({"repeticao": r, "fold": k,
                       "roc_auc": roc_auc_score(y[va], p),
                       "pr_auc": average_precision_score(y[va], p),
                       "ks": ks_statistic(y[va], p)})
    return oof.mean(axis=0), pd.DataFrame(linhas)


def feature_set(name: str) -> list[str]:
    return list(FEATURE_SETS[name])


def tune(model: Pipeline, grid: dict, df: pd.DataFrame, features: list[str],
         n_iter: int = 20, random_state: int = RANDOM_STATE, n_jobs: int = 1):
    """Busca aleatória de hiperparâmetros, com os mesmos folds agrupados por cliente.

    Devolve o objeto RandomizedSearchCV já ajustado (sem refit: o modelo final é
    treinado depois, de forma explícita, no notebook).
    """
    from sklearn.model_selection import RandomizedSearchCV

    folds = [(tr, va) for r, _, tr, va in cv_splits(df, repeats=1, random_state=random_state)]
    total = int(np.prod([len(v) for v in grid.values()]))
    busca = RandomizedSearchCV(model, grid, n_iter=min(n_iter, total), scoring="roc_auc",
                               cv=folds, random_state=random_state, n_jobs=n_jobs, refit=False)
    return busca.fit(df[features], df[TARGET])


# Do mais simples e explicável ao mais complexo: critério de desempate.
ORDEM_SIMPLICIDADE = ["logistic_regression", "decision_tree", "knn", "gradient_boosting",
                      "random_forest"]


def choose_model(auc_por_fold: dict[str, np.ndarray]) -> tuple[str, pd.DataFrame]:
    """Escolhe o modelo final a partir das AUCs por fold da validação cruzada.

    1. O ponto de partida é o modelo de maior AUC média.
    2. Um modelo MAIS SIMPLES fica com a vaga se empata com ele: a diferença média
       pareada (mesmos folds) é menor que o desvio-padrão dessa diferença, ou seja,
       menor que a variação normal entre rodadas.
    3. Entre os que empatam, vence o mais simples (ORDEM_SIMPLICIDADE).

    Em crédito, explicar cada decisão tem valor de negócio e regulatório; com
    desempenho equivalente, o modelo explicável é preferível. Só usa dados de
    treino (folds da validação cruzada); o teste não entra na escolha.
    """
    medias = {n: float(np.mean(v)) for n, v in auc_por_fold.items()}
    melhor = max(medias, key=medias.get)
    linhas = []
    for nome, v in auc_por_fold.items():
        d = np.asarray(auc_por_fold[melhor]) - np.asarray(v)
        dp = float(d.std(ddof=1)) if nome != melhor else 0.0
        linhas.append({"modelo": nome, "roc_auc_media": medias[nome],
                       "diferenca_para_o_melhor": float(d.mean()), "desvio_da_diferenca": dp,
                       "empata_com_o_melhor": nome == melhor or float(d.mean()) < dp,
                       "posicao_em_simplicidade": ORDEM_SIMPLICIDADE.index(nome)})
    tab = pd.DataFrame(linhas).set_index("modelo").sort_values("posicao_em_simplicidade")
    escolhido = tab[tab["empata_com_o_melhor"]].index[0]
    return escolhido, tab
