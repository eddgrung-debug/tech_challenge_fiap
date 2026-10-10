"""Métricas, escolha de limiar e comparação entre modelos."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import METRICS, RANDOM_STATE


def ks_statistic(y_true, y_proba) -> float:
    """KS: maior distância entre as curvas acumuladas de maus e de bons."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def score(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Conjunto de métricas para classificação binária (classe positiva = mau pagador).

    Acurácia sozinha engana em base desbalanceada: com 1,5% de maus, aprovar
    todo mundo dá 98,5% de acurácia e não barra ninguém. Por isso precisão,
    recall, F1, AUC-ROC, PR-AUC e KS vêm junto.
    """
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
        out["pr_auc"] = average_precision_score(y_true, y_proba)
        out["ks"] = ks_statistic(y_true, y_proba)
    return out


def best_threshold(y_true, y_proba, beta: float = 1.0) -> float:
    """Limiar que maximiza F-beta. Use SEMPRE com predições out-of-fold do treino."""
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    b2 = beta ** 2
    f = (1 + b2) * prec * rec / np.clip(b2 * prec + rec, 1e-12, None)
    return float(thr[np.argmax(f[:-1])])


def threshold_for_rejection_rate(y_proba, taxa: float) -> float:
    """Limiar que reprova a fração `taxa` de pedidos com maior risco estimado."""
    return float(np.quantile(y_proba, 1 - taxa))


def policy_table(y_true, y_proba, taxas=(0.02, 0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    """Leitura de negócio: o que acontece ao reprovar os X% de maior risco."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    ordem = np.argsort(-y_proba, kind="stable")
    linhas = []
    for t in taxas:
        n = int(round(t * len(y_true)))
        rej = np.zeros(len(y_true), dtype=bool)
        rej[ordem[:n]] = True
        maus_rej = y_true[rej].sum()
        linhas.append({
            "reprovados_pct": 100 * t,
            "pedidos_reprovados": n,
            "maus_barrados": int(maus_rej),
            "maus_barrados_pct": 100 * maus_rej / y_true.sum(),
            "inadimplencia_reprovados_pct": 100 * y_true[rej].mean(),
            "inadimplencia_aprovados_pct": 100 * y_true[~rej].mean(),
            "bons_reprovados_por_mau": (n - maus_rej) / max(maus_rej, 1),
        })
    df = pd.DataFrame(linhas)
    df.attrs["inadimplencia_base_pct"] = 100 * y_true.mean()
    return df


def band_table(y_true, y_proba, cortes) -> pd.DataFrame:
    """Taxa de maus por faixa de escore. `cortes` são limites em percentil de risco."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_proba)})
    df["pct"] = df["p"].rank(pct=True, method="first")
    df["faixa"] = pd.cut(df["pct"], [0, *cortes, 1.0], include_lowest=True)
    out = df.groupby("faixa", observed=True)["y"].agg(pedidos="size", maus="sum")
    out["taxa_maus_pct"] = 100 * out["maus"] / out["pedidos"]
    out["maus_do_total_pct"] = 100 * out["maus"] / out["maus"].sum()
    return out


def bootstrap_ci(y_true, y_proba, groups, metric=roc_auc_score, n_boot: int = 1000,
                 random_state: int = RANDOM_STATE) -> tuple[float, float]:
    """IC 95% por bootstrap reamostrando CLIENTES (não linhas), pois contas do
    mesmo cliente não são independentes."""
    rng = np.random.default_rng(random_state)
    y_true, y_proba = np.asarray(y_true), np.asarray(y_proba)
    cod, _ = pd.factorize(np.asarray(groups))
    linhas_por_grupo = pd.Series(np.arange(len(cod))).groupby(cod).apply(np.array).to_numpy()
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate(linhas_por_grupo[rng.integers(0, len(linhas_por_grupo),
                                                           len(linhas_por_grupo))])
        if y_true[idx].min() == y_true[idx].max():
            continue
        vals.append(metric(y_true[idx], y_proba[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def comparison_table(results: dict[str, dict], name: str = "comparacao_modelos",
                     sort_by: str = "roc_auc") -> pd.DataFrame:
    """Monta a tabela comparativa e salva em results/metrics/."""
    df = pd.DataFrame(results).T.round(4).sort_values(sort_by, ascending=False)
    METRICS.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS / f"{name}.csv")
    return df


def information_value(df: pd.DataFrame, feature: str, target: str, bins: int = 5) -> float:
    """IV (Information Value) de um atributo: força de separação univariada.

    Referência usual em crédito: < 0,02 inútil; 0,02–0,1 fraco; 0,1–0,3 médio;
    > 0,3 forte. Numéricas são fatiadas em quantis; suavização de 0,5 evita log(0).
    """
    x = df[feature]
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > 10:
        fatias = pd.qcut(x, bins, duplicates="drop")
        # Variável concentrada em um valor (ex.: quase tudo zero): mínimo vs. o resto.
        x = fatias if fatias.nunique() > 1 else (x > x.min())
    tab = pd.crosstab(x, df[target])
    bons = (tab[0] + 0.5) / (tab[0].sum() + 0.5 * len(tab))
    maus = (tab[1] + 0.5) / (tab[1].sum() + 0.5 * len(tab))
    return float(((maus - bons) * np.log(maus / bons)).sum())


def cv_permutation_importance(model, df: pd.DataFrame, features: list[str], target: str,
                              n_repeats: int = 5, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Importância por permutação medida FORA da amostra de ajuste.

    Em cada fold (agrupado por cliente) o modelo é treinado em 4/5 do treino e,
    no 1/5 restante, cada atributo é embaralhado: a queda de AUC-ROC é o quanto
    o modelo depende dele. A permutação é feita na coluna original, antes do
    one-hot, então cada variável categórica conta como uma só.
    """
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance

    from src.models import cv_splits

    quedas = []
    for _, k, tr, va in cv_splits(df, repeats=1, random_state=random_state):
        m = clone(model).fit(df[features].iloc[tr], df[target].iloc[tr])
        r = permutation_importance(m, df[features].iloc[va], df[target].iloc[va],
                                   scoring="roc_auc", n_repeats=n_repeats,
                                   random_state=random_state + k)
        quedas.append(r.importances_mean)
    quedas = np.array(quedas)
    out = pd.DataFrame({"atributo": features, "queda_auc_media": quedas.mean(axis=0),
                        "desvio_entre_folds": quedas.std(axis=0)})
    return out.sort_values("queda_auc_media", ascending=False).reset_index(drop=True)
