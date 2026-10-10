"""Estilo e gráficos reutilizados pelos notebooks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import COR_APOIO, COR_BOM, COR_MAU, COR_NEUTRA, FIGURES, ROTULOS

TINTA = "#1f1f1e"
TINTA_2 = "#5f5e5a"
GRADE = "#e6e5e0"


def set_style() -> None:
    """Estilo único para todos os gráficos: grade discreta, sem moldura, um eixo só."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 130, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": GRADE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.7,
        "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 9.5, "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "text.color": TINTA,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 9.5,
        "lines.linewidth": 2, "font.family": "DejaVu Sans",
    })


def save(fig, nome: str) -> None:
    """Exporta a figura para results/figures/ (versionada)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{nome}.png", bbox_inches="tight")


def rotulo(col: str) -> str:
    return ROTULOS.get(col, col)


def bad_rate_bars(df: pd.DataFrame, col: str, target: str, ax=None, ordem=None,
                  min_n: int = 0, titulo: str | None = None):
    """Taxa de maus por categoria (barras horizontais), com n e linha da taxa média."""
    tab = df.groupby(col, observed=True)[target].agg(taxa="mean", n="size")
    tab = tab[tab["n"] >= min_n]
    # Primeira categoria de `ordem` no topo; sem `ordem`, maior taxa no topo.
    tab = tab.loc[list(ordem)[::-1]] if ordem is not None else tab.sort_values("taxa")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 0.42 * len(tab) + 1.1))
    y = np.arange(len(tab))
    ax.barh(y, 100 * tab["taxa"], color=COR_MAU, height=0.62)
    base = 100 * df[target].mean()
    ax.axvline(base, color=TINTA_2, lw=1, ls="--")
    ax.set_yticks(y, [f"{i}  (n={n:,})".replace(",", ".") for i, n in zip(tab.index, tab["n"])])
    for yi, v in zip(y, 100 * tab["taxa"]):
        ax.annotate(f"{v:.1f}%".replace(".", ","), (v, yi), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=8.5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.set_xlim(0, max(100 * tab["taxa"].max() * 1.22, base * 1.6))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}".replace(".", ","))
    ax.set_xlabel(f"% de maus pagadores (tracejado: média geral, {base:.1f}%)".replace(".", ","))
    ax.set_title(titulo or rotulo(col))
    ax.grid(axis="y", visible=False)
    return ax


def roc_pr_curves(curvas: dict, y_true, titulo_extra: str = ""):
    """ROC e Precisão-Recall lado a lado. `curvas` = {nome: probabilidades}."""
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    cores = [COR_BOM, COR_MAU, COR_APOIO, COR_NEUTRA]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for (nome, p), cor in zip(curvas.items(), cores):
        fpr, tpr, _ = roc_curve(y_true, p)
        a1.plot(fpr, tpr, color=cor, label=f"{nome} (AUC {roc_auc_score(y_true, p):.3f})".replace(".", ","))
        prec, rec, _ = precision_recall_curve(y_true, p)
        a2.plot(rec, prec, color=cor,
                label=f"{nome} (PR-AUC {average_precision_score(y_true, p):.3f})".replace(".", ","))
    a1.plot([0, 1], [0, 1], color=TINTA_2, lw=1, ls="--", label="acaso (AUC 0,500)")
    a1.set(xlabel="Taxa de falsos positivos (bons reprovados)",
           ylabel="Recall (maus barrados)", title=f"Curva ROC{titulo_extra}")
    base = float(np.mean(y_true))
    a2.axhline(base, color=TINTA_2, lw=1, ls="--", label=f"acaso ({base:.3f})".replace(".", ","))
    a2.set(xlabel="Recall (maus barrados)", ylabel="Precisão",
           title=f"Curva Precisão-Recall{titulo_extra}", ylim=(0, 0.4))
    for a in (a1, a2):
        a.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}".replace(".", ","))
        a.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    a1.legend(loc="lower right")
    a2.legend(loc="upper right")
    fig.tight_layout()
    return fig
