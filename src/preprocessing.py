"""Limpeza, definição do alvo e feature engineering.

Nada aqui usa estatísticas da base inteira (médias, frequências, escalas): toda
transformação que "aprende" com os dados fica dentro do Pipeline em src/models.py,
para ser ajustada somente nos folds de treino.
"""

import numpy as np
import pandas as pd

from src.config import GROUP, JANELA_MESES, SEVERIDADE, STATUS_MAU, TARGET

SENTINELA_SEM_EMPREGO = 365243   # valor usado na base para "sem vínculo de emprego"


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo de nulos por coluna, em contagem e percentual."""
    total = df.isna().sum()
    pct = (total / len(df) * 100).round(2)
    return (
        pd.DataFrame({"nulos": total, "pct": pct})
        .query("nulos > 0")
        .sort_values("nulos", ascending=False)
    )


# ------------------------------------------------------------------ cadastro --
def add_client_id(app: pd.DataFrame) -> pd.DataFrame:
    """Cria ID_CLIENTE: solicitações com ficha cadastral idêntica são a mesma pessoa.

    A base tem um ID por conta, não por pessoa. Fichas idênticas em todos os 17
    atributos (incluindo data de nascimento e de admissão em dias) aparecem em
    média 3,7 vezes entre as contas com histórico. Sem este agrupamento a mesma
    pessoa cai no treino e no teste ao mesmo tempo.
    """
    out = app.copy()
    cols = [c for c in out.columns if c != "ID"]
    chave = pd.util.hash_pandas_object(out[cols].astype("string").fillna("NA"), index=False)
    out[GROUP] = "C" + pd.Series(pd.factorize(chave)[0], index=out.index).astype(str).str.zfill(6)
    return out


def clean_application(app: pd.DataFrame) -> pd.DataFrame:
    """Limpeza da base cadastral (regras fixas, sem aprender nada dos dados)."""
    out = app.copy()

    # 47 IDs aparecem duas vezes com fichas diferentes: não dá para saber qual vale.
    out = out[~out["ID"].duplicated(keep=False)]

    # FLAG_MOBIL vale 1 para todo mundo: não discrimina nada.
    out = out.drop(columns=["FLAG_MOBIL"])

    # Dias negativos contados para trás -> anos positivos.
    out["AGE_YEARS"] = (-out["DAYS_BIRTH"] / 365.25).round(2)
    sem_emprego = out["DAYS_EMPLOYED"] == SENTINELA_SEM_EMPREGO
    out["FLAG_NO_JOB"] = sem_emprego.astype(int)
    out["YEARS_EMPLOYED"] = np.where(sem_emprego, 0.0, -out["DAYS_EMPLOYED"] / 365.25).round(2)

    # Ocupação nula: aposentado sem vínculo vira categoria própria; o resto, "não informado".
    ocup = out["OCCUPATION_TYPE"].astype("object")
    ocup = ocup.where(ocup.notna(), np.where(sem_emprego, "Pensioner (no job)", "Not informed"))
    out["OCCUPATION_TYPE"] = ocup.astype(str)

    # Outliers de contagem (até 19 filhos, 20 membros): truncados, não removidos.
    out["CNT_CHILDREN"] = out["CNT_CHILDREN"].clip(upper=3)
    out["CNT_FAM_MEMBERS"] = out["CNT_FAM_MEMBERS"].clip(lower=1, upper=5).astype(int)

    # Binárias em 0/1.
    out["CODE_GENDER"] = (out["CODE_GENDER"] == "F").astype(int)        # 1 = feminino
    out["FLAG_OWN_CAR"] = (out["FLAG_OWN_CAR"] == "Y").astype(int)
    out["FLAG_OWN_REALTY"] = (out["FLAG_OWN_REALTY"] == "Y").astype(int)

    return out.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Atributos derivados da ficha cadastral."""
    out = df.copy()
    # Renda diluída pelo tamanho da família: capacidade de pagamento por pessoa.
    out["INCOME_PER_MEMBER"] = (out["AMT_INCOME_TOTAL"] / out["CNT_FAM_MEMBERS"]).round(2)
    # Estabilidade: fração da vida adulta (desde os 18) passada no emprego atual.
    out["EMPLOYED_SHARE"] = (
        out["YEARS_EMPLOYED"] / (out["AGE_YEARS"] - 18).clip(lower=1)
    ).clip(0, 1).round(4)
    return out


# --------------------------------------------------------------------- alvo --
def prepare_credit(credit: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta severidade ordinal, mês de abertura e meses de vida (MOB) da conta."""
    out = credit.copy()
    out["SEV"] = out["STATUS"].astype(str).map(SEVERIDADE).astype(int)
    out["OPEN_MONTH"] = out.groupby("ID")["MONTHS_BALANCE"].transform("min")
    out["MOB"] = out["MONTHS_BALANCE"] - out["OPEN_MONTH"]
    return out


def build_target(credit: pd.DataFrame, janela: int = JANELA_MESES,
                 status_mau: int = STATUS_MAU) -> pd.DataFrame:
    """Alvo por conta, com janela de desempenho fixa.

    MAU_PAGADOR = 1 se a conta atingiu `status_mau` (60+ dias) em algum dos
    primeiros `janela` meses de vida. Contas com menos de `janela` meses
    observados ficam de fora: ainda não tiveram tempo de mostrar quem são, e
    rotulá-las como "boas" confundiria falta de tempo com bom comportamento.
    """
    cr = credit if "MOB" in credit.columns else prepare_credit(credit)
    contas = cr.groupby("ID").agg(OPEN_MONTH=("OPEN_MONTH", "first"),
                                  MONTHS_OBSERVED=("MOB", "count"))
    pior = cr[cr["MOB"] < janela].groupby("ID")["SEV"].max()
    contas["WORST_STATUS_WINDOW"] = pior
    contas[TARGET] = (pior >= status_mau).astype(int)
    return contas[contas["MONTHS_OBSERVED"] >= janela].reset_index()


# ---------------------------------------------------------------- histórico --
def build_history_features(contas: pd.DataFrame, credit: pd.DataFrame,
                           donos: pd.DataFrame | None = None) -> pd.DataFrame:
    """Comportamento do MESMO cliente em OUTRAS contas, ANTES do pedido atual.

    Para cada conta, olha apenas os meses estritamente anteriores ao mês de
    abertura (MONTHS_BALANCE < OPEN_MONTH) das demais contas do cliente. O alvo
    é medido do mês de abertura em diante, então não há sobreposição no tempo:
    é informação que o banco teria em mãos no dia da solicitação.

    `contas` precisa ter ID, ID_CLIENTE e OPEN_MONTH. `donos` mapeia TODAS as
    contas (ID -> ID_CLIENTE), inclusive as curtas demais para entrar na
    modelagem: uma conta que durou 6 meses não tem rótulo, mas é histórico
    legítimo para o pedido seguinte do mesmo cliente.
    """
    cr = credit if "MOB" in credit.columns else prepare_credit(credit)
    dono = (contas if donos is None else donos)[["ID", GROUP]].drop_duplicates()
    hist = cr[["ID", "MONTHS_BALANCE", "SEV"]].merge(dono, on="ID")
    hist = hist.rename(columns={"ID": "ID_OUTRA"})

    par = contas[["ID", GROUP, "OPEN_MONTH"]].merge(hist, on=GROUP)
    par = par[(par["ID_OUTRA"] != par["ID"]) & (par["MONTHS_BALANCE"] < par["OPEN_MONTH"])]
    par["ATRASO_30"] = (par["SEV"] >= 1).astype(int)
    par["ALGUM_ATRASO"] = (par["SEV"] >= 0).astype(int)

    agg = par.groupby("ID").agg(
        HIST_N_ACCOUNTS=("ID_OUTRA", "nunique"),
        HIST_MONTHS=("SEV", "size"),
        HIST_WORST_STATUS=("SEV", "max"),
        HIST_MONTHS_30PLUS=("ATRASO_30", "sum"),
        HIST_SHARE_LATE=("ALGUM_ATRASO", "mean"),
    )
    ultimo = (par[par["MONTHS_BALANCE"] == par["OPEN_MONTH"] - 1]
              .groupby("ID")["SEV"].max().rename("HIST_LAST_STATUS"))
    agg = agg.join(ultimo)

    out = contas[["ID"]].merge(agg, on="ID", how="left")
    out["HIST_HAS"] = out["HIST_N_ACCOUNTS"].notna().astype(int)
    # Sem histórico: contagens em zero e status em -1 ("nada consta"); HIST_HAS
    # é o que separa "nada consta" de "nunca foi cliente".
    out[["HIST_N_ACCOUNTS", "HIST_MONTHS", "HIST_MONTHS_30PLUS", "HIST_SHARE_LATE"]] = (
        out[["HIST_N_ACCOUNTS", "HIST_MONTHS", "HIST_MONTHS_30PLUS", "HIST_SHARE_LATE"]].fillna(0)
    )
    out[["HIST_WORST_STATUS", "HIST_LAST_STATUS"]] = (
        out[["HIST_WORST_STATUS", "HIST_LAST_STATUS"]].fillna(-1)
    )
    out["HIST_SHARE_LATE"] = out["HIST_SHARE_LATE"].round(4)
    inteiras = ["HIST_N_ACCOUNTS", "HIST_MONTHS", "HIST_MONTHS_30PLUS",
                "HIST_WORST_STATUS", "HIST_LAST_STATUS"]
    out[inteiras] = out[inteiras].astype(int)
    return out


# ----------------------------------------------------------------- dataset --
def build_dataset(app: pd.DataFrame, credit: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de dados: bruto -> uma linha por conta, pronta para modelar."""
    cr = prepare_credit(credit)
    cadastro = build_features(clean_application(add_client_id(app)))
    alvo = build_target(cr)
    df = cadastro.merge(alvo, on="ID", how="inner")
    df = df.merge(build_history_features(df, cr, donos=cadastro[["ID", GROUP]]), on="ID")
    return df.sort_values("ID").reset_index(drop=True)
