"""Uso do modelo treinado para classificar novos pedidos."""

import joblib
import pandas as pd

from src.config import MODELS


def classificar(pedidos: pd.DataFrame, caminho=MODELS / "modelo_final.joblib") -> pd.DataFrame:
    """Devolve escore de risco e decisão para cada pedido.

    `pedidos` deve ter as colunas produzidas por `src.preprocessing.build_dataset`
    (ficha tratada + atributos HIST_*). O escore ordena os pedidos por risco; não é
    uma probabilidade calibrada, porque o treino usa pesos de classe.
    """
    pacote = joblib.load(caminho)
    escore = pacote["modelo"].predict_proba(pedidos[pacote["atributos"]])[:, 1]
    return pd.DataFrame({
        "escore_risco": escore,
        "decisao": ["reprovar" if e >= pacote["limiar"] else "aprovar" for e in escore],
    }, index=pedidos.index)
