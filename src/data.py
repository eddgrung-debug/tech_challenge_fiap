"""Carregamento e persistência de dados."""

import pandas as pd

from src.config import DATA_PROCESSED, PROCESSED_NAME, RAW_APPLICATION, RAW_CREDIT


def _check(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Veja data/README.md para baixar o dataset."
        )


def load_application() -> pd.DataFrame:
    """Lê a base cadastral bruta (uma linha por solicitação), sem transformação."""
    _check(RAW_APPLICATION)
    return pd.read_csv(RAW_APPLICATION)


def load_credit() -> pd.DataFrame:
    """Lê o histórico mensal de pagamento bruto (uma linha por conta e mês)."""
    _check(RAW_CREDIT)
    return pd.read_csv(RAW_CREDIT, dtype={"STATUS": "string"})


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (application_record, credit_record)."""
    return load_application(), load_credit()


def save_processed(df: pd.DataFrame, name: str = PROCESSED_NAME) -> None:
    """Grava um dataset tratado em data/processed/ (CSV, sem dependências extras)."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / f"{name}.csv", index=False)


def load_processed(name: str = PROCESSED_NAME) -> pd.DataFrame:
    path = DATA_PROCESSED / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode notebooks/02_preprocessamento.ipynb "
            "ou `python run_pipeline.py` antes."
        )
    return pd.read_csv(path, dtype={"ID_CLIENTE": "string"})
