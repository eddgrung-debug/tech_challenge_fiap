"""Configuração central do projeto: caminhos, semente e constantes.

Importe daqui em todos os notebooks para que os resultados sejam reproduzíveis.
"""

from pathlib import Path

# --- Semente -----------------------------------------------------------------
# Usada em TODO ponto que envolve aleatoriedade: split, modelos, CV, bootstrap.
RANDOM_STATE = 42

# --- Caminhos ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"
METRICS = RESULTS / "metrics"

# --- Dataset -----------------------------------------------------------------
RAW_APPLICATION = DATA_RAW / "application_record.csv"   # cadastro dos solicitantes
RAW_CREDIT = DATA_RAW / "credit_record.csv"             # histórico mensal de pagamento
PROCESSED_NAME = "dataset_tratado"

TARGET = "MAU_PAGADOR"        # 1 = mau pagador, 0 = bom pagador
GROUP = "ID_CLIENTE"          # mesma pessoa em várias contas -> mesmo grupo no split

# --- Definição do alvo -------------------------------------------------------
# Mau pagador = atraso de 60 dias ou mais (STATUS >= 2) em algum dos primeiros
# JANELA_MESES meses de vida da conta. Só entram contas com a janela completa.
JANELA_MESES = 12
STATUS_MAU = 2

# STATUS do credit_record convertido em escala ordinal de severidade.
SEVERIDADE = {"X": -1, "C": -1, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}

# --- Split -------------------------------------------------------------------
TEST_SIZE = 0.2      # 1 de 5 folds estratificados e agrupados por cliente
CV_FOLDS = 5
CV_REPEATS = 3

# --- Atributos ---------------------------------------------------------------
NUM_CADASTRO = [
    "CNT_CHILDREN", "CNT_FAM_MEMBERS", "AMT_INCOME_TOTAL", "INCOME_PER_MEMBER",
    "AGE_YEARS", "YEARS_EMPLOYED", "EMPLOYED_SHARE",
]
BIN_CADASTRO = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_WORK_PHONE",
    "FLAG_PHONE", "FLAG_EMAIL", "FLAG_NO_JOB",
]
CAT_CADASTRO = [
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]
NUM_HISTORICO = [
    "HIST_N_ACCOUNTS", "HIST_MONTHS", "HIST_WORST_STATUS", "HIST_LAST_STATUS",
    "HIST_MONTHS_30PLUS", "HIST_SHARE_LATE",
]
BIN_HISTORICO = ["HIST_HAS"]

# Variáveis com cauda longa: entram em log nos modelos sensíveis à escala.
LOG_FEATURES = ["AMT_INCOME_TOTAL", "INCOME_PER_MEMBER", "HIST_MONTHS", "HIST_N_ACCOUNTS",
                "HIST_MONTHS_30PLUS"]

FEATURE_SETS = {
    # Só o que está na ficha cadastral: vale para qualquer solicitante.
    "cadastro": NUM_CADASTRO + BIN_CADASTRO + CAT_CADASTRO,
    # Ficha + comportamento do mesmo cliente em contas anteriores à solicitação.
    "cadastro_historico": NUM_CADASTRO + BIN_CADASTRO + CAT_CADASTRO
                          + NUM_HISTORICO + BIN_HISTORICO,
}

# Rótulos em português para gráficos e tabelas.
ROTULOS = {
    "CODE_GENDER": "Gênero", "FLAG_OWN_CAR": "Possui carro",
    "FLAG_OWN_REALTY": "Possui imóvel", "CNT_CHILDREN": "Nº de filhos",
    "CNT_FAM_MEMBERS": "Tamanho da família", "AMT_INCOME_TOTAL": "Renda anual",
    "INCOME_PER_MEMBER": "Renda por membro da família", "AGE_YEARS": "Idade (anos)",
    "YEARS_EMPLOYED": "Tempo de emprego (anos)",
    "EMPLOYED_SHARE": "Fração da vida adulta empregado",
    "FLAG_NO_JOB": "Sem vínculo de emprego", "FLAG_WORK_PHONE": "Telefone comercial",
    "FLAG_PHONE": "Telefone fixo", "FLAG_EMAIL": "E-mail",
    "NAME_INCOME_TYPE": "Tipo de renda", "NAME_EDUCATION_TYPE": "Escolaridade",
    "NAME_FAMILY_STATUS": "Estado civil", "NAME_HOUSING_TYPE": "Tipo de moradia",
    "OCCUPATION_TYPE": "Ocupação",
    "HIST_HAS": "Tem conta anterior", "HIST_N_ACCOUNTS": "Nº de contas anteriores",
    "HIST_MONTHS": "Meses de histórico anterior",
    "HIST_WORST_STATUS": "Pior atraso anterior",
    "HIST_LAST_STATUS": "Situação no mês anterior ao pedido",
    "HIST_MONTHS_30PLUS": "Meses anteriores com atraso 30+",
    "HIST_SHARE_LATE": "Fração dos meses anteriores com algum atraso",
}

# Paleta dos gráficos (validada para daltonismo): bom, mau, apoio.
COR_BOM = "#2a78d6"
COR_MAU = "#eb6834"
COR_APOIO = "#1baf7a"
COR_NEUTRA = "#8a8a86"
