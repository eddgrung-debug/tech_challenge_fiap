#!/usr/bin/env python3
"""Pipeline completo, do dado bruto às métricas de validação, sem abrir notebook.

Uso:
    python run_pipeline.py            # usa os hiperparâmetros já ajustados (results/metrics/)
    python run_pipeline.py --tune     # refaz a busca de hiperparâmetros (mais lento)

Etapas (as mesmas dos notebooks 02, 03 e 04, com as mesmas funções de src/):
    1. data/raw/*.csv -> limpeza, alvo, atributos -> data/processed/dataset_tratado.csv
    2. hold-out de 20% estratificado e agrupado por cliente
    3. validação cruzada repetida (3 x 5 folds agrupados) dos candidatos
    4. escolha pela maior AUC-ROC média, com desempate a favor do modelo mais simples
       (src.models.choose_model); limiar pelo máximo F1 out-of-fold
    5. treino final no treino inteiro, avaliação única no teste
    6. modelo em results/models/, métricas em results/metrics/pipeline_metricas.json
"""

import argparse
import json

import joblib
from sklearn.base import clone

from src.config import FEATURE_SETS, GROUP, METRICS, MODELS, RANDOM_STATE, TARGET
from src.data import load_raw, save_processed
from src.evaluation import best_threshold, bootstrap_ci, score
from src.models import (PARAM_GRIDS, choose_model, cross_validate_model, get_models,
                        split_train_test, tune)
from src.preprocessing import build_dataset

CANDIDATOS = ["logistic_regression", "gradient_boosting", "random_forest"]
N_ITER = {"logistic_regression": 6, "gradient_boosting": 20, "random_forest": 8}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tune", action="store_true", help="refaz a busca de hiperparâmetros")
    args = parser.parse_args()

    print("[1/5] Dados: limpeza, alvo e atributos")
    df = build_dataset(*load_raw())
    save_processed(df)
    print(f"      {len(df):,} contas | {df[GROUP].nunique():,} clientes | "
          f"{df[TARGET].sum()} maus ({df[TARGET].mean():.2%})")

    print("[2/5] Split agrupado por cliente")
    train, test = split_train_test(df)
    assert not set(train[GROUP]) & set(test[GROUP]), "cliente nos dois lados do split"
    feats = FEATURE_SETS["cadastro_historico"]
    modelos = get_models(feats)

    arq_params = METRICS / "melhores_hiperparametros.json"
    if args.tune or not arq_params.exists():
        print("[3/5] Busca de hiperparâmetros")
        params = {n: tune(modelos[n], PARAM_GRIDS[n], train, feats, n_iter=N_ITER[n]).best_params_
                  for n in CANDIDATOS}
        METRICS.mkdir(parents=True, exist_ok=True)
        json.dump(params, open(arq_params, "w"), indent=2)
    else:
        print("[3/5] Hiperparâmetros lidos de", arq_params.relative_to(METRICS.parents[1]))
        params = json.load(open(arq_params))

    print("[4/5] Validação cruzada repetida (agrupada por cliente)")
    cv, oof, candidatos, folds = {}, {}, {}, {}
    for nome in CANDIDATOS:
        candidatos[nome] = clone(modelos[nome]).set_params(**params[nome])
        oof[nome], tab = cross_validate_model(candidatos[nome], train, feats)
        folds[nome] = tab["roc_auc"].to_numpy()
        cv[nome] = {"roc_auc": tab["roc_auc"].mean(), "roc_auc_dp": tab["roc_auc"].std(),
                    "pr_auc": tab["pr_auc"].mean(), "ks": tab["ks"].mean()}
        print(f"      {nome:20s} AUC {cv[nome]['roc_auc']:.3f} ± {cv[nome]['roc_auc_dp']:.3f} | "
              f"PR-AUC {cv[nome]['pr_auc']:.3f} | KS {cv[nome]['ks']:.3f}")
    escolhido, _ = choose_model(folds)
    print(f"      maior AUC média: {max(cv, key=lambda n: cv[n]['roc_auc'])} | "
          f"escolhido pela regra de desempate (empate -> mais simples): {escolhido}")
    limiar = best_threshold(train[TARGET], oof[escolhido])

    print(f"[5/5] Treino final ({escolhido}) e avaliação no teste")
    final = clone(candidatos[escolhido]).fit(train[feats], train[TARGET])
    p = final.predict_proba(test[feats])[:, 1]
    metricas = score(test[TARGET], (p >= limiar).astype(int), p)
    lo, hi = bootstrap_ci(test[TARGET], p, test[GROUP])
    for k, v in metricas.items():
        print(f"      {k:18s} {v:.4f}")
    print(f"      roc_auc IC 95%     [{lo:.3f}; {hi:.3f}]")

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": final, "atributos": feats, "limiar": limiar},
                MODELS / "modelo_final.joblib")
    json.dump({"modelo_escolhido": escolhido, "limiar": limiar, "random_state": RANDOM_STATE,
               "validacao_cruzada": cv, "teste": {**metricas, "roc_auc_ic95": [lo, hi]}},
              open(METRICS / "pipeline_metricas.json", "w"), indent=2)
    print("Modelo salvo em results/models/modelo_final.joblib")


if __name__ == "__main__":
    main()
