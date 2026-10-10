# src/

Funções reutilizadas por mais de um notebook. Se você copiou e colou um bloco de código
entre notebooks, ele deveria estar aqui.

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | caminhos, semente, constantes |
| `data.py` | carregar e salvar datasets |
| `preprocessing.py` | limpeza, definição do alvo, feature engineering |
| `models.py` | split agrupado por cliente, pipelines, validação cruzada, busca de hiperparâmetros |
| `evaluation.py` | métricas, limiar, política de corte, IV, importância por permutação |
| `plots.py` | estilo e gráficos reutilizados |
| `predict.py` | uso do modelo treinado em novos pedidos |

Nos notebooks: `from src.config import RANDOM_STATE`
