# data/

**Nada aqui é versionado.** O `.gitignore` bloqueia o conteúdo destas pastas de propósito:
datasets em Git incham o repositório e frequentemente violam a licença da fonte.

| Pasta | Conteúdo |
|---|---|
| `raw/` | arquivos originais, exatamente como baixados da fonte — nunca editados |
| `processed/` | saída do notebook 02 ou do `run_pipeline.py` (`dataset_tratado.csv`) |

## Como obter

1. Baixe em: https://drive.google.com/file/d/1z4yEyiCE_CGCWbvAAZQZSz-5-E5T5eYd/view?usp=sharing
   (link do enunciado). Os arquivos correspondem ao dataset público
   *Credit Card Approval Prediction*, do Kaggle.
2. Salve como:
   - `data/raw/application_record.csv` (438.557 linhas + cabeçalho, 18 colunas, ~54 MB)
   - `data/raw/credit_record.csv` (1.048.575 linhas + cabeçalho, 3 colunas, ~15 MB)
3. Checksum (SHA-256) dos arquivos usados neste projeto:

```
4833f502d02ad94295de3ffe74f665e726a4b04342d2e94f8cec41dce951925b  application_record.csv
ba0006a4734f74422d68b0a7132ad591850be0a6affb535eb1042d207fe4b27e  credit_record.csv
```

Confira com `shasum -a 256 data/raw/*.csv` (Linux/macOS) ou
`Get-FileHash data\raw\*.csv` (Windows PowerShell).
