"""
Nome do arquivo: make_predicoes.py
Data de criação: 05/11/2025
Autor: Pedro Arthur Maia Damasceno
Matrícula: 01708880

Descrição:
Gera backend/results/predicoes.csv usando a Q-Table treinada (SQLite).
Lê backend/data/test.csv (fallback para train.csv) e produz colunas y_true,y_pred.
"""

import csv
from pathlib import Path

# Importa utilidades do seu Q-Learning
from backend.qlearning_sqlite import (
    _resolve_to_path, open_db, discretize_state,
    argmax_action, _normalize_label, DEFAULT_KEYWORDS,
    RESULTS_DIR_DEFAULT, DB_DIR_DEFAULT, DATA_DIR,
)

def _find_cols(header):
    cols = [c.strip() for c in header]
    lower = [c.lower().strip() for c in cols]
    # tenta achar nomes “text/label” tolerantes
    def pick(cands, default_idx):
        for c in cands:
            if c in lower:
                return cols[lower.index(c)]
        return cols[default_idx]
    text_col  = pick(["text","texto","mensagem","message","body","conteudo"], 0)
    label_col = pick(["label","rotulo","classe","class","target","y_true"], 1 if len(cols) > 1 else 0)
    return text_col, label_col

def main(input_csv:str=None, results_dir:str=None, db_path:str=None):
    # caminhos
    results = _resolve_to_path(results_dir or str(RESULTS_DIR_DEFAULT))
    dbp     = _resolve_to_path(db_path    or str(DB_DIR_DEFAULT / "qtable.sqlite"))
    results.mkdir(parents=True, exist_ok=True)

    # escolhe CSV (test.csv -> train.csv)
    if input_csv:
        csv_path = _resolve_to_path(input_csv)
    else:
        test = (DATA_DIR / "test.csv")
        csv_path = test if test.exists() else (DATA_DIR / "train.csv")

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    # abre DB
    conn = open_db(dbp)

    # lê e gera predições
    out_path = results / "predicoes.csv"
    with open(csv_path, newline="", encoding="utf-8") as f_in, \
         open(out_path, "w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError(f"CSV sem cabeçalho: {csv_path}")
        text_col, label_col = _find_cols(reader.fieldnames)

        writer = csv.writer(f_out)
        writer.writerow(["y_true","y_pred"])

        for row in reader:
            text = (row.get(text_col) or "").strip()
            y_true = _normalize_label(row.get(label_col) or "")
            if not text or not y_true:
                continue
            s = discretize_state(text, DEFAULT_KEYWORDS)
            a, _ = argmax_action(conn, s)  # melhor ação pela Q-Table
            writer.writerow([y_true, a])

    conn.close()
    print(f"[OK] Gerado: {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="", help="CSV de avaliação (default: backend/data/test.csv; fallback: train.csv)")
    ap.add_argument("--results_dir", default=str(RESULTS_DIR_DEFAULT), help="Diretório para salvar predicoes.csv")
    ap.add_argument("--db", default=str(DB_DIR_DEFAULT / "qtable.sqlite"), help="Caminho do SQLite com a Q-Table")
    args = ap.parse_args()
    main(args.input_csv or None, args.results_dir, args.db)
