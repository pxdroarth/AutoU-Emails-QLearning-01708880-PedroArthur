"""
Nome do arquivo: metrics_eval.py
Data de criação: 29/10/2025
Autor: Pedro Arthur Maia (pxdroarth)
Matrícula: 01708880

Descrição:
Avaliação da política aprendida na Q-Table (SQLite). Gera métricas de
classificação e gráficos (matriz de confusão, curva de recompensa por episódio,
e evolução do epsilon).

Funcionalidades:
- Carrega Q(s,a) e infere com política greedy (argmax_a Q(s,a))
- Calcula Accuracy, Precision, Recall, F1 (por classe, macro, weighted) e MCC
- Salva Confusion Matrix (PNG/CSV), relatório em JSON/CSV e predicoes.csv
- Plota recompensas por episódio e ε-curve (train_logs.json OU rewards.csv/epsilon.csv)
"""

from __future__ import annotations
import os, sys, sqlite3, argparse, json, csv
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
import numpy as np

# Backend headless para ambientes sem display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================= Constantes/Paths =======================
ACTIONS = ["produtivo", "improdutivo"]

DEFAULT_KEYWORDS = [
    "reunião", "agenda", "cronograma", "entrega", "prazo", "orçamento",
    "contrato", "proposta", "call", "zoom", "meet", "alinhamento",
    "fatura", "nota", "pagamento", "parceria", "currículo", "vaga",
    "suporte", "erro", "bug", "falha", "senha", "acesso", "boleto",
    "protocolo", "anexo", "documentação", "validação", "cobrança",
]

BACKEND_DIR: Path = Path(__file__).resolve().parent          # backend/
DATA_DIR: Path = BACKEND_DIR / "data"
RESULTS_DIR_DEFAULT: Path = BACKEND_DIR / "results"
DB_DIR_DEFAULT: Path = BACKEND_DIR / "db"

# ======================= Utils de caminho/I/O =======================
def _resolve_to_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    # tenta CWD e depois relativo a backend/
    cand = (Path.cwd() / p).resolve()
    if cand.exists() or cand.parent.exists():
        return cand
    return (BACKEND_DIR / p).resolve()

def safe_mkdir(p: Optional[Path | str]) -> None:
    if not p:
        return
    Path(p).mkdir(parents=True, exist_ok=True)

def _normalize_label(lbl: str) -> str:
    if lbl is None:
        return ""
    x = lbl.strip().lower()
    mapping = {
        "produtivo": "produtivo",
        "improdutivo": "improdutivo",
        "positivo": "produtivo",
        "negativo": "improdutivo",
        "relevante": "produtivo",
        "irrelevante": "improdutivo",
    }
    return mapping.get(x, x)

def _guess_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    lc = [c.lower().strip() for c in cols]
    for cand in candidates:
        if cand in lc:
            return cols[lc.index(cand)]
    return None

def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"CSV sem cabeçalho: {path}")
        cols = [c.strip() for c in r.fieldnames]
        tcol = "text" if "text" in cols else _guess_col(cols, ["texto","conteudo","mensagem","message","body"])
        lcol = "label" if "label" in cols else _guess_col(cols, ["rotulo","classe","class","target"])
        if tcol is None or lcol is None:
            raise ValueError(f"Não encontrei colunas de texto/label no CSV {path}. Campos: {cols}")
        for row in r:
            txt = (row.get(tcol) or "").strip()
            lbl = _normalize_label(row.get(lcol) or "")
            if txt and lbl:
                rows.append({"text": txt, "label": lbl})
    return rows

# ======================= Discretização (igual ao treino) =======================
def bucket_len(n: int) -> str:
    if n < 60: return "len:short"
    if n < 200: return "len:medium"
    if n < 600: return "len:long"
    return "len:xl"

def has_any_keyword(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)

def discretize_state(text: str, keywords: List[str]) -> str:
    t = text.strip()
    blen = bucket_len(len(t))
    kw = "kw:yes" if has_any_keyword(t, keywords) else "kw:no"
    return f"{blen}|{kw}"

# ======================= SQLite / Política =======================
def open_db(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))

def get_q(conn: sqlite3.Connection, state: str, action: str) -> float:
    cur = conn.execute("SELECT q_value FROM QStateAction WHERE state=? AND action=?", (state, action))
    row = cur.fetchone()
    return row[0] if row else 0.0

def argmax_action(conn: sqlite3.Connection, state: str) -> str:
    # Determinístico em empate: escolhe primeira na ordem de ACTIONS
    best_a, best_q = ACTIONS[0], get_q(conn, state, ACTIONS[0])
    for a in ACTIONS[1:]:
        q = get_q(conn, state, a)
        if q > best_q:
            best_a, best_q = a, q
    return best_a

# ======================= Métricas =======================
def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> np.ndarray:
    idx = {lab:i for i, lab in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M

def precision_recall_f1(y_true: List[str], y_pred: List[str], pos_label: str) -> Tuple[float,float,float,int]:
    tp = sum((yt==pos_label and yp==pos_label) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt!=pos_label and yp==pos_label) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt==pos_label and yp!=pos_label) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt!=pos_label and yp!=pos_label) for yt, yp in zip(y_true, y_pred))
    precision = tp / (tp+fp) if (tp+fp) else 0.0
    recall    = tp / (tp+fn) if (tp+fn) else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) else 0.0
    support   = tp + fn
    return precision, recall, f1, support

def mcc_binary(y_true: List[str], y_pred: List[str], pos_label: str) -> float:
    tp = sum((yt==pos_label and yp==pos_label) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt!=pos_label and yp!=pos_label) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt!=pos_label and yp==pos_label) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt==pos_label and yp!=pos_label) for yt, yp in zip(y_true, y_pred))
    num = tp*tn - fp*fn
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (num/den) if den else 0.0

def classification_report(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    per_label: Dict[str, Dict[str, float]] = {}
    macro_p = macro_r = macro_f1 = 0.0
    total = len(y_true)
    weights = []
    weighted_p = weighted_r = weighted_f1 = 0.0

    for lab in labels:
        p, r, f1, support = precision_recall_f1(y_true, y_pred, lab)
        per_label[lab] = {"precision": p, "recall": r, "f1": f1, "support": support}
        macro_p += p; macro_r += r; macro_f1 += f1
        weights.append(support)
        weighted_p += p * support
        weighted_r += r * support
        weighted_f1 += f1 * support

    n_labels = len(labels) if labels else 1
    macro = {"precision": macro_p/n_labels, "recall": macro_r/n_labels, "f1": macro_f1/n_labels}
    if total > 0:
        weighted = {"precision": weighted_p/total, "recall": weighted_r/total, "f1": weighted_f1/total}
    else:
        weighted = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {"per_label": per_label, "macro": macro, "weighted": weighted}

# ======================= Plot helpers =======================
def plot_confusion_matrix(M: np.ndarray, labels: List[str], out_png: Path) -> None:
    fig = plt.figure(figsize=(4.5,4.5))
    plt.imshow(M, interpolation="nearest")
    plt.title("Matriz de confusão")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            plt.text(j, i, str(M[i, j]), ha='center', va='center')
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def plot_curve_xy(xs, ys, title, xlabel, ylabel, out_png: Path) -> None:
    fig = plt.figure(figsize=(5,3))
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(out_png)
    plt
