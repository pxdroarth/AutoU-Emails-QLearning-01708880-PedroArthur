"""
Nome do arquivo: routes_rl.py
Data de criação: 29/10/2025
Autor: Pedro Arthur Maia Damasceno
Matrícula: 01708880

Descrição:
Endpoints FastAPI para integrar o RL (Q-Table) ao projeto:
- POST /rl/train      → roda treino (subprocess) no qlearning_sqlite.py
- GET  /rl/metrics    → roda avaliação e retorna metrics.json
- POST /rl/predict    → inferência greedy na Q-Table (sem rodar scripts)
- POST /rl/feedback   → anexa exemplo rotulado ao CSV (dataset legado)
- POST /rl/feedback_h → anexa feedback humano no CSV dedicado (csv_foruseres.csv) com scoring
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# =========================
# Paths (resolvidos a partir deste arquivo)
# =========================
BASE: Path = Path(__file__).parent.resolve()          # .../backend
DATA_DIR: Path = BASE / "data"                        # backend/data
DB_PATH: Path = BASE / "db" / "qtable.sqlite"         # backend/db/qtable.sqlite
RESULTS_DIR: Path = BASE / "results"                  # backend/results
TRAIN_SCRIPT: Path = BASE / "qlearning_sqlite.py"     # backend/qlearning_sqlite.py
EVAL_SCRIPT: Path = BASE / "metrics_eval.py"          # backend/metrics_eval.py
CSV_FEEDBACK_HUMANO: Path = DATA_DIR / "csv_foruseres.csv"  # feedback humano

# =========================
# Domínio (deve refletir qlearning_sqlite.py)
# =========================
ACTIONS: List[str] = ["produtivo", "improdutivo"]

DEFAULT_KEYWORDS: List[str] = [
    "reunião", "agenda", "cronograma", "entrega", "prazo", "orçamento",
    "contrato", "proposta", "call", "zoom", "meet", "alinhamento",
    "fatura", "nota", "pagamento", "parceria", "currículo", "vaga",
    "suporte", "erro", "bug", "falha", "senha", "acesso", "boleto",
    "protocolo", "anexo", "documentação", "validação", "cobrança",
]

def bucket_len(n: int) -> str:
    if n < 60:
        return "len:short"
    if n < 200:
        return "len:medium"
    if n < 600:
        return "len:long"
    return "len:xl"

def has_any_keyword(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)

def discretize_state(text: str, keywords: List[str]) -> str:
    t = text.strip()
    blen = bucket_len(len(t))
    kw = "kw:yes" if has_any_keyword(t, keywords) else "kw:no"
    return f"{blen}|{kw}"

# =========================
# SQLite / Q-Table helpers
# =========================
DDL = """
CREATE TABLE IF NOT EXISTS QStateAction(
  state TEXT NOT NULL,
  action TEXT NOT NULL,
  q_value REAL NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(state, action)
);
"""

def _open_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(DDL)
    return conn

def _get_q(conn: sqlite3.Connection, state: str, action: str) -> float:
    cur = conn.execute(
        "SELECT q_value FROM QStateAction WHERE state=? AND action=?",
        (state, action),
    )
    row = cur.fetchone()
    return float(row[0]) if row else 0.0

def _argmax_action(conn: sqlite3.Connection, state: str) -> Tuple[str, float, List[Tuple[str, float]]]:
    vals = [(a, _get_q(conn, state, a)) for a in ACTIONS]
    max_q = max(v for _, v in vals)
    # desempate determinístico pela ordem declarada
    for a, v in vals:
        if v == max_q:
            return a, max_q, vals
    # fallback (nunca deve ocorrer)
    return ACTIONS[0], 0.0, [(ACTIONS[0], 0.0), (ACTIONS[1], 0.0)]

# =========================
# CSV helpers
# =========================
def _csv_count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        # subtrai header
        return max(0, sum(1 for _ in f) - 1)

def _csv_append_row(path: Path, header: List[str], row: List[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(header)
        w.writerow(row)

# =========================
# Schemas
# =========================
class TrainRequest(BaseModel):
    train_csv: str = "backend/data/train.csv"
    use_feedback_csv: bool = True
    episodes: int = 20
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    alpha: float = 0.1
    gamma: float = 0.0
    shuffle: bool = True
    extra_keywords: Optional[str] = None
    auto_split: bool = False

class TrainResponse(BaseModel):
    ok: bool
    results_dir: str
    db: str
    logs_file: str

class MetricsResponse(BaseModel):
    ok: bool
    metrics: dict

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    q_values: dict
    confidence: float  # margem normalizada entre as ações
    state: str
    generated_at: str

class FeedbackRequest(BaseModel):
    text: str
    label: str  # "produtivo" | "improdutivo"
    csv_path: str = "backend/data/train.csv"  # compat legado

class FeedbackHumanoRequest(BaseModel):
    text: str
    gold_label: str                 # "produtivo" | "improdutivo"
    pred_label: Optional[str] = None
    source: str = "frontend"

class FeedbackResponse(BaseModel):
    ok: bool
    csv_path: str
    total_after: int

# =========================
# Router
# =========================
router = APIRouter(prefix="/rl", tags=["RL"])

# -------------------------
# /rl/train
# -------------------------
@router.post("/train", response_model=TrainResponse)
def rl_train(req: TrainRequest):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # concatena CSV de feedback humano se existir e for solicitado
    combined_train = req.train_csv
    if req.use_feedback_csv and CSV_FEEDBACK_HUMANO.exists():
        combined_train = f"{req.train_csv},{str(CSV_FEEDBACK_HUMANO)}"

    args = [
        sys.executable, str(TRAIN_SCRIPT),
        "--train_csv", combined_train,
        "--db", str(DB_PATH),
        "--results_dir", str(RESULTS_DIR),
        "--episodes", str(req.episodes),
        "--epsilon_start", str(req.epsilon_start),
        "--epsilon_min", str(req.epsilon_min),
        "--epsilon_decay", str(req.epsilon_decay),
        "--alpha", str(req.alpha),
        "--gamma", str(req.gamma),
    ]
    if req.shuffle:
        args.append("--shuffle")
    if req.extra_keywords:
        args += ["--extra_keywords", req.extra_keywords]
    if req.auto_split:
        args.append("--auto_split")

    try:
        subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(BASE.parent),  # raiz do repo
        )
    except subprocess.CalledProcessError as e:
        # inclui stdout/stderr para diagnóstico
        detail = e.stderr or e.stdout or "Treino falhou sem logs."
        raise HTTPException(status_code=500, detail=f"Treino falhou: {detail}")

    return TrainResponse(
        ok=True,
        results_dir=str(RESULTS_DIR),
        db=str(DB_PATH),
        logs_file=str(RESULTS_DIR / "train_logs.json"),
    )

# -------------------------
# /rl/metrics
# -------------------------
@router.get("/metrics", response_model=MetricsResponse)
def rl_metrics(test_csv: str = "backend/data/test.csv"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable, str(EVAL_SCRIPT),
        "--test_csv", test_csv,
        "--db", str(DB_PATH),
        "--results_dir", str(RESULTS_DIR),
    ]
    try:
        subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(BASE.parent),
        )
    except subprocess.CalledProcessError as e:
        detail = e.stderr or e.stdout or "Avaliação falhou sem logs."
        raise HTTPException(status_code=500, detail=f"Avaliação falhou: {detail}")

    metrics_path = RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=500, detail="metrics.json não encontrado.")
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return MetricsResponse(ok=True, metrics=metrics)

# -------------------------
# /rl/predict
# -------------------------
@router.post("/predict", response_model=PredictResponse)
def rl_predict(req: PredictRequest):
    keywords = list(DEFAULT_KEYWORDS)
    state = discretize_state(req.text, keywords)
    conn = _open_db()
    label, max_q, vals = _argmax_action(conn, state)
    conn.close()

    qmap = {a: float(q) for a, q in vals}
    sorted_vals = sorted(qmap.values(), reverse=True)
    if len(sorted_vals) >= 2:
        margin = sorted_vals[0] - sorted_vals[1]
        denom = abs(sorted_vals[0]) + abs(sorted_vals[1]) + 1e-6
    else:
        margin = abs(sorted_vals[0]) if sorted_vals else 0.0
        denom = abs(sorted_vals[0]) + 1e-6 if sorted_vals else 1.0
    conf = float(max(0.0, min(1.0, margin / denom)))

    return PredictResponse(
        label=label,
        q_values=qmap,
        confidence=conf,
        state=state,
        generated_at=datetime.now(UTC).isoformat(),
    )

# -------------------------
# /rl/feedback (legado)
# -------------------------
@router.post("/feedback", response_model=FeedbackResponse)
def rl_feedback(req: FeedbackRequest):
    lbl = (req.label or "").strip().lower()
    if lbl not in {"produtivo", "improdutivo"}:
        raise HTTPException(status_code=400, detail="label deve ser 'produtivo' ou 'improdutivo'")

    csv_path = Path(req.csv_path)
    _csv_append_row(csv_path, ["text", "label"], [req.text, lbl])
    total = _csv_count_rows(csv_path)

    return FeedbackResponse(ok=True, csv_path=str(csv_path), total_after=total)

# -------------------------
# /rl/feedback_h (feedback humano com scoring)
# -------------------------
@router.post("/feedback_h", response_model=FeedbackResponse)
def rl_feedback_h(req: FeedbackHumanoRequest):
    """
    Registra feedback humano em backend/data/csv_foruseres.csv com as colunas:
    text,label,pred,correct,reward,source,created_at

    - label = gold_label (do humano)
    - pred  = predição atual (opcional)
    - correct/reward: correct=1 (+1.0) se pred==label; caso contrário 0 e -1.0
    """
    lbl = (req.gold_label or "").strip().lower()
    if lbl not in {"produtivo", "improdutivo"}:
        raise HTTPException(status_code=400, detail="gold_label deve ser 'produtivo' ou 'improdutivo'")

    pred = (req.pred_label or "").strip().lower()
    correct = 1 if (pred and pred == lbl) else 0
    reward = 1.0 if correct == 1 else -1.0
    now = datetime.now(UTC).isoformat()

    _csv_append_row(
        CSV_FEEDBACK_HUMANO,
        ["text", "label", "pred", "correct", "reward", "source", "created_at"],
        [req.text, lbl, pred, correct, reward, req.source, now],
    )
    total = _csv_count_rows(CSV_FEEDBACK_HUMANO)

    return FeedbackResponse(ok=True, csv_path=str(CSV_FEEDBACK_HUMANO), total_after=total)
    