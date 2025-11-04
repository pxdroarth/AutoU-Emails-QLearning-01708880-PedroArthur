"""
Nome do arquivo: qlearning_sqlite.py
Data de criação: 29/10/2025
Autor: Pedro Arthur Maia Damasceno
Matrícula: 01708880

Descrição:
Treino de Q-Learning tabular em SQLite para classificar e-mails (produtivo/improdutivo)
por exploração/aproveitamento (ε-greedy) usando estados discretizados simples
a partir do texto. Atualiza Q-Table com a fórmula canônica.

Funcionalidades:
- Aceita múltiplos CSVs (lista separada por vírgulas) ou uma pasta com .csv
- Pode fazer estratificação automática (auto-split) de um CSV único
- Colunas customizáveis (text_col, label_col)
- Q-Table em SQLite (chave composta: state, action)
- Política ε-greedy com ε-decay (ε_t = max(ε_min, ε_0 * decay^t))
- Recompensa: +1 acerto, -1 erro (trocável por custo customizado)
- Discretização leve de estado (comprimento, presença de keywords)
- Logs de treino: recompensas por episódio e evolução de ε
- Geração automática de banco e pastas (db/, results/)
- Pós-processamento automático: curvas/plots e (opcional) métricas via endpoint
"""

from __future__ import annotations
import os, sqlite3, argparse, json, random, glob, csv, sys, subprocess
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional, List, Tuple, Dict

# ------------------------ Constantes ------------------------
ACTIONS = ["produtivo", "improdutivo"]

DEFAULT_KEYWORDS = [
    "reunião", "agenda", "cronograma", "entrega", "prazo", "orçamento",
    "contrato", "proposta", "call", "zoom", "meet", "alinhamento",
    "fatura", "nota", "pagamento", "parceria", "currículo", "vaga",
    "suporte", "erro", "bug", "falha", "senha", "acesso", "boleto",
    "protocolo", "anexo", "documentação", "validação", "cobrança",
]

# Diretórios base resolvidos a partir deste arquivo (independente do CWD)
BACKEND_DIR: Path = Path(__file__).resolve().parent         # backend/
DATA_DIR: Path = BACKEND_DIR / "data"                       # backend/data/
RESULTS_DIR_DEFAULT: Path = BACKEND_DIR / "results"         # backend/results/
DB_DIR_DEFAULT: Path = BACKEND_DIR / "db"                   # backend/db/


# ------------------------ Utilidades de I/O ------------------------
def safe_mkdir(p: Optional[Path | str]) -> None:
    if not p:
        return
    Path(p).mkdir(parents=True, exist_ok=True)

def _normalize_label(lbl: str) -> str:
    if lbl is None:
        return ""
    x = lbl.strip().lower()
    # Mapeamentos úteis (caso recebam "positivo/negativo", "relevante/irrelevante" etc.)
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

def load_csv_generic(path: Path, text_col="text", label_col="label") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"CSV sem cabeçalho: {path}")
        cols = [c.strip() for c in r.fieldnames]
        tcol = text_col if text_col in cols else _guess_col(cols, ["text","texto","conteudo","mensagem","message","body"])
        lcol = label_col if label_col in cols else _guess_col(cols, ["label","rotulo","classe","class","target"])
        if tcol is None or lcol is None:
            raise ValueError(f"Não encontrei colunas de texto/label no CSV {path}. Campos: {cols}")
        for row in r:
            txt = (row.get(tcol) or "").strip()
            lbl = _normalize_label(row.get(lcol) or "")
            if txt and lbl:
                rows.append({"text": txt, "label": lbl})
    return rows

def _resolve_to_path(p: str | Path) -> Path:
    """Resolve caminho relativo para absoluto. Mantém absolutos."""
    p = Path(p)
    if not p.is_absolute():
        # Primeiro tenta relativo ao CWD; se não existir, tenta relativo ao BACKEND_DIR
        cand = (Path.cwd() / p).resolve()
        if cand.exists() or cand.parent.exists():
            return cand
        return (BACKEND_DIR / p).resolve()
    return p

def _locate_default_train_csv() -> Path:
    """
    Fallback quando usuário não passa --train_csv/--train_dir.
    Usa backend/data/train.csv se existir; caso contrário, primeiro *.csv de backend/data/.
    """
    cand = (DATA_DIR / "train.csv").resolve()
    if cand.exists():
        return cand
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if csvs:
        return csvs[0].resolve()
    raise FileNotFoundError(f"Nenhum CSV de treino encontrado. Verifique {DATA_DIR}.")

def _save_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text","label"])
        for r in rows:
            w.writerow([r["text"], r["label"]])

def stratified_split(rows: List[Dict[str, str]], test_size=0.2, seed=42) -> Tuple[List[Dict[str,str]], List[Dict[str,str]]]:
    random.seed(seed)
    by_label: Dict[str, List[Dict[str,str]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    train, test = [], []
    for lbl, grp in by_label.items():
        idxs = list(range(len(grp)))
        random.shuffle(idxs)
        cut = max(1, int(len(idxs)*test_size))
        test_idxs = set(idxs[:cut])
        for i, ex in enumerate(grp):
            (test if i in test_idxs else train).append(ex)
    return train, test

def gather_train_rows(args) -> List[Dict[str, str]]:
    files: List[Path] = []

    # 1) Se passou diretório, pega todos .csv
    if args.train_dir:
        dirp = _resolve_to_path(args.train_dir)
        if not dirp.exists() or not dirp.is_dir():
            raise FileNotFoundError(f"Pasta não encontrada: {dirp}")
        files.extend(sorted(dirp.glob("*.csv")))

    # 2) Se passou lista separada por vírgula
    if args.train_csv:
        for part in str(args.train_csv).split(","):
            part = part.strip()
            if not part:
                continue
            p = _resolve_to_path(part)
            if p.is_dir():
                files.extend(sorted(p.glob("*.csv")))
            elif p.is_file():
                files.append(p)
            else:
                raise FileNotFoundError(f"Arquivo/pasta não encontrado: {p}")

    # 3) Fallback: backend/data/train.csv (ou 1º .csv de backend/data/)
    if not files:
        try:
            files = [ _locate_default_train_csv() ]
            print(f"[INFO] Usando fallback de treino: {files[0]}")
        except FileNotFoundError as e:
            raise FileNotFoundError("Nenhum CSV de treino encontrado (use --train_csv ou --train_dir).") from e

    rows: List[Dict[str,str]] = []
    for fp in files:
        rows.extend(load_csv_generic(fp, args.text_col, args.label_col))

    # Filtra labels alvo
    rows = [r for r in rows if r["label"] in {"produtivo","improdutivo"}]

    # Auto-split opcional
    if args.auto_split:
        split_dir = _resolve_to_path(args.results_dir) / "data_split"
        safe_mkdir(split_dir)
        train_rows, test_rows = stratified_split(rows, test_size=args.test_size, seed=args.seed)
        _save_rows(split_dir / "train.csv", train_rows)
        _save_rows(split_dir / "test.csv", test_rows)
        print(f"[auto-split] Gerados: {split_dir/'train.csv'} e {split_dir/'test.csv'} (test_size={args.test_size})")
        return train_rows

    return rows


# ------------------------ Discretização ------------------------
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


# ------------------------ SQLite Q-Table ------------------------
DDL = """
CREATE TABLE IF NOT EXISTS QStateAction(
    state TEXT NOT NULL,
    action TEXT NOT NULL,
    q_value REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(state, action)
);
"""

def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute(DDL)
    return conn

def get_q(conn: sqlite3.Connection, state: str, action: str) -> float:
    cur = conn.execute("SELECT q_value FROM QStateAction WHERE state=? AND action=?", (state, action))
    row = cur.fetchone()
    return row[0] if row else 0.0

def set_q(conn: sqlite3.Connection, state: str, action: str, qv: float) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute(
        """
        INSERT INTO QStateAction(state,action,q_value,updated_at)
        VALUES(?,?,?,?)
        ON CONFLICT(state,action) DO UPDATE
        SET q_value=excluded.q_value, updated_at=excluded.updated_at
        """,
        (state, action, float(qv), now),
    )

def argmax_action(conn: sqlite3.Connection, state: str) -> Tuple[str, float]:
    values = [(a, get_q(conn, state, a)) for a in ACTIONS]
    max_q = max(v for _, v in values)
    best = [a for a, v in values if v == max_q]
    return random.choice(best), max_q


# ------------------------ Política ε-greedy ------------------------
def epsilon_at(t: int, eps0: float, eps_min: float, decay: float) -> float:
    return max(eps_min, eps0 * (decay ** t))

def choose_action(conn: sqlite3.Connection, state: str, eps_t: float) -> Tuple[str, str]:
    if random.random() < eps_t:
        return random.choice(ACTIONS), "explore"
    else:
        a, _ = argmax_action(conn, state)
        return a, "exploit"


# ------------------------ Recompensa ------------------------
def reward_from_pred(pred: str, gold: str) -> float:
    return 1.0 if pred == gold else -1.0


# ------------------------ Treino ------------------------
def train(args) -> None:
    # Resolve/garante pastas
    db_path = _resolve_to_path(args.db)
    results_dir = _resolve_to_path(args.results_dir)
    safe_mkdir(db_path.parent)
    safe_mkdir(results_dir)

    # Carrega dados
    data = gather_train_rows(args)
    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(data)

    # Keywords (default + arquivo opcional)
    keywords = list(DEFAULT_KEYWORDS)
    if args.extra_keywords:
        extra_path = _resolve_to_path(args.extra_keywords)
        if extra_path.exists():
            with open(extra_path, "r", encoding="utf-8") as f:
                kws = [x.strip().lower() for x in f if x.strip()]
            keywords = list({*keywords, *kws})

    # Banco
    conn = open_db(db_path)

    # Hiperparâmetros
    alpha = float(args.alpha)
    gamma = float(args.gamma)
    eps0 = float(args.epsilon_start)
    eps_min = float(args.epsilon_min)
    decay = float(args.epsilon_decay)

    rewards_per_episode: List[float] = []
    eps_curve: List[float] = []
    t = 0  # step global

    print(f"[INFO] Treinando com {len(data)} exemplos | episodes={args.episodes}")
    print(f"[INFO] db={db_path} | results_dir={results_dir}")

    for ep in range(args.episodes):
        total_reward = 0.0

        for ex in data:
            s = discretize_state(ex["text"], keywords)
            eps_t = epsilon_at(t, eps0, eps_min, decay)
            a, _ = choose_action(conn, s, eps_t)

            r = reward_from_pred(a, ex["label"])
            total_reward += r

            q_sa = get_q(conn, s, a)
            _, max_next = argmax_action(conn, s)
            td_target = r + gamma * max_next
            q_updated = q_sa + alpha * (td_target - q_sa)
            set_q(conn, s, a, q_updated)

            t += 1
            if t % args.commit_every == 0:
                conn.commit()

        rewards_per_episode.append(total_reward)
        eps_curve.append(epsilon_at(t, eps0, eps_min, decay))
        if args.verbose:
            print(f"[EP {ep+1}/{args.episodes}] reward={total_reward:.1f} eps={eps_curve[-1]:.4f}")

    conn.commit()
    conn.close()

    # Logs JSON
    logs = {
        "alpha": alpha, "gamma": gamma,
        "epsilon_start": eps0, "epsilon_min": eps_min, "epsilon_decay": decay,
        "episodes": args.episodes, "seed": args.seed,
        "rewards_per_episode": rewards_per_episode,
        "epsilon_curve": eps_curve,
        "train_sources": {"train_csv": args.train_csv, "train_dir": args.train_dir},
        "text_col": args.text_col, "label_col": args.label_col
    }
    with open(results_dir / "train_logs.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    # CSV rápidos para gráficos
    with open(results_dir / "rewards.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "total_reward"])
        for i, rsum in enumerate(rewards_per_episode, 1):
            w.writerow([i, rsum])

    with open(results_dir / "epsilon.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "epsilon"])
        steps = 0
        for e in eps_curve:
            steps += len(data)  # aproximação: 1 ponto por varredura completa
            w.writerow([steps, e])


# ------------------------ Pós-processamento ------------------------
def _postprocess_results(results_dir: str, eval_endpoint: str | None) -> None:
    """
    (1) Opcionalmente chama um endpoint de métricas para gerar predicoes.csv
    (2) Gera curvas/plots e, se houver predicoes.csv, também metrics.json e confusion_matrix.png
    """
    # 1) Endpoint de métricas (opcional)
    if eval_endpoint:
        try:
            import requests
            print(f"[info] Chamando endpoint de métricas: {eval_endpoint}")
            requests.post(eval_endpoint, timeout=60)
            print("[ok] Endpoint de métricas executado.")
        except Exception as e:
            print(f"[warn] Falha ao chamar métricas: {e}")

    # 2) Geração de relatórios/figuras
    report_py = (BACKEND_DIR / "tools" / "make_results_report.py").resolve()
    cmd = [sys.executable, str(report_py), "--results", str(_resolve_to_path(results_dir))]
    try:
        print(f"[info] Gerando relatórios: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
        print("[ok] Relatórios/figuras atualizados em", results_dir)
    except Exception as e:
        print(f"[warn] make_results_report falhou: {e}")


# ------------------------ CLI ------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Entrada de dados
    p.add_argument("--train_csv", default="", help="Arquivo(s) CSV separados por vírgula OU caminho(s) de pasta. Ex.: backend/data/a.csv,backend/data/b.csv")
    p.add_argument("--train_dir", default="", help="Pasta contendo vários .csv (opcional)")
    p.add_argument("--text_col", default="text", help="Nome da coluna de texto (default: text)")
    p.add_argument("--label_col", default="label", help="Nome da coluna de rótulo (default: label)")

    # Auto split (estratificado) se você só tiver 1 CSV de tudo
    p.add_argument("--auto_split", action="store_true", help="Se habilitado, faz split 80/20 e treina só no conjunto de treino gerado")
    p.add_argument("--test_size", type=float, default=0.2, help="Proporção de teste para auto-split (default 0.2)")

    # Parâmetros de Q-Table / treino
    p.add_argument("--db", default=str(DB_DIR_DEFAULT / "qtable.sqlite"))
    p.add_argument("--results_dir", default=str(RESULTS_DIR_DEFAULT))
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--gamma", type=float, default=0.0)  # 0.0 = bandit puro
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_min", type=float, default=0.05)
    p.add_argument("--epsilon_decay", type=float, default=0.995)
    p.add_argument("--commit_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--extra_keywords", default="", help="Arquivo .txt com uma keyword por linha (opcional)")
    p.add_argument("--verbose", action="store_true")

    # ---- NOVO: endpoint opcional para avaliação/métricas (gera predicoes.csv) ----
    p.add_argument("--eval_endpoint", default="", help="Endpoint opcional para gerar predicoes.csv (ex: http://127.0.0.1:8000/rl/metrics)")

    args = p.parse_args()
    random.seed(args.seed)

    try:
        train(args)
        # Pós-processamento automático (sem .bat)
        _postprocess_results(results_dir=args.results_dir, eval_endpoint=(args.eval_endpoint or None))
    except FileNotFoundError as e:
        # Mensagem mais amigável no console
        print(f"[ERRO] {e}", file=sys.stderr)
        sys.exit(1)
