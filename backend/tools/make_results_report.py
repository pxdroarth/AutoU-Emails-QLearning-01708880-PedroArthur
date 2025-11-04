"""
Nome do arquivo: make_results_report.py
Data de criação: 04/11/2025
Autor: Pedro Arthur Maia Damasceno
Matrícula: 01708880

Descrição:
Geração de relatórios e gráficos a partir dos artefatos de treino do Q-Learning
(arquivos em backend/results). Plota curvas de epsilon e recompensas e, quando
há predições, calcula métricas e gera a matriz de confusão.

Funcionalidades:
- Lê epsilon.csv e rewards.csv e cria epsilon_curve.png e rewards_curve.png
- Se houver predicoes.csv (colunas y_true,y_pred), calcula accuracy, precision,
  recall e f1 (weighted) e salva em metrics.json
- Gera confusion_matrix.png com rótulos inferidos das predições
- Usa backend “headless” (MPLBACKEND=Agg) para evitar abrir janelas
- Diretório de resultados configurável via --results (default: backend/results)
"""

# backend/tools/make_results_report.py
import os, json, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# backend "headless" (evita abrir janela)
os.environ["MPLBACKEND"] = "Agg"

def plot_curve(df, x_col, y_col, out_png, title, xlabel, ylabel):
    plt.figure()
    plt.plot(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="backend/results")
    args = ap.parse_args()

    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)

    # 1) Epsilon
    eps_csv = results / "epsilon.csv"
    if eps_csv.exists():
        eps = pd.read_csv(eps_csv)
        x = "episode" if "episode" in eps.columns else eps.columns[0]
        y = "epsilon" if "epsilon" in eps.columns else eps.columns[-1]
        plot_curve(eps, x, y, results / "epsilon_curve.png", "Epsilon vs Episódio", x, "Epsilon")

    # 2) Rewards
    rew_csv = results / "rewards.csv"
    if rew_csv.exists():
        rew = pd.read_csv(rew_csv)
        x = "episode" if "episode" in rew.columns else rew.columns[0]
        y = "reward" if "reward" in rew.columns else rew.columns[-1]
        plot_curve(rew, x, y, results / "rewards_curve.png", "Recompensa vs Episódio", x, "Recompensa")

    # 3) Métricas (se houver predicoes.csv)
    preds_csv = results / "predicoes.csv"
    if preds_csv.exists():
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        import numpy as np

        preds = pd.read_csv(preds_csv)

        # Tolerante a cabeçalhos diferentes: tenta encontrar y_true/y_pred (case-insensitive).
        header = list(preds.columns)
        lower = [h.strip().lower() for h in header]

        if "y_true" in lower:
            y_true_col = header[lower.index("y_true")]
        else:
            y_true_col = header[0]  # fallback: 1ª coluna é o verdadeiro

        if "y_pred" in lower:
            y_pred_col = header[lower.index("y_pred")]
        else:
            y_pred_col = header[1] if len(header) > 1 else header[0]  # fallback: 2ª coluna (ou 1ª se só houver 1)

        y_true = preds[y_true_col]
        y_pred = preds[y_pred_col]

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        (results / "metrics.json").write_text(json.dumps({
            "accuracy": acc,
            "precision_weighted": prec,
            "recall_weighted": rec,
            "f1_weighted": f1,
            "samples": int(len(preds)),
            "y_true_col": y_true_col,
            "y_pred_col": y_pred_col
        }, indent=2, ensure_ascii=False))

        # Matriz de confusão
        labels = np.unique(pd.concat([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Matriz de Confusão")
        plt.colorbar()
        ticks = range(len(labels))
        plt.xticks(ticks, labels, rotation=45, ha="right")
        plt.yticks(ticks, labels)
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        thresh = cm.max() / 2 if cm.max() else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]:d}",
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(results / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()
