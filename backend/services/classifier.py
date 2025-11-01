from typing import Tuple, List
from pathlib import Path
import joblib, re, unicodedata, traceback

from .responders import resposta_improdutiva, resposta_produtiva

# Onde o modelo local (Pipeline TF-IDF + Classificador) é salvo
CAMINHO_MODELO = Path(__file__).parent.parent / "data" / "model.pkl"
_modelo = None

# ---------------------------------------------------------------------
# Helpers de pré-processamento e heurísticas
# ---------------------------------------------------------------------
def _rm_acentos(s: str) -> str:
    """Remove acentos para simplificar match de regex."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

# Palavras-chave (usamos raízes p/ cobrir variações; _rm_acentos já remove acentos)
PROD_KEYWORDS = [
    # fluxo / acompanhamento
    r"\bprotocolo\b", r"\bchamado\b", r"\bticket\b", r"\bcase\b", r"\bcaso\b",
    r"\bstatus\b", r"\batualiza\w*\b", r"\bretorno\b", r"\bposi\w*\b", r"\bpendenc\w*\b",
    r"\bacompanha\w*\b", r"\bprioridade\b", r"\burgent\w*\b", r"\bprazo\b", r"\bsla\b",

    # suporte / erro
    r"\bsuporte\b", r"\bhelpdesk\b", r"\bproblema\b", r"\bocorr\w*\b", r"\berro\w*\b",
    r"\bfalh\w*\b", r"\bbug\b", r"\binciden\w*\b",

    # documentos e dados
    r"\banex\w*\b", r"\bdocument\w*\b", r"\bnf[e]?\b", r"\bnota\s*fiscal\b", r"\bcontrat\w*\b",
    r"\bcomprovant\w*\b", r"\bbolet\w*\b", r"\bfatur\w*\b", r"\bpropost\w*\b", r"\borcament\w*\b",

    # ações típicas
    r"\bvalid\w*\b", r"\bhomolog\w*\b", r"\bliber\w*\b", r"\bdesbloque\w*\b", r"\bativ\w*\b",
    r"\bdesativ\w*\b", r"\bcancel\w*\b", r"\breembols\w*\b", r"\bregulariz\w*\b", r"\brevis\w*\b",

    # pedidos diretos
    r"\bverific\w*\b", r"\bconfirm\w*\b", r"\binform\w*\b", r"\bencaminh\w*\b", r"\benvi\w*\b",
    r"\bchec\w*\b", r"\bsolicit\w*\b", r"\bprecis\w*\b", r"\bfavor\b", r"\bpor favor\b",

    # padrões numéricos úteis
    r"\bprotocolo\s*\d+\b", r"\bchamado\s*\d+\b",
]

IMPROD_KEYWORDS = [
    # saudações e cortesia
    r"\bobrigad\w*\b", r"\bagrade\w*\b", r"\bvaleu\b", r"\bmuito\s+obrigado\w*\b",
    r"\bparab\w*\b", r"\bboas\s+festas\b", r"\bfeliz\s+natal\b", r"\bfeliz\s+ano\s+novo\b",
    r"\bbo[am]\s+dia\b", r"\bboa\s+tarde\b", r"\bboa\s+noite\b", r"\bbom\s+fim\s+de\s+semana\b",
    r"\babraç\w*\b", r"\batenciosamente\b", r"\bcordialmente\b", r"\batt\b",
    r"\bciente\b", r"\brecebido\b", r"\bok\b", r"\bperfeito\b", r"\bshow\b", r"\bgrato\w*\b",
    r"👍", r"🙏",
]

# Termos fortes de intenção de ação → empurram Produtivo
ACTION_BOOST = re.compile(
    r"\b("
    r"protocolo|chamado|status|suporte|erro|falh\w*|fatura|boleto|nota\s*fiscal|nfe|"
    r"valid\w*|liber\w*|desbloque\w*|desativ\w*|cancel\w*|reembols\w*|"
    r"verific\w*|confirm\w*|inform\w*|encaminh\w*|envi\w*|solicit\w*|"
    r"urgente|prioridade|prazo|sla"
    r")\b",
    re.I,
)
REQUEST_PAT = re.compile(
    r"(pode[m]?\s+(verificar|informar|confirmar|enviar|ver)\b|"
    r"favor\s+(verificar|informar|confirmar|enviar)\b|"
    r"aguardo\s+(retorno|posi\w*)\b)",
    re.I,
)

def _pontuar(texto: str) -> Tuple[int, int]:
    t = _rm_acentos(texto.lower())

    prod = sum(1 for p in PROD_KEYWORDS if re.search(p, t))
    impr = sum(1 for p in IMPROD_KEYWORDS if re.search(p, t))

    # boost por termos de ação
    if ACTION_BOOST.search(t):
        prod += 3  # estava 2

    # pedido direto explícito (pode verificar? favor informar…)
    if REQUEST_PAT.search(t):
        prod += 2

    return prod, impr

def _conf_por_scores(prod: int, impr: int) -> float:
    total = max(1, prod + impr)
    base = prod / total
    delta = 0.1 if abs(prod - impr) >= 2 else 0.0
    bonus = 0.05 if prod - impr >= 3 else 0.0
    return float(min(0.99, max(0.5, base + delta + bonus)))

def _meta_score(proba_prod: float, prod_hits: int, impr_hits: int) -> float:
    s = proba_prod + 0.05 * max(0, prod_hits - impr_hits)
    return max(0.0, min(1.0, s))

# ---------------------------------------------------------------------
# Carregamento de modelo local
# ---------------------------------------------------------------------
def carregar_modelo():
    """Carrega o Pipeline treinado de disco (lazy-load)."""
    global _modelo
    if _modelo is None and CAMINHO_MODELO.exists():
        try:
            _modelo = joblib.load(CAMINHO_MODELO)
        except Exception:
            traceback.print_exc()
            _modelo = None
    return _modelo

# ---------------------------------------------------------------------
# Caminho Modelo Local
# ---------------------------------------------------------------------
def _com_modelo_local(texto: str) -> Tuple[str, float, str, str]:
    modelo = carregar_modelo()
    if not modelo:
        raise RuntimeError("Modelo local indisponível.")

    # Probabilidade do lado "Produtivo"
    try:
        proba = getattr(modelo, "predict_proba", None)
        if proba:
            probs = proba([texto])[0]
            classes: List[str] = list(getattr(modelo, "classes_", ["Improdutivo", "Produtivo"]))
            idx_prod = classes.index("Produtivo") if "Produtivo" in classes else 1
            proba_prod = float(probs[idx_prod])
        else:
            proba_prod = 0.5
    except Exception:
        traceback.print_exc()
        proba_prod = 0.5

    sp, si = _pontuar(texto)
    score_hibrido = _meta_score(proba_prod, sp, si)
    categoria = "Produtivo" if (score_hibrido >= 0.5 or sp >= si) else "Improdutivo"
    confianca = float(max(0.6, min(0.99, score_hibrido)))
    resp = resposta_produtiva() if categoria == "Produtivo" else resposta_improdutiva()
    return str(categoria), float(confianca), resp, "modelo"

# ---------------------------------------------------------------------
# Orquestrador (Modelo -> Heurística)
# ---------------------------------------------------------------------
def classificar_e_sugerir(texto: str) -> Tuple[str, float, str, str]:
    """
    Fluxo LOCAL-ONLY:
      1) Modelo Local (TF-IDF + Classificador)
      2) Heurística (fallback)
    Retorna: (categoria, confiança, resposta_sugerida, origem)
    """
    texto = (texto or "").strip()
    if not texto:
        return "Improdutivo", 0.5, resposta_improdutiva(), "heuristica"

    # 1) Modelo Local
    try:
        return _com_modelo_local(texto)
    except Exception:
        traceback.print_exc()

    # 2) Heurística pura (última linha de defesa)
    sp, si = _pontuar(texto)
    categoria = "Produtivo" if sp >= si else "Improdutivo"
    confianca = _conf_por_scores(sp, si)
    resp = resposta_produtiva() if categoria == "Produtivo" else resposta_improdutiva()
    return categoria, confianca, resp, "heuristica"
