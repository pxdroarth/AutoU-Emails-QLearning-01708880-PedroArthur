"""
Nome do arquivo: app.py
Data de criação: 17/09/2025
Autor: Pedro Arthur Maia Damasceno
Matrícula: 01708880

Descrição:
Aplicação FastAPI do AutoU — Classificador & Auto-Responder.
Expõe endpoints de classificação de e-mails e integra o fluxo de RL (via router /rl/*).
Suporta entrada por texto ou arquivo (.txt, .pdf, .eml), executa fallback de classificação
(Modelo Local → Heurística) e permite CORS para o frontend.

Funcionalidades:
- /health: verificação de saúde
- /config: diagnóstico simples do build e env
- /classify: classifica e retorna categoria, confiança, resposta sugerida e origem
- Integração do router RL (/rl/feedback, /rl/train, /rl/metrics)
- Startup: garante a existência de backend/db/ para a Q-Table (SQLite)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# ======================
#  Paths & .env
# ======================
BASE_DIR = Path(__file__).resolve().parent        # .../backend
ROOT_DIR = BASE_DIR.parent                        # raiz do projeto

try:
    from dotenv import load_dotenv, find_dotenv
    ENV_PATH = find_dotenv(usecwd=True) or str((BASE_DIR / ".env").resolve())
    load_dotenv(ENV_PATH, override=False)
except Exception:
    ENV_PATH = ""

# ======================
#  Imports do projeto (RELATIVOS ao pacote backend)
#  Observação: usar imports relativos evita problemas ao rodar com:
#  uvicorn backend.app:app --reload
# ======================
from .routes_rl import router as rl_router
from .models.schemas import RespostaClassificacao
from .services.classifier import classificar_e_sugerir
from .services.pdf_reader import extract_text_from_pdf  # leve, PyPDF2

# Leitor de EML é opcional
try:
    from .services.eml_reader import extract_text_from_eml
    HAS_EML = True
except Exception:
    HAS_EML = False

# ====== FASTAPI APP ======
app = FastAPI(title="AutoU — Classificador de Emails (Local-Only)")

# Inclui o router de RL (/rl/*)
app.include_router(rl_router)

# ====== CORS ======
# Para dev local, autoriza o front na 127.0.0.1:5500.
# Se quiser deixar aberto, mantenha "*".
allowed_origins = os.getenv("CORS_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500,*")
origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]

# Atenção: allow_credentials=True não pode usar "*" — então deixo False por padrão.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" not in origins else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== CONSTANTES / HELPERS ======
MAX_BYTES = 5 * 1024 * 1024  # 5 MB

def _infer_ext(filename: Optional[str]) -> str:
    if not filename:
        return ""
    lower = filename.lower()
    for ext in (".txt", ".pdf", ".eml"):
        if lower.endswith(ext):
            return ext
    return os.path.splitext(lower)[1].lower()

# ====== STARTUP HOOKS ======
@app.on_event("startup")
def _ensure_dirs() -> None:
    """
    Garante a existência de backend/db e backend/results,
    já que artefatos são ignorados no Git.
    """
    (BASE_DIR / "db").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "results").mkdir(parents=True, exist_ok=True)

# ====== ENDPOINTS ======
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/config")
def config():
    """Diagnóstico simples (HF removida neste build)."""
    return {
        "build": "local+heuristica",
        "cwd": os.getcwd(),
        "has_env_file": (BASE_DIR / ".env").exists(),
        "env_path_loaded": ENV_PATH,
        "has_eml_reader": HAS_EML,
        "base_dir": str(BASE_DIR),
        "cors_origins": origins,
    }

@app.post("/classify", response_model=RespostaClassificacao)
async def classify(
    request: Request,
    arquivo: Optional[UploadFile] = File(None),
    texto: Optional[str] = Form(None),
):
    """
    Aceita:
      - multipart/form-data: campos 'texto' e/ou 'arquivo'
      - application/json:    body {"texto": "..."}
    Prioriza o conteúdo de arquivo quando presente; caso contrário, usa 'texto'.
    """
    # Se não veio via form/multipart, tente JSON { "texto": "..." }
    if texto is None and arquivo is None:
        try:
            data = await request.json()
            if isinstance(data, dict):
                texto_val = data.get("texto")
                if isinstance(texto_val, str):
                    texto = texto_val.strip()
        except Exception:
            # não é JSON; segue fluxo normal
            pass

    # 1) Extrair conteúdo (prioriza arquivo, se enviado)
    conteudo = (texto or "").strip()

    if arquivo is not None:
        data = await arquivo.read()
        if not data:
            raise HTTPException(status_code=400, detail="Arquivo vazio.")
        if len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="Arquivo muito grande (máx. 5MB).")

        ext = _infer_ext(arquivo.filename)

        if ext in (".txt", ""):
            # tenta utf-8; se falhar, usa fallback latin-1 ignorando erros
            try:
                conteudo = data.decode("utf-8")
            except Exception:
                conteudo = data.decode("latin-1", errors="ignore")

        elif ext == ".pdf":
            try:
                conteudo = extract_text_from_pdf(data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Falha ao ler PDF: {e}")

        elif ext == ".eml":
            if not HAS_EML:
                raise HTTPException(status_code=400, detail="Leitor de EML não disponível neste build.")
            try:
                conteudo = extract_text_from_eml(data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Falha ao ler EML: {e}")

        else:
            raise HTTPException(
                status_code=400,
                detail="Tipo de arquivo não suportado. Use .txt, .pdf ou .eml.",
            )

    if not conteudo:
        # Resposta segura para inputs vazios/ilegíveis
        return RespostaClassificacao(
            categoria="Improdutivo",
            confianca=0.5,
            resposta_sugerida="Mensagem vazia ou ilegível. Por favor, reenviar com mais detalhes.",
            origem="heuristica",
        )

    # 2) Classificar e sugerir resposta (Modelo Local → Heurística)
    categoria, confianca, resposta, origem = classificar_e_sugerir(conteudo)

    return RespostaClassificacao(
        categoria=categoria,
        confianca=float(confianca),
        resposta_sugerida=resposta,
        origem=origem,
    )
