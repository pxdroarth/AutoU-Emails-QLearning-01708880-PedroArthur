import email
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# Este módulo cuida da leitura de arquivos .eml (emails salvos em disco).
# Objetivo: extrair o "conteúdo de texto" para ser classificado depois.
# ---------------------------------------------------------------------


def _html_to_text(html: str) -> str:
    """
    Converte HTML → texto simples.

    Usamos o BeautifulSoup para:
      - Remover tags (div, span, etc.).
      - Preservar o conteúdo em texto.
      - Normalizar espaços em branco.

    Se der erro por algum HTML malformado, retorna o HTML original.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return html


def extract_text_from_eml(binary: bytes) -> str:
    """
    Extrai o corpo textual de um email (.eml) recebido em bytes.

    Fluxo:
      1. Parseia o conteúdo binário com o parser oficial do Python (policy default).
      2. Se for multipart:
           - Procura primeiro por "text/plain".
           - Se não achar, tenta "text/html" e converte para texto.
      3. Se não for multipart:
           - Verifica se é plain ou html e trata do mesmo jeito.
      4. Se não conseguir nada, retorna string vazia.

    Essa ordem garante:
      - Preferência por texto puro (quando disponível).
      - Fallback para HTML.
      - Nunca quebra: sempre retorna string.
    """
    msg = BytesParser(policy=policy.default).parsebytes(binary)

    if msg.is_multipart():
        # Emails podem ter várias "partes" (texto, HTML, anexos…)
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain":
                return part.get_content().strip()
            if ctype == "text/html":
                return _html_to_text(part.get_content()).strip()
    else:
        # Caso simples: apenas uma parte
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            return msg.get_content().strip()
        if ctype == "text/html":
            return _html_to_text(msg.get_content()).strip()

    # Última linha de defesa: nada encontrado
    return ""
