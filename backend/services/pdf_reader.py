from io import BytesIO
from PyPDF2 import PdfReader

# ---------------------------------------------------------------------
# Este módulo cuida da leitura de arquivos PDF.
# Objetivo: extrair o texto de cada página e devolvê-lo como string.
# ---------------------------------------------------------------------


def extract_text_from_pdf(binary: bytes) -> str:
    """
    Extrai o texto de um arquivo PDF (recebido em bytes).

    Passos:
      1. Lê o binário com PyPDF2 (usando BytesIO como buffer em memória).
      2. Percorre todas as páginas do PDF.
      3. Para cada página, tenta extrair texto com `page.extract_text()`.
         - Se não conseguir (PDF escaneado ou erro), ignora a página.
      4. Junta todo o texto das páginas em uma única string.

    Observações:
      - PDFs baseados em imagem (scans) não terão texto extraído aqui
        → para esses casos seria necessário OCR (ex.: Tesseract).
      - O retorno é sempre uma string (pode ser vazia se não extrair nada).

    Exemplo de uso:
      with open("fatura.pdf", "rb") as f:
          conteudo = extract_text_from_pdf(f.read())
    """
    reader = PdfReader(BytesIO(binary))
    parts = []

    for page in reader.pages:
        try:
            # Extrai o texto da página (pode retornar None se vazio)
            parts.append(page.extract_text() or "")
        except Exception:
            # Silencia erros de parsing em páginas específicas
            pass

    # Junta todas as partes, separadas por quebras de linha
    text = "\n".join(parts).strip()
    return text
