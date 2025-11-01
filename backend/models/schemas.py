from pydantic import BaseModel

class RespostaClassificacao(BaseModel):
    categoria: str        # Produtivo | Improdutivo
    confianca: float
    resposta_sugerida: str
    origem: str           # "modelo" | "heuristica"