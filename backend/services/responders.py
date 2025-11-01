# -------------------------------------------------------------------
# responders.py
# Este módulo contém respostas padrão ("fallbacks") usadas quando:
#   - Não há conexão com Hugging Face (HF) OU ia gerativa desabilitada
#   - O modelo local é usado
#   - Ou quando a heurística simples decide a categoria
#
# A ideia é sempre retornar uma resposta profissional mínima,
# mesmo sem IA generativa disponível.
# -------------------------------------------------------------------

def resposta_produtiva() -> str:
    """
    Resposta padrão para emails classificados como PRODUTIVOS.
    - Confirma recebimento
    - Solicita dados objetivos (ex: protocolo, anexos)
    - Define expectativa de prazo (SLA curto)
    """
    return (
        "Olá! Recebemos sua solicitação e já estamos analisando. "
        "Se possível, informe o número do protocolo e anexos relevantes. "
        "Daremos retorno até o fim do dia útil."
    )


def resposta_improdutiva() -> str:
    """
    Resposta padrão para emails classificados como IMPRODUTIVOS.
    - Agradece cordialmente
    - Orienta o usuário caso precise de suporte real
    """
    return (
        "Olá! Agradecemos a mensagem. Caso precise de suporte ou acompanhamento "
        "de alguma requisição, envie os detalhes para acelerarmos o atendimento."
    )
