## AutoU — Classificador & Auto‑Responder de Emails 🚀

[![Python](https://img.shields.io/badge/python-3.x-blue)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/FastAPI‑backend-green)](https://fastapi.tiangolo.com/) [![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

## 🎯 Objetivo

Aplicar conceitos de Machine Learning e Aprendizado por Reforço (Reinforcement Learning – RL)
na classificação automática de e-mails produtivos ou improdutivos, integrando um fluxo de feedback humano
que retroalimenta o agente Q-Learning em tempo real.

## 📚. Introdução

O projeto AutoU é um classificador inteligente de e-mails que usa Q-Learning armazenado em SQLite para aprender padrões de produtividade em mensagens.
Cada mensagem é representada como um estado discreto (tamanho + presença de palavras-chave), e o agente aprende qual ação (“produtivo” ou “improdutivo”) maximiza a recompensa, com base em acertos e feedback humano.

O sistema possui:

Backend FastAPI com Q-Learning tabular persistente.

Frontend leve (HTML + JS) para classificação e revisão humana.

Módulo de feedback que registra as correções do usuário e as reintegra nos próximos 
treinos.

## Sobre o Sistema
    
Aplicação **Web + API** (FastAPI + HTML/JS) desenvolvida como entrega do  
**Desafio de Machine Learning / Reforço (ADS 2025)**.

> Sistema capaz de **classificar e-mails como produtivos/improdutivos**,  
> **gerar respostas automáticas** e **aprender com feedback humano**,  
> usando **Q-Learning tabular** e **recompensas +1/-1**.

- Classifica emails em **Produtivo** ou **Improdutivo**
- Gera **resposta sugerida** conforme a categoria
- Aceita **texto** ou **arquivos** (.txt, .pdf, .eml)


## ⚙️ Tecnologias
- **Backend:** Python 3.x, FastAPI, Uvicorn, SQLite, NumPy, scikit-learn, PyPDF2  
- **Frontend:** HTML, CSS, JavaScript vanilla (sem frameworks)  
- **Aprendizado de Máquina:** Q-Learning (tabular), TF-IDF + Logistic Regression (baseline legado)  
- **Automação:** Scripts `.bat` (Windows),  fiz so pra windows mesmo
- **Armazenamento:** Q-Table persistida em `backend/db/qtable.sqlite`, e arquivos csv já preparados pro retreino
- Visualização Matplotlib (recompensa, epsilon, matriz de confusão)



## 🧩 Arvore Principal

```tree
AutoU-ClassificacaoEmails-01708880-PedroArthur/
│
├── .gitignore                    # Ignora artefatos temporários, cache, venv etc.
├── image.png                     # Print de capa / resultado (para README)
├── LICENSE                       # Licença do projeto
├── README.md                     # Documentação completa do projeto
│
├── backend/                      # Backend principal (FastAPI + Q-Learning)
│   │
│   ├── .env                      # Variáveis de ambiente (porta, CORS, configs)
│   ├── app.py                    # Aplicação principal FastAPI (endpoints + CORS)
│   ├── metrics_eval.py           # Script de avaliação e geração de métricas/plots
│   ├── qlearning_sqlite.py       # Implementação do agente Q-Learning tabular
│   ├── requirements.txt          # Dependências Python para instalação via pip
│   ├── routes_rl.py              # Endpoints /rl/* (treino, métricas, feedback)
│   ├── train_classifier.py       # Treino inicial supervisionado (baseline opcional)
│   ├── __init__.py               # Marca o backend como pacote Python
│   │
│   ├── data/                     # Dados de treino, teste e feedback
│   │   ├── csv_foruseres.csv     # CSV de feedback humano (persistente)
│   │   ├── full.csv              # Dataset completo (base unificada)
│   │   ├── model.pkl             # Modelo legado scikit-learn
│   │   ├── samples.csv           # Exemplos usados para testes rápidos
│   │   ├── test.csv              # Conjunto de teste
│   │   └── train.csv             # Conjunto de treino
│   │
│   ├── db/                       # Banco SQLite da Q-Table
│   │   └── qtable.sqlite         # Armazena Q(s,a) e histórico de aprendizado
│   │
│   ├── models/                   # Schemas Pydantic e estruturas de resposta
│   │   ├── schemas.py            # Classes de validação e resposta da API
│   │   └── __init__.py
│   │
│   ├── results/                  # Resultados e métricas do agente
│   │   ├── confusion_matrix.png  # Matriz de confusão (produtivo x improdutivo)
│   │   ├── epsilon.csv           # Log da taxa de exploração ε por episódio
│   │   ├── epsilon_curve.png     # Curva de decaimento do epsilon
│   │   ├── metrics.json          # Métricas finais (accuracy, f1-score, reward etc.)
│   │   ├── predicoes.csv         # Predições e resultados detalhados
│   │   ├── rewards.csv           # Histórico de recompensas por episódio
│   │   ├── rewards_curve.png     # Curva de recompensas (média móvel)
│   │   └── train_logs.json       # antigo treino supervisionado (baseline)
│   │
│   ├── scripts/                  # Scripts auxiliares para execução rápida
│   │   ├── run_api.bat           # Atalho para iniciar a API local (Uvicorn)
│   │   └── treinar_uma_vez.bat   # Executa o treino e grava resultados em /results
│   │
│   └── services/                 # Serviços e utilitários internos (camada de negócio)
│       ├── classifier.py         # Classificador principal (modelo + heurística)
│       ├── eml_reader.py         # Extrator de texto de e-mails (.eml)
│       ├── nlp_preprocess.py     # Pré-processamento NLP (limpeza e tokenização)
│       ├── pdf_reader.py         # Extrator de texto de PDFs (PyPDF2)
│       ├── responders.py         # Gerador de resposta automática com base na classe
│       └── __init__.py
│
└── frontend/                     # Interface web leve (HTML + JS puro)
    ├── index.html                # Página principal do classificador (UI)
    ├── app.js                    # Lógica principal da interface (API + feedback)
    ├── config.js                 # Resolução dinâmica do backend (API_BASE)
    └── styles.css                # Estilos visuais da UI (modo escuro elegante)

```


## Formulas
![formulas](https://github.com/user-attachments/assets/44e9397d-8fda-4c01-8485-c33378baa91e)

---

## 🔄 Fluxo de funcionamento
  Usuário cola texto ou envia arquivo (.txt, .pdf, .eml).

```tree
API /classify executa pipeline local + heurística

Frontend exibe categoria, confiança e resposta sugerida

Usuário fornece feedback (✅ Correto | ❌ Errado)

Feedback é salvo em backend/data/train.csv,csv_foruseres.csv 

Script treinar_uma_vez.bat roda e atualiza a Q-Table

Gráficos (results/) mostram evolução das recompensas, epsilon e matriz de confusão
```



## ⚙️ Execução local
1️⃣ Prepara o Ambiente virtual

    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r backend/requirements.txt

2️⃣ Rodar API ou abrir o bat run_api.bat

    python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000


3️⃣ Frontend
```bash
  cd frontend
  python -m http.server 8080
 
```
4️⃣ Abrir frontend
```bash
abrir http://localhost:8080/
```
## 🧪 Treinamento e avaliação
```bash
Executar treinamento:
cd backend/scripts
treinar_uma_vez.bat
```
## Gera:
```tree
backend/results/
 ├── confusion_matrix.png
 ├── rewards_curve.png
 ├── epsilon_curve.png
 └── metrics.json
```
## Avaliar métricas:
```bash
GET /rl/metrics
```

## Retroalimentação humana:

Cada clique “Correto” ou “Errado” no frontend grava em:
```bash
backend/data/csv_foruseres.csv
```
Esses dados são combinados automaticamente no próximo treino.
## 📊 Resultados e visualizações


rewards_curve.png	evolução média das recompensas por episódio

epsilon_curve.png	decaimento do ε (trade-off exploração/aproveitamento)

confusion_matrix.png	desempenho do agente nas classes produtivo/improdutivo

## 🧠  Dificuldades e decisões

Garantir persistência da Q-Table no SQLite sem sobrescrita.

Implementar parsing robusto de .pdf e .eml sem dependências pesadas.

Desenhar recompensas simples e simétricas (+1/−1).

Normalizar o fluxo de feedback humano (csv_foruseres.csv) e integrar automaticamente no treino.
## 🧾  Conclusão

O projeto cumpre todos os critérios de Machine Learning com Reinforcement Learning tabular, aplicando os conceitos de exploration vs. exploitation, epsilon-decay, recompensa, e retroalimentação humana.

O resultado é um sistema funcional que aprende com experiência e feedback, utilizando apenas Python puro, SQLite e FastAPI — totalmente executável e interpretável.
## Proximos passos

- Pipeline Auto
  Re-treino automático via trigger de feedback

- Deploy

👤 MATRICULA 
--- 

Pedro Arthur Maia Damasceno
Matrícula: 01708880
Curso: Análise e Desenvolvimento de Sistemas (ADS – UNINASSAU)
Fortaleza-CE, 2025

ME DA UM 10 VALA DEU MT TRABALHO 
```markdown
🤝 Contribuições
Contribuições são bem-vindas!  
Abra uma *issue* para sugestões ou envie um *pull request*. 🚀
```
