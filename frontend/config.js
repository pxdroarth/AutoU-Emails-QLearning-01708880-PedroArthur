/*
Nome do arquivo: config.js
Data de criação: 29/10/2025
Autor: Pedro Arthur Maia Damasceno
Matrícula: 01708880

Descrição:
Resolve a URL base da API para o frontend do AutoU.
Prioridade: override manual (localStorage) → injeção (window.API_BASE) → default local.

Funcionalidades:
- Lê override opcional em localStorage: API_BASE_OVERRIDE
- Aceita window.API_BASE se injetado pela página
- Define window.APP_CONFIG = { API_BASE }
*/

(function () {
  // 1) Override manual para testes (opcional)
  const saved = localStorage.getItem("API_BASE_OVERRIDE");

  // 2) Injeção via window (caso você queira setar no index.html)
  let injected = (typeof window.API_BASE === "string") ? window.API_BASE.trim() : "";
  const looksLikeTemplate = injected.includes("{{") || injected.includes("}}");
  if (!injected || looksLikeTemplate) injected = "";

  // 3) Default local (backend FastAPI)
  const LOCAL_DEFAULT = "http://127.0.0.1:8000";

  const resolved = saved || injected || LOCAL_DEFAULT;
  window.APP_CONFIG = { API_BASE: resolved };
})();
