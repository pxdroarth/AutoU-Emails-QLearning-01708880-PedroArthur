/**
 * Nome do arquivo: app.js
 * Data de criação: 17/09/2025
 * Autor: Pedro Arthur Maia Damasceno
 * Matrícula: 01708880
 *
 * Descrição:
 * Camada de UI e orquestração do frontend (AutoU). Consome a API local/Render
 * para classificar e-mails e aciona o laço de RL com feedback humano.
 *
 * Funcionalidades:
 * - Resolver API_BASE via config.js (Render/local)
 * - Enviar texto/arquivo para /classify
 * - Renderizar categoria, confiança e resposta sugerida
 * - Exibir painel de revisão manual (correto/errado) e enviar para /rl/feedback
 * - Disparar treino (/rl/train) e visualizar métricas (/rl/metrics)
 */

/* ====================== BLOQUEIO TOTAL DE RELOAD/NAVEGAÇÃO ====================== */
(function hardBlockNavigation() {
  // impede unload/reload (útil p/ diagnosticar qualquer tentativa de recarregar)
  window.addEventListener('beforeunload', (e) => {
    e.preventDefault();
    e.returnValue = '';
  });

  // log de eventos de navegação/visibilidade p/ diagnóstico
  ['pagehide', 'visibilitychange', 'freeze', 'resume', 'unload'].forEach(ev => {
    window.addEventListener(ev, () => console.debug('[nav]', ev, document.visibilityState));
  });

  // evita submit por Enter fora de TEXTAREA
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target && e.target.tagName !== 'TEXTAREA') {
      e.preventDefault();
      e.stopPropagation();
      console.debug('[block] Enter prevenido fora de TEXTAREA');
    }
  }, true);

  // bloqueia qualquer submit de <form> (capturing)
  document.addEventListener('submit', (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.debug('[block] submit prevenido ->', e.target);
    return false;
  }, true);

  // força todo <button> sem type a virar "button"
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('button:not([type])').forEach(b => b.setAttribute('type', 'button'));
  });

  // bloqueia cliques em <a href=""> ou "#"
  document.addEventListener('click', (e) => {
    const a = e.target.closest('a');
    if (a && (!a.getAttribute('href') || a.getAttribute('href') === '#')) {
      e.preventDefault();
      e.stopPropagation();
      console.debug('[block] anchor vazia/# prevenido');
      return false;
    }
  }, true);
})();
/* =============================================================================== */

// ========================= Config & Refs =========================//

// Lê a base da API da config injetada
const API_BASE = window.APP_CONFIG?.API_BASE;
if (!API_BASE) alert("Config inválida: API_BASE não definida.");

// Helpers e refs
const $ = (q) => document.querySelector(q);
const txt = $("#emailText");
const file = $("#emailFile");
const btn = $("#btnProcessar");
const box = $("#result");
const outCat = $("#outCategory");
const outConf = $("#outConfidence");
const outReply = $("#outReply");
const btnCopy = $("#btnCopiar");
const originBadge = $("#originBadge");
const originTitle = $("#originTitle");

// badge visual da origem
function setBadge(origin) {
  const map = {
    hf: ["HF", "badge b-hf", "via Hugging Face"],
    modelo: ["Local", "badge b-ml", "via Modelo Local"],
    heuristica: ["Heurística", "badge b-h", "via Heurística"],
  };
  const [label, cls, titleText] = map[origin] || ["?", "badge", ""];
  if (originBadge) { originBadge.textContent = label; originBadge.className = cls; }
  if (originTitle) { originTitle.textContent = titleText ? ` (${titleText})` : ""; }
}

// ============== UI: Classificar ==============

btn.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();

  const f = file.files[0];
  const t = (txt.value || "").trim();

  if (!f && !t) {
    alert("Cole um texto ou selecione um arquivo .txt/.pdf/.eml");
    return;
  }

  // limpa UI anterior
  outCat.textContent = "-";
  outConf.textContent = "-";
  outReply.value = "";
  setBadge(null);
  box.classList.add("hidden");

  btn.disabled = true;
  btn.textContent = "Processando...";
  try {
    // Envia SEMPRE como FormData (texto e/ou arquivo)
    const fd = new FormData();
    if (t) fd.append("texto", t);
    if (f) fd.append("arquivo", f);

    const res = await fetch(`${API_BASE}/classify`, {
      method: "POST",
      headers: { "Accept": "application/json" }, // não defina Content-Type manualmente em FormData
      body: fd
    });

    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { msg = await res.text(); } catch {}
      throw new Error(msg || `HTTP ${res.status}`);
    }

    const text = await res.text();
    if (!text) throw new Error("Resposta vazia da API.");
    let data;
    try { data = JSON.parse(text); } catch { throw new Error("Resposta não é JSON válido."); }

    outCat.textContent = data.categoria ?? "-";
    outConf.textContent = (typeof data.confianca === "number") ? `${(data.confianca * 100).toFixed(1)}%` : "-";
    outReply.value = data.resposta_sugerida ?? "";
    setBadge(data.origem);
    box.classList.remove("hidden");

    // MOSTRAR painel de revisão manual
    exibirSecaoFeedback((txt.value || "").trim(), outCat.textContent);
  } catch (e2) {
    alert("Erro ao classificar: " + (e2?.message || e2));
  } finally {
    btn.disabled = false;
    btn.textContent = "Processar";
  }
});

// Copiar resposta
btnCopy.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();
  try {
    await navigator.clipboard.writeText(outReply.value || "");
    btnCopy.textContent = "Copiado!";
    setTimeout(() => (btnCopy.textContent = "Copiar resposta"), 1200);
  } catch {
    alert("Não foi possível copiar.");
  }
});

// ===== FEEDBACK HUMANO (integração RL) =====

// elementos
const feedbackSection   = document.getElementById("feedback-section");
const btnCorreto        = document.getElementById("btnCorreto");
const btnErrado         = document.getElementById("btnErrado");
const correcaoWrap      = document.getElementById("correcao-wrap");
const selectCorrecao    = document.getElementById("selectCorrecao");
const btnEnviarCorrecao = document.getElementById("btnEnviarCorrecao");
const btnTreinar        = document.getElementById("btnTreinar");
const btnVerMetricas    = document.getElementById("btnVerMetricas");
const metricasBox       = document.getElementById("metricasBox");
const toast             = document.getElementById("toast");

// guarda último texto/categoria exibidos
let ultimoTexto = "";
let ultimaCategoria = "";

// util
function showToast(msg) {
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.remove("hidden");
  toast.style.opacity = 1;
  setTimeout(() => { toast.style.opacity = 0; toast.classList.add("hidden"); }, 2000);
}

// exibir painel após resultado
function exibirSecaoFeedback(texto, categoria) {
  ultimoTexto = (texto || "").trim();
  ultimaCategoria = (categoria || "").toLowerCase();
  if (!ultimoTexto || !ultimaCategoria) return;
  feedbackSection.classList.remove("hidden");
  correcaoWrap.classList.add("hidden");
  selectCorrecao.value = "";
}

// Confirmar como CORRETO
btnCorreto?.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();

  if (!ultimoTexto || !ultimaCategoria) {
    return showToast("Sem contexto para feedback.");
  }

  try {
    const resp = await fetch(`${API_BASE}/rl/feedback_h`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: ultimoTexto,
        gold_label: ultimaCategoria,   // humano confirma
        pred_label: ultimaCategoria,   // o modelo acertou
        source: "frontend"
      })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    showToast("Feedback (correto) registrado!");
  } catch (err) {
    console.error(err);
    showToast("Falha ao enviar feedback.");
  }
});

// Marcar como ERRADO (abre seleção de correção)
btnErrado?.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  correcaoWrap.classList.remove("hidden");
});

// Enviar correção de rótulo
btnEnviarCorrecao?.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();

  const novo = selectCorrecao.value;
  if (!novo) return showToast("Escolha o rótulo correto.");

  try {
    const resp = await fetch(`${API_BASE}/rl/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: ultimoTexto, label: novo.toLowerCase(), csv_path: "data/train.csv" })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    showToast("Correção enviada!");
    correcaoWrap.classList.add("hidden");
  } catch (err) {
    console.error(err);
    showToast("Falha ao enviar correção.");
  }
});

// Disparar TREINO
btnTreinar?.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();

  try {
    const r = await fetch(`${API_BASE}/rl/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        train_csv: "backend/data/train.csv",   // caminho do backend (server side)
        use_feedback_csv: true,                // inclui csv_foruseres.csv automaticamente
        episodes: 20,
        shuffle: true
      })
    });
    showToast(r.ok ? "Treino concluído!" : "Falha ao treinar.");
  } catch (err) {
    console.error(err);
    showToast("Falha ao treinar.");
  }
});

// Ver MÉTRICAS
btnVerMetricas?.addEventListener("click", async (e) => {
  e.preventDefault();
  e.stopPropagation();

  try {
    const r = await fetch(`${API_BASE}/rl/metrics?test_csv=backend/data/test.csv`);
    if (!r.ok) {
      showToast("Erro ao carregar métricas.");
      return;
    }
    const d = await r.json();
    metricasBox.textContent = JSON.stringify(d, null, 2);
    metricasBox.classList.remove("hidden");
    showToast("Métricas atualizadas!");
  } catch (err) {
    console.error(err);
    showToast("Falha ao carregar métricas.");
  }
});

// Blindagem extra se algum submit rolar por engano
document.addEventListener("submit", (e) => e.preventDefault());
