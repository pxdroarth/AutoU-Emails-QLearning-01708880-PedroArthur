
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

// badge
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

btn.addEventListener("click", async () => {
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
      headers: { "Accept": "application/json" }, // NÃO defina Content-Type manualmente
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
  } catch (e) {
    alert("Erro ao classificar: " + (e?.message || e));
  } finally {
    btn.disabled = false;
    btn.textContent = "Processar";
  }
});

btnCopy.addEventListener("click", async () => {
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

// binds
btnCorreto?.addEventListener("click", async () => {
  if (!ultimoTexto || !ultimaCategoria) {
    return showToast("Sem contexto para feedback.");
  }

  try {
    await fetch(`${API_BASE}/rl/feedback_h`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: ultimoTexto,
        gold_label: ultimaCategoria,   // o humano confirma
        pred_label: ultimaCategoria,   // o modelo acertou
        source: "frontend"
      })
    });
    showToast("Feedback (correto) registrado!");
  } catch (err) {
    console.error(err);
    showToast("Falha ao enviar feedback.");
  }
});


btnErrado?.addEventListener("click", () => {
  correcaoWrap.classList.remove("hidden");
});

btnEnviarCorrecao?.addEventListener("click", async () => {
  const novo = selectCorrecao.value;
  if (!novo) return showToast("Escolha o rótulo correto.");
  await fetch(`${API_BASE}/rl/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: ultimoTexto, label: novo.toLowerCase(), csv_path: "data/train.csv" })
  });
  showToast("Correção enviada!");
  correcaoWrap.classList.add("hidden");
});

btnTreinar?.addEventListener("click", async () => {
  const r = await fetch(`${API_BASE}/rl/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ train_csv: "data/train.csv", episodes: 20, shuffle: true })
  });
  showToast(r.ok ? "Treino concluído!" : "Falha ao treinar.");
});

btnVerMetricas?.addEventListener("click", async () => {
  const r = await fetch(`${API_BASE}/rl/metrics?test_csv=data/test.csv`);
  if (!r.ok) return showToast("Erro ao carregar métricas.");
  const d = await r.json();
  metricasBox.textContent = JSON.stringify(d, null, 2);
  metricasBox.classList.remove("hidden");
  showToast("Métricas atualizadas!");
});
