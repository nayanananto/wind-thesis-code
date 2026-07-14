const $ = (id) => document.getElementById(id);
const messages = $("messages");
const sessionKey = "hitl_thread_id";
let threadId = sessionStorage.getItem(sessionKey);
let runtimeDefaults = {};
let selectedWindowId = null;

const FIXED_CONFIG = {
  topK: 5,
  enableLlm: true,
  enableAgentGraph: true,
  enableRag: true,
  preferLivePhase: true,
  phaseHistory: 3,
  phaseStep: 1,
  phaseTopK: 3,
  phaseAnalogK: 5,
  phaseMinSupport: 5,
  phaseModelMode: "transition",
  numericSteps: 6,
  numericMode: "disabled",
};

if (!threadId) {
  threadId = `hitl_${crypto.randomUUID()}`;
  sessionStorage.setItem(sessionKey, threadId);
}

function esc(value) {
  const normalized = value && typeof value === "object"
    ? JSON.stringify(value, null, 2)
    : String(value ?? "");
  return normalized
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function addMessage(role, html) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.innerHTML = html;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

function pill(text, className = "") {
  return `<span class="pill ${className}">${esc(text)}</span>`;
}

function renderRows(title, rows, type) {
  if (!rows || rows.length === 0) return "";
  const body = rows
    .map((row) => {
      if (type === "candidate") {
        const prob = typeof row.probability === "number" ? row.probability.toFixed(3) : row.probability;
        return `<div class="row row--candidate"><strong>Token ${esc(row.token_id)}</strong><span>${esc(row.regime_name)}</span><span>p=${esc(prob)}</span><span>n=${esc(row.count)}</span></div>`;
      }
      if (type === "analog") {
        return `<div class="row row--analog"><strong>${esc(row.evidence_id)}</strong><span>${esc(row.history_tokens)}</span><span>next ${esc(row.next_token_id)}</span><span>${esc(row.future_window_start)}</span></div>`;
      }
      const dist = typeof row.retrieval_distance === "number" ? row.retrieval_distance.toFixed(4) : row.retrieval_distance;
      return `<div class="row row--similar"><strong title="${esc(row.window_id)}">${esc(row.window_id)}</strong><span>${esc(row.regime_name)}</span><span>d=${esc(dist)}</span></div>`;
    })
    .join("");
  return `<div class="section"><h3>${esc(title)}</h3><div class="rows">${body}</div></div>`;
}

function renderReviewState(reviewState) {
  if (!reviewState) return "";
  const meta = [
    pill(`status: ${reviewState.status || reviewState.action || "n/a"}`, "strong"),
    reviewState.window_id ? pill(`forecast input window: ${reviewState.window_id}`) : null,
    reviewState.model_predicted_token !== null && reviewState.model_predicted_token !== undefined
      ? pill(`model next token: ${reviewState.model_predicted_token}`)
      : null,
  ].filter(Boolean).join("");

  return `<div class="section">
    <h3>Human Review State</h3>
    <div class="meta">${meta}</div>
  </div>`;
}

function renderMemoryDebug(debug) {
  if (!debug) return "";
  const chunks = (debug.selected_chunk_ids || []).join(", ");
  const kinds = (debug.selected_chunk_kinds || []).map((item) => pill(item)).join("");
  const blockers = (debug.llm_blockers || []).join(", ");
  return `<details class="details-box">
    <summary>Technical memory/RAG trace</summary>
    <div class="details-body">
      <div class="meta">
        ${pill(`LLM attempted: ${Boolean(debug.llm_attempted)}`)}
        ${pill(`LLM used: ${Boolean(debug.llm_used)}`)}
        ${debug.llm_output_format ? pill(`format: ${debug.llm_output_format}`) : ""}
      </div>
      ${kinds ? `<div class="meta">${kinds}</div>` : ""}
      ${chunks ? `<div class="answer">Chunks: ${esc(chunks)}</div>` : ""}
      ${blockers ? `<div class="answer">LLM blockers: ${esc(blockers)}</div>` : ""}
      ${debug.llm_fallback_reason ? `<div class="answer">Fallback: ${esc(debug.llm_fallback_reason)}</div>` : ""}
    </div>
  </details>`;
}

function renderTechnicalDetails(s) {
  const details = [
    `intent: ${s.intent ?? "n/a"}`,
    `confidence: ${s.confidence ?? "n/a"}`,
    `mode: ${s.mode ?? "n/a"}`,
    s.retrieval_context ? `retrieval: ${s.retrieval_context}` : null,
    s.explanation_context ? `explain context: ${s.explanation_context}` : null,
    s.explanation_mode ? `explain mode: ${s.explanation_mode}` : null,
    s.phase_transition_source ? `phase source: ${s.phase_transition_source}` : null,
    s.forecast_model_source ? `model: ${s.forecast_model_source}` : null,
    s.agent_graph ? `agent: ${s.agent_graph.engine}/${s.agent_graph.route}` : null,
    s.router && s.router.source ? `router: ${s.router.source}` : null,
    s.thread_id ? `thread: ${s.thread_id}` : null,
    s.model_warning ? `warning: ${s.model_warning}` : null,
  ].filter(Boolean);

  if (!details.length) return "";
  return `<details class="details-box">
    <summary>Run details</summary>
    <div class="details-body"><div class="meta">${details.map((item) => pill(item)).join("")}</div></div>
  </details>`;
}

function renderResponse(payload) {
  const s = payload.summary || {};
  const liveState = s.current_live_state || {};
  const route = s.agent_graph ? `${s.agent_graph.route}` : (s.intent || "hitl");
  const windowLabel = s.window_id && String(s.window_id).startsWith("live_")
    ? `live window: ${s.window_id}`
    : null;
  const primaryMeta = [
    pill(`route: ${route}`, "strong"),
    windowLabel ? pill(windowLabel) : null,
    s.live_token_sequence ? pill(`live tokens: ${s.live_token_sequence.join(", ")}`) : null,
    liveState.regime_name ? pill(`live regime: ${liveState.regime_name}`) : null,
    s.llm_requested ? pill(`LLM: ${s.llm_available ? "available" : "unavailable"}`) : pill("LLM: off"),
  ].filter(Boolean).join("");

  const warnings = [
    ...(s.critic_warnings || []),
    s.phase_low_support ? `low support: n=${s.phase_support}` : null,
    s.model_warning || null,
  ].filter(Boolean);
  const hasSimilarTable = Array.isArray(s.similar_windows) && s.similar_windows.length > 0;
  const hideRedundantRetrievalAnswer = s.intent === "retrieve_similar" && hasSimilarTable;

  return `
    <div class="meta">${primaryMeta}</div>
    ${warnings.length ? `<div class="meta">${warnings.map((item) => pill(item, "warning")).join("")}</div>` : ""}
    ${hideRedundantRetrievalAnswer ? "" : `<div class="answer">${esc(s.answer || "No answer returned.")}</div>`}
    ${renderRows("Phase candidates", s.top_phase_candidates, "candidate")}
    ${renderRows("Similar historical windows", s.similar_windows, "similar")}
    ${renderRows("Transition analogs", s.transition_analogs, "analog")}
    ${renderReviewState(s.review_state)}
    ${renderMemoryDebug(s.memory_rag_debug)}
    ${renderTechnicalDetails(s)}
  `;
}

function requestBody(question) {
  const cfg = { ...FIXED_CONFIG, ...runtimeDefaults };
  return {
    question,
    thread_id: threadId,
    metadata: cfg.metadata || null,
    window_id: selectedWindowId,
    latest: !selectedWindowId,
    top_k: cfg.topK,
    enable_llm: cfg.enableLlm,
    enable_agent_graph: cfg.enableAgentGraph,
    enable_rag: cfg.enableRag,
    phase_tokens: null,
    phase_history_length: cfg.phaseHistory,
    phase_horizon_steps: cfg.phaseStep,
    phase_top_k: cfg.phaseTopK,
    phase_analog_k: cfg.phaseAnalogK,
    phase_min_support: cfg.phaseMinSupport,
    live_raw_path: cfg.live_raw_path || null,
    numeric_forecast_steps: cfg.numericSteps,
    numeric_forecast_mode: cfg.numericMode,
    numeric_model_path: cfg.numeric_model_path || null,
    phase_model_mode: cfg.phaseModelMode,
    phase_model_path: cfg.phase_model_path || null,
    live_phase_history_path: cfg.live_phase_history_path || null,
    live_phase_state_path: cfg.live_phase_state_path || null,
    prefer_live_phase: cfg.preferLivePhase,
  };
}

async function send(question) {
  addMessage("user", `<div class="answer">${esc(question)}</div>`);
  const pending = addMessage("assistant", `<div class="answer">Running semantic review workflow...</div>`);

  const response = await fetch("/api/hitl", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody(question)),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    pending.innerHTML = `<div class="answer error">${esc(err.detail || response.statusText)}</div>`;
    return;
  }

  pending.innerHTML = renderResponse(await response.json());
}

async function loadWindows() {
  const select = $("windowSelect");
  if (!select) return;

  const params = new URLSearchParams({
    metadata: runtimeDefaults.metadata || "",
    limit: $("windowLimit")?.value || "30",
  });
  const tokenFilter = $("tokenFilter")?.value?.trim();
  if (tokenFilter) params.set("token_id", tokenFilter);

  select.innerHTML = `<option>Loading windows...</option>`;
  const response = await fetch(`/api/windows?${params.toString()}`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    select.innerHTML = `<option>${esc(err.detail || "Failed to load windows")}</option>`;
    return;
  }

  const payload = await response.json();
  select.innerHTML = "";
  payload.windows.forEach((row) => {
    const option = document.createElement("option");
    option.value = row.window_id;
    option.textContent = `${row.window_start} | token ${row.token_id} | ${row.regime_name || "unlabeled"} | mean ${row.wind_speed_mean ?? "n/a"}`;
    select.appendChild(option);
  });
}

async function init() {
  const defaults = await fetch("/api/defaults").then((r) => r.json());
  runtimeDefaults = {
    metadata: defaults.metadata,
    live_phase_history_path: defaults.live_phase_history_path || null,
    live_phase_state_path: defaults.live_phase_state_path || null,
    live_raw_path: defaults.live_raw_path || null,
    numericSteps: defaults.numeric_forecast_steps || FIXED_CONFIG.numericSteps,
    numericMode: defaults.numeric_forecast_mode || FIXED_CONFIG.numericMode,
    numeric_model_path: defaults.numeric_model_path || null,
    phaseModelMode: "transition",
    phase_model_path: defaults.phase_model_path || null,
  };

  const sync = defaults.live_sync || {};
  const syncText = sync.attempted
    ? `Live sync ${sync.ok ? "ready" : "needs review"}`
    : `Live sync ${sync.status || "not run"}`;
  $("metadataStatus").textContent = `${defaults.metadata_exists ? "Metadata ready" : "Metadata missing"} | ${syncText}`;

  const syncMessage = sync.message ? `<div class="meta">${pill(sync.message)}</div>` : "";
  addMessage("assistant", `
    <div class="meta">${pill("workflow ready", "strong")}${pill("LLM follow-up on")}${pill("Memory/RAG on")}${pill("live METAR state")}</div>
    <div class="answer">Use the core action buttons or ask naturally. The interface now runs a fixed optimal HITL configuration, so no manual checkboxes or advanced path settings are required.</div>
    ${syncMessage}
  `);
}

$("chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  await send(question);
});

$("question").addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("chatForm").requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    $("question").value = button.dataset.prompt;
    $("question").focus();
  });
});

$("loadWindows")?.addEventListener("click", loadWindows);
$("metadata")?.addEventListener("change", loadWindows);
$("windowSelect")?.addEventListener("change", () => {
  selectedWindowId = $("windowSelect").value || null;
});

init().catch((error) => {
  addMessage("assistant", `<div class="answer error">${esc(error.message)}</div>`);
});
