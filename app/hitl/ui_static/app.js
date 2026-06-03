const $ = (id) => document.getElementById(id);
const messages = $("messages");
const sessionKey = "hitl_thread_id";
let threadId = sessionStorage.getItem(sessionKey);
if (!threadId) {
  threadId = `hitl_${crypto.randomUUID()}`;
  sessionStorage.setItem(sessionKey, threadId);
}

function parseTokens(value) {
  const clean = value.trim();
  if (!clean) return null;
  const tokens = clean
    .replace("[", "")
    .replace("]", "")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number.parseInt(part, 10))
    .filter(Number.isFinite);
  return tokens.length ? tokens : null;
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

function renderRows(title, rows, type) {
  if (!rows || rows.length === 0) return "";
  const body = rows
    .map((row) => {
      if (type === "candidate") {
        return `<div class="row"><strong>Token ${esc(row.token_id)}</strong><span>${esc(row.regime_name)}</span><span>${esc(row.probability)}</span><span>n=${esc(row.count)}</span></div>`;
      }
      if (type === "analog") {
        return `<div class="row"><strong>${esc(row.evidence_id)}</strong><span>${esc(row.history_tokens)}</span><span>Next ${esc(row.next_token_id)}</span><span>${esc(row.future_window_start)}</span></div>`;
      }
      if (type === "numeric") {
        return `<div class="row"><strong>Step ${esc(row.step)}</strong><span>${esc(row.datetime)}</span><span>${esc(row.wind_speed)} m/s</span><span>${esc(row.lower)}-${esc(row.upper)}</span></div>`;
      }
      const dist = typeof row.retrieval_distance === "number" ? row.retrieval_distance.toFixed(3) : row.retrieval_distance;
      return `<div class="row"><strong>${esc(row.window_id)}</strong><span>${esc(row.regime_name)}</span><span>${esc(dist)}</span><button type="button" class="use-window" data-window="${esc(row.window_id)}">Use</button></div>`;
    })
    .join("");
  return `<div class="section"><h2>${esc(title)}</h2><div class="rows">${body}</div></div>`;
}

function renderResponse(payload) {
  const s = payload.summary || {};
  const meta = [
    `intent: ${s.intent ?? "n/a"}`,
    `confidence: ${s.confidence ?? "n/a"}`,
    `mode: ${s.mode ?? "n/a"}`,
    s.retrieval_context ? `retrieval: ${s.retrieval_context}` : null,
    s.explanation_context ? `explain context: ${s.explanation_context}` : null,
    s.explanation_mode ? `explain: ${s.explanation_mode}` : null,
    s.forecast_model_source ? `model: ${s.forecast_model_source}` : null,
    s.model_warning ? `warning: ${s.model_warning}` : null,
    `llm: ${s.llm_requested ? (s.llm_available ? "available" : "unavailable") : "off"}`,
    `window: ${s.window_id ?? "n/a"}`,
    s.phase_transition_source ? `phase source: ${s.phase_transition_source}` : null,
    s.agent_graph ? `agent: ${s.agent_graph.engine}/${s.agent_graph.route}` : null,
    s.router && s.router.source ? `router: ${s.router.source}` : null,
    s.thread_id ? `thread: ${s.thread_id}` : null,
  ]
    .filter(Boolean)
    .map((item) => `<span class="pill">${esc(item)}</span>`)
    .join("");
  const liveState = s.current_live_state || {};
  const liveInfo = s.live_token_sequence
    ? `<div class="meta"><span class="pill">live tokens: ${esc(s.live_token_sequence.join(", "))}</span><span class="pill">live state: ${esc(liveState.window_id || "n/a")}</span><span class="pill">live regime: ${esc(liveState.regime_name || "n/a")}</span></div>`
    : "";
  const criticWarnings = s.critic_warnings && s.critic_warnings.length
    ? `<div class="meta">${s.critic_warnings.map((item) => `<span class="pill warning">${esc(item)}</span>`).join("")}</div>`
    : "";
  const memoryDebug = s.memory_rag_debug
    ? `<div class="section">
        <h2>Memory RAG Debug</h2>
        <div class="meta">
          <span class="pill">LLM attempted: ${esc(String(Boolean(s.memory_rag_debug.llm_attempted)))}</span>
          <span class="pill">LLM used: ${esc(String(Boolean(s.memory_rag_debug.llm_used)))}</span>
          ${s.memory_rag_debug.llm_output_format ? `<span class="pill">format: ${esc(s.memory_rag_debug.llm_output_format)}</span>` : ""}
        </div>
        ${(s.memory_rag_debug.llm_blockers || []).length
          ? `<div class="answer">LLM blockers: ${esc((s.memory_rag_debug.llm_blockers || []).join(", "))}</div>`
          : ""}
        <div class="meta">
          ${(s.memory_rag_debug.selected_chunk_kinds || []).map((item) => `<span class="pill">${esc(item)}</span>`).join("")}
        </div>
        ${(s.memory_rag_debug.selected_chunk_ids || []).length
          ? `<div class="answer">Chunks: ${esc((s.memory_rag_debug.selected_chunk_ids || []).join(", "))}</div>`
          : ""}
        ${s.memory_rag_debug.llm_fallback_reason
          ? `<div class="answer">Fallback: ${esc(s.memory_rag_debug.llm_fallback_reason)}</div>`
          : ""}
      </div>`
    : "";
  const reviewState = s.review_state
    ? `<div class="section">
        <h2>Review State</h2>
        <div class="meta">
          <span class="pill">status: ${esc(s.review_state.status || s.review_state.action || "n/a")}</span>
          ${s.review_state.model_predicted_token !== null && s.review_state.model_predicted_token !== undefined
            ? `<span class="pill">model next token: ${esc(s.review_state.model_predicted_token)}</span>`
            : ""}
          ${s.review_state.human_preferred_token !== null && s.review_state.human_preferred_token !== undefined
            ? `<span class="pill">human token: ${esc(s.review_state.human_preferred_token)}</span>`
            : ""}
        </div>
        ${s.review_state.note ? `<div class="answer">${esc(s.review_state.note)}</div>` : ""}
      </div>`
    : "";

  return `
    <div class="meta">${meta}</div>
    ${liveInfo}
    ${criticWarnings}
    ${s.phase_low_support ? `<div class="meta"><span class="pill warning">low support: n=${esc(s.phase_support)}</span></div>` : ""}
    <div class="answer">${esc(s.answer || "No answer returned.")}</div>
    ${renderRows("Phase candidates", s.top_phase_candidates, "candidate")}
    ${renderRows("Transition analogs", s.transition_analogs, "analog")}
    ${renderRows("Similar windows for selected query state", s.similar_windows, "similar")}
    ${reviewState}
    ${memoryDebug}
    ${s.human_review_prompt ? `<div class="section"><h2>Review</h2><div class="answer">${esc(s.human_review_prompt)}</div></div>` : ""}
  `;
}

async function send(question) {
  addMessage("user", `<div class="answer">${esc(question)}</div>`);
  const pending = addMessage("assistant", `<div class="answer">Running HITL pipeline...</div>`);

  const response = await fetch("/api/hitl", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      thread_id: threadId,
      metadata: $("metadata").value,
      window_id: $("windowId").value || null,
      latest: !$("windowId").value,
      top_k: Number.parseInt($("topK").value, 10),
      enable_llm: $("enableLlm").checked,
      enable_agent_graph: $("enableAgentGraph").checked,
      enable_rag: $("enableRag").checked,
      phase_tokens: null,
      phase_history_length: Number.parseInt($("phaseHistory").value, 10),
      phase_horizon_steps: Number.parseInt($("phaseStep").value, 10),
      phase_top_k: Number.parseInt($("phaseTopK").value, 10),
      phase_analog_k: Number.parseInt($("phaseAnalogK").value, 10),
      phase_min_support: Number.parseInt($("phaseMinSupport").value, 10),
      live_raw_path: $("liveRawPath").value || null,
      numeric_forecast_steps: Number.parseInt($("numericSteps").value, 10),
      numeric_forecast_mode: $("numericMode").value,
      numeric_model_path: $("numericModelPath").value || null,
      phase_model_mode: $("phaseModelMode").value,
      phase_model_path: $("phaseModelPath").value || null,
      live_phase_history_path: $("livePhaseHistory").value || null,
      live_phase_state_path: $("livePhaseState").value || null,
      prefer_live_phase: $("preferLivePhase").checked,
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    pending.innerHTML = `<div class="answer error">${esc(err.detail || response.statusText)}</div>`;
    return;
  }

  pending.innerHTML = renderResponse(await response.json());
}

async function init() {
  const defaults = await fetch("/api/defaults").then((r) => r.json());
  $("metadata").value = defaults.metadata;
  $("livePhaseHistory").value = defaults.live_phase_history_path || "";
  $("livePhaseState").value = defaults.live_phase_state_path || "";
  $("liveRawPath").value = defaults.live_raw_path || "";
  $("numericSteps").value = defaults.numeric_forecast_steps || 6;
  $("numericMode").value = defaults.numeric_forecast_mode || "auto";
  $("numericModelPath").value = defaults.numeric_model_path || "";
  $("phaseModelMode").value = defaults.phase_model_mode || "auto";
  $("phaseModelPath").value = defaults.phase_model_path || "";
  const sync = defaults.live_sync || {};
  const syncText = sync.attempted
    ? `Live sync: ${sync.ok ? "updated" : "failed"}`
    : `Live sync: ${sync.status || "not attempted"}`;
  $("metadataStatus").textContent = `${defaults.metadata_exists ? "Metadata ready" : "Metadata missing"} | ${syncText}`;
  await loadWindows();
  const syncMessage = sync.message ? `<div class="meta"><span class="pill">${esc(sync.message)}</span></div>` : "";
  addMessage("assistant", `<div class="answer">HITL review interface ready.</div>${syncMessage}`);
}

async function loadWindows() {
  const params = new URLSearchParams({
    metadata: $("metadata").value,
    limit: $("windowLimit").value || "30",
  });
  const tokenFilter = $("tokenFilter").value.trim();
  if (tokenFilter) params.set("token_id", tokenFilter);

  const select = $("windowSelect");
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

$("chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) return;
  $("question").value = "";
  await send(question);
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    $("question").value = button.dataset.prompt;
    $("question").focus();
  });
});

$("loadWindows").addEventListener("click", loadWindows);
$("metadata").addEventListener("change", loadWindows);
$("windowSelect").addEventListener("change", () => {
  $("windowId").value = $("windowSelect").value;
});

messages.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const windowId = target.dataset.window;
  if (!windowId) return;
  $("windowId").value = windowId;
  addMessage("assistant", `<div class="answer">Selected query window: ${esc(windowId)}</div>`);
});

init().catch((error) => {
  addMessage("assistant", `<div class="answer error">${esc(error.message)}</div>`);
});
