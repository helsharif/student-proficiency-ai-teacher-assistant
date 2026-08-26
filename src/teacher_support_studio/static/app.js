const state = { scope: "class", classId: null, studentId: null, summary: null };

const $ = (selector) => document.querySelector(selector);
const elements = {
  classSelect: $("#class-select"),
  studentSelect: $("#student-select"),
  studentField: $("#student-field"),
  workspace: $("#workspace"),
  loading: $("#loading"),
  error: $("#error"),
  metricGrid: $("#metric-grid"),
  skillChart: $("#skill-chart"),
  trendChart: $("#trend-chart"),
  questions: $("#suggested-questions"),
  messages: $("#messages"),
  form: $("#chat-form"),
  input: $("#question-input"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function setBusy(isBusy) {
  elements.loading.classList.toggle("hidden", !isBusy);
  if (isBusy) elements.workspace.classList.add("hidden");
}

function showError(error) {
  elements.error.textContent = error.message || String(error);
  elements.error.classList.remove("hidden");
  elements.loading.classList.add("hidden");
}

async function initialize() {
  try {
    const classes = await api("/api/v1/classes");
    elements.classSelect.innerHTML = classes
      .map((item) => `<option value="${item.id}">${escapeHtml(item.label)} · ${escapeHtml(item.detail)}</option>`)
      .join("");
    state.classId = Number(classes[0].id);
    await refreshStudents();
    await refreshSummary();
  } catch (error) {
    showError(error);
  }
}

async function refreshStudents() {
  const students = await api(`/api/v1/classes/${state.classId}/students`);
  elements.studentSelect.innerHTML = students
    .map((item) => `<option value="${item.id}">${escapeHtml(item.label)} · ${escapeHtml(item.detail)}</option>`)
    .join("");
  state.studentId = students.length ? Number(students[0].id) : null;
}

async function refreshSummary() {
  setBusy(true);
  elements.error.classList.add("hidden");
  try {
    const path = state.scope === "class"
      ? `/api/v1/classes/${state.classId}/summary`
      : `/api/v1/classes/${state.classId}/students/${state.studentId}/summary`;
    state.summary = await api(path);
    renderSummary(state.summary);
    resetConversation();
    elements.workspace.classList.remove("hidden");
  } catch (error) {
    showError(error);
  } finally {
    elements.loading.classList.add("hidden");
  }
}

function renderSummary(summary) {
  $("#context-label").textContent = summary.context_label;
  $("#entity-label").textContent = summary.entity_label;
  $("#headline").textContent = summary.headline;
  elements.input.placeholder = summary.scope === "class" ? "Ask about this class…" : "Ask about this student…";

  elements.metricGrid.innerHTML = summary.cards.map((card) => `
    <article class="metric-card ${card.tone}">
      <div class="metric-label">${escapeHtml(card.label)}</div>
      <strong class="metric-value">${escapeHtml(card.value)}</strong>
      <div class="metric-detail">${escapeHtml(card.detail)}</div>
    </article>`).join("");

  elements.skillChart.innerHTML = summary.skills.length
    ? summary.skills.map((skill) => `
      <div class="skill-row" title="${escapeHtml(skill.interactions)} recent interactions">
        <span class="skill-name">${escapeHtml(skill.label)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, skill.success_rate * 100)}%"></div></div>
        <span class="skill-value">${Math.round(skill.success_rate * 100)}%</span>
        <span class="skill-meta">${skill.interactions} interactions${skill.students ? ` · ${skill.students} students` : ""}</span>
      </div>`).join("")
    : "<p>No recent skill data is available.</p>";

  renderTrend(summary.trend);
  elements.questions.innerHTML = summary.suggested_questions.map((question) =>
    `<button class="question-chip" type="button" data-question="${escapeHtml(question)}">${escapeHtml(question)}</button>`
  ).join("");
  $("#dashboard-evidence").innerHTML = summary.evidence.map((item) => `<li>${escapeHtml(item)}</li>`).join("");

  document.querySelectorAll(".question-chip").forEach((button) => {
    button.addEventListener("click", () => askQuestion(button.dataset.question));
  });
}

function renderTrend(points) {
  if (!points.length) {
    elements.trendChart.innerHTML = "<p>No trend data is available.</p>";
    return;
  }
  const width = 420;
  const height = 205;
  const pad = 25;
  const xStep = points.length > 1 ? (width - pad * 2) / (points.length - 1) : 0;
  const coordinates = points.map((point, index) => ({
    x: pad + index * xStep,
    y: height - pad - point.success_rate * (height - pad * 2),
    value: Math.round(point.success_rate * 100),
  }));
  const line = coordinates.map((point) => `${point.x},${point.y}`).join(" ");
  const area = `${pad},${height - pad} ${line} ${width - pad},${height - pad}`;
  elements.trendChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="First-attempt success trend from earlier to recent interactions">
      <defs><linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#62aaa4" stop-opacity=".35"/><stop offset="100%" stop-color="#62aaa4" stop-opacity="0"/></linearGradient></defs>
      <line class="trend-grid" x1="${pad}" y1="${height * .25}" x2="${width - pad}" y2="${height * .25}" />
      <line class="trend-grid" x1="${pad}" y1="${height * .5}" x2="${width - pad}" y2="${height * .5}" />
      <line class="trend-grid" x1="${pad}" y1="${height * .75}" x2="${width - pad}" y2="${height * .75}" />
      <polygon class="trend-area" points="${area}" />
      <polyline class="trend-line" points="${line}" />
      ${coordinates.map((point) => `<circle class="trend-dot" cx="${point.x}" cy="${point.y}" r="4"/><text class="trend-label" x="${point.x}" y="${point.y - 10}" text-anchor="middle">${point.value}%</text>`).join("")}
    </svg>`;
}

function resetConversation() {
  elements.messages.innerHTML = `<div class="assistant-message intro-message"><p>Ask about the patterns shown for <strong>${escapeHtml(state.summary.entity_label)}</strong>.</p></div>`;
}

function addUserMessage(question) {
  elements.messages.insertAdjacentHTML("beforeend", `<div class="user-message">${escapeHtml(question)}</div>`);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function addAssistantMessage(answer) {
  const modeLabel = answer.response_mode === "openai" ? "OpenAI response" : "Guided local response";
  elements.messages.insertAdjacentHTML("beforeend", `
    <div class="assistant-message">
      <h4>What I noticed</h4><p>${escapeHtml(answer.what_i_noticed)}</p>
      <h4>What you might try</h4><p>${escapeHtml(answer.what_you_might_try)}</p>
      <h4>What to keep in mind</h4><p>${escapeHtml(answer.what_to_keep_in_mind)}</p>
      <details class="message-evidence"><summary>Show supporting evidence</summary><ul>${answer.supporting_evidence.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></details>
      <span class="response-mode">${modeLabel}</span>
    </div>`);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

async function askQuestion(question) {
  const cleanQuestion = String(question || "").trim();
  if (!cleanQuestion) return;
  addUserMessage(cleanQuestion);
  elements.input.value = "";
  const sendButton = elements.form.querySelector("button");
  sendButton.disabled = true;
  sendButton.textContent = "Thinking…";
  try {
    const answer = await api("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({
        scope: state.scope,
        class_id: state.classId,
        student_id: state.scope === "student" ? state.studentId : null,
        question: cleanQuestion,
      }),
    });
    addAssistantMessage(answer);
  } catch (error) {
    elements.messages.insertAdjacentHTML("beforeend", `<div class="assistant-message"><p>I couldn't prepare a response: ${escapeHtml(error.message)}</p></div>`);
  } finally {
    sendButton.disabled = false;
    sendButton.innerHTML = 'Send <span aria-hidden="true">→</span>';
  }
}

document.querySelectorAll(".focus-button").forEach((button) => {
  button.addEventListener("click", async () => {
    state.scope = button.dataset.scope;
    document.querySelectorAll(".focus-button").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    elements.studentField.classList.toggle("hidden", state.scope !== "student");
    await refreshSummary();
  });
});

elements.classSelect.addEventListener("change", async () => {
  state.classId = Number(elements.classSelect.value);
  await refreshStudents();
  await refreshSummary();
});

elements.studentSelect.addEventListener("change", async () => {
  state.studentId = Number(elements.studentSelect.value);
  await refreshSummary();
});

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await askQuestion(elements.input.value);
});

initialize();
