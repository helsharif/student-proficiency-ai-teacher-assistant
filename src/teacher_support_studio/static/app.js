const state = {
  scope: "student",
  classId: null,
  studentId: null,
  summary: null,
  skillEmojis: {},
};

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

function skillEmoji(label) {
  const skill = String(label).toLowerCase();
  return state.skillEmojis[skill] || "✏️🔢";
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
    const [classes, skillEmojis] = await Promise.all([
      api("/api/v1/classes"),
      api("/api/v1/skill-emojis"),
    ]);
    state.skillEmojis = skillEmojis;
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
    const path = `/api/v1/classes/${state.classId}/students/${state.studentId}/summary`;
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
  $("#readiness-copy").textContent =
    `Estimated chance of first-attempt success on plausible next-practice scenarios. ` +
    `Only skills with at least ${summary.readiness_min_interactions} prior interactions are included.`;
  elements.input.placeholder = "Ask about this student…";

  elements.metricGrid.innerHTML = summary.cards.map((card) => `
    <article class="metric-card ${card.tone}">
      <div class="metric-label">${escapeHtml(card.label)}</div>
      <strong class="metric-value">${escapeHtml(card.value)}</strong>
      <div class="metric-detail">${escapeHtml(card.detail)}</div>
    </article>`).join("");
  elements.metricGrid.classList.toggle("hidden", summary.cards.length === 0);

  const orderedReadiness = [...summary.readiness]
    .sort((left, right) => right.estimated_readiness - left.estimated_readiness);
  const highestReadiness = orderedReadiness.slice(0, 5);
  const highestLabels = new Set(highestReadiness.map((skill) => skill.label));
  const lowestReadiness = orderedReadiness
    .slice(-5)
    .filter((skill) => !highestLabels.has(skill.label));
  const readinessGroups = orderedReadiness.length
    ? [
      { label: "Highest estimated readiness", skills: highestReadiness },
      { label: "Lowest estimated readiness", skills: lowestReadiness },
    ].filter((group) => group.skills.length)
    : [];
  elements.skillChart.innerHTML = readinessGroups.length
    ? readinessGroups.map((group) => `
      <section class="readiness-group">
        <h4>${group.label}</h4>
        ${group.skills.map((skill) => `
          <div class="readiness-row">
            <strong class="readiness-name"><span class="skill-emoji" aria-hidden="true">${skillEmoji(skill.label)}</span>${escapeHtml(skill.label)}</strong>
            <div class="bar-track">
              <div class="bar-fill" style="width:${Math.max(2, skill.estimated_readiness * 100)}%"></div>
            </div>
            <strong class="readiness-value">${Math.round(skill.estimated_readiness * 100)}%</strong>
          </div>`).join("")}
      </section>`).join("")
    : `<p>No skills currently meet the minimum of ${summary.readiness_min_interactions} prior interactions.</p>`;
  elements.questions.innerHTML = summary.suggested_questions.map((question) =>
    `<button class="question-chip" type="button" data-question="${escapeHtml(question)}">${escapeHtml(question)}</button>`
  ).join("");
  $("#dashboard-evidence").innerHTML = summary.evidence.map((item) => `<li>${escapeHtml(item)}</li>`).join("");

  document.querySelectorAll(".question-chip").forEach((button) => {
    button.addEventListener("click", () =>
      askQuestion(button.dataset.question, { showUserMessage: false })
    );
  });
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
  const latestMessage = elements.messages.lastElementChild;
  elements.messages.scrollTop = Math.max(
    0,
    latestMessage.offsetTop - elements.messages.offsetTop,
  );
}

async function askQuestion(question, { showUserMessage = true } = {}) {
  const cleanQuestion = String(question || "").trim();
  if (!cleanQuestion) return;
  if (showUserMessage) addUserMessage(cleanQuestion);
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
        student_id: state.studentId,
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
