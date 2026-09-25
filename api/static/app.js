const conversation = [];
const form = document.querySelector("#composer");
const question = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const chat = document.querySelector("#chat");
const welcome = document.querySelector("#welcome");
const scroller = document.querySelector("#chat-scroller");

function setText(element, text) {
  element.textContent = text ?? "";
}

function metadataLines(item) {
  const lines = [];

  if (item.source_file)
    lines.push(item.source_file);

  if (item.case_name)
    lines.push(item.case_name);

  if (item.citation)
    lines.push(`Citation: ${item.citation}`);

  if (item.court)
    lines.push(`Court: ${item.court}`);

  if (item.judgment_date)
    lines.push(`Judgment date: ${item.judgment_date}`);

  if (item.document_id)
    lines.push(`Document: ${item.document_id}`);

  if (item.chunk_index != null)
    lines.push(`Chunk: ${item.chunk_index}`);

  if (item.role)
    lines.push(`Role: ${item.role}`);

  if (item.char_start != null && item.char_end != null)
    lines.push(`Text range: ${item.char_start}–${item.char_end}`);

  return lines;
}

function appendText(parent, tag, text, className = "") {
  const node = document.createElement(tag);
  node.textContent = text ?? "";
  if (className) node.className = className;
  parent.append(node);
  return node;
}

function renderSources(parent, citations, chunks) {
  if (!citations?.length && !chunks?.length) return;

  const wrap = document.createElement("div");
  wrap.className = "sources";

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "sources-toggle";
  toggle.textContent = `View sources${citations?.length ? ` · ${citations.length}` : ""}`;
  wrap.append(toggle);

  const panel = document.createElement("div");
  panel.className = "source-panel";
  toggle.addEventListener("click", () => {
    panel.classList.toggle("open");
    toggle.textContent = panel.classList.contains("open")
      ? "Hide sources"
      : `View sources${citations?.length ? ` · ${citations.length}` : ""}`;
  });

  if (citations?.length) {
    const list = document.createElement("div");
    list.className = "citation-list";
    citations.forEach((citation) => {
      const card = document.createElement("article");
      card.className = "citation";
      appendText(
        card,
        "div",
        citation.case_name ? "JUDGMENT" : "SOURCE",
        "citation-kind"
      );
      metadataLines(citation).forEach((line) => appendText(card, "p", line));
      if (citation.element_type)
        appendText(card, "p", `Structure type: ${citation.element_type}`);
      list.append(card);
    });
    panel.append(list);
  }

  if (chunks?.length) {
    const details = document.createElement("details");
    details.className = "context";
    appendText(details, "summary", `Retrieved context · ${chunks.length} passages`);
    const list = document.createElement("div");
    list.className = "context-list";
    chunks.forEach((chunk) => {
      const card = document.createElement("article");
      card.className = "chunk";
      appendText(card, "p",
        `Rank ${chunk.rank}${typeof chunk.score === "number" ? ` · Score ${chunk.score.toFixed(3)}` : ""}`,
        "metadata");
      metadataLines(chunk).forEach((line) => appendText(card, "p", line, "metadata"));
      if (chunk.element_type) appendText(card, "p", `Type: ${chunk.element_type}`, "metadata");
      if (chunk.text) appendText(card, "p", chunk.text, "chunk-text");
      list.append(card);
    });
    details.append(list);
    panel.append(details);
  }

  wrap.append(panel);
  parent.append(wrap);
}

function addMessage(role, content, extras = {}) {
  const message = document.createElement("article");
  message.className = `message ${role}${extras.error ? " error" : ""}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Y" : "E";

  const contentWrap = document.createElement("div");
  contentWrap.className = "message-content";

  const body = document.createElement("div");
  body.className = "message-body";

  if (extras.loading) {
    const row = document.createElement("div");
    row.className = "loading-row";
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    spinner.setAttribute("aria-hidden", "true");
    row.append(spinner);
    appendText(row, "span", "Searching the EPFO legal corpus…");
    body.append(row);
  } else {
    body.textContent = content ?? "";
  }

  contentWrap.append(body);

  if (role === "assistant" && !extras.loading) {
    renderSources(contentWrap, extras.citations, extras.retrieved_chunks);
  }

  message.append(avatar, contentWrap);
  chat.append(message);

  welcome.hidden = conversation.length > 0;
  requestAnimationFrame(() => {
    scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
  });

  return message;
}

function apiError(status, body) {
  if (body?.error?.message) return body.error.message;
  if (status === 422) return "Please enter a valid question.";
  if (status === 503) return "The legal assistant service is temporarily unavailable. Please try again.";
  if (status === 504) return "The legal assistant took too long to respond. Please try again.";
  return "I couldn't complete that request. Please try again.";
}

async function sendQuestion(value) {
  const trimmed = value.trim();
  if (!trimmed || sendButton.disabled) return;

  const requestConversation = conversation.map(({ role, content }) => ({ role, content }));
  conversation.push({ role: "user", content: trimmed });
  addMessage("user", trimmed);

  question.value = "";
  autoResize();
  sendButton.disabled = true;

  const loading = addMessage("assistant", "", { loading: true });

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: trimmed,
        conversation: requestConversation
      })
    });

    const body = await response.json().catch(() => null);
    loading.remove();

    if (!response.ok) throw { status: response.status, body };

    conversation.push({
      role: "assistant",
      content: body.answer,
      citations: body.citations,
      retrieved_chunks: body.retrieved_chunks
    });

    addMessage("assistant", body.answer, body);
  } catch (error) {
    loading.remove();
    const content = error.status
      ? apiError(error.status, error.body)
      : "I couldn't complete that request. Please try again.";
    conversation.push({ role: "assistant", content });
    addMessage("assistant", content, { error: true });
  } finally {
    sendButton.disabled = false;
    question.focus();
  }
}

function resetConversation() {
  conversation.splice(0);
  chat.replaceChildren();
  welcome.hidden = false;
  question.value = "";
  autoResize();
  question.focus();
}

function autoResize() {
  question.style.height = "auto";
  question.style.height = `${Math.min(question.scrollHeight, 180)}px`;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuestion(question.value);
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

question.addEventListener("input", autoResize);

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => sendQuestion(button.dataset.question));
});

document.querySelector("#new-conversation").addEventListener("click", resetConversation);
document.querySelector("#mobile-new-chat").addEventListener("click", resetConversation);

fetch("/health")
  .then((response) => {
    if (!response.ok) throw new Error();
    setText(document.querySelector("#service-status"), "Service available");
    document.querySelector("#status-dot").className = "status-dot available";
  })
  .catch(() => {
    setText(document.querySelector("#service-status"), "Service unavailable");
    document.querySelector("#status-dot").className = "status-dot unavailable";
  });

autoResize();
question.focus();
