const $ = (id) => document.getElementById(id);
// Same name, letter and colour as the connector cards on the Sources page.
const SOURCES = {
  messages: { name: "iMessage", letter: "M", color: "#34C759" },
  whatsapp: { name: "WhatsApp", letter: "W", color: "#25D366" },
};
const PAGE = 60; // sessions rendered at a time in a chat

let chats = [];
let srcFilter = "all";
let current = null; // { row, data, from }

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  Object.assign(node, attrs);
  if (text !== undefined) node.textContent = text;
  return node;
}

function logo(source) {
  const s = SOURCES[source];
  const node = el("span", { className: "logo", title: s.name }, s.letter);
  node.style.setProperty("--c", s.color);
  return node;
}

async function getJSON(path) {
  const res = await fetch(path);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Something went wrong");
  return data;
}

const fmtDate = (t) => new Date(t * 1000).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
const fmtTime = (t) => new Date(t * 1000).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
const fmtFull = (t) => `${fmtDate(t)}, ${fmtTime(t)}`;
const n = (x) => x.toLocaleString();
const plural = (x, word) => `${n(x)} ${word}${x === 1 ? "" : "s"}`;

// The extractors only have numbers and ids, never contact names. Make them readable.
function phone(raw) {
  const d = raw.replace(/\D/g, "");
  if (d.length === 11 && d[0] === "1") return `+1 (${d.slice(1, 4)}) ${d.slice(4, 7)}-${d.slice(7)}`;
  if (d.length === 10 && !raw.startsWith("+")) return `(${d.slice(0, 3)}) ${d.slice(3, 6)}-${d.slice(6)}`;
  return d.length >= 7 ? `+${d}` : raw;
}
function person(id) {
  if (!id) return "";
  const local = id.split("@")[0];
  if (id.includes("@s.whatsapp.net") || /^\+?[\d\s()-]+$/.test(id)) return phone(local);
  return id; // an email address
}
function chatLabel(row) {
  if (row.group) return `Group of ${row.people}`;
  return person(row.chat);
}
const senderLabel = person;

const DAY = 86400000;
function when(t) {
  const d = new Date(t * 1000), now = new Date();
  const days = Math.floor((new Date(now.toDateString()) - new Date(d.toDateString())) / DAY);
  if (days === 0) return fmtTime(t);
  if (days === 1) return "Yesterday";
  if (days < 7) return d.toLocaleDateString(undefined, { weekday: "long" });
  if (d.getFullYear() === now.getFullYear()) return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

// ---- stats + list ----

function statCard(icon, big, label, extraClass = "") {
  const card = el("div", { className: `stat ${extraClass}` });
  const body = el("div", { className: "stat-body" });
  body.append(el("span", { className: "stat-big" }, big), el("span", { className: "stat-label" }, label));
  card.append(icon, body);
  return card;
}

function renderStats() {
  const stats = $("stats");
  stats.replaceChildren();
  for (const src of Object.keys(SOURCES)) {
    const rows = chats.filter((c) => c.source === src);
    if (!rows.length) continue;
    const msgs = rows.reduce((a, c) => a + c.messages, 0);
    const mine = rows.reduce((a, c) => a + c.mine, 0);
    const sessions = rows.reduce((a, c) => a + c.sessions, 0);
    stats.append(statCard(logo(src), n(msgs),
      `${SOURCES[src].name} messages, ${n(mine)} of them yours. ${plural(rows.length, "chat")}, ${plural(sessions, "session")}.`));
  }
  const quiet = chats.filter((c) => c.mine === 0).length;
  stats.append(statCard(el("span", { className: "logo quiet" }, "!"), n(quiet),
    "chats you never replied in, like codes and alerts. Probably not worth training on.", "stat-quiet"));
}

function renderList() {
  const q = $("chat-filter").value.trim().toLowerCase();
  const onlyMine = $("only-mine").checked;
  const rows = chats.filter((c) =>
    (srcFilter === "all" || c.source === srcFilter) &&
    (!onlyMine || c.mine > 0) &&
    (!q || c.chat.toLowerCase().includes(q) || chatLabel(c).toLowerCase().includes(q) || c.chat.replace(/\D/g, "").includes(q.replace(/\D/g, "") || "\u0000"))
  );
  $("list-count").textContent = plural(rows.length, "chat");
  const list = $("chat-list");
  list.replaceChildren();
  for (const c of rows) {
    const b = el("button", { type: "button", className: "chat-row" });
    if (current && current.row.source === c.source && current.row.chat_id === c.chat_id) b.classList.add("is-current");
    const text = el("span", { className: "chat-text" });
    const top = el("span", { className: "chat-row-top" });
    top.append(el("span", { className: "chat-name" }, chatLabel(c)));
    top.append(el("span", { className: "chat-date" }, when(c.last)));
    const meta = el("span", { className: "chat-meta" }, plural(c.messages, "message"));
    if (c.mine === 0) meta.append(" · ", el("span", { className: "quiet-text" }, "never replied"));
    text.append(top, el("span", { className: "chat-preview" }, c.preview), meta);
    b.append(logo(c.source), text);
    b.addEventListener("click", () => openChat(c));
    const li = el("li");
    li.append(b);
    list.append(li);
  }
}

// ---- chat viewer ----

// Scroll only the message pane (never the page), leaving room for its sticky header.
function scrollPaneTo(node) {
  if (!node) return;
  const viewer = $("viewer");
  const head = viewer.querySelector(".viewer-head");
  viewer.scrollTop = node.offsetTop - (head ? head.offsetHeight : 0) - 8;
}

async function openChat(row, focusSession) {
  const viewer = $("viewer");
  viewer.replaceChildren(el("p", { className: "muted viewer-hint" }, "Loading…"));
  let data;
  try {
    data = await getJSON(`/api/data/${row.source}/${row.chat_id}`);
  } catch (err) {
    viewer.replaceChildren(el("p", { className: "error viewer-hint" }, err.message));
    return;
  }
  const total = data.sessions.length;
  let from = Math.max(0, total - PAGE);
  if (focusSession !== undefined) from = Math.min(from, focusSession);
  current = { row, data, from };
  renderList();
  renderChat();
  const target = focusSession !== undefined ? $(`s-${focusSession}`) : viewer.lastElementChild;
  if (focusSession !== undefined) scrollPaneTo(target);
  else viewer.scrollTop = viewer.scrollHeight;
  if (focusSession !== undefined && target) target.classList.add("flash");
}

function renderChat() {
  const { row, data, from } = current;
  const viewer = $("viewer");
  viewer.replaceChildren();

  const head = el("div", { className: "viewer-head" });
  const info = el("div");
  info.append(el("h2", {}, chatLabel(row)));
  info.append(el("p", { className: "muted" },
    `${SOURCES[row.source].name} · ${plural(row.messages, "message")}, ${n(row.mine)} from you · since ${fmtDate(row.first)}`));
  head.append(logo(row.source), info);
  viewer.append(head);

  if (from > 0) {
    const more = el("button", { type: "button", className: "btn more" }, `Show earlier messages (${plural(from, "session")} hidden)`);
    more.addEventListener("click", () => {
      current.from = Math.max(0, from - PAGE);
      renderChat();
      scrollPaneTo($(`s-${from}`));
    });
    viewer.append(more);
  }

  data.sessions.slice(from).forEach((s, k) => {
    const i = from + k;
    const sec = el("section", { className: "session", id: `s-${i}` });
    const sameDay = fmtDate(s.start) === fmtDate(s.end);
    const head = el("div", { className: "session-head" });
    const day = (t) => { const w = when(t); return /\d:\d\d/.test(w) ? "Today" : w; };
    const label = el("span", {}, `${day(s.start)} · ${fmtTime(s.start)}${s.end !== s.start ? ` – ${sameDay ? fmtTime(s.end) : `${day(s.end)}, ${fmtTime(s.end)}`}` : ""}`);
    if (!s.messages.some((m) => m.me)) label.append(" · ", el("span", { className: "quiet-text" }, "you didn't reply"));
    head.append(label);
    sec.append(head);
    let prevFrom = null;
    for (const m of s.messages) {
      const wrap = el("div", { className: `msg ${m.me ? "me" : "them"}` });
      if (row.group && !m.me && m.from !== prevFrom) wrap.append(el("span", { className: "msg-from" }, senderLabel(m.from)));
      prevFrom = m.me ? null : m.from;
      wrap.append(el("div", { className: "bubble", title: fmtFull(m.t) }, m.text));
      sec.append(wrap);
    }
    viewer.append(sec);
  });
}

// ---- search ----

async function runSearch(q) {
  const viewer = $("viewer");
  current = null;
  renderList();
  viewer.replaceChildren(el("p", { className: "muted viewer-hint" }, "Searching…"));
  let hits;
  try {
    ({ hits } = await getJSON(`/api/data/search?q=${encodeURIComponent(q)}`));
  } catch (err) {
    viewer.replaceChildren(el("p", { className: "error viewer-hint" }, err.message));
    return;
  }
  viewer.replaceChildren();
  const head = el("div", { className: "viewer-head" });
  const info = el("div");
  info.append(el("h2", {}, `“${q}”`));
  info.append(el("p", { className: "muted" }, hits.length >= 200 ? "The 200 newest matches" : plural(hits.length, "match").replace("matchs", "matches")));
  head.append(info);
  viewer.append(head);
  if (!hits.length) {
    viewer.append(el("p", { className: "muted viewer-hint" }, "No messages contain that text."));
    return;
  }
  const lower = q.toLowerCase();
  const ul = el("ul", { className: "hits" });
  for (const h of hits) {
    const row = chats.find((c) => c.source === h.source && c.chat_id === h.chat_id);
    const b = el("button", { type: "button", className: "hit" });
    const text = el("span", { className: "chat-text" });
    const top = el("span", { className: "chat-row-top" });
    top.append(el("span", { className: "chat-name" }, `${h.me ? "You" : senderLabel(h.from) || "Them"}${row && row.group ? ` in ${chatLabel(row)}` : ""}`));
    top.append(el("span", { className: "chat-date" }, when(h.t)));
    const body = el("span", { className: "hit-text" });
    const at = h.text.toLowerCase().indexOf(lower);
    const start = Math.max(0, at - 60);
    body.append((start ? "…" : "") + h.text.slice(start, at), el("mark", {}, h.text.slice(at, at + q.length)), h.text.slice(at + q.length, at + q.length + 120));
    text.append(top, body);
    b.append(logo(h.source), text);
    if (row) b.addEventListener("click", () => openChat(row, h.session));
    const li = el("li");
    li.append(b);
    ul.append(li);
  }
  viewer.append(ul);
}

// ---- wiring ----

$("chat-filter").addEventListener("input", renderList);
$("only-mine").addEventListener("change", renderList);
document.querySelectorAll(".chip-btn").forEach((b) =>
  b.addEventListener("click", () => {
    srcFilter = b.dataset.src;
    document.querySelectorAll(".chip-btn").forEach((x) => x.classList.toggle("is-on", x === b));
    renderList();
  })
);
$("search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("search-q").value.trim();
  if (q) runSearch(q);
});

(async () => {
  try {
    ({ chats } = await getJSON("/api/data"));
  } catch (err) {
    $("empty").hidden = false;
    $("empty").querySelector("p").textContent = err.message;
    return;
  }
  if (!chats.length) {
    $("empty").hidden = false;
    return;
  }
  $("browser").hidden = false;
  renderStats();
  renderList();
  openChat(chats[0]);
})();
