const $ = (id) => document.getElementById(id);
// Same name, letter, colour and icon as the connector cards on the Sources page.
const SOURCES = {
  messages: { name: "Messages", letter: "M", color: "#34C759" },
  whatsapp: { name: "WhatsApp", letter: "W", color: "#25D366" },
};
const PAGE = 60; // sessions fetched and shown at a time in a chat

let chats = [];
let srcFilter = "all";
let current = null; // { row, total, from, to, sessions }: sessions [from, to) of total are loaded and on screen
// Bumped by every chat open and search. Both draw into the same pane, so a slow answer that comes back
// after a newer click or search is dropped instead of replacing what the user asked for last.
let viewSeq = 0;

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  Object.assign(node, attrs);
  if (text !== undefined) node.textContent = text;
  return node;
}

function logo(source) {
  const s = SOURCES[source];
  const node = el("span", { className: "logo has-icon", title: s.name }, s.letter);
  node.style.setProperty("--c", s.color);
  node.style.setProperty("--icon", `url("/static/icons/${source}.svg")`);
  return node;
}

async function getJSON(path) {
  let res;
  try {
    res = await fetch(path);
  } catch {
    throw new Error("Can't reach the Nybble server. Is it still running?");
  }
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

function renderStats() {
  const quiet = chats.filter((c) => c.mine === 0).length;
  const stats = $("stats");
  stats.hidden = quiet === 0;
  if (!quiet) return;
  const body = el("div", { className: "stat-body" });
  body.append(el("span", { className: "stat-big" }, n(quiet)), el("span", { className: "stat-label" }, "chats you never replied in"));
  const card = el("div", { className: "stat stat-quiet" });
  card.append(el("span", { className: "logo quiet", ariaHidden: "true" }, "!"), body);
  stats.replaceChildren(card);
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
    b.dataset.key = key(c);
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
  markCurrent();
}

const key = (c) => `${c.source}:${c.chat_id}`;

// Highlight the open chat in place. Rebuilding the list here would destroy the row the user just
// clicked, and with it their keyboard focus.
function markCurrent() {
  const open = current && key(current.row);
  for (const b of $("chat-list").children) {
    const row = b.firstChild;
    const on = row.dataset.key === open;
    row.classList.toggle("is-current", on);
    if (on) row.setAttribute("aria-current", "true");
    else row.removeAttribute("aria-current");
  }
}

// The viewer isn't a live region (a screen reader would read out the whole chat), so say what
// changed here in a sentence instead.
function announce(text) {
  $("status").textContent = text;
}

// ---- chat viewer ----

// Scroll only the message pane (never the page), leaving room for its sticky header.
function scrollPaneTo(node) {
  if (!node) return;
  const viewer = $("viewer");
  const head = viewer.querySelector(".viewer-head");
  viewer.scrollTop = node.offsetTop - (head ? head.offsetHeight : 0) - 8;
}

// Sessions [start, end) of a chat (0 = oldest). With no range the server sends the newest PAGE.
// A chat is never fetched whole: long ones run to thousands of sessions.
function fetchWindow(row, start, end) {
  const range = start === undefined ? "" : `?start=${start}&end=${end}`;
  return getJSON(`/api/data/${row.source}/${row.chat_id}${range}`);
}

async function openChat(row, focusSession) {
  const seq = ++viewSeq;
  const viewer = $("viewer");
  // Opened from a search hit: that hit is about to be replaced, which would drop focus to the page.
  const fromSearch = viewer.contains(document.activeElement);
  viewer.replaceChildren(el("p", { className: "muted viewer-hint" }, "Loading…"));
  // From a search hit, a window starting just before the hit, however old it is; otherwise the newest sessions.
  const start = focusSession === undefined ? undefined : Math.max(0, focusSession - 2);
  let data;
  try {
    data = await fetchWindow(row, start, start === undefined ? undefined : start + PAGE);
  } catch (err) {
    if (seq === viewSeq) viewer.replaceChildren(el("p", { className: "error viewer-hint" }, err.message));
    return;
  }
  if (seq !== viewSeq) return; // something newer was opened while this loaded
  current = { row, total: data.total, from: data.start, to: data.end, sessions: data.sessions };
  markCurrent();
  renderChat();
  announce(`Opened ${chatLabel(row)}, ${plural(row.messages, "message")}`);
  if (fromSearch) viewer.querySelector("h2").focus({ preventScroll: true });
  if (focusSession === undefined) {
    viewer.scrollTop = viewer.scrollHeight;
    return;
  }
  const target = $(`s-${focusSession}`);
  scrollPaneTo(target);
  if (target) target.classList.add("flash");
}

// Show earlier/later: fetch the next PAGE sessions in that direction and add them to what's on screen.
async function loadMore(which, btn) {
  const seq = ++viewSeq;
  const { row, from, to } = current;
  const [start, end] = which === "earlier" ? [Math.max(0, from - PAGE), from] : [to, to + PAGE];
  btn.disabled = true;
  btn.textContent = "Loading…";
  let data;
  try {
    data = await fetchWindow(row, start, end);
  } catch (err) {
    if (seq === viewSeq) {
      btn.disabled = false;
      btn.textContent = `${err.message} Try again.`;
    }
    return;
  }
  if (seq !== viewSeq) return; // another chat or search took over the pane meanwhile
  current.total = data.total;
  if (which === "earlier") {
    current.sessions = data.sessions.concat(current.sessions);
    current.from = data.start;
  } else {
    current.sessions = current.sessions.concat(data.sessions);
    current.to = data.end;
  }
  const viewer = $("viewer");
  const top = viewer.scrollTop;
  renderChat();
  viewer.scrollTop = top;
  if (which === "earlier") scrollPaneTo($(`s-${from}`)); // keep the session that was first in view
  // The clicked button was replaced: focus its replacement, or the heading if there's no more to load.
  (viewer.querySelector(`.more-${which}`) || viewer.querySelector("h2")).focus({ preventScroll: true });
}

function moreButton(which, hidden) {
  const label = which === "earlier" ? "Show earlier messages" : "Show later messages";
  const btn = el("button", { type: "button", className: `btn more more-${which}` }, `${label} (${plural(hidden, "session")} hidden)`);
  btn.addEventListener("click", () => loadMore(which, btn));
  return btn;
}

function renderChat() {
  const { row, total, from, to, sessions } = current;
  const viewer = $("viewer");
  viewer.replaceChildren();

  const head = el("div", { className: "viewer-head" });
  const info = el("div");
  info.append(el("h2", { tabIndex: -1 }, chatLabel(row)));
  info.append(el("p", { className: "muted" },
    `${SOURCES[row.source].name} · ${plural(row.messages, "message")}, ${n(row.mine)} from you · since ${fmtDate(row.first)}`));
  head.append(logo(row.source), info);
  viewer.append(head);

  if (from > 0) viewer.append(moreButton("earlier", from));

  for (const s of sessions) {
    const sec = el("section", { className: "session", id: `s-${s.seq}` });
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
  }

  if (to < total) viewer.append(moreButton("later", total - to));
}

// ---- search ----

async function runSearch(q) {
  const seq = ++viewSeq;
  const viewer = $("viewer");
  current = null;
  markCurrent();
  viewer.replaceChildren(el("p", { className: "muted viewer-hint" }, "Searching…"));
  let hits;
  try {
    ({ hits } = await getJSON(`/api/data/search?q=${encodeURIComponent(q)}`));
  } catch (err) {
    if (seq === viewSeq) viewer.replaceChildren(el("p", { className: "error viewer-hint" }, err.message));
    return;
  }
  if (seq !== viewSeq) return; // something newer was opened while this loaded
  viewer.replaceChildren();
  const head = el("div", { className: "viewer-head" });
  const info = el("div");
  info.append(el("h2", {}, `“${q}”`));
  const found = hits.length >= 200 ? "The 200 newest matches" : plural(hits.length, "match").replace("matchs", "matches");
  info.append(el("p", { className: "muted" }, found));
  announce(`${found} for ${q}`);
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
