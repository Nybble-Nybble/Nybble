const mcps = new Map(window.MCPS.map((m) => [m.id, m]));

const $ = (id) => document.getElementById(id);
const connectDialog = $("connect-dialog");
const finetuneDialog = $("finetune-dialog");
let current = null;

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  Object.assign(node, attrs);
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Something went wrong");
  return data;
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove("show"), 2400);
}

function renderCard(m) {
  const card = document.querySelector(`.card[data-id="${m.id}"]`);
  card.classList.toggle("is-connected", m.connected);
  card.querySelector(".status").textContent = m.connected ? "Connected" : "Connect";
}

// ---- Connect popup ----

function openConnect(id) {
  current = mcps.get(id);
  const m = current;
  $("cd-logo").textContent = m.name[0];
  $("cd-logo").style.setProperty("--c", m.color);
  $("cd-title").textContent = m.name;
  $("cd-desc").textContent = m.description;
  $("cd-error").textContent = "";
  $("cd-disconnect").hidden = !m.connected;

  const content = $("cd-content");
  content.replaceChildren();

  if (m.connected) {
    content.append(el("p", { className: "connected-msg" }, `${m.name} is connected.`));
    $("cd-submit").hidden = true;
  } else if (m.auth === "oauth") {
    content.append(el("p", { className: "muted" }, `Nybble will be able to:`));
    const ul = el("ul", { className: "scopes" });
    (m.scopes || []).forEach((s) => ul.append(el("li", {}, s)));
    content.append(ul);
    $("cd-submit").hidden = false;
    $("cd-submit").textContent = `Sign in with ${m.name}`;
  } else {
    (m.fields || []).forEach((f, i) => {
      const wrap = el("div", { className: "field" });
      const inputId = `f-${f.name}`;
      wrap.append(el("label", { htmlFor: inputId }, f.label));
      wrap.append(el("input", {
        id: inputId,
        name: f.name,
        type: f.secret ? "password" : "text",
        placeholder: f.placeholder || "",
        required: f.required !== false,
        autocomplete: "off",
        autofocus: i === 0,
      }));
      content.append(wrap);
    });
    if (m.note) content.append(el("p", { className: "note" }, m.note));
    $("cd-submit").hidden = false;
    $("cd-submit").textContent = "Connect";
  }

  connectDialog.showModal();
}

$("connect-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!current || current.connected) return;
  const fields = {};
  $("cd-content").querySelectorAll("input").forEach((i) => (fields[i.name] = i.value));
  const btn = $("cd-submit");
  btn.disabled = true;
  try {
    const updated = await api(`/api/mcps/${current.id}/connect`, { fields });
    mcps.set(updated.id, updated);
    renderCard(updated);
    connectDialog.close();
    toast(`${updated.name} connected`);
  } catch (err) {
    $("cd-error").textContent = err.message;
  } finally {
    btn.disabled = false;
  }
});

$("cd-disconnect").addEventListener("click", async () => {
  try {
    const updated = await api(`/api/mcps/${current.id}/disconnect`);
    mcps.set(updated.id, updated);
    renderCard(updated);
    connectDialog.close();
    toast(`${updated.name} disconnected`);
  } catch (err) {
    $("cd-error").textContent = err.message;
  }
});

// ---- Finetune popup ----

function openFinetune() {
  const list = $("ft-sources");
  list.replaceChildren();
  $("ft-error").textContent = "";
  const connected = [...mcps.values()].filter((m) => m.connected);

  if (connected.length === 0) {
    list.append(el("p", { className: "muted" }, "No sources connected yet. Click a card to connect one."));
    $("ft-submit").disabled = true;
  } else {
    connected.forEach((m) => {
      const row = el("label", { className: "source" });
      row.append(el("input", { type: "checkbox", value: m.id, checked: true }));
      const logo = el("span", { className: "logo" }, m.name[0]);
      logo.style.setProperty("--c", m.color);
      row.append(logo, el("span", {}, m.name));
      list.append(row);
    });
    $("ft-submit").disabled = false;
  }
  finetuneDialog.showModal();
}

$("finetune-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const sources = [...$("ft-sources").querySelectorAll("input:checked")].map((i) => i.value);
  try {
    const job = await api("/api/finetune", { sources });
    finetuneDialog.close();
    toast(`Finetune job ${job.id} queued`);
  } catch (err) {
    $("ft-error").textContent = err.message;
  }
});

// ---- Wiring ----

$("grid").addEventListener("click", (e) => {
  const card = e.target.closest(".card");
  if (card) openConnect(card.dataset.id);
});
$("finetune-btn").addEventListener("click", openFinetune);

document.querySelectorAll("[data-close]").forEach((b) =>
  b.addEventListener("click", () => b.closest("dialog").close())
);
// Click on the backdrop closes the dialog.
[connectDialog, finetuneDialog].forEach((d) =>
  d.addEventListener("click", (e) => { if (e.target === d) d.close(); })
);
