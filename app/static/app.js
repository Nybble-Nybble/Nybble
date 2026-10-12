const mcps = new Map(window.MCPS.map((m) => [m.id, m]));

const $ = (id) => document.getElementById(id);
const connectDialog = $("connect-dialog");
const finetuneDialog = $("finetune-dialog");
let current = null;
// Why the last connect attempt failed, by connector id. Shown on the card and in its popup until it connects.
const connectErrors = new Map();

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  Object.assign(node, attrs);
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, body) {
  let res;
  try {
    res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body ?? {}),
    });
  } catch {
    throw new Error("Can't reach the Nybble server. Is it still running?");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Something went wrong");
  return data;
}

function toast(msg, ms = 2400) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove("show"), ms);
}

// Put a connector's full-colour logo (app/static/icons/<id>.svg) on a .logo tile.
function paintLogo(node, m) {
  node.textContent = m.name[0];
  node.classList.add("has-icon");
  node.style.setProperty("--c", m.color);
  node.style.setProperty("--icon", `url("/static/icons/${m.id}.svg")`);
}

function renderCard(m) {
  const card = document.querySelector(`.card[data-id="${m.id}"]`);
  const failed = !m.connected && connectErrors.has(m.id);
  card.classList.toggle("is-connected", m.connected);
  card.classList.toggle("has-error", failed);
  card.querySelector(".status").textContent =
    !m.available ? "Coming soon" : m.connected ? "Connected" : failed ? "Couldn't connect" : "Connect";
}

// ---- Connect popup ----

function openConnect(id) {
  current = mcps.get(id);
  const m = current;
  paintLogo($("cd-logo"), m);
  $("cd-title").textContent = m.name;
  $("cd-desc").textContent = m.description;
  $("cd-error").textContent = m.connected ? "" : connectErrors.get(m.id) || "";
  $("cd-disconnect").hidden = !m.connected;
  $("cd-refresh").hidden = !m.connected;

  const content = $("cd-content");
  content.replaceChildren();

  if (!m.available) {
    content.append(el("p", { className: "unavailable-msg" },
      `Nybble can't connect to ${m.name} yet. Right now only Messages and WhatsApp can be connected.`));
    if (m.scopes?.length) {
      content.append(el("p", { className: "muted" }, "Once it's ready, Nybble will be able to:"));
      const ul = el("ul", { className: "scopes" });
      m.scopes.forEach((s) => ul.append(el("li", {}, s)));
      content.append(ul);
    }
    $("cd-submit").hidden = true;
  } else if (m.connected) {
    content.append(el("p", { className: "connected-msg" },
      `${m.name} is connected. Its messages are on the Your data page; use Update data to pull in new ones. ` +
      `Disconnecting deletes them from this Mac.`));
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
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Connecting…";
  try {
    const updated = await api(`/api/mcps/${current.id}/connect`, { fields });
    connectErrors.delete(updated.id);
    mcps.set(updated.id, updated);
    renderCard(updated);
    connectDialog.close();
    toast(updated.detail ? `${updated.name} connected · ${updated.detail}` : `${updated.name} connected`);
  } catch (err) {
    $("cd-error").textContent = err.message;
    connectErrors.set(current.id, err.message);
    renderCard(current);
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
});

// Re-run the extractor for a connected source to pick up new messages.
$("cd-refresh").addEventListener("click", async () => {
  const btn = $("cd-refresh");
  btn.disabled = $("cd-disconnect").disabled = true; // the server would make a Disconnect wait anyway
  btn.textContent = "Updating…";
  try {
    const updated = await api(`/api/mcps/${current.id}/extract`);
    connectDialog.close();
    toast(`${updated.name} updated · ${updated.detail}`);
  } catch (err) {
    $("cd-error").textContent = err.message;
  } finally {
    btn.disabled = $("cd-disconnect").disabled = false;
    btn.textContent = "Update data";
  }
});

$("cd-disconnect").addEventListener("click", async () => {
  const btn = $("cd-disconnect");
  btn.disabled = $("cd-refresh").disabled = true;
  btn.textContent = "Disconnecting…";
  try {
    const updated = await api(`/api/mcps/${current.id}/disconnect`);
    mcps.set(updated.id, updated);
    renderCard(updated);
    connectDialog.close();
    toast(`${updated.name} disconnected · its data was deleted`);
  } catch (err) {
    $("cd-error").textContent = err.message;
  } finally {
    btn.disabled = $("cd-refresh").disabled = false;
    btn.textContent = "Disconnect";
  }
});

// ---- Connect all ----

async function connectAll() {
  const btn = $("connect-all-btn");
  const before = [...mcps.values()].filter((m) => m.connected).length;
  btn.disabled = true;
  btn.textContent = "Connecting…";
  try {
    const { mcps: all, skipped, failed = [] } = await api("/api/mcps/connect-all");
    failed.forEach((f) => connectErrors.set(f.id, f.error));
    all.forEach((m) => { if (m.connected) connectErrors.delete(m.id); mcps.set(m.id, m); renderCard(m); });
    const added = all.filter((m) => m.connected).length - before;
    let msg = added ? `Connected ${added} source${added === 1 ? "" : "s"}` : failed.length ? "Nothing connected" : "Nothing new to connect";
    if (skipped.length) msg += `. Needs details: ${skipped.join(", ")}`;
    if (failed.length === 1) msg += `. ${failed[0].name}: ${failed[0].error}`;
    else if (failed.length) msg += `. Couldn't connect ${failed.map((f) => f.name).join(" or ")}; open a card to see why`;
    toast(msg, failed.length ? 8000 : 2400);
  } catch (err) {
    toast(err.message, 5000);
  } finally {
    btn.disabled = false;
    btn.textContent = "Connect all";
  }
}

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
      const logo = el("span", { className: "logo" });
      paintLogo(logo, m);
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
$("connect-all-btn").addEventListener("click", connectAll);
$("finetune-btn").addEventListener("click", openFinetune);

document.querySelectorAll("[data-close]").forEach((b) =>
  b.addEventListener("click", () => b.closest("dialog").close())
);
// Click on the backdrop closes the dialog.
[connectDialog, finetuneDialog].forEach((d) =>
  d.addEventListener("click", (e) => { if (e.target === d) d.close(); })
);
