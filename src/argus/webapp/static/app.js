/* Argus web — shared client library (no framework, no build). */
window.A = (() => {
  // ---------- fetch wrapper with auth handling ----------
  async function api(path, opts = {}) {
    const init = { headers: {}, ...opts };
    if (init.body !== undefined && typeof init.body !== "string") {
      init.body = JSON.stringify(init.body);
      init.headers["Content-Type"] = "application/json";
    }
    const res = await fetch(path, init);
    if (res.status === 401) {
      showLogin();
      throw new Error("sign in required");
    }
    let data = null;
    try { data = await res.json(); } catch (_) { /* no body */ }
    if (!res.ok) {
      const detail = data && data.detail ? data.detail : res.status + " " + res.statusText;
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return data;
  }

  // ---------- toasts ----------
  function toast(text, kind = "") {
    let host = document.getElementById("toasts");
    if (!host) { host = el("div", { id: "toasts" }); document.body.appendChild(host); }
    const t = el("div", { class: "toast " + kind }, text);
    host.appendChild(t);
    setTimeout(() => t.remove(), kind === "err" ? 7000 : 4200);
  }

  // ---------- tiny DOM helper ----------
  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== undefined && v !== null) node.setAttribute(k, v);
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined) continue;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---------- formatting ----------
  function fmtTime(iso) {
    if (!iso) return "—";
    const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
    if (isNaN(d)) return iso;
    return d.toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
  }
  function ago(iso) {
    if (!iso) return "—";
    const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : iso + "Z");
    const s = Math.max(0, (Date.now() - d.getTime()) / 1000);
    if (s < 90) return "just now";
    if (s < 3600) return Math.round(s / 60) + " min ago";
    if (s < 86400) return Math.round(s / 3600) + " h ago";
    return Math.round(s / 86400) + " d ago";
  }
  const short = (s, n = 10) => (s ? String(s).slice(0, n) : "—");

  function statusBadge(status) {
    const map = {
      done: "good", ok: "good", active: "good", confirmed: "good",
      running: "warn", queued: "dim", degraded: "warn", paused: "warn",
      attributed: "warn", not_modified: "dim", pending: "dim", unknown: "dim",
      failed: "bad", failing: "bad", error: "bad", retired: "dim",
      contradicted: "bad", skipped: "dim",
    };
    return el("span", { class: "badge " + (map[status] || "dim") }, status || "—");
  }
  const tierBadge = (t) => el("span", { class: "tier" + (t === 1 ? " t1" : "") },
    ({ 1: "Ⅰ", 2: "Ⅱ", 3: "Ⅲ" })[t] || "?");

  // ---------- masthead + provenance strip ----------
  let overviewPromise = null;
  function overview(force = false) {
    if (!overviewPromise || force) overviewPromise = api("/api/overview");
    return overviewPromise;
  }

  const EYE = `<svg class="eye" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M4 6 v7 a8 8 0 0 0 16 0 V6" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/>
    <circle cx="12" cy="12.5" r="3.1" fill="currentColor"/></svg>`;

  function masthead(active) {
    const host = document.getElementById("mast");
    if (!host) return;
    host.innerHTML = `
      <div class="mast">
        <a class="wordmark" href="/">ARG${EYE}S<small>Controlled-source research</small></a>
        <nav>
          <a href="/" data-p="home">Research desk</a>
          <a href="/admin" data-p="admin">Admin</a>
          <a href="/maintenance" data-p="maintenance">Maintenance</a>
        </nav>
      </div>
      <div class="stamps" id="stamps">
        <span class="stamp"><b>registry</b> <span class="val">…</span></span>
      </div>`;
    host.querySelectorAll("nav a").forEach((a) => {
      if (a.dataset.p === active) a.classList.add("active");
    });
    overview().then(renderStamps).catch(() => {});
  }

  function renderStamps(ov) {
    const s = document.getElementById("stamps");
    if (!s) return;
    const reg = ov.registry || {};
    const commit = reg.commit || "uncommitted";
    const dirty = commit.endsWith("-dirty");
    const stamps = [
      ["registry", (dirty ? short(commit, 7) + " ·dirty" : short(commit, 7)),
       reg.error ? "bad" : dirty ? "warn" : ""],
      ["sources", String(reg.sources ?? "—"), ""],
      ["synth", ov.mode.llm, ov.mode.llm === "stub" ? "warn" : "good"],
      ["embed", ov.mode.embedder, ov.mode.embedder === "hashing" ? "warn" : "good"],
      ["snapshots", String(ov.counts.snapshots ?? 0), ""],
      ["queue", ov.jobs_active.length ? ov.jobs_active.length + " running" : "idle",
       ov.jobs_active.length ? "warn" : ""],
    ];
    if (ov.azure_missing && ov.azure_missing.length)
      stamps.push(["azure", ov.azure_missing.length + " settings missing", "bad"]);
    if (ov.restart_required && ov.restart_required.length)
      stamps.push(["restart", "required", "bad"]);
    s.innerHTML = stamps.map(([k, v, cls]) =>
      `<span class="stamp"><b>${k}</b> <span class="val ${cls}">${esc(v)}</span></span>`
    ).join("");
  }

  // ---------- login overlay ----------
  function showLogin() {
    if (document.getElementById("login-overlay")) return;
    const input = el("input", { type: "password", placeholder: "Password", autofocus: "true" });
    const err = el("p", { class: "msg-error", style: "min-height:18px;margin:8px 0 0;font-size:12.5px" });
    const submit = async () => {
      try {
        await api("/api/login", { method: "POST", body: { password: input.value } });
        location.reload();
      } catch (e) { err.textContent = e.message; }
    };
    const box = el("div", { class: "box" },
      el("h3", {}, "Argus"),
      el("p", {}, "This desk is password-protected. Sign in to continue."),
      el("label", { class: "field" }, el("span", {}, "Password"), input),
      err,
      el("div", { style: "margin-top:12px" },
        el("button", { class: "btn primary", style: "width:100%", onclick: submit }, "Sign in")));
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
    document.body.appendChild(el("div", { id: "login-overlay" }, box));
  }
  async function authCheck() {
    try {
      const s = await fetch("/api/authstate").then((r) => r.json());
      if (s.required && !s.ok) showLogin();
    } catch (_) { /* server down; page fetches will surface it */ }
  }

  // ---------- modal / drawer ----------
  function modal(title, contentNode, { drawer = false, footer = null, sub = "" } = {}) {
    const close = () => wrap.remove();
    const box = el("div", { class: "modal" },
      el("h3", {}, title),
      sub ? el("p", { class: "sub" }, sub) : null,
      contentNode,
      el("div", { class: "modal-foot" },
        ...(footer ? footer(close) : [el("button", { class: "btn", onclick: close }, "Close")])));
    const wrap = el("div", { class: "overlay" + (drawer ? " drawer" : "") }, box);
    wrap.addEventListener("mousedown", (e) => { if (e.target === wrap) close(); });
    document.addEventListener("keydown", function onEsc(e) {
      if (e.key === "Escape") { close(); document.removeEventListener("keydown", onEsc); }
    });
    document.body.appendChild(wrap);
    return { close, box };
  }

  // ---------- job tray + watcher ----------
  function tray() {
    let host = document.getElementById("tray");
    if (!host) { host = el("div", { id: "tray" }); document.body.appendChild(host); }
    return host;
  }
  function watchJob(jobId, { title = "Working…", onDone, onFail } = {}) {
    const label = el("span", {}, "queued");
    const chip = el("div", { class: "tray-job" },
      el("span", { class: "dot live" }),
      el("div", { class: "t" }, el("b", {}, title), label));
    tray().appendChild(chip);
    const timer = setInterval(async () => {
      let job;
      try { job = await api("/api/jobs/" + jobId); }
      catch (e) { clearInterval(timer); chip.remove(); toast(e.message, "err"); return; }
      label.textContent = job.status === "running" ? (job.note || "running") : job.status;
      if (job.status === "done" || job.status === "failed") {
        clearInterval(timer);
        setTimeout(() => chip.remove(), 400);
        overview(true).then(renderStamps).catch(() => {});
        if (job.status === "done") {
          toast(title + " — done", "ok");
          onDone && onDone(job);
        } else {
          toast(title + " failed: " + (job.error || "unknown error"), "err");
          onFail && onFail(job);
        }
      }
    }, 1500);
  }

  return { api, toast, el, esc, fmtTime, ago, short, statusBadge, tierBadge,
           masthead, overview, renderStamps, modal, watchJob, authCheck };
})();
