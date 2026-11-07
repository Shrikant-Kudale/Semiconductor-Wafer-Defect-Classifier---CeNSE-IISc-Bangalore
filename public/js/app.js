/* WaferFab Intelligence — application logic (vanilla JS, no build step). */
(async function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const { h: sv, css } = Charts;

  /* ---------- load ---------- */
  let model, meta;
  try {
    [model, meta] = await Promise.all([fetch("/data/model.json").then((r) => r.json()), fetch("/data/meta.json").then((r) => r.json())]);
  } catch (e) {
    document.body.insertAdjacentHTML("afterbegin", '<p style="padding:24px;color:#ef4444">Failed to load model files. Run a local server (e.g. <code>npx serve public</code>) instead of opening index.html directly.</p>');
    throw e;
  }
  const eng = WaferEngine.create(model);
  const F = meta.features, C = meta.classes, K = C.length, NF = F.length;
  const classIdx = Object.fromEntries(C.map((c, i) => [c.name, i]));
  const ref = model.reference;
  const used = F.map((f) => f.used);
  const decimals = (step) => (String(step).split(".")[1] || "").length;
  const fmt = (v, i) => (+v).toFixed(Math.max(decimals(F[i].step), v >= 100 ? 0 : 0));
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const statusOf = (c) => ["STABLE", "WATCH", "ALERT", "CRITICAL · HOLD LOT"][C[classIdx[c]].severity];

  /* ---------- state ---------- */
  const defaults = F.map((f) => f.default);
  let x = defaults.slice();
  const locked = new Array(NF).fill(false);
  const st = { tab: "diagnose", mapMode: "die", impKind: "gain", shapClass: "auto", sweepF: 2, mcNoise: 25, cfGoal: 80, cfTarget: "Normal", batch: null, dirty: new Set() };

  const q = new URLSearchParams(location.search).get("v");
  if (q) { const a = q.split(",").map(Number); if (a.length === NF && a.every(Number.isFinite)) x = a.map((v, i) => clamp(v, F[i].min, F[i].max)); }

  /* ---------- toast ---------- */
  let toastT;
  const toast = (m) => { const t = $("#toast"); t.textContent = m; t.classList.add("show"); clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("show"), 2200); };

  /* ============ CONTROLS ============ */
  function buildControls() {
    // presets
    const pc = $("#presets");
    meta.classes.forEach((c) => {
      const b = document.createElement("button");
      b.className = "pchip"; b.dataset.cls = c.name; b.innerHTML = `<i style="background:${c.color}"></i>${c.name}`;
      b.title = `Load an archetype ${c.name} wafer — ${c.rootCause}`;
      b.onclick = () => { setVector(meta.archetypes[c.name], c.name); };
      pc.append(b);
    });
    // sliders
    const box = $("#sliders");
    [["spatial", "Inspection-derived (spatial)"], ["process", "Process metrology"]].forEach(([g, title]) => {
      box.insertAdjacentHTML("beforeend", `<div class="grp">${title}</div>`);
      F.forEach((f, i) => {
        if (f.group !== g) return;
        const span = f.max - f.min;
        const lo = clamp(((ref.mean[i] - 3 * ref.std[i] - f.min) / span) * 100, 0, 100);
        const hi = clamp(((ref.mean[i] + 3 * ref.std[i] - f.min) / span) * 100, 0, 100);
        let spec = "";
        if (f.spec) {
          const a = clamp(((f.spec.lsl - f.min) / span) * 100, 0, 100), b = clamp(((f.spec.usl - f.min) / span) * 100, 0, 100);
          spec = `<div class="spec" style="left:${a}%;width:${b - a}%" title="Illustrative spec window"></div>`;
        }
        const row = document.createElement("div");
        row.className = "srow";
        row.innerHTML = `
          <div class="lab"><span>${f.label}<small>${f.unit}</small>${f.used ? "" : '<em class="unused" title="Not used by any tree in the trained model">unused by model</em>'}</span>
            <span class="ctl"><button class="lock" data-i="${i}" title="Lock during counterfactual search" aria-label="Lock ${f.label}">🔒</button>
            <input type="number" data-n="${i}" min="${f.min}" max="${f.max}" step="${f.step}" aria-label="${f.label} value"></span></div>
          <div class="track" title="${f.description}"><div class="band" style="left:${lo}%;width:${hi - lo}%"></div>${spec}
            <input type="range" data-r="${i}" min="${f.min}" max="${f.max}" step="${f.step}" aria-label="${f.label}"></div>`;
        box.append(row);
      });
    });
    box.addEventListener("input", (e) => {
      const r = e.target.dataset.r, n = e.target.dataset.n;
      if (r !== undefined) { x[+r] = +e.target.value; $(`[data-n="${r}"]`).value = fmt(x[+r], +r); changed(); }
      if (n !== undefined && e.target.value !== "") { const i = +n; x[i] = clamp(+e.target.value, F[i].min, F[i].max); $(`[data-r="${i}"]`).value = x[i]; changed(); }
    });
    box.addEventListener("click", (e) => { const b = e.target.closest(".lock"); if (b) { const i = +b.dataset.i; locked[i] = !locked[i]; b.classList.toggle("on", locked[i]); } });
    $("#resetBtn").onclick = () => setVector(defaults);
  }
  function syncControls() {
    x.forEach((v, i) => { $(`[data-r="${i}"]`).value = v; $(`[data-n="${i}"]`).value = fmt(v, i); });
  }
  function setVector(v, presetName) {
    x = v.map((val, i) => clamp(val, F[i].min, F[i].max));
    syncControls(); changed();
    $$(".pchip").forEach((b) => b.classList.toggle("on", b.dataset.cls === presetName));
  }

  /* ============ WAFER MAP ============ */
  function genDefects(v) {
    const [count, , mrad, srad, edge, center, ccount, maxc] = v;
    const rnd = eng.util.mulberry32(20260608), g = () => eng.util.gauss(rnd);
    const N = Math.min(900, Math.round(count));
    let nE = Math.round(N * edge), nC = Math.round(N * center);
    if (nE + nC > N) { const s = N / (nE + nC); nE = Math.floor(nE * s); nC = Math.floor(nC * s); }
    let rest = N - nE - nC;
    const pts = [];
    const put = (r, t) => pts.push([r * Math.cos(t), r * Math.sin(t)]);
    const arc0 = rnd() * 2 * Math.PI, localized = count < 90;
    for (let i = 0; i < nE; i++) put(clamp(1 - 0.09 * Math.abs(g()), 0.8, 0.995), localized ? arc0 + (rnd() - 0.5) * 1.1 : rnd() * 2 * Math.PI);
    for (let i = 0; i < nC; i++) put(clamp(0.17 * Math.abs(g()), 0, 0.32), rnd() * 2 * Math.PI);
    const Cn = Math.round(ccount);
    if (Cn > 0 && rest > 0) {
      const big = Math.min(Math.round(maxc), Math.round(rest * 0.7));
      const sizes = [big];
      let remain = Math.min(rest - big, Math.round((rest - big) * 0.85));
      const w = Array.from({ length: Cn - 1 }, () => 0.3 + rnd());
      const ws = w.reduce((a, b) => a + b, 0) || 1;
      w.forEach((wi) => sizes.push(Math.round((remain * wi) / ws)));
      sizes.forEach((sz) => {
        const rc = clamp(mrad + srad * g(), 0.05, 0.92), tc = rnd() * 2 * Math.PI, sg = 0.025 + 0.009 * Math.sqrt(sz);
        for (let i = 0; i < sz; i++) {
          let px = rc * Math.cos(tc) + g() * sg, py = rc * Math.sin(tc) + g() * sg;
          const rr = Math.hypot(px, py); if (rr > 0.985) { px *= 0.985 / rr; py *= 0.985 / rr; }
          pts.push([px, py]); rest--;
        }
      });
    }
    for (let i = 0; i < Math.max(0, rest); i++) put(clamp(mrad + srad * g(), 0, 0.99), rnd() * 2 * Math.PI);
    return pts;
  }

  function drawWafer(pred) {
    const cv = $("#waferCanvas"), dpr = Math.min(2, window.devicePixelRatio || 1);
    const size = cv.clientWidth || 400;
    cv.width = cv.height = Math.round(size * dpr);
    const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const R = size / 2 - 8, cx = size / 2, cy = size / 2;
    const color = C[pred.index].color, pts = genDefects(x), G = 32;
    c.clearRect(0, 0, size, size);
    // substrate
    c.beginPath(); c.arc(cx, cy, R, 0, 7); c.fillStyle = css("--wafer-bg"); c.fill();
    c.lineWidth = 2; c.strokeStyle = css("--wafer-line"); c.stroke();
    const cells = new Uint16Array(G * G);
    pts.forEach(([px, py]) => { const gx = Math.floor(((px + 1) / 2) * G), gy = Math.floor(((py + 1) / 2) * G); if (gx >= 0 && gx < G && gy >= 0 && gy < G) cells[gy * G + gx]++; });
    const cs = (2 * R) / G;
    if (st.mapMode !== "scatter") {
      for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
        const mx = (gx + 0.5) / G * 2 - 1, my = (gy + 0.5) / G * 2 - 1;
        if (Math.hypot(mx, my) > 0.985) continue;
        const n = cells[gy * G + gx], X = cx - R + gx * cs, Y = cy - R + gy * cs;
        c.globalAlpha = 1;
        c.fillStyle = n ? color : (Math.hypot(mx, my) > 0.88 ? css("--die-edge") : css("--die-pass"));
        if (n) c.globalAlpha = Math.min(1, 0.5 + n * 0.17);
        c.fillRect(X + 0.7, Y + 0.7, cs - 1.4, cs - 1.4);
      }
      c.globalAlpha = 1;
    }
    if (st.mapMode !== "die") {
      c.fillStyle = color; c.strokeStyle = "rgba(255,255,255,.7)"; c.lineWidth = 0.6;
      c.globalAlpha = st.mapMode === "both" ? 0.95 : 0.85;
      pts.forEach(([px, py]) => { c.beginPath(); c.arc(cx + px * R, cy + py * R, st.mapMode === "both" ? 1.9 : 2.6, 0, 7); c.fill(); if (st.mapMode === "scatter") c.stroke(); });
      c.globalAlpha = 1;
    }
    // edge-exclusion ring, notch, crosshair
    c.setLineDash([4, 4]); c.lineWidth = 1; c.strokeStyle = css("--wafer-line"); c.beginPath(); c.arc(cx, cy, R * 0.88, 0, 7); c.stroke(); c.setLineDash([]);
    c.fillStyle = css("--surface"); c.beginPath(); c.arc(cx, cy + R, 7, Math.PI, 0); c.fill();
    c.lineWidth = 2; c.strokeStyle = css("--wafer-line"); c.beginPath(); c.arc(cx, cy + R, 7, Math.PI, 0); c.stroke();
    c.fillStyle = css("--muted"); c.font = "600 10px system-ui,sans-serif"; c.textAlign = "left";
    c.fillText(`${pts.length} defects · ${cells.reduce((a, b) => a + (b > 0), 0)} failing dies`, 8, 14);

    // radial profile
    const rc = $("#radialCanvas"), W = rc.clientWidth || 400, H = 70;
    rc.width = Math.round(W * dpr); rc.height = Math.round(H * dpr);
    const r2 = rc.getContext("2d"); r2.setTransform(dpr, 0, 0, dpr, 0, 0); r2.clearRect(0, 0, W, H);
    const B = 24, bins = new Array(B).fill(0);
    pts.forEach(([px, py]) => bins[Math.min(B - 1, Math.floor(Math.hypot(px, py) * B))]++);
    const dens = bins.map((n, i) => n / ((i + 1) ** 2 - i ** 2)), mx = Math.max(...dens, 1e-9), bw = (W - 8) / B;
    dens.forEach((d, i) => { const bh = (d / mx) * (H - 22); r2.fillStyle = color; r2.globalAlpha = 0.85; r2.fillRect(4 + i * bw + 1, H - 14 - bh, bw - 2, bh); });
    r2.globalAlpha = 1; r2.fillStyle = css("--muted"); r2.font = "10px system-ui,sans-serif";
    r2.textAlign = "left"; r2.fillText("centre", 4, H - 2); r2.textAlign = "right"; r2.fillText("edge", W - 4, H - 2);
    r2.textAlign = "center"; r2.fillText("radial defect density (area-normalised)", W / 2, H - 2);
  }

  /* ============ DIAGNOSE ============ */
  let pred;
  function renderDiagnose() {
    const t0 = performance.now(); pred = eng.predict(x); const dt = performance.now() - t0;
    $("#latency").textContent = `in‑browser · ${dt < 0.1 ? "<0.1" : dt.toFixed(2)} ms`;
    const cls = C[pred.index], vc = $("#verdictCard");
    vc.style.setProperty("--vc", cls.color);
    $("#vClass").textContent = pred.label;
    const stEl = $("#vStatus"); stEl.textContent = statusOf(pred.label);
    stEl.className = "status " + (cls.severity === 0 ? "ok" : cls.severity === 3 ? "sev3" : "bad");
    Charts.ring($("#ring"), pred.confidence, cls.color);
    $("#kConf").textContent = pct(pred.confidence); $("#kMargin").textContent = pct(pred.margin);
    $("#kRunner").textContent = pred.runnerUp; $("#kEnt").textContent = `${pred.normalizedEntropy.toFixed(2)} / 1`;
    // OOD + confidence banner
    const z = eng.zscores(x), ood = z.map((v, i) => [i, v]).filter(([i, v]) => used[i] && Math.abs(v) > 3);
    const msgs = [];
    if (ood.length) msgs.push(`⚠️ <b>${ood.length} input${ood.length > 1 ? "s" : ""} outside the training range</b> (|z|&gt;3): ${ood.map(([i, v]) => `${F[i].label} (${v > 0 ? "+" : ""}${v.toFixed(1)}σ)`).join(", ")}. Treat the prediction as an extrapolation.`);
    if (pred.confidence < 0.6) msgs.push(`🔎 <b>Low confidence</b> — the top two classes (${pred.label}, ${pred.runnerUp}) are close; consider an SEM / manual review.`);
    const bn = $("#oodBanner"); bn.hidden = !msgs.length; bn.innerHTML = msgs.join("<br>");
    $("#causeBox").innerHTML = `<div class="row"><span>Root cause</span><span><b>${cls.rootCause}</b></span></div><div class="row"><span>Process area</span><span>${cls.processStep}</span></div><div class="row"><span>Suggested action</span><span>${cls.action}</span></div>`;
    // probabilities
    const pb = $("#probBars");
    if (!pb.children.length) pb.innerHTML = C.map((c, i) => `<div class="brow" data-i="${i}"><div class="nm"><i style="background:${c.color}"></i>${c.name}</div><div class="tr"><div class="fl" style="background:${c.color}"></div></div><div class="v"></div></div>`).join("");
    $$(".brow", pb).forEach((r, i) => { r.classList.toggle("top", i === pred.index); $(".fl", r).style.width = pred.proba[i] * 100 + "%"; $(".v", r).textContent = pct(pred.proba[i]); });
    // radar
    Charts.radar($("#radar"), F.map((f) => f.short), z);
    drawWafer(pred);
    $("#miniDot").style.background = cls.color; $("#miniClass").textContent = pred.label; $("#miniClass").style.color = cls.color;
    $("#miniConf").textContent = `${pct(pred.confidence)} · ${statusOf(pred.label)}`;
    document.title = `${pred.label} (${pct(pred.confidence, 0)}) — WaferFab Intelligence`;
  }

  /* ============ EXPLAIN ============ */
  function renderExplain() {
    const sel = $("#shapClass");
    if (!sel.children.length) { sel.innerHTML = `<option value="auto">Predicted class</option>` + C.map((c) => `<option>${c.name}</option>`).join(""); sel.onchange = () => { st.shapClass = sel.value; renderExplain(); }; }
    const ci = st.shapClass === "auto" ? pred.index : classIdx[st.shapClass], cls = C[ci];
    const s = eng.shap(x, ci), m = eng.margins(x)[ci];
    const rank = s.values.map((v, i) => ({ i, v })).sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const top = rank.slice(0, 8), restN = rank.slice(8);
    const items = top.map(({ i, v }) => ({ label: F[i].label, sub: `= ${fmt(x[i], i)} ${F[i].unit}`.trim(), shap: v }));
    const restSum = restN.reduce((a, b) => a + b.v, 0);
    if (restN.length) items.push({ label: `${restN.length} other features`, sub: "", shap: restSum });
    Charts.waterfall($("#waterfall"), { base: s.bias, final: m, items, color: cls.color, classLabel: `→ P(${cls.name}) = ${pct(eng.proba(x)[ci])}` });
    const lead = top[0];
    $("#shapLead").innerHTML = `For <b style="color:${cls.color}">${cls.name}</b> the strongest driver is <b>${F[lead.i].label}</b> (${lead.v >= 0 ? "+" : ""}${lead.v.toFixed(2)} log‑odds at ${fmt(x[lead.i], lead.i)} ${F[lead.i].unit}), ${lead.v >= 0 ? "pushing the model towards" : "pulling the model away from"} this class.`;
    renderImportance();
    // profile table
    const z = eng.zscores(x);
    $("#profileTable").innerHTML = `<thead><tr><th>Feature</th><th>Value</th><th>Train mean</th><th>z</th><th>Spec (illustr.)</th><th>Used</th></tr></thead><tbody>` + F.map((f, i) => {
      let specTxt = "—";
      if (f.spec) { const ok = x[i] >= f.spec.lsl && x[i] <= f.spec.usl; specTxt = `<span class="${ok ? "" : "zbad"}">${ok ? "✓ in" : "✗ out"}</span> [${f.spec.lsl}, ${f.spec.usl}]`; }
      const zc = Math.abs(z[i]) > 3 && used[i] ? "zbad" : Math.abs(z[i]) > 2 && used[i] ? "zwarn" : "";
      return `<tr><td>${f.label} <small style="color:var(--muted)">${f.unit}</small></td><td>${fmt(x[i], i)}</td><td>${ref.mean[i].toFixed(2)}</td><td class="${zc}">${z[i] >= 0 ? "+" : ""}${z[i].toFixed(2)}</td><td>${specTxt}</td><td>${f.used ? "✓" : "—"}</td></tr>`;
    }).join("") + "</tbody>";
  }
  function renderImportance() {
    const v = model.importance[st.impKind], tot = v.reduce((a, b) => a + b, 0) || 1;
    const order = v.map((val, i) => i).sort((a, b) => v[b] - v[a]), mx = v[order[0]];
    $("#importance").innerHTML = order.map((i) => `<div class="brow"><div class="nm" title="${F[i].label}">${F[i].short}</div><div class="tr"><div class="fl" style="width:${(v[i] / mx) * 100}%;background:var(--accent)"></div></div><div class="v">${pct(v[i] / tot)}</div></div>`).join("");
  }

  /* ============ WHAT-IF ============ */
  let sweepState = null;
  function renderWhatIf() {
    const sf = $("#sweepFeature");
    if (!sf.children.length) {
      sf.innerHTML = F.map((f, i) => (f.used ? `<option value="${i}">${f.label}</option>` : "")).join("");
      sf.value = st.sweepF; sf.onchange = () => { st.sweepF = +sf.value; renderSweep(); };
      $("#cfTarget").innerHTML = C.map((c) => `<option ${c.name === "Normal" ? "selected" : ""}>${c.name}</option>`).join("");
      $("#cfTarget").onchange = (e) => { st.cfTarget = e.target.value; $("#cfOut").innerHTML = ""; };
      $("#cfGoal").oninput = (e) => { st.cfGoal = +e.target.value; $("#cfGoalVal").textContent = st.cfGoal + "%"; };
      $("#cfRun").onclick = runCounterfactual;
      let mt; $("#mcNoise").oninput = (e) => { st.mcNoise = +e.target.value; $("#mcVal").textContent = st.mcNoise + "%"; clearTimeout(mt); mt = setTimeout(renderMC, 120); };
    }
    renderSweep(); renderMC();
  }
  function renderSweep() {
    const i = st.sweepF, f = F[i];
    const { xs, ps } = eng.sweep(x, i, f.min, f.max, 90);
    const series = C.map((c, k) => ({ name: c.name, color: c.color, ys: ps.map((p) => p[k]), bold: k === pred.index, off: sweepState && sweepState.off && sweepState.off.has(k) }));
    const off = sweepState && sweepState.feat === i ? sweepState.off : new Set();
    series.forEach((s, k) => (s.off = off.has(k)));
    sweepState = { feat: i, off, xs, ps };
    let boundaryHTML = "";
    Charts.lines($("#sweepChart"), { xs, series, xLabel: `${f.label} (${f.unit || "–"})`, band: [ref.mean[i] - 3 * ref.std[i], ref.mean[i] + 3 * ref.std[i]], marker: x[i] }, (idx) => {
      const ro = $("#sweepReadout");
      if (idx < 0) { ro.innerHTML = boundaryHTML; return; }
      const top3 = ps[idx].map((p, k) => [p, k]).sort((a, b) => b[0] - a[0]).slice(0, 3);
      ro.innerHTML = `<b>${f.short} = ${xs[idx].toFixed(2)}</b> → ` + top3.map(([p, k]) => `<span style="color:${C[k].color}">●</span> ${C[k].name} ${pct(p)}`).join(" · ");
    });
    $("#sweepLegend").innerHTML = C.map((c, k) => `<span data-k="${k}" class="${off.has(k) ? "off" : ""}"><i style="background:${c.color}"></i>${c.name}</span>`).join("");
    $$("#sweepLegend span").forEach((el) => (el.onclick = () => { const k = +el.dataset.k; off.has(k) ? off.delete(k) : off.add(k); renderSweep(); }));
    // transition points
    const cls = ps.map((p) => eng.util.argmax(p)), edges = [];
    for (let j = 1; j < cls.length; j++) if (cls[j] !== cls[j - 1]) edges.push(`${C[cls[j - 1]].name} → ${C[cls[j]].name} near ${xs[j].toFixed(2)}`);
    $("#sweepReadout").innerHTML = boundaryHTML = edges.length ? `<b>Decision boundaries:</b> ${edges.slice(0, 5).join(" · ")}` : "No class change across this range — the diagnosis is insensitive to this parameter here.";
  }
  function renderMC() {
    const r = eng.monteCarlo(x, st.mcNoise / 100, 400, 7, locked);
    const mi = eng.util.argmax(r.freq);
    $("#mcBars").innerHTML = C.map((c, k) => ({ c, k, f: r.freq[k] })).filter((o) => o.f > 0 || o.k === pred.index).sort((a, b) => b.f - a.f).map(({ c, f }) => `<div class="brow ${c.name === C[mi].name ? "top" : ""}"><div class="nm"><i style="background:${c.color}"></i>${c.name}</div><div class="tr"><div class="fl" style="width:${f * 100}%;background:${c.color}"></div></div><div class="v">${pct(f)}</div></div>`).join("");
    const stab = r.freq[pred.index], lvl = stab > 0.95 ? "robust" : stab > 0.75 ? "moderately stable" : "fragile";
    $("#mcSummary").innerHTML = `With ${st.mcNoise}% of training σ noise, <b>${pred.label}</b> persists in <b>${pct(stab, 0)}</b> of 400 trials — the diagnosis is <b>${lvl}</b>.`;
  }
  function runCounterfactual() {
    const ti = classIdx[st.cfTarget], bounds = F.map((f) => [f.min, f.max]);
    const mask = locked.map((l, i) => l || ($("#cfScope").value === "process" && F[i].group !== "process"));
    const res = eng.counterfactual(x, ti, bounds, { goal: st.cfGoal / 100, locked: mask, maxIter: 8 });
    const out = $("#cfOut");
    if (!res.steps.length) { out.innerHTML = pred.index === ti ? `<div class="cf-step"><span class="n">✓</span><span>Already classified as <b>${st.cfTarget}</b> (${pct(res.p)}).</span><span></span></div>` : `<div class="cf-step"><span class="n">!</span><span>No improving change found with the current locks.</span><span></span></div>`; return; }
    out.innerHTML = res.steps.map((s, n) => `<div class="cf-step"><span class="n">${n + 1}</span><span><b>${F[s.feature].label}</b>: ${s.from.toFixed(2)} → <b>${s.to.toFixed(2)}</b> ${F[s.feature].unit}<div class="d">${((s.to - s.from) / ref.std[s.feature]).toFixed(1)}σ change</div></span><span><b>${pct(s.pAfter, 0)}</b> <small>P(${st.cfTarget})</small></span></div>`).join("")
      + `<div class="btn-row"><button class="btn" id="cfApply">Apply these changes</button></div><p class="fineprint">${res.reached ? "Goal reached." : "Goal not reached within 8 steps."} Final P(${st.cfTarget}) = ${pct(res.p)}.</p>`;
    $("#cfApply").onclick = () => { setVector(res.x.map((v, i) => Math.round(v / F[i].step) * F[i].step)); toast("Applied counterfactual"); };
  }

  /* ============ BATCH ============ */
  const NORM = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");
  function parseCSV(text) {
    const rows = []; let row = [], cur = "", inQ = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQ) { if (ch === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else inQ = false; } else cur += ch; }
      else if (ch === '"') inQ = true;
      else if (ch === ",") { row.push(cur); cur = ""; }
      else if (ch === "\n" || ch === "\r") { if (ch === "\r" && text[i + 1] === "\n") i++; row.push(cur); cur = ""; if (row.some((c) => c.trim() !== "")) rows.push(row); row = []; }
      else cur += ch;
    }
    row.push(cur); if (row.some((c) => c.trim() !== "")) rows.push(row);
    return rows;
  }
  function loadBatch(text) {
    const rows = parseCSV(text);
    if (rows.length < 1) return toast("Empty file");
    const isNum = (v) => v.trim() !== "" && Number.isFinite(+v);
    const hasHeader = rows[0].some((c) => !isNum(c));
    const header = hasHeader ? rows[0] : null, body = hasHeader ? rows.slice(1) : rows;
    let cols = null, idCol = -1;
    if (header) {
      const nh = header.map(NORM);
      const idx = F.map((f) => nh.findIndex((h) => h === NORM(f.key) || h === NORM(f.short) || h === NORM(f.label)));
      if (idx.every((v) => v >= 0)) cols = idx;
      idCol = nh.findIndex((h) => ["waferid", "id", "wafer", "lot", "name"].includes(h));
    }
    if (!cols) {
      // positional: first 15 numeric-looking columns
      const numeric = (body[0] || []).map((v, i) => [isNum(v), i]).filter(([ok]) => ok).map(([, i]) => i);
      if (numeric.length < NF) return toast(`Need ${NF} feature columns (found ${numeric.length}).`);
      cols = numeric.slice(0, NF);
    }
    const res = [];
    body.forEach((r, n) => {
      const v = cols.map((c) => +r[c]);
      if (v.some((a) => !Number.isFinite(a))) { res.push({ n, id: idCol >= 0 ? r[idCol] : "", err: true, v }); return; }
      const p = eng.predict(v), z = eng.zscores(v), ood = z.filter((zz, i) => used[i] && Math.abs(zz) > 3).length;
      res.push({ n, id: idCol >= 0 ? r[idCol] : "", v, p, ood });
    });
    st.batch = { res, header: header ? header : F.map((f) => f.key) };
    renderBatch();
  }
  function renderBatch() {
    const b = st.batch; if (!b) return;
    $("#batchOut").hidden = false;
    const ok = b.res.filter((r) => !r.err), bad = b.res.length - ok.length;
    const counts = new Array(K).fill(0); ok.forEach((r) => counts[r.p.index]++);
    const defective = ok.filter((r) => r.p.label !== "Normal").length;
    const low = ok.filter((r) => r.p.confidence < 0.6).length, oodN = ok.filter((r) => r.ood).length;
    const meanConf = ok.reduce((a, r) => a + r.p.confidence, 0) / (ok.length || 1);
    $("#batchKpis").innerHTML = [["Wafers scored", ok.length + (bad ? ` (+${bad} invalid)` : "")], ["Defective", `${defective} · ${pct(defective / (ok.length || 1), 0)}`], ["Mean confidence", pct(meanConf)], ["Low‑confidence (<60%)", low], ["Out‑of‑distribution", oodN]].map(([a, v]) => `<div><span>${a}</span><b>${v}</b></div>`).join("");
    const mx = Math.max(...counts, 1);
    $("#batchBars").innerHTML = C.map((c, k) => `<div class="brow"><div class="nm"><i style="background:${c.color}"></i>${c.name}</div><div class="tr"><div class="fl" style="width:${(counts[k] / mx) * 100}%;background:${c.color}"></div></div><div class="v">${counts[k]}</div></div>`).join("");
    Charts.hist($("#batchHist"), ok.map((r) => r.p.confidence), 0, 1, 20, css("--accent"));
    $("#batchTable").innerHTML = `<thead><tr><th>#</th><th>ID</th><th>Predicted</th><th>Confidence</th><th>Status</th><th>Runner‑up</th><th>OOD feats</th></tr></thead><tbody>` +
      b.res.slice(0, 300).map((r) => r.err ? `<tr><td>${r.n + 1}</td><td>${r.id || ""}</td><td colspan="5" class="zbad">invalid row</td></tr>` :
        `<tr><td>${r.n + 1}</td><td>${r.id || ""}</td><td><span class="dot" style="background:${C[r.p.index].color}"></span>${r.p.label}</td><td class="${r.p.confidence < 0.6 ? "zwarn" : ""}">${pct(r.p.confidence)}</td><td>${statusOf(r.p.label)}</td><td>${r.p.runnerUp}</td><td class="${r.ood ? "zbad" : ""}">${r.ood || "—"}</td></tr>`).join("") +
      (b.res.length > 300 ? `<tr><td colspan="7" style="color:var(--muted)">… ${b.res.length - 300} more rows (download CSV for everything)</td></tr>` : "") + "</tbody>";
  }
  function downloadText(name, text, type = "text/csv") {
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 500);
  }
  const esc = (s) => (/[",\n]/.test(s) ? '"' + String(s).replace(/"/g, '""') + '"' : s);
  function bindBatch() {
    const drop = $("#drop"), fi = $("#fileIn");
    fi.onchange = () => fi.files[0] && fi.files[0].text().then(loadBatch);
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) f.text().then(loadBatch); });
    $("#batchTemplate").onclick = () => downloadText("wafer_template.csv", ["wafer_id," + F.map((f) => f.key).join(","), "W-001," + x.map((v, i) => fmt(v, i)).join(",")].join("\n"));
    $("#batchSample").onclick = () => {
      const rnd = eng.util.mulberry32(99), lines = ["wafer_id," + F.map((f) => f.key).join(",")];
      let n = 1;
      for (let r = 0; r < 240; r++) {
        const cls = C[Math.floor(rnd() * K)].name, base = meta.archetypes[cls];
        const v = base.map((val, i) => clamp(val + eng.util.gauss(rnd) * 0.5 * ref.std[i], F[i].min, F[i].max));
        lines.push(`W-${String(n++).padStart(3, "0")},` + v.map((a, i) => fmt(a, i)).join(","));
      }
      loadBatch(lines.join("\n"));
    };
    $("#batchDownload").onclick = () => {
      const b = st.batch; if (!b) return;
      const head = ["row", "wafer_id", ...F.map((f) => f.key), "predicted", "confidence", "status", "runner_up", "ood_features", ...C.map((c) => "p_" + c.name)];
      const out = [head.join(",")];
      b.res.forEach((r) => out.push(r.err ? [r.n + 1, esc(r.id || ""), ...r.v.map(() => ""), "invalid"].join(",") :
        [r.n + 1, esc(r.id || ""), ...r.v, r.p.label, r.p.confidence.toFixed(5), statusOf(r.p.label), r.p.runnerUp, r.ood, ...r.p.proba.map((p) => p.toFixed(5))].join(",")));
      downloadText("wafer_predictions.csv", out.join("\n"));
    };
  }

  /* ============ MODEL & API ============ */
  function renderModel() {
    const tg = model.importance.gain;
    $("#modelCard").innerHTML = [
      ["Algorithm", "XGBoost — gradient-boosted decision trees"], ["Objective", "multi:softprob (9 classes)"],
      ["Boosting rounds", model.n_estimators + " × 9 classes = " + model.trees.length + " trees"], ["Max depth", model.max_depth],
      ["Learning rate", model.learning_rate], ["Inputs", NF + " features (" + used.filter(Boolean).length + " used in splits)"],
      ["Top feature", F[tg.indexOf(Math.max(...tg))].label], ["Training data", "≈5,400 simulated wafers (600/class), 70/15/15 stratified split"],
      ["Inference", "In-browser JavaScript (float32-exact), exact TreeSHAP"], ["Model size", "≈ 490 KB JSON"],
    ].map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("");
    $("#taxTable").innerHTML = `<thead><tr><th>Class</th><th>Severity</th><th>Likely root cause</th><th>Process area</th><th>Suggested investigation</th></tr></thead><tbody>` +
      C.map((c) => `<tr><td><span class="dot" style="background:${c.color}"></span><b>${c.name}</b></td><td>${["Stable", "Watch", "Alert", "Critical"][c.severity]}</td><td class="wrap">${c.rootCause}</td><td>${c.processStep}</td><td class="wrap">${c.action}</td></tr>`).join("") + "</tbody>";
    renderCurl();
  }
  const curlBody = () => JSON.stringify({ features: x.map((v) => +(+v).toFixed(4)), explain: true });
  const curlText = () => `curl -X POST ${location.origin}/api/predict \\\n  -H "Content-Type: application/json" \\\n  -d '${curlBody()}'`;
  function renderCurl() { $("#curlBox").textContent = curlText(); }
  function bindApi() {
    $("#copyCurl").onclick = () => navigator.clipboard.writeText(curlText()).then(() => toast("curl copied"));
    $("#apiTry").onclick = async () => {
      const out = $("#apiOut"); out.hidden = false; out.textContent = "Sending…";
      try {
        const t0 = performance.now();
        const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: curlBody() });
        const txt = await r.text(); let pretty = txt; try { pretty = JSON.stringify(JSON.parse(txt), null, 2); } catch (e) { }
        out.textContent = `HTTP ${r.status} · ${(performance.now() - t0).toFixed(0)} ms round-trip\n\n` + pretty;
      } catch (e) { out.textContent = "Request failed: " + e.message + "\n(The API is only available on the deployed Vercel site or via `vercel dev`.)"; }
    };
  }

  /* ============ TABS / THEME / SHARE ============ */
  function showTab(name) {
    st.tab = name;
    $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
    $$(".tabpanel").forEach((p) => (p.hidden = p.id !== "tab-" + name));
    if (history.replaceState) history.replaceState(null, "", location.pathname + location.search + (name === "diagnose" ? "" : "#" + name));
    renderActive(true);
    if (name === "diagnose") drawWafer(pred);
  }
  function renderActive(force) {
    if (st.tab === "explain" && (force || st.dirty.has("explain"))) renderExplain();
    if (st.tab === "whatif" && (force || st.dirty.has("whatif"))) renderWhatIf();
    if (st.tab === "model") renderCurl();
    st.dirty.delete(st.tab);
  }
  let raf = 0, urlT;
  function changed() {
    st.dirty.add("explain"); st.dirty.add("whatif");
    $$(".pchip.on").forEach((b) => b.classList.remove("on"));
    if (raf) return;
    raf = requestAnimationFrame(() => { raf = 0; renderDiagnose(); renderActive(false); });
    clearTimeout(urlT); urlT = setTimeout(() => history.replaceState(null, "", "?v=" + x.map((v) => +(+v).toFixed(4)).join(",") + (st.tab === "diagnose" ? "" : "#" + st.tab)), 400);
  }
  function bindShell() {
    $$(".tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
    $("#miniGo").onclick = () => $("#content").scrollIntoView({ behavior: "smooth" });
    $("#themeBtn").onclick = () => {
      const t = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", t); localStorage.setItem("wf-theme", t);
      renderDiagnose(); st.dirty.add("explain"); st.dirty.add("whatif"); renderActive(true); if (st.batch) renderBatch();
    };
    $("#shareBtn").onclick = () => { const u = location.origin + location.pathname + "?v=" + x.map((v) => +(+v).toFixed(4)).join(","); navigator.clipboard.writeText(u).then(() => toast("Link copied — opens this exact wafer"), () => toast(u)); };
    $("#mapMode").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; st.mapMode = b.dataset.m; $$("#mapMode button").forEach((k) => k.classList.toggle("on", k === b)); drawWafer(pred); };
    $("#impKind").onclick = (e) => { const b = e.target.closest("button"); if (!b) return; st.impKind = b.dataset.k; $$("#impKind button").forEach((k) => k.classList.toggle("on", k === b)); renderImportance(); };
    new ResizeObserver(() => pred && st.tab === "diagnose" && drawWafer(pred)).observe($(".wafer-wrap"));
  }

  /* ---------- boot ---------- */
  buildControls(); bindShell(); bindBatch(); bindApi(); renderModel();
  syncControls(); renderDiagnose();
  const hash = location.hash.slice(1);
  if (["explain", "whatif", "batch", "model"].includes(hash)) showTab(hash);
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) navigator.serviceWorker.register("/sw.js").catch(() => { });
  window.__wafer = { eng, get x() { return x; }, setVector };
})();
