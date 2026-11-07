/* Tiny dependency-free SVG chart toolkit (ring, radar, waterfall, lines, histogram). */
(function (root) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  function h(tag, attrs, ...kids) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
    return e;
  }
  const svg = (w, hgt, extra) => h("svg", Object.assign({ viewBox: `0 0 ${w} ${hgt}`, role: "img", preserveAspectRatio: "xMidYMid meet" }, extra));
  const nf = (v, d = 2) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(d));

  /* ----- confidence ring ----- */
  function ring(el, p, color) {
    const r = 48, c = 2 * Math.PI * r;
    const s = svg(120, 120, { "aria-label": `Confidence ${(p * 100).toFixed(1)} percent` });
    s.append(
      h("circle", { cx: 60, cy: 60, r, fill: "none", stroke: css("--surface2"), "stroke-width": 11 }),
      h("circle", { cx: 60, cy: 60, r, fill: "none", stroke: color, "stroke-width": 11, "stroke-linecap": "round",
        "stroke-dasharray": `${c * p} ${c}`, transform: "rotate(-90 60 60)", style: "transition:stroke-dasharray .5s cubic-bezier(.2,.8,.2,1)" }),
      h("text", { x: 60, y: 62, "text-anchor": "middle", "font-size": 19, "font-weight": 800, fill: css("--text") }, (p * 100).toFixed(1) + "%"),
      h("text", { x: 60, y: 80, "text-anchor": "middle", "font-size": 9.5, fill: css("--muted"), "letter-spacing": ".08em" }, "CONFIDENCE")
    );
    el.replaceChildren(s);
  }

  /* ----- radar of z-scores (−4…+4σ) ----- */
  function radar(el, labels, zs, opts) {
    opts = opts || {};
    const W = 560, H = 440, cx = W / 2, cy = H / 2 + 6, R = 150, lim = 4, n = labels.length;
    const s = svg(W, H, { "aria-label": "Radar chart of z-scores" });
    const ang = (i) => (-Math.PI / 2) + (2 * Math.PI * i) / n;
    const rad = (z) => ((Math.max(-lim, Math.min(lim, z)) + lim) / (2 * lim)) * R;
    const pt = (i, z) => [cx + rad(z) * Math.cos(ang(i)), cy + rad(z) * Math.sin(ang(i))];
    const grid = css("--border"), muted = css("--muted");
    [-3, -2, -1, 0, 1, 2, 3].forEach((z) => {
      const poly = labels.map((_, i) => pt(i, z).join(",")).join(" ");
      s.append(h("polygon", { points: poly, fill: z === 0 ? css("--accent-soft") : "none", stroke: z === 0 ? css("--accent") : grid, "stroke-width": z === 0 ? 1.4 : 1, "stroke-dasharray": z === 0 ? "4 3" : "" }));
    });
    s.append(h("text", { x: cx + 4, y: cy - rad(3) + 10, "font-size": 8.5, fill: muted }, "+3σ"), h("text", { x: cx + 4, y: cy - rad(0) + 10, "font-size": 8.5, fill: css("--accent") }, "mean"));
    labels.forEach((l, i) => {
      const [x, y] = pt(i, lim);
      s.append(h("line", { x1: cx, y1: cy, x2: x, y2: y, stroke: grid }));
      const lx = cx + (R + 24) * Math.cos(ang(i)), ly = cy + (R + 20) * Math.sin(ang(i));
      const bad = Math.abs(zs[i]) > 3;
      s.append(h("text", { x: lx, y: ly + 3, "text-anchor": Math.abs(lx - cx) < 8 ? "middle" : lx > cx ? "start" : "end", "font-size": 10.5, "font-weight": bad ? 800 : 600, fill: bad ? css("--bad") : muted }, l));
    });
    const poly = zs.map((z, i) => pt(i, z).join(",")).join(" ");
    s.append(h("polygon", { points: poly, fill: css("--accent"), "fill-opacity": .2, stroke: css("--accent"), "stroke-width": 2, "stroke-linejoin": "round" }));
    zs.forEach((z, i) => {
      const [x, y] = pt(i, z), bad = Math.abs(z) > 3;
      const c = h("circle", { cx: x, cy: y, r: bad ? 5 : 3.4, fill: bad ? css("--bad") : css("--accent"), stroke: css("--surface"), "stroke-width": 1.5 });
      c.append(h("title", {}, `${labels[i]}: z = ${z.toFixed(2)}`));
      s.append(c);
    });
    el.replaceChildren(s);
  }

  /* ----- SHAP waterfall ----- */
  function waterfall(el, d) {
    // d: {base, final, items:[{label, value, shap}], color, classLabel}
    const W = 520, rowH = 32, padL = 150, padR = 56, top = 30, n = d.items.length;
    const H = top + (n + 2) * rowH + 28;
    const s = svg(W, H, { "aria-label": "SHAP waterfall" });
    let cum = d.base;
    const segs = d.items.map((it) => { const a = cum; cum += it.shap; return { ...it, a, b: cum }; });
    const all = [d.base, d.final, ...segs.flatMap((g) => [g.a, g.b])];
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
    const X = (v) => padL + ((v - lo) / (hi - lo)) * (W - padL - padR);
    const pos = d.color, neg = css("--muted"), txt = css("--text"), grid = css("--border");
    // axis ticks
    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4, x = X(v);
      s.append(h("line", { x1: x, x2: x, y1: top - 6, y2: H - 26, stroke: grid, "stroke-dasharray": "2 4" }), h("text", { x, y: H - 10, "text-anchor": "middle", "font-size": 9.5, fill: css("--muted") }, v.toFixed(1)));
    }
    s.append(h("text", { x: W - padR, y: H - 1, "text-anchor": "end", "font-size": 10, fill: css("--muted") }, "log-odds (margin) →"));
    const row = (i, label, sub, a, b, fill, valTxt, bold) => {
      const y = top + i * rowH;
      const x0 = X(Math.min(a, b)), w = Math.max(2, Math.abs(X(b) - X(a)));
      s.append(
        h("text", { x: padL - 10, y: y + 13, "text-anchor": "end", "font-size": 12, "font-weight": bold ? 800 : 600, fill: txt }, label),
        sub ? h("text", { x: padL - 10, y: y + 24, "text-anchor": "end", "font-size": 10, fill: css("--muted") }, sub) : null,
        h("rect", { x: x0, y: y + 3, width: w, height: 17, rx: 4, fill, opacity: bold ? 1 : .92 }),
        h("text", { x: X(Math.max(a, b)) + 6, y: y + 16, "font-size": 11.5, "font-weight": 700, fill: txt }, valTxt)
      );
    };
    row(0, "Base value", "E[margin] over training", lo + pad * 0.0001, d.base, css("--border"), d.base.toFixed(2), true);
    segs.forEach((g, i) => row(i + 1, g.label, g.sub, g.a, g.b, g.shap >= 0 ? pos : neg, (g.shap >= 0 ? "+" : "") + g.shap.toFixed(2)));
    row(n + 1, "Model margin f(x)", d.classLabel, lo + pad * 0.0001, d.final, pos, d.final.toFixed(2), true);
    // connectors
    s.append(h("line", { x1: X(d.base), x2: X(d.base), y1: top + 20, y2: top + rowH, stroke: grid }));
    el.replaceChildren(s);
  }

  /* ----- multi-line chart with hover ----- */
  function lines(el, d, onHover) {
    // d: {xs, series:[{name,color,ys,w,dim}], xLabel, band:[lo,hi]|null, marker, xmin, xmax}
    const W = 760, H = 330, L = 46, R = 16, T = 14, B = 42;
    const s = svg(W, H, { "aria-label": "Sensitivity chart" });
    const X = (v) => L + ((v - d.xs[0]) / (d.xs[d.xs.length - 1] - d.xs[0])) * (W - L - R);
    const Y = (p) => T + (1 - p) * (H - T - B);
    const grid = css("--border"), muted = css("--muted");
    for (let i = 0; i <= 5; i++) {
      const p = i / 5, y = Y(p);
      s.append(h("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: grid }), h("text", { x: L - 8, y: y + 3.5, "text-anchor": "end", "font-size": 10, fill: muted }, (p * 100).toFixed(0) + "%"));
    }
    for (let i = 0; i <= 6; i++) {
      const v = d.xs[0] + ((d.xs[d.xs.length - 1] - d.xs[0]) * i) / 6;
      s.append(h("text", { x: X(v), y: H - B + 16, "text-anchor": "middle", "font-size": 10, fill: muted }, nf(v, v < 10 ? 2 : 1)));
    }
    s.append(h("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle", "font-size": 11, fill: muted, "font-weight": 600 }, d.xLabel));
    if (d.band) {
      const x1 = Math.max(L, X(d.band[0])), x2 = Math.min(W - R, X(d.band[1]));
      if (x2 > x1) s.append(h("rect", { x: x1, y: T, width: x2 - x1, height: H - T - B, fill: css("--accent-soft") }), h("text", { x: x1 + 4, y: T + 12, "font-size": 9, fill: css("--accent") }, "training ±3σ"));
    }
    d.series.forEach((se) => {
      if (se.off) return;
      const pts = se.ys.map((y, i) => `${X(d.xs[i]).toFixed(1)},${Y(y).toFixed(1)}`).join(" ");
      s.append(h("polyline", { points: pts, fill: "none", stroke: se.color, "stroke-width": se.bold ? 3 : 1.6, "stroke-linejoin": "round", opacity: se.bold ? 1 : .75 }));
    });
    s.append(h("line", { x1: X(d.marker), x2: X(d.marker), y1: T, y2: H - B, stroke: css("--text"), "stroke-width": 1.4, "stroke-dasharray": "5 4" }),
      h("text", { x: X(d.marker), y: H - B + 30, "text-anchor": "middle", "font-size": 10, "font-weight": 800, fill: css("--text") }, "▲ now"));
    const cross = h("line", { y1: T, y2: H - B, stroke: muted, "stroke-width": 1, opacity: 0 });
    const hit = h("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent", style: "cursor:crosshair" });
    s.append(cross, hit);
    const move = (ev) => {
      const r = s.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      const t = Math.max(0, Math.min(1, (px - L) / (W - L - R)));
      const i = Math.round(t * (d.xs.length - 1));
      cross.setAttribute("x1", X(d.xs[i])); cross.setAttribute("x2", X(d.xs[i])); cross.setAttribute("opacity", 1);
      onHover && onHover(i);
    };
    hit.addEventListener("pointermove", move);
    hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); onHover && onHover(-1); });
    el.replaceChildren(s);
  }

  /* ----- histogram ----- */
  function hist(el, values, lo, hi, bins, color) {
    const W = 420, H = 190, L = 30, R = 8, T = 10, B = 28;
    const counts = new Array(bins).fill(0);
    values.forEach((v) => counts[Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)))]++);
    const mx = Math.max(1, ...counts), bw = (W - L - R) / bins;
    const s = svg(W, H, { "aria-label": "Histogram" });
    counts.forEach((c, i) => {
      const bh = (c / mx) * (H - T - B);
      s.append(h("rect", { x: L + i * bw + 1.5, y: H - B - bh, width: bw - 3, height: bh, rx: 3, fill: color, opacity: i / bins < 0.6 ? 0.55 : 0.95 }));
    });
    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4;
      s.append(h("text", { x: L + ((W - L - R) * i) / 4, y: H - 10, "text-anchor": "middle", "font-size": 9.5, fill: css("--muted") }, (v * 100).toFixed(0) + "%"));
    }
    s.append(h("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: css("--border") }), h("text", { x: L - 6, y: T + 8, "text-anchor": "end", "font-size": 9.5, fill: css("--muted") }, mx));
    el.replaceChildren(s);
  }

  root.Charts = { h, ring, radar, waterfall, lines, hist, css };
})(window);
