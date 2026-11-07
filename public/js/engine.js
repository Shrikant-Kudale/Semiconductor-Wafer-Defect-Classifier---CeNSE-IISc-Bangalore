/*!
 * WaferEngine — dependency-free XGBoost inference for the browser and Node.
 *
 *  - exact multi-class softprob inference (float32 comparisons, as in XGBoost)
 *  - exact TreeSHAP attributions (Lundberg et al., 2018), cover-weighted
 *  - 1-D sensitivity sweeps, Monte-Carlo robustness, greedy counterfactual search
 *
 * UMD: window.WaferEngine in the browser, module.exports in Node / Vercel.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.WaferEngine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const f32 = Math.fround;

  /* ---------- small utilities ---------- */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rnd) {
    let u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function softmax(m) {
    let mx = -Infinity;
    for (const v of m) if (v > mx) mx = v;
    const e = m.map((v) => Math.exp(v - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }
  function argmax(a) {
    let bi = 0;
    for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
    return bi;
  }
  function entropy(p) {
    let h = 0;
    for (const v of p) if (v > 1e-12) h -= v * Math.log(v);
    return h;
  }

  /* ---------- engine ---------- */
  function create(model) {
    const K = model.classes.length;
    const F = model.n_features;
    const trees = model.trees;
    const byClass = Array.from({ length: K }, () => []);
    trees.forEach((t, i) => byClass[t.c].push(i));

    // pre-compute per-tree expected value (cover-weighted mean of leaves)
    const treeExp = trees.map((t) => {
      const rec = (n) => {
        if (t.l[n] === -1) return t.t[n];
        const wl = t.w[t.l[n]], wr = t.w[t.r[n]];
        return (wl * rec(t.l[n]) + wr * rec(t.r[n])) / (wl + wr);
      };
      return rec(0);
    });

    function leafValue(t, x) {
      let n = 0;
      while (t.l[n] !== -1) {
        const v = x[t.f[n]];
        if (Number.isNaN(v)) n = t.d[n] ? t.l[n] : t.r[n];
        else n = v < t.t[n] ? t.l[n] : t.r[n];
      }
      return t.t[n];
    }

    function toF32(x) {
      const o = new Array(F);
      for (let i = 0; i < F; i++) o[i] = f32(+x[i]);
      return o;
    }

    function margins(x) {
      const xf = toF32(x);
      const m = model.base_score.slice();
      for (let i = 0; i < trees.length; i++) m[trees[i].c] += leafValue(trees[i], xf);
      return m;
    }

    function proba(x) { return softmax(margins(x)); }

    function predict(x) {
      const p = proba(x);
      const idx = argmax(p);
      const sorted = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
      return {
        index: idx,
        label: model.classes[idx],
        confidence: p[idx],
        proba: p,
        margin: sorted[0][0] - sorted[1][0],
        runnerUp: model.classes[sorted[1][1]],
        entropy: entropy(p),
        normalizedEntropy: entropy(p) / Math.log(K),
        top3: sorted.slice(0, 3).map(([v, i]) => ({ label: model.classes[i], p: v })),
      };
    }

    /* ----- exact TreeSHAP (margin / log-odds space, one class) ----- */
    function shap(x, cls) {
      const xf = toF32(x);
      const phi = new Array(F).fill(0);
      let bias = model.base_score[cls];
      for (const ti of byClass[cls]) {
        const t = trees[ti];
        bias += treeExp[ti];
        if (t.l[0] === -1) continue;

        const recurse = (node, parentPath, depth, pz, po, pi) => {
          const path = parentPath.map((e) => ({ ...e }));
          // extend
          path[depth] = { d: pi, z: pz, o: po, w: depth === 0 ? 1 : 0 };
          for (let i = depth - 1; i >= 0; i--) {
            path[i + 1].w += (po * path[i].w * (i + 1)) / (depth + 1);
            path[i].w = (pz * path[i].w * (depth - i)) / (depth + 1);
          }
          if (t.l[node] === -1) {
            for (let i = 1; i <= depth; i++) {
              // unwound sum
              const o = path[i].o, z = path[i].z;
              let next = path[depth].w, total = 0;
              for (let j = depth - 1; j >= 0; j--) {
                if (o !== 0) {
                  const tmp = (next * (depth + 1)) / ((j + 1) * o);
                  total += tmp;
                  next = path[j].w - tmp * z * ((depth - j) / (depth + 1));
                } else if (z !== 0) {
                  total += path[j].w / z / ((depth - j) / (depth + 1));
                }
              }
              phi[path[i].d] += total * (path[i].o - path[i].z) * t.t[node];
            }
            return;
          }
          const f = t.f[node];
          const hot = xf[f] < t.t[node] ? t.l[node] : t.r[node];
          const cold = hot === t.l[node] ? t.r[node] : t.l[node];
          let iz = 1, io = 1, d = depth;
          let k = -1;
          for (let i = 1; i <= d; i++) if (path[i].d === f) { k = i; break; }
          if (k >= 0) {
            iz = path[k].z; io = path[k].o;
            // unwind path element k
            const o = path[k].o, z = path[k].z;
            let next = path[d].w;
            for (let j = d - 1; j >= 0; j--) {
              if (o !== 0) {
                const tmp = path[j].w;
                path[j].w = (next * (d + 1)) / ((j + 1) * o);
                next = tmp - (path[j].w * z * (d - j)) / (d + 1);
              } else {
                path[j].w = (path[j].w * (d + 1)) / (z * (d - j));
              }
            }
            for (let j = k; j < d; j++) { path[j].d = path[j + 1].d; path[j].z = path[j + 1].z; path[j].o = path[j + 1].o; }
            d -= 1;
          }
          const cn = t.w[node];
          recurse(hot, path, d + 1, (iz * t.w[hot]) / cn, io, f);
          recurse(cold, path, d + 1, (iz * t.w[cold]) / cn, 0, f);
        };
        recurse(0, [], 0, 1, 1, -1);
      }
      return { values: phi, bias };
    }

    /* ----- what-if tools ----- */
    function sweep(x, fi, lo, hi, n) {
      n = n || 60;
      const xs = [], ps = [];
      const v = x.slice();
      for (let i = 0; i < n; i++) {
        const val = lo + ((hi - lo) * i) / (n - 1);
        v[fi] = val;
        xs.push(val);
        ps.push(proba(v));
      }
      return { xs, ps };
    }

    // Monte-Carlo: Gaussian measurement noise sigma = frac * training std
    function monteCarlo(x, frac, n, seed, lockedMask) {
      const rnd = mulberry32(seed || 1);
      const counts = new Array(K).fill(0);
      const meanP = new Array(K).fill(0);
      const sd = model.reference.std;
      n = n || 300;
      for (let i = 0; i < n; i++) {
        const v = x.map((val, j) => (lockedMask && lockedMask[j] ? val : val + gauss(rnd) * frac * sd[j]));
        const p = proba(v);
        counts[argmax(p)]++;
        for (let k = 0; k < K; k++) meanP[k] += p[k] / n;
      }
      return { freq: counts.map((c) => c / n), meanP, n };
    }

    // Greedy coordinate search for the smallest standardized change raising P(target)
    function counterfactual(x, target, bounds, opts) {
      opts = opts || {};
      const maxIter = opts.maxIter || 8;
      const goal = opts.goal || 0.8;
      const locked = opts.locked || [];
      const sd = model.reference.std;
      const cur = x.slice();
      const steps = [];
      let p = proba(cur)[target];
      for (let it = 0; it < maxIter && p < goal; it++) {
        let best = null;
        for (let j = 0; j < F; j++) {
          if (locked[j] || model.importance.gain[j] === 0) continue;
          const [lo, hi] = bounds[j];
          for (const mult of [-2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2]) {
            const v = cur.slice();
            v[j] = Math.min(hi, Math.max(lo, cur[j] + mult * sd[j]));
            if (v[j] === cur[j]) continue;
            const pn = proba(v)[target];
            const cost = Math.abs(v[j] - cur[j]) / sd[j];
            const gainPer = (pn - p) / (0.35 + cost);
            if (pn > p + 1e-6 && (!best || gainPer > best.gainPer)) best = { j, v: v[j], pn, gainPer };
          }
        }
        if (!best) break;
        steps.push({ feature: best.j, from: cur[best.j], to: best.v, pBefore: p, pAfter: best.pn });
        cur[best.j] = best.v;
        p = best.pn;
      }
      return { x: cur, steps, p, reached: p >= goal };
    }

    // Out-of-distribution score: per-feature z vs training population
    function zscores(x) {
      return x.map((v, i) => (v - model.reference.mean[i]) / model.reference.std[i]);
    }

    return {
      model, K, F, classes: model.classes,
      margins, proba, predict, shap, sweep, monteCarlo, counterfactual, zscores,
      util: { softmax, argmax, entropy, mulberry32, gauss },
    };
  }

  return { create };
});
