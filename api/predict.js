"use strict";
const { engine, model, KEYS, cors, toVector, describe } = require("./_lib.js");

/**
 * POST /api/predict
 *   { "features": [15 numbers] | { name: value, ... }, "explain": true? }
 *   { "batch": [ [15 numbers] | {...}, ... ] }          (max 5000 rows)
 */
module.exports = (req, res) => {
  cors(res);
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST, OPTIONS");
    return res.status(405).json({ error: "Use POST", features: KEYS });
  }
  try {
    let body = req.body;
    if (typeof body === "string") body = JSON.parse(body || "{}");
    body = body || {};
    const t0 = process.hrtime.bigint();

    if (Array.isArray(body.batch)) {
      if (body.batch.length > 5000) return res.status(413).json({ error: "Batch limited to 5000 rows" });
      const results = body.batch.map((row, i) => {
        try { return { index: i, ...describe(toVector(row)) }; }
        catch (e) { return { index: i, error: e.message }; }
      });
      return res.status(200).json({ count: results.length, results, ms: Number(process.hrtime.bigint() - t0) / 1e6 });
    }

    const x = toVector(body.features);
    const out = { prediction: describe(x) };
    if (body.explain) {
      const cls = model.classes.indexOf(out.prediction.label);
      const s = engine.shap(x, cls);
      out.explanation = {
        method: "TreeSHAP (exact), log-odds of predicted class",
        class: out.prediction.label,
        baseValue: s.bias,
        features: KEYS,
        shap: s.values,
        ranked: KEYS.map((k, i) => ({ feature: k, shap: s.values[i], value: x[i] }))
          .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap)).slice(0, 8),
      };
    }
    out.ms = Number(process.hrtime.bigint() - t0) / 1e6;
    res.setHeader("Cache-Control", "no-store");
    return res.status(200).json(out);
  } catch (e) {
    return res.status(400).json({ error: e.message });
  }
};
