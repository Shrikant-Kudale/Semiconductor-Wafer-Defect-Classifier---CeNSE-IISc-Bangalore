const test = require("node:test");
const assert = require("node:assert");
const path = require("node:path");
const Engine = require("../public/js/engine.js");
const model = require("../public/data/model.json");
const golden = require("./golden.json");

const eng = Engine.create(model);

test("probabilities match Python XGBoost (|Δ| < 1e-5)", () => {
  let worst = 0;
  golden.X.forEach((x, i) => {
    const p = eng.proba(x);
    p.forEach((v, k) => { worst = Math.max(worst, Math.abs(v - golden.P[i][k])); });
  });
  console.log("    max |Δp| =", worst.toExponential(2));
  assert.ok(worst < 1e-5);
});

test("argmax identical on every golden vector", () => {
  golden.X.forEach((x, i) => {
    const g = golden.P[i].indexOf(Math.max(...golden.P[i]));
    assert.strictEqual(eng.predict(x).index, g);
  });
});

test("TreeSHAP matches xgboost pred_contribs (|Δ| < 1e-4)", () => {
  let worst = 0;
  golden.X.slice(0, 30).forEach((x, i) => {
    for (const cls of [0, 3, 6, 8]) {
      const s = eng.shap(x, cls);
      const ref = golden.C[i][cls];
      s.values.forEach((v, f) => { worst = Math.max(worst, Math.abs(v - ref[f])); });
      worst = Math.max(worst, Math.abs(s.bias - ref[model.n_features]));
    }
  });
  console.log("    max |Δshap| =", worst.toExponential(2));
  assert.ok(worst < 1e-4);
});

test("SHAP efficiency: bias + Σφ equals the class margin", () => {
  const x = golden.X[10];
  const m = eng.margins(x);
  for (let c = 0; c < eng.K; c++) {
    const s = eng.shap(x, c);
    assert.ok(Math.abs(s.bias + s.values.reduce((a, b) => a + b, 0) - m[c]) < 1e-4);
  }
});

test("API handler: predict, batch, validation", async () => {
  const handler = require("../api/predict.js");
  const call = (method, body) => new Promise((resolve) => {
    const res = { headers: {}, setHeader(k, v) { this.headers[k] = v; }, status(c) { this.code = c; return this; },
      json(o) { resolve({ code: this.code || 200, body: o }); }, end() { resolve({ code: this.code || 204 }); } };
    handler({ method, body, query: {} }, res);
  });
  const ok = await call("POST", { features: golden.X[0], explain: true });
  assert.strictEqual(ok.code, 200);
  assert.strictEqual(ok.body.prediction.label, model.classes[golden.P[0].indexOf(Math.max(...golden.P[0]))]);
  assert.ok(ok.body.explanation.shap.length === 15);
  const batch = await call("POST", { batch: [golden.X[1], golden.X[2]] });
  assert.strictEqual(batch.body.results.length, 2);
  const bad = await call("POST", { features: [1, 2, 3] });
  assert.strictEqual(bad.code, 400);
  const named = await call("POST", { features: Object.fromEntries(require("../public/data/meta.json").features.map((f, i) => [f.key, golden.X[0][i]])) });
  assert.strictEqual(named.code, 200);
});
