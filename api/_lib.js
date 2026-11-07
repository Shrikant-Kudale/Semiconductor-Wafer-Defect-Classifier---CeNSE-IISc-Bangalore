"use strict";
// Shared helpers for all serverless functions. Loaded once per warm instance.
const Engine = require("../public/js/engine.js");
const model = require("../public/data/model.json");
const meta = require("../public/data/meta.json");

const engine = Engine.create(model);
const KEYS = meta.features.map((f) => f.key);
const SEV = Object.fromEntries(meta.classes.map((c) => [c.name, c.severity]));

function cors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

// Accept a 15-array or an object keyed by feature name. Returns number[] or throws.
function toVector(input) {
  let v;
  if (Array.isArray(input)) v = input;
  else if (input && typeof input === "object") {
    const miss = KEYS.filter((k) => !(k in input));
    if (miss.length) throw new Error("Missing feature(s): " + miss.join(", "));
    v = KEYS.map((k) => input[k]);
  } else throw new Error("`features` must be an array of 15 numbers or an object keyed by feature name");
  if (v.length !== KEYS.length) throw new Error(`Expected ${KEYS.length} features, received ${v.length}`);
  v = v.map(Number);
  if (v.some((n) => !Number.isFinite(n))) throw new Error("All features must be finite numbers");
  return v;
}

function describe(x) {
  const p = engine.predict(x);
  const z = engine.zscores(x);
  const ood = z.map((v, i) => ({ feature: KEYS[i], z: v })).filter((e) => Math.abs(e.z) > 3 && model.importance.gain[KEYS.indexOf(e.feature)] > 0);
  return {
    label: p.label,
    confidence: p.confidence,
    severity: SEV[p.label],
    status: ["STABLE", "WATCH", "ALERT", "CRITICAL"][SEV[p.label]],
    margin: p.margin,
    entropy: p.entropy,
    runnerUp: p.runnerUp,
    probabilities: Object.fromEntries(model.classes.map((c, i) => [c, p.proba[i]])),
    outOfDistribution: ood,
  };
}

module.exports = { engine, model, meta, KEYS, cors, toVector, describe };
