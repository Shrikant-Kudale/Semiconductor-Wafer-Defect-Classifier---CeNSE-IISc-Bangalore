"use strict";
const { model, meta, cors } = require("./_lib.js");
module.exports = (req, res) => {
  cors(res);
  res.setHeader("Cache-Control", "public, s-maxage=86400, stale-while-revalidate=604800");
  res.status(200).json({
    algorithm: "XGBoost (gradient-boosted decision trees)",
    objective: model.objective,
    n_estimators: model.n_estimators,
    max_depth: model.max_depth,
    learning_rate: model.learning_rate,
    n_trees_total: model.trees.length,
    classes: model.classes,
    features: meta.features.map((f, i) => ({
      key: f.key, unit: f.unit, usedByModel: f.used,
      importanceGain: model.importance.gain[i],
      trainingMean: model.reference.mean[i], trainingStd: model.reference.std[i],
    })),
    note: "Trained on simulated wafer data (proof of concept); not validated on proprietary fab data.",
  });
};
