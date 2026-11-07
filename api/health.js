"use strict";
const { model, cors } = require("./_lib.js");
module.exports = (req, res) => {
  cors(res);
  res.setHeader("Cache-Control", "no-store");
  res.status(200).json({ status: "ok", trees: model.trees.length, classes: model.classes.length, runtime: process.version, time: new Date().toISOString() });
};
