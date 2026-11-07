# ⚡ WaferFab Intelligence — Explainable Semiconductor Defect Diagnostics

**Vercel-ready rewrite** of the *Semiconductor Wafer Defect Classifier* built during the MoTA Foundation Training Program on Semiconductor Fabrication & Characterization (CeNSE, IISc Bengaluru, 8–19 June 2026).

The original Streamlit app slept after inactivity and had to be woken up. This version has **no server to wake**: the trained XGBoost model is exported to a 490 KB JSON file and executed **inside the browser** by a small, dependency-free JavaScript engine. Vercel only serves static files, plus an optional serverless REST API that runs the same engine.

| | Streamlit (before) | Vercel (now) |
|---|---|---|
| Cold start | 10–60 s "wake up" | **none** (static + in-browser inference) |
| Inference latency | server round-trip | **~0.05 ms** locally |
| Offline | no | **yes** (PWA / service worker) |
| Explainability | none | **exact TreeSHAP** per prediction |
| Python runtime on host | required | **not needed** |
| API | none | `/api/predict`, `/api/model`, `/api/health` |

## Why the model was converted instead of "just deploying Python"

Vercel's Python functions have a 250 MB unzipped limit. A current `pip install xgboost` pulls in `nvidia-nccl` (~345 MB on disk) on top of xgboost itself (~85 MB), so the XGBoost wheel alone exceeds it. Even if it fit, each cold start would re-import numpy/sklearn/xgboost.

Instead, `scripts/export_model.py` converts the 1,800 trees (200 rounds × 9 classes) to JSON and `public/js/engine.js` walks them. This is **numerically verified against the original Python model** (below).

## What's new (everything, basically)

**Diagnose** — verdict card with confidence ring, margin to runner-up, normalised entropy, graded status (STABLE / WATCH / ALERT / CRITICAL), root cause + process area + suggested investigation for each of the 9 classes (from your presentation's taxonomy); reconstructed 32×32 die-level wafer map (die / scatter / both) with radial density profile; probability bars; z-score "process fingerprint" radar against the training population; automatic **out-of-distribution warnings** and low-confidence flags.

**Explain** — exact **TreeSHAP waterfall** for any class, global importance (gain / cover / splits), full input profile with z-scores and illustrative spec windows.

**What-if** — one-at-a-time **sensitivity sweeps** with detected decision boundaries; **Monte-Carlo robustness** under measurement noise; **counterfactual search** ("smallest changes to reach class X") with parameter locks and a *process-metrology-only* mode.

**Batch** — drag-and-drop CSV scoring (in-browser, data never leaves the device), class mix, confidence histogram, OOD counts, downloadable results CSV, demo lot.

**Model & API** — model card, taxonomy table, live API tester with copy-able `curl`, OpenAPI spec.

**Platform** — day/night theme, responsive/mobile layout, shareable links (`?v=` encodes the exact wafer), installable PWA, security headers, CI.

### Findings from reading the original model (worth knowing)
* The model was trained on **raw features**; `feature_scaler.joblib` is not used at inference (it is used here only as the *training-population reference* for z-scores).
* `defect_density` is **never used in any tree split** (importance 0). The UI marks it "unused by model".
* Training features occupy narrow bands (e.g. oxide 100.4 ± 2.2 nm), whereas the old sliders allowed 80–120 nm. The new sliders show the training band (mean ± 3σ) and the app warns when inputs are extrapolations.
* Slider ranges for cluster count / largest cluster were widened (0–80, 0–300): the old maxima (20, 100) were below the training statistics (mean 28 and 70).

## Deploy to Vercel

### Option A — GitHub import (recommended)
1. Replace the contents of your repository with this project (or push it to a new repo).
2. Go to <https://vercel.com/new>, import the repo.
3. Leave **Framework Preset = Other**, **Build Command = empty**, **Output Directory = `public`** (already set in `vercel.json`).
4. Deploy. Every `git push` redeploys automatically.

### Option B — CLI
```bash
npm i -g vercel
vercel          # preview
vercel --prod   # production
```

No environment variables are required.

## Run locally
```bash
npm start            # zero-dependency server → http://localhost:3000  (static + /api)
npm test             # parity + TreeSHAP + API tests
# or, to emulate Vercel exactly:
vercel dev
```

## REST API
```bash
curl -X POST https://<your-app>.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[159,0.265,0.988,0.236,0.95,0.167,36,27,99.9,44.5,3.48,1.6,92.8,5.6,74.1],"explain":true}'
```
`features` may be an array of 15 numbers (order below) or an object keyed by feature name. Send `{"batch":[[...],[...]]}` (≤5000 rows) for batch scoring. Feature order:
`defect_count, defect_density, mean_radial_position, std_radial_position, edge_ratio, center_ratio, cluster_count, max_cluster_size, oxide_thickness_nm, critical_dimension_nm, overlay_error_nm, line_edge_roughness_nm, etch_uniformity_pct, particle_index, wafer_yield_pct`.

Other endpoints: `GET /api/model`, `GET /api/health`, and the spec at `/openapi.json`.

## Verification (what was actually tested)
`npm test` compares the JS engine to the original Python XGBoost on 69 reference wafers (9 archetypes + 60 random draws):

* class probabilities: max |Δp| = **6.6 × 10⁻⁷**, identical argmax on all vectors
* TreeSHAP vs `pred_contribs`: max |Δφ| = **1.3 × 10⁻⁶**; efficiency axiom (bias + Σφ = margin) holds
* serverless handler: single, batch, named-feature and validation paths

The UI was additionally smoke-tested in headless Chromium (light/dark, all tabs, mobile viewport, zero console errors).

## Project layout
```
public/                 static site served by Vercel
  index.html  css/style.css
  js/engine.js          XGBoost inference + TreeSHAP + what-if tools (browser & Node)
  js/charts.js          dependency-free SVG charts
  js/app.js             application logic
  data/model.json       exported model (1,800 trees)
  data/meta.json        feature/class metadata + archetype wafers
  openapi.json  sw.js  manifest.webmanifest  favicon.svg
api/                    Vercel serverless functions (Node 22)
  predict.js  model.js  health.js  _lib.js
models/                 your original .joblib artefacts (provenance)
scripts/                export_model.py · build_meta.py · make_presets.py · make_golden.py · serve.js
tests/                  engine.test.js + golden.json (Python reference outputs)
legacy/                 the original Streamlit app, untouched
vercel.json  package.json  .github/workflows/ci.yml
```

## Re-training or swapping the model
1. Train and save as before (`xgb_defect_model.joblib`, `feature_scaler.joblib`, `label_encoder.joblib`) into `models/`.
2. `pip install -r scripts/requirements.txt`
3. `npm run export` (regenerates `model.json` and `meta.json`), then `python scripts/make_presets.py && python scripts/make_golden.py && npm test`.

Class names and the 15-feature order must stay the same, or update `scripts/build_meta.py`.

## Scope & honest limitations
* Trained on **simulated** wafers (≈5,400 samples); proof of concept, not validated on fab data (see the presentation's Limitations slide). I did not have the held-out test metrics, so none are quoted here — add yours to the Model card in `public/js/app.js` (`renderModel`).
* The wafer map is a **schematic reconstruction** from summary statistics, not a measured inspection map. The CNN branch from the presentation is not part of this deployment.
* Spec windows (e.g. oxide 95–105 nm) are illustrative defaults, not process specifications.
* Counterfactuals describe the model's decision surface, not physical causality.
* Suggested investigation text is general guidance based on the class root causes in the presentation.

## Credits
Shrikant Kudale · MIT-ADT University, Pune · MoTA Foundation Training Program, CeNSE, IISc Bengaluru.
Method references: Lundberg et al. (2018) *Consistent Individualized Feature Attribution for Tree Ensembles* (TreeSHAP); Chen & Guestrin (2016) *XGBoost*; WM-811K wafer-map taxonomy (Wu et al., 2015).
