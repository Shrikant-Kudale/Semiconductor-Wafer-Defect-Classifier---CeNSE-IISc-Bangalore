#!/usr/bin/env python3
"""Derive one representative, high-confidence 'archetype' wafer per class by
stochastic hill-climbing from domain-informed seeds (stays near the seed so the
archetype remains physically plausible). Output: scripts/_presets.json"""
import json, warnings, pathlib
import joblib, numpy as np
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
xgb = joblib.load(ROOT / "models/xgb_defect_model.joblib")
le = joblib.load(ROOT / "models/label_encoder.joblib")
ref = json.loads((ROOT / "public/data/model.json").read_text())["reference"]
sd = np.array(ref["std"])
names = [str(c) for c in le.classes_]
#          count dens  mrad  srad  edge  ctr  clus maxcl oxide  CD   ovl  rough etch part yield
seeds = {
 "Normal":    [12, .02, .54, .26, .20, .20,  1,   2, 99.0, 44.3, 3.3, 2.4, 95.2, 5.4, 78.9],
 "Center":    [213,.35, .36, .29, .48, .84, 32, 139,102.8, 45.0, 3.7, 2.2, 93.6, 4.7, 81.8],
 "Donut":     [170,.28, .37, .07, .49, .12, 40, 137, 99.9, 44.9, 3.5, 2.0, 94.3, 5.6, 83.5],
 "Edge-Loc":  [ 25,.05, .83, .24, .55, .38, 30, 144, 99.4, 44.9, 4.0, 1.9, 94.7, 5.6, 81.3],
 "Edge-Ring": [227,.38, .91, .24, .90, .35, 33, 141, 99.4, 45.0, 3.5, 2.0, 94.6, 5.7, 80.6],
 "Loc":       [ 47,.08, .35, .22, .18, .30, 26, 112, 99.0, 44.9, 3.2, 1.7, 95.8, 7.5, 87.0],
 "Near-Full": [365,.60, .40, .28, .50, .34, 30, 170, 99.3, 44.8, 3.5, 2.0, 94.5, 5.8, 65.0],
 "Random":    [237,.40, .49, .26, .49, .39, 71, 147, 99.2, 45.0, 3.4, 2.2, 94.9, 5.7, 82.6],
 "Scratch":   [ 63,.10, .52, .28, .50, .38, 26, 163, 99.9, 45.3, 3.5, 3.0, 94.6, 5.0, 82.6],
}
lo = np.array([0,0,0,0,0,0,0,0,80,30,0,0,80,0,0.]); hi = np.array([480,1,1,.5,.95,1,65,300,120,60,10,5,100,20,100.])
rng = np.random.default_rng(7)
out = {}
for k, s in seeds.items():
    ci = names.index(k); x = np.array(s, float)
    p = lambda v: xgb.predict_proba(v.reshape(1, -1).astype(np.float32))[0, ci]
    best = p(x)
    for _ in range(1500):
        cand = np.clip(x + rng.normal(0, 0.12, 15) * sd * (np.arange(15) != 1), lo, hi)
        # stay within ~0.8 sd of the seed
        if np.any(np.abs(cand - np.array(s)) > 0.8 * sd * np.where(np.arange(15) == 1, 99, 1)):
            continue
        pc = p(cand)
        if pc >= best:
            x, best = cand, pc
    x[1] = round(min(1.0, x[0] / 600.0), 3)  # density is unused by the model; keep it coherent
    out[k] = {"values": [round(float(v), 3) for v in x], "p": round(float(p(x)), 4)}
    print(k, out[k]["p"], out[k]["values"])
(ROOT / "scripts/_presets.json").write_text(json.dumps(out))
