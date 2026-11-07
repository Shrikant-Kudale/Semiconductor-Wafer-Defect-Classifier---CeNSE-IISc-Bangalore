#!/usr/bin/env python3
"""Create tests/golden.json: reference outputs from the *original* Python
XGBoost model (probabilities + exact TreeSHAP) used to prove JS parity."""
import json, warnings, pathlib
import joblib, numpy as np, xgboost as xgb_lib
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
xgb = joblib.load(ROOT / "models/xgb_defect_model.joblib")
ref = json.loads((ROOT / "public/data/model.json").read_text())["reference"]
presets = json.loads((ROOT / "scripts/_presets.json").read_text())
rng = np.random.default_rng(2026)
mu, sd = np.array(ref["mean"]), np.array(ref["std"])
X = [np.array(v["values"]) for v in presets.values()]
X += list(np.clip(mu + rng.normal(0, 1.2, (60, 15)) * sd, 0, None))
X = np.array(X, dtype=np.float32)
P = xgb.predict_proba(X)
C = xgb.get_booster().predict(xgb_lib.DMatrix(X), pred_contribs=True)  # (n, K, F+1)
json.dump({"X": X.tolist(), "P": P.tolist(), "C": C.tolist()},
          open(ROOT / "tests/golden.json", "w"))
print("golden vectors:", X.shape, "contribs", C.shape)
