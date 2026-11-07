#!/usr/bin/env python3
"""Export the trained XGBoost classifier + scaler + label encoder into one
dependency-free JSON file (public/data/model.json) that the JS engine can run.

Usage:  python scripts/export_model.py
Needs:  pip install xgboost scikit-learn joblib numpy
"""
import json, warnings, pathlib
import joblib
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
xgb = joblib.load(ROOT / "models/xgb_defect_model.joblib")
scaler = joblib.load(ROOT / "models/feature_scaler.joblib")
le = joblib.load(ROOT / "models/label_encoder.joblib")

booster = xgb.get_booster()
n_class = len(le.classes_)
raw = json.loads(booster.save_raw("json"))
gb = raw["learner"]["gradient_booster"]["model"]
lmp = raw["learner"]["learner_model_param"]
bs = lmp["base_score"]
if bs.startswith("["):
    base = [float(v) for v in json.loads(bs.replace("E", "e"))]
else:
    base = [float(bs)] * n_class

trees = []
for t, cls in zip(gb["trees"], gb["tree_info"]):
    trees.append({
        "c": int(cls),
        "l": t["left_children"], "r": t["right_children"],
        "f": t["split_indices"],
        "t": [float(x) for x in t["split_conditions"]],  # threshold OR leaf value
        "d": t["default_left"],
        "w": [round(float(x), 6) for x in t["sum_hessian"]],  # cover (for TreeSHAP)
    })

nf = int(xgb.n_features_in_)
def imp(kind):
    d = booster.get_score(importance_type=kind)
    return [float(d.get(f"f{i}", 0.0)) for i in range(nf)]

model = {
    "version": 1,
    "classes": [str(c) for c in le.classes_],
    "n_features": nf,
    "base_score": base,
    "n_estimators": int(xgb.n_estimators),
    "max_depth": int(xgb.max_depth),
    "learning_rate": float(xgb.learning_rate),
    "objective": "multi:softprob",
    "trees": trees,
    "importance": {k: imp(k) for k in ("gain", "cover", "weight")},
    "reference": {  # training-population statistics from the fitted StandardScaler
        "mean": [float(x) for x in scaler.mean_],
        "std": [float(x) for x in scaler.scale_],
    },
}
out = ROOT / "public/data/model.json"
out.write_text(json.dumps(model, separators=(",", ":")))
print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB, {len(trees)} trees)")
