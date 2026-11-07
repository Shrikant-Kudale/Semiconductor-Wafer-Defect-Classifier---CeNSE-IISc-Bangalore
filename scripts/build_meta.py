#!/usr/bin/env python3
"""Compose public/data/meta.json (UI metadata: features, classes, archetypes)."""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
model = json.loads((ROOT / "public/data/model.json").read_text())
presets = json.loads((ROOT / "scripts/_presets.json").read_text())
gain = model["importance"]["gain"]

# key, label, unit, group, min, max, step, default, (target, lsl, usl) illustrative spec, description
F = [
 ("defect_count","Defect count","dies","spatial",0,500,1,50,None,"Total number of failing dies / detected defects on the wafer."),
 ("defect_density","Defect density","","spatial",0,1,0.005,0.05,None,"Defects per unit wafer area (normalised). Not used by the trained trees — kept for API compatibility."),
 ("mean_radial_position","Mean radial position","r/R","spatial",0,1,0.005,0.5,None,"Mean normalised radius of defects (0 = centre, 1 = edge)."),
 ("std_radial_position","Radial spread (σ)","r/R","spatial",0,0.5,0.005,0.1,None,"Standard deviation of defect radius — low values indicate ring-like bands."),
 ("edge_ratio","Edge ratio","","spatial",0,1,0.005,0.2,None,"Fraction of defects inside the edge-exclusion zone."),
 ("center_ratio","Centre ratio","","spatial",0,1,0.005,0.2,None,"Fraction of defects inside the central zone."),
 ("cluster_count","Cluster count","","spatial",0,80,1,2,None,"Number of spatially connected defect clusters."),
 ("max_cluster_size","Largest cluster","dies","spatial",0,300,1,10,None,"Size of the largest connected defect cluster."),
 ("oxide_thickness_nm","Oxide thickness","nm","process",80,120,0.1,100,(100,95,105),"Gate / field oxide thickness from ellipsometry."),
 ("critical_dimension_nm","Critical dimension","nm","process",30,60,0.1,45,(45,42,48),"Printed feature width from CD-SEM."),
 ("overlay_error_nm","Overlay error","nm","process",0,10,0.05,3,(0,0,5),"Layer-to-layer registration error from the lithography scanner."),
 ("line_edge_roughness_nm","Line-edge roughness","nm","process",0,5,0.05,2,(0,0,3),"3σ edge roughness of printed lines."),
 ("etch_uniformity_pct","Etch uniformity","%","process",80,100,0.1,96,(100,92,100),"Across-wafer etch-rate uniformity."),
 ("particle_index","Particle index","","process",0,20,0.1,5,(0,0,10),"Particle contamination index from cleanroom / tool monitors."),
 ("wafer_yield_pct","Wafer yield","%","process",0,100,0.1,90,(100,85,100),"Probe-test yield of the wafer."),
]
feats = []
for i, (k, lab, unit, grp, lo, hi, st, df, spec, desc) in enumerate(F):
    d = dict(key=k, label=lab, unit=unit, group=grp, min=lo, max=hi, step=st, default=df, description=desc,
             used=gain[i] > 0, short=["Count","Density","MeanRad","StdRad","Edge","Center","Clusters","MaxClust","Oxide","CD","Overlay","Rough","Etch","PartIdx","Yield"][i])
    if spec: d["spec"] = dict(target=spec[0], lsl=spec[1], usl=spec[2])
    feats.append(d)

C = {
 "Normal":   ("#10b981","Healthy process — no systematic pattern","None (baseline)",0,"Continue routine SPC monitoring. No action required."),
 "Center":   ("#8b5cf6","CMP pad centre-wear, deposition non-uniformity","CMP / Deposition",2,"Inspect CMP pad conditioning and wear profile; check deposition showerhead / gas distribution."),
 "Donut":    ("#d946ef","Etch / CMP non-uniformity at mid-radius","Etch / CMP",2,"Review etch chamber radial uniformity, ESC temperature zones and CMP zone pressures."),
 "Edge-Loc": ("#f59e0b","Edge-bead removal issue, edge handling","Lithography / Handling",1,"Check edge-bead removal (EBR) recipe and wafer-edge handling contact points."),
 "Edge-Ring":("#ef4444","Edge-exclusion-zone process drift","Etch / Deposition",2,"Check focus-ring condition, edge gas flow and edge-exclusion process settings."),
 "Loc":      ("#06b6d4","Local contamination or particle event","Contamination",1,"Trace particle source: tool PM history, FOUP / chamber events, localized contamination."),
 "Near-Full":("#be123c","Major process excursion (gross film failure)","Whole-flow excursion",3,"STOP & HOLD lot. Escalate: gross film / process excursion — verify recipe, chemicals and tool state."),
 "Random":   ("#64748b","Background defectivity, random particles","Cleanroom / Background",1,"Review baseline particle monitors, cleanroom classification and chemical purity."),
 "Scratch":  ("#3b82f6","Wafer handling, robot-arm scratch","Handling / CMP",2,"Inspect robot blade / end-effector, cassette slots and CMP slurry agglomerates."),
}
classes = []
for c in model["classes"]:
    col, cause, step, sev, action = C[c]
    classes.append(dict(name=c, color=col, rootCause=cause, processStep=step, severity=sev, action=action))

meta = dict(features=feats, classes=classes,
            archetypes={k: v["values"] for k, v in presets.items()},
            project=dict(title="WaferFab Intelligence", author="Shrikant Kudale",
                         program="MoTA Foundation Training — Semiconductor Fabrication & Characterization",
                         institute="CeNSE, IISc Bengaluru", dates="8–19 June 2026"))
(ROOT / "public/data/meta.json").write_text(json.dumps(meta, indent=1))
print("meta.json ok", len(feats), "features,", len(classes), "classes")
