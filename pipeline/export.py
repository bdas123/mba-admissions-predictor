"""Export the trained model (parameters + evaluation only, no row-level data) for the static site."""
import json, shutil
M = json.load(open("model.json")); ref = json.load(open("school_ref.json"))
r = lambda v: round(v, 5)
keys = ["features", "labels", "eval", "coef_table", "gb_shap_importance", "sch_eval", "n_train", "n_applicants", "n_schools",
        "intake_counts", "train_admit_rate", "sch_features", "app_schools"]
out = {k: M[k] for k in keys}
out["mu"] = [r(x) for x in M["mu"]]; out["sd"] = [r(x) for x in M["sd"]]
out["admit_boot"] = [[r(x) for x in b] for b in M["admit_boot"]]
out["sch_boot"] = [{"a0": r(b["a0"]), "a": [r(x) for x in b["a"]], "b0": r(b["b0"]), "b": [r(x) for x in b["b"]], "sigma": r(b["sigma"])} for b in M["sch_boot"]]
APP = M["app_schools"]
out["prior_correction"] = {k: M["prior_correction"][k] for k in APP}
out["prior_correction"]["_global_offset"] = M["prior_correction"]["_global_offset"]
out["sch_school_offsets"] = {k: M["sch_school_offsets"].get(k) for k in APP}
out["school_ref"] = {k: ref[k] for k in APP if k in ref}
out["descriptives"] = {k: M["descriptives_focus5"][k] for k in APP}
json.dump(out, open("/home/user/workspace/mba-predictor-site/data/model.json", "w"), separators=(",", ":"))
for f in ["parse.py", "build_ref.py", "train.py", "gen_schools.py", "export.py"]:
    shutil.copy(f, "/home/user/workspace/mba-predictor-site/pipeline/" + f)
print("exported", len(APP), "schools")
