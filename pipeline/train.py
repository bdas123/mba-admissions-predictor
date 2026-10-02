"""Train interpretable MBA admission + scholarship models.

Admission:   L2 logistic regression, pooled across ~36 US programs, with school-cycle
             context features (official acceptance rate, application volume, class GMAT).
             Self-report bias is corrected per school with King & Zeng (2001) prior correction.
             Uncertainty: 300 applicant-level bootstrap refits -> distribution of probability.
Comparison:  Histogram gradient boosting + SHAP, same features, same CV folds.
Scholarship: two-part model on admitted applicants at merit-aid schools:
             P(any award) logistic + award share (% of tuition | award>0) linear on logit share.
"""
import json, math, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)

rows = pd.DataFrame(json.load(open("rows.json")))
ref = json.load(open("school_ref.json"))
FOCUS5 = {"Stanford GSB": "Stanford GSB", "Sloan MIT": "MIT Sloan", "Booth": "Chicago Booth",
          "Jones Rice": "Rice Jones", "McCombs": "UT Austin McCombs"}
NEED_BASED = {"Stanford GSB", "Harvard"}
# Schools offered in the app: 2026 U.S. News top 25 (26 with ties) plus Rice Jones (#29, requested).
APP_SCHOOLS = ["Stanford GSB", "Wharton", "Booth", "Kellogg", "Harvard", "Sloan MIT", "Columbia", "Stern", "Tuck", "Haas",
               "Yale", "Darden", "Ross (Michigan)", "Fuqua", "Johnson (Cornell)", "Tepper", "Owen Vanderbilt", "McCombs",
               "Anderson", "Foster", "Kelley", "Kenan-Flagler", "Goizueta", "Marshall (USC)", "Terry Georgia", "Jones Rice"]  # fellowships awarded on financial need only

# ---------- cleaning ----------
df = rows[rows.gc_school.isin(ref)].copy()
df = df[df.intake.isin([2023, 2024, 2025, 2026])]
df = df[df.gmat_focus.notna()]                      # GMAT Focus or Classic (converted); GRE/waiver excluded
df = df[(df.gmat_focus >= 455) & (df.gmat_focus <= 805)]
df["gpa_missing"] = df.gpa.isna().astype(int)
df["gpa"] = df.gpa.fillna(df.gpa.median()).clip(2.0, 4.0)
df["yoe_missing"] = df.yoe.isna().astype(int)
df["yoe"] = df.yoe.fillna(df.yoe.median()).clip(0, 15)
df["intl"] = (df.country.fillna("United States") != "United States").astype(int)
df["round"] = df["round"].fillna(2).clip(0, 4)


def ind_group(s):
    s = s.lower() if isinstance(s, str) else ""
    if "consult" in s: return "consulting"
    if "financ" in s or "bank" in s or "equity" in s or "venture" in s or "invest" in s: return "finance"
    if "tech" in s or "software" in s or "internet" in s or "e-commerce" in s: return "tech"
    if "energy" in s or "oil" in s or "utilit" in s: return "energy"
    if "military" in s or "government" in s or "non-profit" in s: return "public_service"
    return "other"


df["ind"] = df.pre_industry.map(ind_group)
for g in ["consulting", "finance", "tech", "energy", "public_service"]:
    df[f"ind_{g}"] = (df.ind == g).astype(int)


def ctx(r):
    c = ref[r.gc_school]["cycles"][str(int(r.intake))]
    return pd.Series({"acc_rate": c["acceptance_rate"] / 100, "apps": c["applications"],
                      "gmat_ref": c["gmat_ref_focus"] or 670.0})


df = df[df.gc_school.map(lambda g: ref[g]["cycles"].get("2025", {}).get("gmat_ref_focus") is not None)]
df = pd.concat([df, df.apply(ctx, axis=1)], axis=1)
df["logit_acc"] = np.log(df.acc_rate / (1 - df.acc_rate))
df["log_apps"] = np.log(df.apps)
df["gmat_gap"] = (df.gmat_focus - df.gmat_ref) / 10.0          # per 10 Focus points
df["gpa_c"] = (df.gpa - 3.5) * 10                              # per 0.1 GPA
df["yoe_c"] = df.yoe - 5
df["yoe_sq"] = df.yoe_c ** 2
df["r1"] = (df["round"] <= 1).astype(int)
df["r3plus"] = (df["round"] >= 3).astype(int)

FEATS = ["gmat_gap", "gpa_c", "gpa_missing", "yoe_c", "yoe_sq", "yoe_missing", "r1", "r3plus", "intl",
         "ind_consulting", "ind_finance", "ind_tech", "ind_energy", "ind_public_service",
         "logit_acc", "log_apps"]
EC_FEATS = ["ec_volunteer", "ec_leadership", "ec_military", "ec_entrepreneur", "ec_athletics"]
LABELS = {
    "gmat_gap": "GMAT Focus vs class reference (+10 pts)", "gpa_c": "Undergrad GPA (+0.1)",
    "gpa_missing": "GPA not reported", "yoe_c": "Work experience (+1 yr from 5)",
    "yoe_sq": "Work experience squared (distance from ~5 yrs)", "yoe_missing": "Experience not reported",
    "r1": "Applied Round 1 / early", "r3plus": "Applied Round 3+", "intl": "Non-US applicant",
    "ind_consulting": "Pre-MBA: consulting", "ind_finance": "Pre-MBA: finance",
    "ind_tech": "Pre-MBA: tech", "ind_energy": "Pre-MBA: energy", "ind_public_service": "Pre-MBA: military/government/non-profit",
    "ec_volunteer": "Notes mention volunteering/mentoring", "ec_leadership": "Notes mention leadership role",
    "ec_military": "Notes mention military", "ec_entrepreneur": "Notes mention startup/founding",
    "ec_athletics": "Notes mention athletics", "logit_acc": "School acceptance rate that cycle (logit)",
    "log_apps": "Applications to school that cycle (log)",
}

X = df[FEATS].values.astype(float)
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd
y = df.admit.values
groups = df.rid.values
C = 0.5

# ---------- cross-validated evaluation (grouped by applicant) ----------
gkf = GroupKFold(n_splits=5)
p_lr = np.zeros(len(y)); p_gb = np.zeros(len(y)); p_base = np.zeros(len(y))
for tr, te in gkf.split(Xs, y, groups):
    lr = LogisticRegression(C=C, max_iter=2000).fit(Xs[tr], y[tr]); p_lr[te] = lr.predict_proba(Xs[te])[:, 1]
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                        l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
    p_gb[te] = gb.predict_proba(X[te])[:, 1]
    # baseline: school-cycle context only
    b = LogisticRegression(max_iter=2000).fit(Xs[tr][:, [-2, -1]], y[tr]); p_base[te] = b.predict_proba(Xs[te][:, [-2, -1]])[:, 1]


def metrics(p):
    return {"auc": round(roc_auc_score(y, p), 3), "brier": round(brier_score_loss(y, p), 4),
            "log_loss": round(log_loss(y, p), 4)}


XE = np.column_stack([Xs, (df[EC_FEATS].values - df[EC_FEATS].values.mean(0)) / (df[EC_FEATS].values.std(0) + 1e-9)])
p_ec = np.zeros(len(y))
for tr, te in gkf.split(XE, y, groups):
    p_ec[te] = LogisticRegression(C=C, max_iter=2000).fit(XE[tr], y[tr]).predict_proba(XE[te])[:, 1]
p_gm = np.zeros(len(y)); keep = [i for i, f in enumerate(FEATS) if f != "gmat_gap"]
for tr, te in gkf.split(Xs, y, groups):
    p_gm[te] = LogisticRegression(C=C, max_iter=2000).fit(Xs[tr][:, keep], y[tr]).predict_proba(Xs[te][:, keep])[:, 1]
evals = {"logistic_plus_extracurricular_keywords": None, "logistic_without_gmat": None}
evals = {"context_only_baseline": metrics(p_base), "logistic_regression": metrics(p_lr),
         "gradient_boosting": metrics(p_gb),
         "logistic_plus_extracurricular_keywords": metrics(p_ec), "logistic_without_gmat": metrics(p_gm)}
frac, mean_pred = calibration_curve(y, p_lr, n_bins=8, strategy="quantile")
evals["lr_calibration"] = [{"mean_pred": round(a, 3), "observed": round(b, 3)} for a, b in zip(mean_pred, frac)]
m5 = df.gc_school.isin(FOCUS5).values
if m5.sum() > 20 and len(set(y[m5])) == 2:
    evals["logistic_regression_focus5_only"] = {"n": int(m5.sum()), "auc": round(roc_auc_score(y[m5], p_lr[m5]), 3)}

# ---------- final fit + bootstrap ----------
final = LogisticRegression(C=C, max_iter=3000).fit(Xs, y)
uniq = np.unique(groups)
boots = []
for b in range(300):
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([np.where(groups == g)[0] for g in pick])
    if len(set(y[idx])) < 2: continue
    m = LogisticRegression(C=C, max_iter=3000).fit(Xs[idx], y[idx])
    boots.append([float(m.intercept_[0])] + [float(v) for v in m.coef_[0]])
boots = np.array(boots)

# ---------- prior correction per school (King & Zeng 2001) ----------
# offset_s = logit(true rate) - logit(sample rate); sample rate shrunk toward pooled with k=10 pseudo-obs
pooled = y.mean()
corr = {}
for s_, g in df.groupby("gc_school"):
    n, a = len(g), int(g.admit.sum())
    samp = (a + 2 * pooled) / (n + 2)           # light smoothing so 24/24 is not treated as 100%
    true = float(np.mean(g.acc_rate))
    corr[s_] = {"n": int(n), "admits": a, "sample_rate": round(float(samp), 3), "official_rate": round(true, 3),
                "raw_offset": math.log(true / (1 - true)) - math.log(samp / (1 - samp))}
ntot = sum(v["n"] for v in corr.values())
global_off = sum(v["raw_offset"] * v["n"] for v in corr.values()) / ntot
K_SHRINK = 50
for v in corr.values():
    w = v["n"] / (v["n"] + K_SHRINK)
    v["shrink_weight"] = round(w, 3)
    v["offset"] = round(w * v["raw_offset"] + (1 - w) * global_off, 4)
    v["raw_offset"] = round(v["raw_offset"], 4)
corr["_global_offset"] = round(global_off, 4)
for s_ in APP_SCHOOLS:   # schools with no tracker decisions fall back to the pooled correction
    if s_ not in corr:
        corr[s_] = {"n": 0, "admits": 0, "sample_rate": None, "official_rate": None, "raw_offset": None,
                    "shrink_weight": 0.0, "offset": round(global_off, 4)}

# global SHAP-like explanation for LR: coefficient (standardized) and odds ratios in raw units
coef_tbl = []
for i, f in enumerate(FEATS):
    raw = final.coef_[0][i] / sd[i]
    lo, hi = np.percentile(boots[:, i + 1] / sd[i], [5, 95])
    coef_tbl.append({"feature": f, "label": LABELS[f], "std_coef": round(float(final.coef_[0][i]), 4),
                     "odds_ratio_per_unit": round(math.exp(raw), 3),
                     "or_90ci": [round(math.exp(lo), 3), round(math.exp(hi), 3)],
                     "prevalence_or_mean": round(float(mu[i]), 3)})
coef_tbl.sort(key=lambda r: -abs(r["std_coef"]))

# SHAP for gradient boosting comparison
try:
    import shap
    gbf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                         random_state=0).fit(X, y)
    sv = shap.TreeExplainer(gbf).shap_values(X)
    shap_imp = sorted([{"feature": f, "label": LABELS[f], "mean_abs_shap": round(float(np.abs(sv[:, i]).mean()), 4)}
                       for i, f in enumerate(FEATS)], key=lambda r: -r["mean_abs_shap"])
except Exception as e:
    shap_imp = [{"error": str(e)}]

# ---------- scholarship model ----------
sch = df[(df.admit == 1) & df.scholarship_pct.notna() & ~df.gc_school.isin(NEED_BASED)].copy()
SF = ["gmat_gap", "gpa_c", "gpa_missing", "yoe_c", "intl", "ind_consulting", "ind_finance", "ind_tech",
      "ind_energy", "ind_public_service", "r1", "r3plus", "logit_acc", "log_apps"]
idxF = [FEATS.index(f) for f in SF]
Xsch = (sch[SF].values - mu[idxF]) / sd[idxF]
any_y = (sch.scholarship_pct > 0).astype(int).values
pos = sch.scholarship_pct.values > 0
share = np.clip(sch.scholarship_pct.values[pos] / 100, 0.05, 0.95)
z = np.log(share / (1 - share))
sch_boot = []
sg = sch.rid.values; su = np.unique(sg)
for b in range(300):
    pick = rng.choice(su, size=len(su), replace=True)
    idx = np.concatenate([np.where(sg == g)[0] for g in pick])
    if len(set(any_y[idx])) < 2: continue
    m1 = LogisticRegression(C=C, max_iter=3000).fit(Xsch[idx], any_y[idx])
    pi = idx[pos[idx]]
    zi = np.log(np.clip(sch.scholarship_pct.values[pi] / 100, .05, .95) / (1 - np.clip(sch.scholarship_pct.values[pi] / 100, .05, .95)))
    m2 = Ridge(alpha=5.0).fit(Xsch[pi], zi)
    resid = float(np.std(zi - m2.predict(Xsch[pi])))
    sch_boot.append({"a0": float(m1.intercept_[0]), "a": [float(v) for v in m1.coef_[0]],
                     "b0": float(m2.intercept_), "b": [float(v) for v in m2.coef_], "sigma": resid})
# CV for the "any award" part and for share
pa = np.zeros(len(any_y))
for tr, te in GroupKFold(n_splits=5).split(Xsch, any_y, sg):
    pa[te] = LogisticRegression(C=C, max_iter=3000).fit(Xsch[tr], any_y[tr]).predict_proba(Xsch[te])[:, 1]
zcv = np.zeros(pos.sum()); Xp = Xsch[pos]; gp = sg[pos]
for tr, te in GroupKFold(n_splits=5).split(Xp, z, gp):
    zcv[te] = Ridge(alpha=5.0).fit(Xp[tr], z[tr]).predict(Xp[te])
share_cv = 100 / (1 + np.exp(-zcv))
sch_eval = {"n_admits_with_scholarship_info": int(len(sch)), "share_receiving_any": round(float(any_y.mean()), 3),
            "any_award_auc": round(roc_auc_score(any_y, pa), 3) if len(set(any_y)) == 2 else None,
            "share_mae_pct_points": round(float(np.mean(np.abs(share_cv - share * 100))), 1),
            "share_mae_naive_median": round(float(np.mean(np.abs(np.median(share * 100) - share * 100))), 1),
            "observed_share_distribution": {str(k): int(v) for k, v in sch.scholarship_pct.value_counts().sort_index().items()}}
# school-specific calibration of the scholarship model.
# Self-reported awards skew high (big awards get posted), so where a school publishes official
# aggregates we solve for offsets that reproduce them on that school's admitted sample:
#   any-award offset  -> mean P(any award) = official share of admits/students receiving aid
#   share offset      -> mean award share among recipients = official average award / annual tuition
# Where nothing is published (Booth, MIT Sloan) the targets are the tracker sample's own means
# and the app flags that those estimates may run high.
from scipy.optimize import brentq
OFFICIAL = {"Jones Rice": {"any": 0.96, "share": 40000 / 79116, "src": "Rice: 96% receive; average award > $40k/yr; tuition $79,116"},
            "McCombs": {"any": 0.88, "share": 19080 / 61214, "src": "McCombs Class of 2027: 88% offered; average $19,080/yr; non-resident cost $61,214"}}
mfin = LogisticRegression(C=C, max_iter=3000).fit(Xsch, any_y)
mshare = Ridge(alpha=5.0).fit(Xsch[pos], z)
sig_full = float(np.std(z - mshare.predict(Xsch[pos])))
eps = np.random.default_rng(3).standard_normal(400)
sch_off = {}
for s_ in APP_SCHOOLS:
    if s_ in NEED_BASED: continue
    ii = np.where(sch.gc_school.values == s_)[0]
    if len(ii) < 8:   # too few admits with scholarship info: use the pooled model unadjusted
        sch_off[s_] = {"n": int(len(ii)), "n_pos": int(pos[ii].sum()) if len(ii) else 0, "any_offset": 0.0, "share_offset": 0.0,
                       "target_any": None, "target_mean_share": None, "basis": "pooled_only", "observed_median_share": None}
        continue
    za = mfin.decision_function(Xsch[ii]); zs = mshare.predict(Xsch[ii])
    if s_ in OFFICIAL:
        t_any, t_share, basis = OFFICIAL[s_]["any"], OFFICIAL[s_]["share"], "official"
    else:
        t_any = (any_y[ii].sum() + 1) / (len(ii) + 2)
        pi = ii[pos[ii]]
        t_share = float(np.mean(np.clip(sch.scholarship_pct.values[pi] / 100, .05, .95))) if len(pi) else 0.3
        basis = "tracker_sample"
    oa = brentq(lambda o: np.mean(1 / (1 + np.exp(-(za + o)))) - t_any, -10, 10)
    os_ = brentq(lambda o: np.mean(1 / (1 + np.exp(-(zs[:, None] + o + sig_full * eps[None, :])))) - t_share, -10, 10)
    sch_off[s_] = {"n": int(len(ii)), "n_pos": int(pos[ii].sum()), "any_offset": round(float(oa), 4),
                   "share_offset": round(float(os_), 4), "target_any": round(float(t_any), 3),
                   "target_mean_share": round(float(t_share), 3), "basis": basis,
                   "observed_median_share": float(np.median(sch.scholarship_pct.values[ii]))}

# ---------- descriptive distributions for the five target schools ----------
desc = {}
for s in APP_SCHOOLS:
    g = df[df.gc_school == s]
    a = g[g.admit == 1]
    def q(v):
        v = v.dropna()
        return {k: (round(float(np.percentile(v, p)), 2) if len(v) else None) for k, p in
                [("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90)]} | {"n": int(len(v))}
    desc[s] = {"admitted_gmat_focus": q(a.gmat_focus), "admitted_gpa": q(a.gpa[a.gpa_missing == 0]),
               "denied_gmat_focus": q(g[g.admit == 0].gmat_focus),
               "admitted_scholarship_pct": q(a.scholarship_pct)}

out = {
    "features": FEATS, "labels": LABELS, "mu": mu.tolist(), "sd": sd.tolist(),
    "admit_final": {"b0": float(final.intercept_[0]), "b": final.coef_[0].tolist()},
    "admit_boot": boots.tolist(), "prior_correction": corr,
    "sch_features": SF, "sch_boot": sch_boot, "sch_school_offsets": sch_off,
    "eval": evals, "coef_table": coef_tbl, "gb_shap_importance": shap_imp, "sch_eval": sch_eval,
    "n_train": int(len(df)), "n_applicants": int(len(uniq)), "n_schools": int(df.gc_school.nunique()),
    "intake_counts": {str(k): int(v) for k, v in df.intake.value_counts().sort_index().items()},
    "descriptives_focus5": desc,
    "train_admit_rate": round(float(y.mean()), 3),
}
json.dump(out, open("model.json", "w"))
print(json.dumps({k: out[k] for k in ["n_train", "n_applicants", "n_schools", "intake_counts", "eval", "sch_eval", "train_admit_rate"]}, indent=1))
print(pd.DataFrame(coef_tbl)[["label", "std_coef", "odds_ratio_per_unit", "or_90ci"]].to_string())
print(pd.DataFrame(shap_imp).head(10).to_string())
print({s: corr[s]["offset"] for s in APP_SCHOOLS}, corr["_global_offset"])
print({s: (sch_off.get(s) or {}).get("basis") for s in APP_SCHOOLS})
out["app_schools"] = APP_SCHOOLS
json.dump(out, open("model.json", "w"))
