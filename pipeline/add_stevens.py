"""Add Stevens Institute of Technology (U.S. News 2026 #72, tie) to the explorer.

Stevens has no GMAT Club tracker decisions in the training set, so it is added the
same way as Georgia Terry: the pooled admission model with the pooled self-report
correction. The scholarship uses the pooled two-part model with evidence-anchored offsets
(see the block below). Nothing is retrained; the shipped coefficients
are unchanged. Run from the repository root after export.py and gen_schools.py:

    python pipeline/add_stevens.py

Sources
- Applications / admits / acceptance rate (P&Q 2026 rank #77), entering 2023-2025:
  https://poetsandquants.com/2026/08/02/acceptance-rates-yield-apps-at-the-top-100-u-s-mba-programs-for-2026/
  (page 2 acceptance rates, page 4 applications, page 5 admits)
- Class GMAT: Stevens and P&Q publish no class GMAT figure. The reference uses the
  third-party average of 594 (GMAT Classic) reported by CampusReel, converted with the
  GMAC concordance to Focus 555. BeatTheGMAT reports 570; a lower reference would raise
  estimates. This is a proxy, and the card says so.
  https://www.campusreel.org/mba-programs/guides/stevens-institute-of-technology-mba-mba-profile-acceptance-rate-gmat-scores
- Tuition: 2026-27 graduate full-time rate, $22,995 per semester:
  https://www.stevens.edu/office-of-student-accounts/tuition-and-fees
- Aid policy: https://www.stevens.edu/page-basic/graduate-financing-frequently-asked-questions
- Reported awards: https://www.reddit.com/r/stevens/comments/1imajhm/decision_for_stevens/
                   https://www.reddit.com/r/stevens/comments/1kbi34d/can_i_negotiate_a_higher_scholarship_at_stevens/
- Rank: https://www.stevens.edu/news/stevens-schools-ranked-in-top-80-nationally-in-2026-u-s-news-best-graduate
"""
import json

KEY = "Stevens"
GMAT_REF_FOCUS = 555.0  # Classic 594 -> Focus 555 (GMAC concordance, same table as the app)
CYCLES = {  # entering year: (applications, admits)
    "2023": (516, 193), "2024": (436, 147), "2025": (687, 384),
}

M = json.load(open("data/model.json"))
S = json.load(open("data/schools.json"))

cycles = {}
for y, (apps, admits) in CYCLES.items():
    cycles[y] = {"applications": apps, "acceptance_rate": round(100 * admits / apps, 1),
                 "gmat_ref_focus": GMAT_REF_FOCUS, "proxied": False, "gmat_ref_proxy": True}
cycles["2026"] = dict(cycles["2025"], proxied=True)   # 2026 entry reuses the latest published cycle
M["school_ref"][KEY] = {"pq_name": "Stevens Institute of Technology", "cycles": cycles}

g = M["prior_correction"]["_global_offset"]
M["prior_correction"][KEY] = {"n": 0, "admits": 0, "sample_rate": None, "official_rate": None,
                              "raw_offset": None, "shrink_weight": 0.0, "offset": g}
# ---------- scholarship: evidence-anchored calibration of the pooled two-part model ----------
# There are no Stevens scholarship reports, so the pooled model's school offsets are solved to hit
# target ranges taken from Stevens's published award structure and the few individual awards found:
#   - Official FAQ: a merit award for first-year, full-time, on-campus master's students, disclosed at
#     admission, paid over three semesters, plus a $4,000 Office of Graduate Education supplement.
#   - Reported master's awards: "$7,000 for the first three semesters, plus an additional $4,000"
#     (MS Computer Science, Fall 2025; $11k total, or $25k if the $7k is per semester) and a scholarship
#     that "reduces my tuition by $14,000". Against two years of tuition ($91,980) these are 12%, 27% and 15%.
#   - The Provost Master's Scholarship (a competitive tuition scholarship) is the documented upper tail.
# Assumed target ranges (uniform, one draw per bootstrap refit so the assumption spread reaches the chart):
#   P(any award | admitted)                 in [0.50, 0.90]   no published rate; described as standard
#   mean award share among recipients       in [0.10, 0.28]   brackets the 12%-27% reported awards
# Offsets are solved for a reference admit: the pooled-average applicant (standardized applicant features = 0)
# placed in Stevens's school context. The applicant's own stats then move the estimate through the pooled
# coefficients, exactly as at every other school.
import math, random
from scipy.optimize import brentq
ANY_RANGE, SHARE_RANGE = (0.50, 0.90), (0.10, 0.28)
# Applicant features are clipped to +/-2 training SDs in the app, because Stevens's low GMAT reference would otherwise
# put strong applicants far outside the data (a Focus 705 sits 3.9 training SDs above the mean GMAT gap).
CLIP_Z = 2.0
TWO_YEAR_TUITION = 2 * 45990
feats, sch_feats = M["features"], M["sch_features"]
c26 = cycles["2026"]
ctx = {"logit_acc": math.log((c26["acceptance_rate"] / 100) / (1 - c26["acceptance_rate"] / 100)),
       "log_apps": math.log(c26["applications"])}
x_ref = [((ctx[f] - M["mu"][feats.index(f)]) / M["sd"][feats.index(f)]) if f in ctx else 0.0 for f in sch_feats]
rng = random.Random(2026)
eps = [rng.gauss(0, 1) for _ in range(400)]
sig = lambda z: 1 / (1 + math.exp(-z))
boot = []
for b in M["sch_boot"]:
    t_any, t_share = rng.uniform(*ANY_RANGE), rng.uniform(*SHARE_RANGE)
    za = b["a0"] + sum(a * x for a, x in zip(b["a"], x_ref))
    zs = b["b0"] + sum(a * x for a, x in zip(b["b"], x_ref))
    oa = math.log(t_any / (1 - t_any)) - za
    os_ = brentq(lambda o: sum(sig(zs + o + b["sigma"] * e) for e in eps) / len(eps) - t_share, -15, 15)
    boot.append([round(oa, 4), round(os_, 4)])
med = lambda v: sorted(v)[len(v) // 2]
M["sch_school_offsets"][KEY] = {
    "n": 0, "n_pos": 0, "any_offset": med([o[0] for o in boot]), "share_offset": med([o[1] for o in boot]),
    "target_any": list(ANY_RANGE), "target_mean_share": list(SHARE_RANGE), "basis": "evidence_anchored",
    "observed_median_share": None, "boot_offsets": boot, "clip_z": CLIP_Z,
    "evidence": {"reported_awards_usd": [11000, 14000, 25000], "two_year_tuition_usd": TWO_YEAR_TUITION},
}
empty = {"p10": None, "p25": None, "median": None, "p75": None, "p90": None, "n": 0}
M["descriptives"][KEY] = {k: dict(empty) for k in ["admitted_gmat_focus", "admitted_gpa", "denied_gmat_focus", "admitted_scholarship_pct"]}
if KEY not in M["app_schools"]:
    M["app_schools"].append(KEY)

S[KEY] = {
    "name": "Stevens Institute of Technology",
    "usnews_rank": "72 (tie)",
    "state": "NJ",
    "tuition": 45990,
    "tuition_resident": None,
    "tuition_label": "2026–27 graduate full-time rate ($22,995 per semester; Stevens lists no MBA-specific rate)",
    "tuition_src": "https://www.stevens.edu/office-of-student-accounts/tuition-and-fees",
    "aid_basis": "merit",
    "aid_fact": ("Stevens gives a merit award to first-year, full-time, on-campus master’s students, disclosed at admission and paid "
                 "over three semesters, plus a $4,000 Graduate Education supplement. It publishes no MBA award rates or averages, "
                 "so this estimate is anchored to that structure and a few reported master’s awards ($11k–$25k total)."),
    "aid_src": "https://www.stevens.edu/page-basic/graduate-financing-frequently-asked-questions",
    "profile": "Entering 2025: 687 applications, 55.9% admitted",
    "profile_src": "https://poetsandquants.com/2026/08/02/acceptance-rates-yield-apps-at-the-top-100-u-s-mba-programs-for-2026/",
    "usnews_src": "https://www.stevens.edu/news/stevens-schools-ranked-in-top-80-nationally-in-2026-u-s-news-best-graduate",
    "caveat": ("Extrapolated estimate: there are no Stevens tracker decisions, its 55.9% acceptance rate is far above every "
               "other school in the app (the highest is 37.9%), and its class GMAT reference (Focus 555) is a third-party proxy, "
               "not a published figure. Admit rates also jumped from 33.7% (2024) to 55.9% (2025). Read this as a rough range."),
    "caveat_src": "https://www.campusreel.org/mba-programs/guides/stevens-institute-of-technology-mba-mba-profile-acceptance-rate-gmat-scores",
}

json.dump(M, open("data/model.json", "w"), separators=(",", ":"))
json.dump(S, open("data/schools.json", "w"), indent=1)
print("added", KEY, cycles["2026"])
