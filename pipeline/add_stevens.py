"""Add Stevens Institute of Technology (U.S. News 2026 #72, tie) to the explorer.

Stevens has no GMAT Club tracker decisions in the training set, so it is added the
same way as Georgia Terry: the pooled admission model with the pooled self-report
correction, and no scholarship model. Nothing is retrained; the shipped coefficients
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
M["sch_school_offsets"][KEY] = {"n": 0, "n_pos": 0, "any_offset": 0.0, "share_offset": 0.0, "target_any": None,
                                "target_mean_share": None, "basis": "pooled_only", "observed_median_share": None}
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
    "aid_basis": "policy_only",
    "aid_fact": ("Stevens offers merit awards to first-year master’s students who apply as full-time, on-campus students. "
                 "Any award is disclosed at admission and paid over three semesters, and the Office of Graduate Education adds "
                 "$4,000 in the third semester. Stevens publishes no MBA award rates or averages and there are no tracker "
                 "reports, so the scholarship isn’t modeled."),
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
