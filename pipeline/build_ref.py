"""Build school x intake-year reference features from Poets&Quants tables.

P&Q's 2026 article labels cycles by entering year (2025 = class entering fall 2025,
Class of 2027). Intake 2026 is not yet published there, so it reuses 2025 and is
flagged `proxied`.
"""
import json, statistics

pq = json.load(open("pq_tables.json"))
conc = json.load(open("concordance.json"))
c2f = {}
for c, f, _ in conc:
    c2f.setdefault(c, []).append(f)
c2f = {k: statistics.median(v) for k, v in c2f.items()}


def classic_to_focus(x):
    k = min(c2f, key=lambda z: abs(z - x))
    return float(c2f[k])


# GMAT Club tracker name -> P&Q name
MAP = {
    "Stanford GSB": "Stanford GSB", "Sloan MIT": "MIT (Sloan)", "Booth": "Chicago (Booth)",
    "Jones Rice": "Rice (Jones)", "McCombs": "Texas-Austin (McCombs)", "Harvard": "Harvard Business School",
    "Wharton": "Pennsylvania (Wharton)", "Kellogg": "Northwestern (Kellogg)", "Columbia": "Columbia Business School",
    "Haas": "UC-Berkeley (Haas)", "Tuck": "Dartmouth (Tuck)", "Yale": "Yale SOM", "Ross (Michigan)": "Michigan (Ross)",
    "Fuqua": "Duke (Fuqua)", "Darden": "Virginia (Darden)", "Stern": "New York (Stern)",
    "Johnson (Cornell)": "Cornell (Johnson)", "Anderson": "UCLA (Anderson)", "Tepper": "Carnegie Mellon (Tepper)",
    "Kenan-Flagler": "North Carolina (Kenan-Flagler)", "Marshall (USC)": "Southern California (Marshall)",
    "McDonough": "Georgetown (McDonough)", "Goizueta": "Emory (Goizueta)", "Owen Vanderbilt": "Vanderbilt (Owen)",
    "Olin St. Louis": "Washington (Olin)", "Kelley": "Indiana (Kelley)", "Foster": "Washington (Foster)",
    "Scheller": "Georgia Institute of Technology (Scheller)", "Mendoza": "Notre Dame (Mendoza)",
    "Simon": "Rochester (Simon)", "Terry Georgia": "Georgia (Terry)", "Carlson": "Minnesota (Carlson)",
    "Cox (SMU)": "Southern Methodist (Cox)", "W.P. Carey": "Arizona State (Carey)", "Fisher": "Ohio State (Fisher)",
    "BU Questrom": "Boston University (Questrom)", "Broad": "Michigan State (Broad)",
    "Jindal": "Texas-Dallas (Jindal)",
}


def get(school, metric):
    for x in pq:
        if x["school"].replace(" **", "") == school and x["metric"] == metric and x["values"]:
            return x["values"]
    return {}


ref = {}
for gc, pqn in MAP.items():
    apps = get(pqn, "applications")
    acc = get(pqn, "acceptance_rate")
    gm = {}
    for x in pq:
        if x["school"].replace(" **", "") == pqn and x["metric"] == "gmat_classic_avg" and "2025" in x["values"]:
            gm = x["values"]
    # fallbacks when no year-by-year Classic average exists: a single Classic average, else the Focus average
    classic_any = next((v for x in pq if x["school"].replace(" **", "") == pqn and x["metric"] == "gmat_classic_avg"
                        for v in x["values"].values()), None)
    focus_any = next((v for x in pq if x["school"].replace(" **", "") == pqn and x["metric"] == "gmat_focus_avg"
                      for v in x["values"].values()), None)
    rows = {}
    for y in (2023, 2024, 2025, 2026):
        yy = min(y, 2025)
        a = apps.get(f"Applications {yy}")
        r = acc.get(f"Acceptance Rate {yy}")
        g = gm.get(str(yy))
        if a is None or r is None:
            continue
        rows[str(y)] = {
            "applications": a, "acceptance_rate": r,
            "gmat_ref_focus": classic_to_focus(g) if g else (classic_to_focus(classic_any) if classic_any else (float(focus_any) if focus_any else None)),
            "proxied": y == 2026,
        }
    if rows:
        ref[gc] = {"pq_name": pqn, "cycles": rows}

json.dump(ref, open("school_ref.json", "w"), indent=1)
print(len(ref), "schools with cycle features")
for k in ["Stanford GSB", "Sloan MIT", "Booth", "Jones Rice", "McCombs"]:
    print(k, ref.get(k))
missing = [k for k in MAP if k not in ref]
print("missing:", missing)
