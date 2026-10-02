"""Parse GMAT Club decision-tracker profile segments into applicant-school rows.

Usernames and comments are discarded. Only structured profile fields, school
outcomes and a few keyword flags derived from the applicant's own notes are kept.
"""
import json, re, glob, statistics
from datetime import datetime

FOCUS5 = {
    "Stanford GSB": "Stanford GSB",
    "Sloan MIT": "MIT Sloan",
    "Booth": "Chicago Booth",
    "Jones Rice": "Rice Jones",
    "McCombs": "UT Austin McCombs",
}
TARGET = {k: v["pq_name"] for k, v in json.load(open("school_ref.json")).items()}
TARGET.update(FOCUS5)

conc = json.load(open("concordance.json"))
classic_to_focus = {}
for c, f, _ in conc:
    classic_to_focus.setdefault(c, []).append(f)
classic_to_focus = {k: statistics.median(v) for k, v in classic_to_focus.items()}


def to_focus(score, kind):
    if kind == "GMAT Focus":
        return float(score)
    if kind == "GMAT Classic":
        keys = sorted(classic_to_focus)
        s = min(keys, key=lambda k: abs(k - score))
        return float(classic_to_focus[s])
    return None


EC_KEYS = {
    "ec_volunteer": r"volunteer|non-?profit|ngo|community service|mentor|tutor",
    "ec_leadership": r"president|founder|co-?founder|captain|led a team|team lead|manag(ed|ing) a team|board member|chair",
    "ec_military": r"military|army|navy|air force|marine|veteran",
    "ec_entrepreneur": r"startup|start-up|founded|entrepreneur",
    "ec_athletics": r"athlet|varsity|rugby|marathon|national level|state level",
}

STATUS = ["Accepted", "Admitted from WL", "Admitted", "Matriculating", "Denied without Interview",
          "Denied with Interview", "Denied", "Waitlisted with Interview", "Waitlisted without Interview",
          "Waitlisted", "Interviewed", "Invited to interview", "Applied", "Withdrawn Application",
          "Deferred to Next Year"]
POS = {"Accepted", "Admitted", "Admitted from WL", "Matriculating"}
NEG = {"Denied", "Denied without Interview", "Denied with Interview"}
DATE_RE = re.compile(r"^([A-Z][a-z]{2}) (\d{1,2}), (\d{2})$")


def parse_seg(seg):
    lines = [l.strip() for l in seg.split("\n") if l.strip()]
    if "Share" not in lines:
        return None
    i = lines.index("Share") + 1
    prof = {}
    # header until first school block
    hdr = lines[i:i + 40]
    for j, l in enumerate(hdr):
        m = re.match(r"^(\d+) years?$", l)
        if m and "yoe" not in prof:
            prof["yoe"] = int(m.group(1))
        if l == "Score:" and j + 1 < len(hdr):
            m = re.match(r"(\d{3}) (GMAT Focus|GMAT Classic|GRE|EA)", hdr[j + 1])
            if m:
                prof["score_raw"], prof["score_kind"] = int(m.group(1)), m.group(2)
        if l == "GPA:" and j + 1 < len(hdr):
            m = re.match(r"^(\d(?:\.\d+)?)(?:/(\d+(?:\.\d+)?))?$", hdr[j + 1])
            if m:
                g, scale = float(m.group(1)), float(m.group(2) or 4)
                if scale in (4, 4.0) and g <= 4.0:
                    prof["gpa"] = g
        if l in ("Male", "Female"):
            prof["gender"] = l
        if l == "Pre-MBA industry:" and j + 1 < len(hdr):
            prof["pre_industry"] = hdr[j + 1]
        if l == "Post-MBA industry:" and j + 1 < len(hdr):
            prof["post_industry"] = hdr[j + 1]
    # country = the line just before the 4-digit year line (works for 'Private' headers and missing YOE)
    for j in range(1, min(len(hdr), 8)):
        if re.match(r"^20\d\d$", hdr[j]) and not re.match(r"^\d+ years?$", hdr[j - 1]) and hdr[j - 1] not in ("Private", "Share"):
            prof["country"] = hdr[j - 1]
            break
    if "country" not in prof and len(hdr) > 2 and "yoe" in prof:
        k = [n for n, l in enumerate(hdr) if re.match(r"^\d+ years?$", l)][0]
        if k + 1 < len(hdr):
            prof["country"] = hdr[k + 1]
    # school blocks: a school line followed by a repeated program line
    blocks, starts = [], []
    for n in range(len(lines) - 2):
        if lines[n + 1] == lines[n + 2] and lines[n + 1] in (
                "Full Time MBA", "January Intake", "One Year MBA", "Deferred Enrollment Program",
                "Part Time MBA", "Executive MBA", "Online MBA", "Masters") or (
                lines[n + 1] == lines[n + 2] and "MBA" in lines[n + 1] and n > i):
            starts.append(n)
    starts.append(len(lines))
    for a, b in zip(starts, starts[1:]):
        blk = lines[a:b]
        school, program = blk[0], blk[1]
        rnd, events, sch = None, [], None
        for q, l in enumerate(blk):
            if re.match(r"^(Round \d|Early|Rolling|January Round|\w+ deadline|\d\w\w deadline)", l):
                rnd = l
            if l in STATUS and q + 1 < len(blk):
                dm = DATE_RE.match(blk[q + 1])
                d = datetime.strptime(blk[q + 1], "%b %d, %y") if dm else None
                events.append((l, d))
            if l == "Scholarship" and q + 1 < len(blk):
                sch = blk[q + 1]
        blocks.append({"school": school, "program": program, "round": rnd, "events": events, "scholarship": sch})
    # free-text notes = trailing lines of the last block after timestamps (heuristic)
    text = " ".join(lines).lower()
    flags = {k: int(bool(re.search(p, text))) for k, p in EC_KEYS.items()}
    return prof, blocks, flags


def round_num(r):
    if not r:
        return None
    m = re.match(r"Round (\d)", r)
    if m:
        return int(m.group(1))
    if r.lower().startswith("early"):
        return 0
    return None


def main():
    rows = []
    seen = set()
    for f in sorted(glob.glob("raw*.jsonl")):
        for line in open(f):
            rec = json.loads(line)
            if rec["id"] in seen or not rec["seg"]:
                continue
            seen.add(rec["id"])
            p = parse_seg(rec["seg"])
            if not p:
                continue
            prof, blocks, flags = p
            focus = to_focus(prof.get("score_raw"), prof.get("score_kind")) if prof.get("score_raw") else None
            for b in blocks:
                if b["school"] not in TARGET or b["program"] != "Full Time MBA":
                    continue
                st = [e for e in b["events"]]
                final = None
                for s, d in st:
                    if s in POS:
                        final = (1, d)
                    elif s in NEG and final is None:
                        final = (0, d)
                if final is None:
                    continue
                d = final[1] or next((dd for _, dd in st if dd), None)
                intake = (d.year + (1 if d.month >= 9 else 0)) if d else None
                sch_pct = None
                if final[0] == 1 and b["scholarship"]:
                    m = re.match(r"^(\d+)%$", b["scholarship"])
                    if m:
                        sch_pct = float(m.group(1))
                    elif b["scholarship"] == "None":
                        sch_pct = 0.0
                rows.append({
                    "rid": rec["id"], "school": TARGET[b["school"]], "gc_school": b["school"], "focus5": int(b["school"] in FOCUS5), "admit": final[0],
                    "intake": intake, "round": round_num(b["round"]),
                    "interviewed": int(any(s in ("Interviewed", "Invited to interview", "Denied with Interview",
                                                 "Waitlisted with Interview") for s, _ in st)),
                    "gmat_focus": focus, "score_kind": prof.get("score_kind"), "gpa": prof.get("gpa"),
                    "yoe": prof.get("yoe"), "country": prof.get("country"), "gender": prof.get("gender"),
                    "pre_industry": prof.get("pre_industry"), "post_industry": prof.get("post_industry"),
                    "scholarship_pct": sch_pct, **flags,
                })
    json.dump(rows, open("rows.json", "w"), indent=0)
    print(len(seen), "profiles;", len(rows), "target-school decisions")
    import collections
    print(collections.Counter((r["school"], r["admit"]) for r in rows if r["focus5"]))
    print("with GMAT", sum(r["gmat_focus"] is not None for r in rows), "with GPA", sum(r["gpa"] is not None for r in rows))
    print("scholarship known", sum(r["scholarship_pct"] is not None for r in rows))


if __name__ == "__main__":
    main()
