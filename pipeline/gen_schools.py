"""Build data/schools.json for every school in the app dropdown (official tuition + aid facts + sources)."""
import json
ref = json.load(open("school_ref.json"))
PQ = "https://poetsandquants.com/2026/08/02/acceptance-rates-yield-apps-at-the-top-100-u-s-mba-programs-for-2026/"
GEN = "Merit scholarships are offered, but the school publishes no award-rate or average-award statistics. Estimates rely on self-reported tracker data, which likely skew high."
# key: (display name, US News 2026 rank, state, tuition, resident tuition, tuition label, tuition src, aid basis, aid fact, aid src)
T = {
 "Stanford GSB": ("Stanford GSB", "1", "CA", 89187, None, "2026–27 tuition", "https://www.gsb.stanford.edu/programs/mba/tuition-financial-aid/types-aid", "need",
                  "Fellowships are need-based only; about half of students receive one, averaging about $47k per year.", "https://www.gsb.stanford.edu/programs/mba/tuition-financial-aid/types-aid"),
 "Wharton": ("Wharton (Penn)", "2", "PA", 87970, None, "2026–27 tuition", "https://mba-inside.wharton.upenn.edu/financial-aid/tuition-fees/", "merit", GEN, "https://mba-inside.wharton.upenn.edu/financial-aid/tuition-fees/"),
 "Booth": ("Chicago Booth", "3", "IL", 89976, None, "2026–27 tuition", "https://www.chicagobooth.edu/mba/full-time/admissions/cost", "merit",
           "Merit scholarships are considered automatically; Booth does not publish award rates or averages. Estimates rely on tracker data.", "https://www.chicagobooth.edu/mba/full-time/admissions/scholarships-and-aid"),
 "Kellogg": ("Northwestern Kellogg", "4 (tie)", "IL", 88536, None, "2026–27 tuition", "https://www.kellogg.northwestern.edu/admissions/financial-aid/ft-fin-aid/", "merit", GEN, "https://www.kellogg.northwestern.edu/admissions/financial-aid/ft-fin-aid/"),
 "Harvard": ("Harvard Business School", "4 (tie)", "MA", 84760, None, "2026–27 tuition", "https://www.hbs.edu/mba/financial-aid/tuition-assistance/cost-of-attendance", "need",
             "HBS fellowships are awarded on financial need, not test scores.", "https://www.hbs.edu/mba/financial-aid/tuition-assistance/cost-of-attendance"),
 "Sloan MIT": ("MIT Sloan", "6", "MA", 91892, None, "2026–27 tuition", "https://mitsloan.mit.edu/mba/admissions/financing-your-education", "merit",
               "Sloan offers fellowships but publishes no award rates or averages. Estimates rely on tracker data.", "https://mitsloan.mit.edu/mba/admissions/financing-your-education"),
 "Columbia": ("Columbia Business School", "7 (tie)", "NY", 93908, None, "2026–27 tuition (via Clear Admit)", "https://www.clearadmit.com/schools/columbia/", "merit", GEN, "https://www.clearadmit.com/schools/columbia/"),
 "Stern": ("NYU Stern", "7 (tie)", "NY", 92658, None, "2026–27 tuition", "https://www.stern.nyu.edu/programs-admissions/full-time-mba/financial-aid/tuition-cost-attendance", "merit", GEN, "https://www.stern.nyu.edu/programs-admissions/full-time-mba/financial-aid/tuition-cost-attendance"),
 "Tuck": ("Dartmouth Tuck", "9", "NH", 87536, None, "2026–27 tuition", "https://tuck.dartmouth.edu/admissions/finance-your-degree/cost-attendance", "merit", GEN, "https://tuck.dartmouth.edu/admissions/finance-your-degree/cost-attendance"),
 "Haas": ("Berkeley Haas", "10", "CA", 92755, 80510, "2026–27 first-year tuition & fees (non-resident; CA resident $80,510)", "https://haas.berkeley.edu/financial-aid/full-time-mba/cost/", "merit", GEN, "https://haas.berkeley.edu/financial-aid/full-time-mba/cost/"),
 "Yale": ("Yale SOM", "11 (tie)", "CT", 90900, None, "2026–27 tuition", "https://som.yale.edu/programs/mba/affording-your-mba/cost-information", "merit", GEN, "https://som.yale.edu/programs/mba/affording-your-mba/cost-information"),
 "Darden": ("UVA Darden", "11 (tie)", "VA", 83328, 79010, "2026–27 tuition (non-resident; VA resident $79,010)", "https://www.darden.virginia.edu/mba/tuition-financial-aid/tuition-fees-billing", "merit", GEN, "https://www.darden.virginia.edu/mba/tuition-financial-aid/tuition-fees-billing"),
 "Ross (Michigan)": ("Michigan Ross", "13", "MI", 83946, 78586, "2026–27 tuition (non-resident; MI resident $78,586)", "https://rossweb.bus.umich.edu/financial-aid/program-overview/mba/", "merit", GEN, "https://rossweb.bus.umich.edu/financial-aid/program-overview/mba/"),
 "Fuqua": ("Duke Fuqua", "14", "NC", 83700, None, "2026–27 tuition", "https://www.fuqua.duke.edu/programs/daytime-mba/tuition-costs", "merit", GEN, "https://www.fuqua.duke.edu/programs/daytime-mba/tuition-costs"),
 "Johnson (Cornell)": ("Cornell Johnson", "15", "NY", 88934, None, "2026–27 tuition", "https://business.cornell.edu/admissions/graduate/mba/tuition-financing/", "merit", GEN, "https://business.cornell.edu/admissions/graduate/mba/tuition-financing/"),
 "Tepper": ("Carnegie Mellon Tepper", "16 (tie)", "PA", 84186, None, "2026–27 tuition", "https://www.cmu.edu/sfs/tuition/graduate/tepper/full-accel-mba.html", "merit", GEN, "https://www.cmu.edu/sfs/tuition/graduate/tepper/full-accel-mba.html"),
 "Owen Vanderbilt": ("Vanderbilt Owen", "16 (tie)", "TN", 76700, None, "2026–27 tuition", "https://business.vanderbilt.edu/mba/admissions/tuition-financing-scholarships/", "merit", GEN, "https://business.vanderbilt.edu/mba/admissions/tuition-financing-scholarships/"),
 "McCombs": ("UT Austin McCombs", "18 (tie)", "TX", 61214, 55196, "annual program cost (non-resident; TX resident $55,196)", "https://blogs.mccombs.utexas.edu/mba-insider/funding-your-texas-mccombs-full-time-mba-investment-impact/", "merit",
             "88% of Class of 2027 admits received scholarship offers, averaging $19,080 per year (range: $5k to full tuition).", "https://blogs.mccombs.utexas.edu/mba-insider/funding-your-texas-mccombs-full-time-mba-investment-impact/"),
 "Anderson": ("UCLA Anderson", "18 (tie)", "CA", 82734, None, "2026–27 tuition", "https://www.anderson.ucla.edu/degrees/full-time-mba/financing", "merit", GEN, "https://www.anderson.ucla.edu/degrees/full-time-mba/financing"),
 "Foster": ("UW Foster", "20", "WA", 61389, 43707, "2026–27 program cost ÷ 2 (non-resident $122,778, WA resident $87,414 total)", "https://foster.uw.edu/academics/degree-programs/full-time-mba/tuition-and-fees/", "merit", GEN, "https://foster.uw.edu/academics/degree-programs/full-time-mba/tuition-and-fees/"),
 "Kelley": ("Indiana Kelley", "21 (tie)", "IN", 55695, 29573, "2026–27 tuition (non-resident; IN resident $29,573)", "https://kelley.iu.edu/programs/full-time-mba/admissions/tuition-aid-scholarships.html", "merit", GEN, "https://kelley.iu.edu/programs/full-time-mba/admissions/tuition-aid-scholarships.html"),
 "Kenan-Flagler": ("UNC Kenan-Flagler", "21 (tie)", "NC", 74192, 55470, "2026–27 tuition (non-resident; NC resident $55,470)", "https://www.kenan-flagler.unc.edu/programs/mba/full-time-mba/admissions/tuition-financial-aid/", "merit", GEN, "https://www.kenan-flagler.unc.edu/programs/mba/full-time-mba/admissions/tuition-financial-aid/"),
 "Goizueta": ("Emory Goizueta", "23 (tie)", "GA", 79200, None, "total tuition $158,400 ÷ 2 years", "https://goizueta.emory.edu/full-time-mba/two-year-mba/admissions/tuition", "merit",
              "Scholarships range from partial to full tuition plus stipend; no award-rate statistics are published. Estimates rely on tracker data.", "https://goizueta.emory.edu/full-time-mba/two-year-mba/admissions/tuition"),
 "Jindal": ("UT Dallas Jindal", "23 (tie)", "TX", None, None, "tuition not published on the program pages", "https://mba.utdallas.edu/academics/full-time-mba/", "official_only",
            "Every admitted full-time cohort student receives a guaranteed merit scholarship that qualifies them for Texas resident tuition. In 2025, 41% received $10k–$20k, 52% received $21k–$35k and 7% received full tuition plus stipend.", "https://mba.utdallas.edu/academics/full-time-mba/"),
 "Marshall (USC)": ("USC Marshall", "25 (tie)", "CA", 86295, None, "2026–27 tuition", "https://www.marshall.usc.edu/programs/graduate-programs/mba-programs/full-time-mba/tuition-fees", "merit", GEN, "https://www.marshall.usc.edu/programs/graduate-programs/mba-programs/full-time-mba/tuition-fees"),
 "Terry Georgia": ("Georgia Terry", "25 (tie)", "GA", 34920, 13918, "2025–26 tuition (non-resident; GA resident $13,918)", "https://www.terry.uga.edu/mba/fulltime/admissions/tuition/", "merit",
                   "Scholarships range from $5,000 to $20,000, all applicants are considered, and up to 80% of the incoming class receives an assistantship. No tracker decisions are available, so the estimate is pooled only.", "https://www.terry.uga.edu/mba/fulltime/admissions/tuition/"),
 "Jones Rice": ("Rice Jones", "29", "TX", 79116, None, "annual tuition", "https://business.rice.edu/rice-mba/full-time-mba/tuition-and-financial-aid", "merit",
                "96% of students receive scholarships; the average award exceeds $40k per year.", "https://business.rice.edu/admissions-blog/guide-to-2025-full-time-MBA-scholarships"),
}
out = {}
for k, (name, rank, st, tu, res, lab, tsrc, basis, fact, asrc) in T.items():
    c = ref.get(k, {}).get("cycles", {}).get("2025")
    prof = f"Entering 2025: {c['applications']:,} applications, {c['acceptance_rate']}% admitted" if c else "No P&Q cycle data"
    out[k] = {"name": name, "usnews_rank": rank, "state": st, "tuition": tu, "tuition_resident": res, "tuition_label": lab,
              "tuition_src": tsrc, "aid_basis": basis, "aid_fact": fact, "aid_src": asrc, "profile": prof, "profile_src": PQ,
              "usnews_src": "https://www.clearadmit.com/mba-rankings/usnews/"}
json.dump(out, open("/home/user/workspace/mba-predictor-site/data/schools.json", "w"), indent=1)
print(len(out))
