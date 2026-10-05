# MBA Odds Explorer

The MBA Odds Explorer is an explainable estimator of **admission chances** and **scholarships given admission**. It covers the 2026 U.S. News top 25 full-time MBA programs (26 schools, because of a tie at #25), plus Rice Jones and Stevens Institute of Technology (U.S. News #72, tie; added as a pooled, extrapolated estimate with no school-specific tracker data; its scholarship estimate is anchored to Stevens's award policy and a few reported awards under stated assumptions, see `methodology.html#stevens`).

- **App:** `index.html` (GitHub Pages: `https://bdas123.github.io/mba-admissions-predictor/`)
- **Methodology and transparency:** `methodology.html`, which documents data sources, modeling, evaluation, explainability, limitations and build credits.

## What it does
- Paste a sentence about your profile. A rule-based interpreter that runs entirely in your browser reads your GMAT score, GPA, experience, industry, round and home state, then sets the sliders.
- Move the GMAT, GPA and other controls, then build a school list from the dropdown.
- Each school shows two distributions with the **median** highlighted:
  1. Estimated chance of admission (300 bootstrap refits of the model)
  2. Scholarship if admitted, as a share of tuition and in dollars (simulated outcomes)

## How it works (short version)
- **Data:** self-reported GMAT Club Decision Tracker outcomes (structured fields only; usernames and comments are discarded, and no row-level data is published). School-cycle applications and acceptance rates come from Poets&Quants; rankings come from U.S. News; tuition and aid facts come from official school pages.
- **Admission model:** pooled L2 logistic regression with King & Zeng prior correction for self-report bias, shrunk for small samples. The app shows bootstrap uncertainty.
- **Scholarship model:** a two-part model (whether you receive an award, then award size), calibrated to official aggregate aid statistics when a school publishes them. Stanford and Harvard are shown as need-based and are not modeled.
- **Evaluation:** 5-fold cross-validation grouped by applicant, reporting AUC, Brier score, log loss and calibration. Gradient boosting with SHAP is included as a benchmark.

See `methodology.html` for the numbers and caveats. This tool gives estimates, not admissions decisions.

## Repository layout
```
index.html, methodology.html, assets/   static site (no backend; nothing leaves the browser)
data/model.json                         model parameters, evaluation, aggregates
data/schools.json                       tuition, aid facts and source links per school
data/credits.json                       measured build cost
pipeline/                               reproducible collection, parsing, training and export scripts
pipeline/add_stevens.py                 adds Stevens to the exported data (no retraining; evidence-anchored scholarship offsets)
```

## Credits
Built with [Perplexity Computer](https://www.perplexity.ai/computer). Model: Claude Opus 5.5 (Medium). Build cost: **1,284.63 Perplexity Computer credits** (about $12.85), as reported by the author; details are in `methodology.html#credits`.

Not affiliated with GMAT Club, Poets&Quants, U.S. News, GMAC or any school.
