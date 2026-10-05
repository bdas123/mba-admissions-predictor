// MBA Admissions & Scholarship Explorer — all computation runs in the browser.
const DEFAULT_LIST = ["Stanford GSB", "Sloan MIT", "Booth", "Jones Rice", "McCombs"];
let M, S, charts = {};
let selected = [];

const $ = (id) => document.getElementById(id);
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
// aid bases with no scholarship model: need-based, official range only, official policy only
const noSchModel = (info) => ["need", "range_only", "policy_only"].includes(info.aid_basis);
const pct = (v, d = 0) => (100 * v).toFixed(d) + "%";
const usd = (v) => (v < 500 ? "$0" : "$" + Math.round(v / 1000).toLocaleString() + "k");
function quantile(arr, q) {
  const a = [...arr].sort((x, y) => x - y);
  const i = (a.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
}
// deterministic RNG so the same inputs always produce the same chart
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

function inputs() {
  return {
    gmat: +$("gmat").value, gpa: +$("gpa").value, yoe: +$("yoe").value,
    intl: $("intl").checked ? 1 : 0, ind: $("ind").value, round: +$("round").value,
    appsDelta: +$("apps").value / 100, home: $("home").value,
  };
}

function featureVector(school, x) {
  const c = M.school_ref[school].cycles["2026"];   // latest published cycle (entering 2025) used as proxy
  const acc = c.acceptance_rate / 100;
  const apps = c.applications * (1 + x.appsDelta);
  const d = {
    gmat_gap: (x.gmat - c.gmat_ref_focus) / 10, gpa_c: (x.gpa - 3.5) * 10, gpa_missing: 0,
    yoe_c: x.yoe - 5, yoe_sq: (x.yoe - 5) ** 2, yoe_missing: 0,
    r1: x.round <= 1 ? 1 : 0, r3plus: x.round >= 3 ? 1 : 0, intl: x.intl,
    ind_consulting: +(x.ind === "consulting"), ind_finance: +(x.ind === "finance"), ind_tech: +(x.ind === "tech"),
    ind_energy: +(x.ind === "energy"), ind_public_service: +(x.ind === "public_service"),
    logit_acc: Math.log(acc / (1 - acc)), log_apps: Math.log(apps),
  };
  return M.features.map((f, i) => (d[f] - M.mu[i]) / M.sd[i]);
}

function admitDistribution(school, x) {
  const xs = featureVector(school, x);
  const off = M.prior_correction[school].offset;
  return M.admit_boot.map((b) => sigmoid(b[0] + xs.reduce((s, v, i) => s + b[i + 1] * v, 0) + off));
}

function scholarshipDistribution(school, x) {
  const xsFull = featureVector(school, x);
  const xs = M.sch_features.map((f) => xsFull[M.features.indexOf(f)]);
  const o = M.sch_school_offsets[school] || { any_offset: 0, share_offset: 0 };
  const r = mulberry32(12345);
  const draws = [];
  for (const b of M.sch_boot) {
    const pAny = sigmoid(b.a0 + xs.reduce((s, v, i) => s + b.a[i] * v, 0) + o.any_offset);
    const mu = b.b0 + xs.reduce((s, v, i) => s + b.b[i] * v, 0) + o.share_offset;
    for (let k = 0; k < 10; k++) {
      draws.push(r() < pAny ? Math.min(1, sigmoid(mu + b.sigma * gauss(r))) : 0);
    }
  }
  return draws;
}

function histAdmit(ps) {
  const edges = Array.from({ length: 21 }, (_, i) => i * 0.05);
  const hi = Math.min(1, Math.ceil((quantile(ps, 0.995) + 0.05) * 20) / 20);
  const lo = Math.max(0, Math.floor((quantile(ps, 0.005) - 0.05) * 20) / 20);
  const width = Math.max(0.01, (hi - lo) / 15);
  const bins = []; for (let a = lo; a < hi - 1e-9; a += width) bins.push([a, a + width]);
  return { labels: bins.map(([a]) => (100 * a).toFixed(0)), counts: bins.map(([a, b]) => ps.filter((p) => p >= a && p < b).length), bins };
}
const SBINS = [["None", 0, 0], ["1–25%", 1e-9, 0.25], ["26–50%", 0.25, 0.5], ["51–75%", 0.5, 0.75], ["76–100%", 0.75, 1.0001]];
function histSch(draws) {
  return { labels: SBINS.map((b) => b[0]), counts: SBINS.map(([, a, b]) => draws.filter((d) => (b === 0 ? d === 0 : d > a && d <= b)).length / draws.length) };
}

function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

function drawChart(id, labels, data, medianIdx, opts) {
  const ctx = $(id);
  const colors = data.map((_, i) => (i === medianIdx ? css("--accent") : css("--bar")));
  if (charts[id]) {
    charts[id].data.labels = labels; charts[id].data.datasets[0].data = data; charts[id].data.datasets[0].backgroundColor = colors;
    charts[id].update("none"); return;
  }
  charts[id] = new Chart(ctx, {
    type: "bar",
    data: { labels, datasets: [{ data, backgroundColor: colors, borderRadius: 3, barPercentage: 0.95, categoryPercentage: 0.95 }] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: opts.tip } } },
      scales: {
        x: { title: { display: true, text: opts.xTitle, color: css("--muted"), font: { size: 11 } }, ticks: { color: css("--muted"), font: { size: 10 }, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }, grid: { display: false } },
        y: { display: false, beginAtZero: true },
      },
    },
  });
}

function render() {
  const x = inputs();
  $("gmatVal").textContent = x.gmat; $("gpaVal").textContent = x.gpa.toFixed(2); $("yoeVal").textContent = x.yoe;
  $("appsVal").textContent = (x.appsDelta >= 0 ? "+" : "") + Math.round(x.appsDelta * 100) + "%";
  for (const s of selected) {
    const info = S[s];
    if (info.aid_basis === "official_only") continue;
    const ps = admitDistribution(s, x);
    const med = quantile(ps, 0.5), p10 = quantile(ps, 0.1), p90 = quantile(ps, 0.9);
    $(`adm-med-${s}`).textContent = pct(med, med < 0.1 ? 1 : 0);
    $(`adm-rng-${s}`).textContent = `80% interval ${pct(p10, p10 < 0.1 ? 1 : 0)}–${pct(p90, p90 < 0.1 ? 1 : 0)}`;
    const h = histAdmit(ps);
    const mi = h.bins.findIndex(([a, b]) => med >= a && med < b);
    drawChart(`ca-${s}`, h.labels, h.counts, mi, { xTitle: "Estimated chance of admission (%)", tip: (c) => `${c.raw} of 300 model refits` });

    if (noSchModel(info)) {
      $(`sch-med-${s}`).textContent = { need: "Need-based", range_only: "Official range only", policy_only: "Official policy only" }[info.aid_basis];
      $(`sch-rng-${s}`).textContent = "Not predicted from your stats";
      continue;
    }
    const tuition = x.home === info.state && info.tuition_resident ? info.tuition_resident : info.tuition;
    const d = scholarshipDistribution(s, x);
    const smed = quantile(d, 0.5), s25 = quantile(d, 0.25), s75 = quantile(d, 0.75);
    const pAny = d.filter((v) => v > 0).length / d.length;
    $(`sch-med-${s}`).textContent = `${pct(smed)} · ${usd(smed * tuition * 2)}`;
    $(`sch-rng-${s}`).textContent = `middle 50% ${pct(s25)}–${pct(s75)} of tuition · ${pct(pAny)} chance of any award`;
    const hs = histSch(d);
    const smi = smed === 0 ? 0 : SBINS.findIndex(([, a, b], i) => i > 0 && smed > a && smed <= b);
    drawChart(`cs-${s}`, hs.labels, hs.counts.map((v) => +(v * 100).toFixed(1)), smi, { xTitle: "Scholarship given admission (% of tuition)", tip: (c) => `${c.raw}% of simulated outcomes` });
  }
}

// ---------- rule-based profile interpreter (runs locally; nothing is sent anywhere) ----------
function interpret(text) {
  const t = text.toLowerCase();
  const found = [];
  const set = (id, v, label) => { const el = $(id); el.value = v; found.push(label); };
  let m = t.match(/(?:gmat(?:\s*focus)?|focus|classic|legacy)[^0-9]{0,14}(\d{3})/);
  if (m) {
    let g = +m[1], note = "";
    // Focus scores always end in 5; a score ending in 0 is read as the older GMAT Classic and converted.
    if (g % 10 === 0 || /classic|legacy|old gmat/.test(t)) { note = ` (from Classic ${g})`; g = classicToFocus(g); }
    g = Math.round((g - 5) / 10) * 10 + 5;   // Focus scores end in 5; snap to the nearest valid score
    g = Math.max(505, Math.min(805, g)); set("gmat", g, `GMAT Focus ${g}${note}`);
  }
  m = t.match(/gpa[^0-9]{0,10}([1-4]\.\d{1,2})/) || t.match(/([1-4]\.\d{1,2})\s*(?:\/\s*4(?:\.0)?)?\s*gpa/);
  if (m) set("gpa", Math.min(4, +m[1]), `GPA ${(+m[1]).toFixed(2)}`);
  m = t.match(/(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?|yoe)/);
  if (m) set("yoe", Math.min(15, Math.round(+m[1])), `${Math.round(+m[1])} years of experience`);
  const inds = [["consulting", /consult|mckinsey|bain|bcg|deloitte|accenture/], ["finance", /bank|private equity|\bpe\b|hedge|invest|venture|finance|asset management/],
    ["energy", /petroleum|oil|gas|energy|utility|refin|sinopec|exxon|chevron|shell|bp\b/], ["public_service", /military|army|navy|air force|marine|government|non-?profit|teach for/],
    ["tech", /software|tech|machine learning|\bml\b|\bai\b|data scien|google|amazon|microsoft|meta|apple/]];
  for (const [k, re] of inds) if (re.test(t)) { $("ind").value = k; found.push(`industry: ${$("ind").selectedOptions[0].text}`); break; }
  if (/international|non-us|non us|visa|from (india|china|brazil|nigeria|mexico|korea|japan|canada|uk|europe)/.test(t)) { $("intl").checked = true; found.push("non-US applicant"); }
  else if (/\bus citizen|american|green card|texas|houston|austin/.test(t)) { $("intl").checked = false; found.push("US applicant"); }
  const states = [["TX", /texas|houston|austin|dallas/], ["CA", /california|los angeles|san francisco|bay area/], ["VA", /virginia/], ["MI", /michigan/], ["WA", /washington state|seattle/], ["IN", /indiana/], ["NC", /north carolina/], ["GA", /georgia|atlanta/]];
  for (const [k, re] of states) if (re.test(t)) { $("home").value = k; found.push(`home state ${k} (in-state tuition where it applies)`); break; }
  m = t.match(/round\s*([123])|\br([123])\b/); if (m) set("round", +(m[1] || m[2]), `Round ${m[1] || m[2]}`);
  const ecs = [];
  if (/volunteer|mentor|tutor|pets alive|shelter/.test(t)) ecs.push("volunteering");
  if (/shift lead|captain|president|founded|lead(er|ing)?/.test(t)) ecs.push("leadership");
  if (/rugby|marathon|athlet|varsity|sport/.test(t)) ecs.push("athletics");
  if (/open[- ]source|github|startup|founder/.test(t)) ecs.push("builder/open-source");
  $("interp-out").innerHTML = (found.length ? `<strong>Read from your text:</strong> ${found.join(" · ")}.` : "No stats recognized. Try “GMAT 705, GPA 3.43, 6 years in energy tech”.") +
    (ecs.length ? `<br><strong>Extracurriculars noted:</strong> ${ecs.join(", ")}. These are shown for context and do not change the numbers; <a href="methodology.html#extracurriculars">here’s why</a>.` : "");
  render();
}
function classicToFocus(c) { // GMAC concordance: median Focus score for each Classic score (see methodology)
  const t = [[800, 805], [790, 795], [780, 770], [770, 745], [760, 725], [750, 705], [740, 690], [730, 680], [720, 670], [710, 660], [700, 650], [690, 640], [680, 625], [670, 615], [660, 615], [650, 605], [640, 590], [630, 585], [620, 580], [610, 570], [600, 560], [590, 555], [580, 550], [570, 540], [560, 530], [550, 520]];
  for (const [a, f] of t) if (c >= a) return f; return 515;
}

function cardHTML(s) {
  const info = S[s];
  const rm = `<button class="rm" type="button" data-rm="${s}" aria-label="Remove ${info.name}">Remove</button>`;
  const head = `<header><div><h3>${info.name}</h3><p class="meta">U.S. News 2026 #${info.usnews_rank} · ${info.profile} (<a href="${info.profile_src}" target="_blank" rel="noopener">P&amp;Q</a>)</p></div>${rm}</header>`;
  if (info.aid_basis === "official_only") {
    return head + `<p class="need">No model estimate: this program publishes no class GMAT figure and has almost no tracker decisions, so an estimate would be invented. Official facts: ${info.aid_fact} <a href="${info.aid_src}" target="_blank" rel="noopener">UT Dallas</a></p>`;
  }
  const off = M.sch_school_offsets[s] || {};
  const flag = noSchModel(info) ? "" : off.basis === "official" ? "Calibrated to official aid statistics." : off.basis === "pooled_only" ? "Too few school-specific reports: pooled estimate only." : "Calibrated to tracker reports, which likely skew high.";
  const n = (M.prior_correction[s] || {}).n || 0;
  return head + `
    <div class="pair">
      <section>
        <div class="kicker">Chance of admission</div>
        <div class="big" id="adm-med-${s}">–</div>
        <div class="sub" id="adm-rng-${s}"></div>
        <div class="chart"><canvas id="ca-${s}"></canvas></div>
      </section>
      <section>
        <div class="kicker">Scholarship if admitted · median</div>
        <div class="big" id="sch-med-${s}">–</div>
        <div class="sub" id="sch-rng-${s}"></div>
        ${noSchModel(info) ? `<p class="need">${info.aid_fact} <a href="${info.aid_src}" target="_blank" rel="noopener">official source</a></p>` : `<div class="chart"><canvas id="cs-${s}"></canvas></div>`}
      </section>
    </div>
    <footer>${info.caveat ? `<strong>${info.caveat}</strong> (<a href="${info.caveat_src}" target="_blank" rel="noopener">GMAT proxy source</a>) ` : ""}${n} school-specific tracker decisions in training${n < 20 ? " (thin: leans on the pooled model)" : ""}. ${flag} ${noSchModel(info) ? "" : `Official: ${info.aid_fact} <a href="${info.aid_src}" target="_blank" rel="noopener">source</a> · `}Dollar figures use ${info.tuition_label}, $${info.tuition.toLocaleString()} × 2 years (<a href="${info.tuition_src}" target="_blank" rel="noopener">source</a>).</footer>`;
}

function buildCards() {
  const wrap = $("cards");
  for (const id of Object.keys(charts)) { charts[id].destroy(); delete charts[id]; }
  wrap.innerHTML = "";
  if (!selected.length) { wrap.innerHTML = `<p class="empty">Add a school from the list to see estimates.</p>`; }
  for (const s of selected) {
    const card = document.createElement("article");
    card.className = "card"; card.innerHTML = cardHTML(s); wrap.appendChild(card);
  }
  wrap.querySelectorAll("[data-rm]").forEach((b) => b.addEventListener("click", () => { selected = selected.filter((x) => x !== b.dataset.rm); syncPicker(); buildCards(); render(); }));
}

function syncPicker() {
  const sel = $("pick");
  sel.innerHTML = Object.keys(S).filter((k) => !selected.includes(k))
    .sort((a, b) => parseInt(S[a].usnews_rank) - parseInt(S[b].usnews_rank))
    .map((k) => `<option value="${k}">#${S[k].usnews_rank} · ${S[k].name}</option>`).join("");
  $("add").disabled = !sel.options.length;
}

async function main() {
  [M, S] = await Promise.all([fetch("data/model.json").then((r) => r.json()), fetch("data/schools.json").then((r) => r.json())]);
  selected = DEFAULT_LIST.filter((k) => S[k]);
  syncPicker(); buildCards();
  $("add").addEventListener("click", () => { const v = $("pick").value; if (v && !selected.includes(v)) { selected.push(v); syncPicker(); buildCards(); render(); } });
  $("add-all").addEventListener("click", () => { selected = Object.keys(S).sort((a, b) => parseInt(S[a].usnews_rank) - parseInt(S[b].usnews_rank)); syncPicker(); buildCards(); render(); });
  for (const id of ["gmat", "gpa", "yoe", "intl", "ind", "round", "apps", "home"]) $(id).addEventListener("input", render);
  $("interp-btn").addEventListener("click", () => interpret($("profile").value));
  $("n-train").textContent = M.n_train.toLocaleString();
  $("n-app").textContent = M.n_applicants.toLocaleString();
  $("n-sch").textContent = M.n_schools;
  $("auc").textContent = M.eval.logistic_regression.auc.toFixed(2);
  render();
}
main();
