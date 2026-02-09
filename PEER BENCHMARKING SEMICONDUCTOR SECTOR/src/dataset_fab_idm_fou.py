# Next-step fab model classifier (HTML->text clean, section focus, weighted keywords + negation/proximity,
# + financial sanity check (PPE / CapEx / D&A), + evidence snippets)
#
# pip install beautifulsoup4 lxml

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

headers = {"User-Agent": "laksolajoni@gmail.com"}

# ---------- SEC helpers ----------

def latest_annual_filing_info(cik10: str):
    j = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik10}.json", headers=headers
    ).json()
    recent = j["filings"]["recent"]
    forms = recent["form"]
    for i, f in enumerate(forms):
        if f in ("10-K", "10-K/A", "20-F", "20-F/A"):
            accn_nodash = recent["accessionNumber"][i].replace("-", "")
            primary = recent["primaryDocument"][i]
            filed = recent["filingDate"][i]
            form = f
            return accn_nodash, primary, filed, form
    return None, None, None, None


def download_filing_html(cik10: str, accn_nodash: str, primary_doc: str) -> str:
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{accn_nodash}/{primary_doc}"
    r = requests.get(
        url,
        headers={
            "User-Agent": headers["User-Agent"],
            "Accept-Encoding": "gzip, deflate",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.text


# ---------- HTML -> text clean ----------

def html_to_text_clean(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # remove script/style/noscript
    for tag in soup(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()

    # drop tables (often XBRL/financial tables noise). comment out if you want tables included.
    for tag in soup.find_all("table"):
        tag.decompose()

    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- Section extraction (10-K Items / 20-F items) ----------

ITEM_PATTERNS_10K = [
    ("item1",  r"\bitem\s+1[\.\:\- ]\s*business\b"),
    ("item1a", r"\bitem\s+1a[\.\:\- ]\s*risk\s+factors\b"),
]
ITEM_PATTERNS_20F = [
    ("item4",  r"\bitem\s+4[\.\:\- ]\s*information\s+on\s+the\s+company\b"),
    ("item3d", r"\bitem\s+3d[\.\:\- ]\s*risk\s+factors\b"),
]

def extract_sections(text: str, form: str) -> dict:
    t = text.lower()
    patterns = ITEM_PATTERNS_20F if form.startswith("20-F") else ITEM_PATTERNS_10K

    hits = []
    for name, pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            hits.append((name, m.start()))

    # if nothing found, fall back to first N chars (still better than whole doc)
    if not hits:
        return {"fallback": t[:300_000]}

    hits.sort(key=lambda x: x[1])
    sections = {}
    for idx, (name, start) in enumerate(hits):
        end = hits[idx + 1][1] if idx + 1 < len(hits) else min(len(t), start + 300_000)
        sections[name] = t[start:end]
    return sections


# ---------- Weighted keyword scoring + negation/proximity ----------

NEGATION = r"(?:no|not|never|without|do\s+not|does\s+not|did\s+not|none|lack|lacks|lacking)"

def find_snippets(text: str, pattern: str, window: int = 120, max_snips: int = 3):
    snips = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        a = max(0, m.start() - window)
        b = min(len(text), m.end() + window)
        snip = text[a:b]
        snip = re.sub(r"\s+", " ", snip).strip()
        snips.append(snip)
        if len(snips) >= max_snips:
            break
    return snips

# Strong / weak evidence patterns (regex-friendly)
FABLESS_RULES = [
    # strong
    (r"\bfabless\b", 5),
    (rf"{NEGATION}.*\b(fab|fabrication\s+facility|wafer\s+fab|manufacturing\s+facility)\b", 5),
    (r"\brely\s+on\s+(?:third[- ]party|external)\s+(?:wafer\s+)?foundr(?:y|ies)\b", 5),
    (r"\boutsourc(?:e|ed|ing)\s+(?:wafer\s+)?fabrication\b", 5),
    # medium
    (r"\bthird[- ]party\s+(?:wafer\s+)?foundr(?:y|ies)\b", 3),
    (r"\bcontract\s+manufacturer(?:s)?\b", 2),
]

FOUNDRY_RULES = [
    # strong
    (r"\bpure[- ]play\s+foundry\b", 6),
    (r"\bfoundry\s+services\b", 6),
    (r"\bwafer\s+foundry\b", 5),
    (r"\bmanufacture\s+wafers\s+for\s+customers\b", 6),
    (r"\bfabricate\s+wafers\s+for\s+customers\b", 6),
    # medium
    (r"\bprocess\s+technology\s+node(?:s)?\b", 2),
    (r"\bwafer\s+starts?\b", 2),
    (r"\bcapacity\s+utilization\b", 2),
]

IDM_RULES = [
    # strong
    (r"\bown\s+and\s+operate\s+(?:wafer\s+)?fabrication\b", 6),
    (r"\b(wafer\s+)?fabrication\s+facilit(?:y|ies)\b", 5),
    (r"\bfront[- ]end\s+manufacturing\b", 4),
    (r"\b300\s*mm\b|\b200\s*mm\b", 3),
    (r"\bclean\s*room\b|\bcleanroom\b", 3),
    (r"\bfab(?:s)?\b.*\b(located|location|in)\b", 3),
    # medium/weak (keep low weight)
    (r"\bassembly\s+and\s+test\b", 1),
]

# proximity heuristic: "manufactur*" near "outsource/third-party/foundry/contract" => fabless boost, idm penalty
OUTSOURCE_CONTEXT = r"(?:outsource|third[- ]party|external|foundr(?:y|ies)|contract)"
def proximity_outsource_boost(text: str) -> int:
    # look for manufact* within ~12 words of outsource context
    pat = r"(manufactur\w+(?:\W+\w+){0,12}\W+" + OUTSOURCE_CONTEXT + r")|(" + OUTSOURCE_CONTEXT + r"(?:\W+\w+){0,12}\W+manufactur\w+)"
    return 3 if re.search(pat, text, flags=re.IGNORECASE) else 0

def score_rules(text: str, rules):
    score = 0
    evid = []
    for pat, w in rules:
        if re.search(pat, text, flags=re.IGNORECASE):
            score += w
            evid.extend(find_snippets(text, pat))
    return score, evid


def classify_fab_model(text: str):
    # text should already be focused sections (lowercase ok)
    fab_score, fab_e = score_rules(text, FABLESS_RULES)
    fnd_score, fnd_e = score_rules(text, FOUNDRY_RULES)
    idm_score, idm_e = score_rules(text, IDM_RULES)

    # proximity tweak
    boost = proximity_outsource_boost(text)
    if boost:
        fab_score += boost
        idm_score -= 1  # small penalty

    scores = {"fabless": fab_score, "foundry": fnd_score, "idm": idm_score}
    evid = {"fabless": fab_e[:5], "foundry": fnd_e[:5], "idm": idm_e[:5]}

    best = max(scores, key=scores.get)
    top = scores[best]

    if top <= 0:
        return "unknown", scores, evid

    # hybrid if we have strong-ish evidence for >=2 categories
    strong_hits = sum(1 for v in scores.values() if v >= 6)
    if strong_hits >= 2:
        return "hybrid", scores, evid

    # low confidence if close race
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_scores) >= 2 and (sorted_scores[0][1] - sorted_scores[1][1] <= 2):
        return "hybrid", scores, evid

    return best, scores, evid


# ---------- Financial sanity check (PPE / CapEx / D&A) ----------
# You can compute these from companyfacts; here are robust fact fallbacks for a quick signal.

FACT_FALLBACKS = {
    "assets": ["Assets"],
    "ppe": ["PropertyPlantAndEquipmentNet", "PropertyPlantAndEquipmentGross"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"],
    "capex": [
        # US-GAAP common
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "CapitalExpenditures",  # sometimes present
        "PurchaseOfPropertyPlantAndEquipment",
        # IFRS-ish variants (sometimes mapped under ifrs-full; still try same names)
        "PaymentsToAcquirePropertyPlantAndEquipmentIntangibleAssets",
    ],
}

def extract_fact_units_any_currency(facts_json, standard_tag, fact_name):
    try:
        units = facts_json["facts"][standard_tag][fact_name]["units"]
    except KeyError:
        return None, None
    # prioritize USD, else first numeric currency unit
    if "USD" in units:
        return units["USD"], "USD"
    for k, v in units.items():
        if isinstance(v, list) and k.isalpha() and len(k) in (3,4):  # crude: currency-like
            return v, k
    return None, None

def latest_annual_value(facts_json, fact_names):
    # returns (val, unit, end, accn, form) latest annual (10-K/20-F) based on 'filed'
    best = None
    for std in ("us-gaap", "ifrs-full"):
        for fn in fact_names:
            rows, unit = extract_fact_units_any_currency(facts_json, std, fn)
            if not rows:
                continue
            df = pd.DataFrame(rows)
            if df.empty:
                continue
            df = df[df["form"].isin(["10-K", "10-K/A", "20-F", "20-F/A"])].copy()
            if df.empty:
                continue
            df["filed_dt"] = pd.to_datetime(df.get("filed"), errors="coerce")
            df["end_dt"] = pd.to_datetime(df.get("end"), errors="coerce")
            df = df.sort_values("filed_dt")
            # prefer FY if exists, else keep
            if "fp" in df.columns:
                df_fy = df[df["fp"].astype(str).str.upper().eq("FY")]
                if not df_fy.empty:
                    df = df_fy
            row = df.iloc[-1]
            cand = (row.get("val"), unit, row.get("end"), row.get("accn"), row.get("form"), row.get("filed"))
            if best is None:
                best = cand
            else:
                # pick later filed
                if pd.to_datetime(cand[5], errors="coerce") > pd.to_datetime(best[5], errors="coerce"):
                    best = cand
    return best  # or None

def financial_sanity_scores(facts_json):
    assets = latest_annual_value(facts_json, FACT_FALLBACKS["assets"])
    ppe    = latest_annual_value(facts_json, FACT_FALLBACKS["ppe"])
    da     = latest_annual_value(facts_json, FACT_FALLBACKS["da"])
    capex  = latest_annual_value(facts_json, FACT_FALLBACKS["capex"])

    # compute simple intensities if same currency (or just compute ratio if numeric)
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    a = to_float(assets[0]) if assets else None
    p = to_float(ppe[0]) if ppe else None
    d = to_float(da[0]) if da else None
    c = to_float(capex[0]) if capex else None

    # ratios (rough sanity)
    ppe_int = (p / a) if (a and p is not None and a != 0) else None

    # heuristic: high PPE intensity => IDM/foundry; low => fabless
    hint = {}
    if ppe_int is not None:
        if ppe_int >= 0.20:
            hint["idm_foundry_boost"] = 2
        elif ppe_int <= 0.08:
            hint["fabless_boost"] = 2
    # if capex exists and large relative to assets, similar
    if c is not None and a:
        capex_int = c / a if a != 0 else None
        if capex_int is not None and capex_int >= 0.05:
            hint["idm_foundry_boost"] = hint.get("idm_foundry_boost", 0) + 1

    return {
        "assets": assets, "ppe": ppe, "da": da, "capex": capex,
        "ppe_intensity": ppe_int,
        "hint": hint
    }


def reconcile_text_and_fin(text_label, text_scores, fin_hint):
    scores = dict(text_scores)  # copy
    if fin_hint.get("fabless_boost"):
        scores["fabless"] += fin_hint["fabless_boost"]
    if fin_hint.get("idm_foundry_boost"):
        scores["idm"] += fin_hint["idm_foundry_boost"]
        scores["foundry"] += fin_hint["idm_foundry_boost"]

    best = max(scores, key=scores.get)
    top = scores[best]
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # If still close => hybrid
    if top <= 0:
        return "unknown", scores
    if len(sorted_scores) >= 2 and (sorted_scores[0][1] - sorted_scores[1][1] <= 2):
        return "hybrid", scores
    return best, scores


# ---------- Main loop: classify + print evidence ----------

def classify_company(cik10: str):
    accn, primary, filed, form = latest_annual_filing_info(cik10)
    if not accn:
        return {
            "cik_str": cik10, "label": "unknown", "form": None, "filed": None,
            "scores_text": None, "scores_final": None, "evidence": None, "ppe_intensity": None
        }

    html = download_filing_html(cik10, accn, primary)
    text = html_to_text_clean(html)

    sections = extract_sections(text, form)
    focused_text = " ".join(sections.values())  # concat item1+item1a or 20F items

    label_text, scores_text, evid = classify_fab_model(focused_text)

    # financial sanity-check from companyfacts
    facts_json = requests.get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json", headers=headers, timeout=60
    ).json()
    fin = financial_sanity_scores(facts_json)
    label_final, scores_final = reconcile_text_and_fin(label_text, scores_text, fin["hint"])

    return {
        "cik_str": cik10,
        "label_text": label_text,
        "label": label_final,
        "form": form,
        "filed": filed,
        "scores_text": scores_text,
        "scores_final": scores_final,
        "evidence": evid,
        "ppe_intensity": fin["ppe_intensity"],
        "assets_latest": fin["assets"][0] if fin["assets"] else None,
        "ppe_latest": fin["ppe"][0] if fin["ppe"] else None,
        "capex_latest": fin["capex"][0] if fin["capex"] else None,
        "da_latest": fin["da"][0] if fin["da"] else None,
    }


# ---- Run on your seed list (use your existing seed dataframe with cik_str) ----
seed = pd.read_csv("/Users/kayttaja/Desktop/PEER BENCHMARKING SEMICONDUCTOR SECTOR/data/interim/semiconductor_companies.csv", dtype={"cik_str": str})
seed["cik_str"] = seed["cik_str"].astype(str).str.zfill(10)

def run_classification(seed: pd.DataFrame, sleep_s=0.25, max_rows=None, verbose=True):
    out = []
    n = len(seed) if max_rows is None else min(len(seed), max_rows)
    for i, cik10 in enumerate(seed["cik_str"].astype(str).str.zfill(10).tolist()[:n], start=1):
        try:
            res = classify_company(cik10)
            out.append(res)

            if verbose:
                print(f"[{i}/{n}] {cik10} => {res['label']} (text={res.get('label_text')}) "
                      f"scores={res.get('scores_final')} ppe_int={res.get('ppe_intensity')}")
                # print a couple snippets from the winning class
                if res.get("evidence") and res.get("label") in res["evidence"]:
                    snips = res["evidence"][res["label"]][:2]
                    for s in snips:
                        print("  -", s[:240], "...")
        except Exception as e:
            out.append({"cik_str": cik10, "label": "error", "error": str(e)})
            if verbose:
                print(f"[{i}/{n}] {cik10} => ERROR: {e}")

        time.sleep(sleep_s)
    return pd.DataFrame(out)



result_df = run_classification(seed, sleep_s=0.25, max_rows=50, verbose=True)
result_df.to_csv("/Users/kayttaja/Desktop/PEER BENCHMARKING SEMICONDUCTOR SECTOR/data/interim/semiconductor_companies_with_fab_model_v2.csv", index=False)


