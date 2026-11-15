"""
Step 7b: turn a scheme's eligibility text into a rule (same format as data/rag_schemes.json).

    python src/extract_rules.py --eval                 # score against the 1,082 known rules
    python src/extract_rules.py --write                # write data/rules_extracted.json for ALL schemes

Version 1 = plain patterns (regex). It is deliberately cautious: when a sentence is unclear it
leaves the field out (= unknown), because a missing condition is safe and a wrong one is not.
State and level come from the scheme's own metadata, not from the text.
"""
import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path

NUM = r"(\d{1,3})"
OCC = {  # keyword -> occupation label
    "farmer": ["farmer", "cultivator", "agricultur", "grower", "horticultur", "fisherm", "dairy", "livestock"],
    "student": ["student", "pursuing", "scholar", "enrolled in", "studying", "class ", "graduat"],
    "worker": ["worker", "labour", "laborer", "artisan", "weaver", "employee", "driver", "construction"],
    "entrepreneur": ["entrepreneur", "start-up", "startup", "msme", "enterprise", "self-employed", "business"],
    "unemployed": ["unemployed", "job seeker", "jobseeker", "employment exchange"],
    "vendor": ["vendor", "hawker"],
}


def clean(t):
    t = html.unescape(html.unescape(t or ""))
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[*_#>]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def money(txt):
    """'Rs. 2.5 lakh' / '1,50,000' -> int rupees."""
    m = re.search(r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lac|crore)?", txt, re.I)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    u = (m.group(2) or "").lower()
    v *= 10_000_000 if u == "crore" else 100_000 if u in ("lakh", "lac") else 1
    return int(v)


def extract(text, tags=()):
    t = clean(text)
    low = t.lower()
    rule = {}

    # ---- age
    lo = hi = None
    m = re.search(rf"(?:between|aged?|age of)\s*{NUM}\s*(?:and|to|-|–)\s*{NUM}\s*years", low) or \
        re.search(rf"{NUM}\s*(?:to|-|–)\s*{NUM}\s*years", low)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
    for m in re.finditer(rf"(?:not (?:be )?(?:below|less than|under)|at least|minimum age(?: of)?|"
                         rf"(?:above|over|more than|completed)|min\.? age)\s*{NUM}\s*(?:years|yrs)", low):
        lo = int(m.group(1)) if lo is None else lo
    for m in re.finditer(rf"(?:not (?:be )?(?:above|more than|exceed(?:ing)?|older than)|"
                         rf"maximum age(?: of)?|(?:below|under|up to|upto)|max\.? age)\s*{NUM}\s*(?:years|yrs)", low):
        hi = int(m.group(1)) if hi is None else hi
    if lo and 3 <= lo <= 90:
        rule["min_age"] = lo
    if hi and 3 <= hi <= 100 and (not lo or hi >= lo):
        rule["max_age"] = hi

    # ---- income
    m = re.search(r"(?:annual |family |household |yearly )?income[^.]{0,80}?(?:not exceed|less than|below|up to|upto|"
                  r"does not exceed|should not be more than|not more than|within)\s*(?:rs\.?|inr|₹)?\s*[\d,\.]+\s*(?:lakh|lac|crore)?", low)
    if m:
        v = money(re.sub(r"^.*?(?:exceed|less than|below|up to|upto|more than|within)", "", m.group(0)))
        if v and v >= 10000:
            rule["max_income"] = v

    # ---- caste / category
    castes = []
    if re.search(r"\bscheduled castes?\b|(?<![a-z]\.)\bsc\b(?!\.)", low): castes.append("sc")
    if re.search(r"\bscheduled tribes?\b|(?<![a-z]\.)\bst\b(?!\.)|\btribal\b", low): castes.append("st")
    if re.search(r"\bobc\b|other backward|backward class", low): castes.append("obc")
    if castes and re.search(r"belong|belonging|categor|caste|tribe|tribal|community|\bsc\b|\bst\b|\bobc\b", low):
        rule["castes"] = sorted(set(castes))

    # ---- gender
    if re.search(r"\b(women|woman|female|girl|girls|widow|lady|ladies)\b", low) and \
       not re.search(r"\bmen and women\b|\bboth (?:men|male)|\bmale and female\b|\bwomen and men\b", low):
        rule["gender"] = "female"

    # ---- disability, BPL, residence
    if re.search(r"disab|divyang|differently.abled|handicap|pwd|blind|deaf|locomotor", low):
        rule["disability"] = True
    if re.search(r"below poverty line|\bbpl\b|economically weaker|\bews\b|antyodaya|\baay\b", low):
        rule["bpl"] = True
    rural = re.search(r"\brural\b|\bvillages?\b|gram panchayat", low)
    urban = re.search(r"\burban\b|\bmunicipal(?:ity|ities)?\b|\bcity\b|\btown\b|\bcities\b", low)
    if rural and not urban:
        rule["residence"] = "rural"
    elif urban and not rural:
        rule["residence"] = "urban"

    # ---- occupation (text + tags)
    blob = low + " " + " ".join(tags).lower()
    hits = [o for o, kws in OCC.items() if any(k in blob for k in kws)]
    if hits:
        rule["occupation"] = hits[0] if len(hits) == 1 else hits[:2]
    return rule


def eligibility_text(rec):
    e = rec["en"].get("eligibilityCriteria", {}) or {}
    return e.get("eligibilityDescription_md") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    d = Path(args.data_dir)
    known = {s["id"]: s["rule"] for s in json.load(open(d / "rag_schemes.json", encoding="utf-8"))}
    recs = {}
    for l in open(d / "en" / "schemes_detail.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            recs[r["slug"]] = r
    out = {s: extract(eligibility_text(r), r["en"]["basicDetails"].get("tags") or []) for s, r in recs.items()}

    if args.eval:
        keys = ["min_age", "max_age", "max_income", "castes", "gender", "disability", "bpl", "residence", "occupation"]
        print(f"{'field':12}{'gold':>6}{'found':>7}{'right':>7}{'precision':>11}{'recall':>8}")
        for k in keys:
            g = f = ok = 0
            for s, gold in known.items():
                if s not in out:
                    continue
                a, b = gold.get(k), out[s].get(k)
                g += a is not None
                f += b is not None
                ok += a is not None and a == b
            print(f"{k:12}{g:6}{f:7}{ok:7}{ok / max(1, f):11.2f}{ok / max(1, g):8.2f}")
        exact = sum({k: v for k, v in known[s].items() if k != 'state'} == out[s] for s in known if s in out)
        print(f"\nwhole rule exactly right: {exact}/{len(known)}")
    if args.write:
        json.dump(out, open(d / "rules_extracted.json", "w", encoding="utf-8"), indent=1)
        print(f"wrote data/rules_extracted.json  ({len(out)} schemes, "
              f"{sum(1 for v in out.values() if v)} with at least one condition)")


if __name__ == "__main__":
    main()