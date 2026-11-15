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


DISJOINT = re.compile(r"(?:any|one)(?: one)? of the following (?:categories|groups|types|criteria|conditions)|\beither\b|any of these")
RELAX = re.compile(r"relaxation|woman or man|women and men|men and women|including women")


def positive_part(low):
    """Drop sentences that say what the applicant must NOT be (so 'not an employee' is not read as 'employee')."""
    parts = re.split(r"\s\d\.\s|\s-\s|(?<=\.)\s", low)
    return " ".join(p for p in parts if not re.search(r"\bnot\b", p))


def extract(text, tags=()):
    t = clean(text)
    low = t.lower()
    pos = positive_part(low)
    disj = bool(DISJOINT.search(low))
    rule = {}

    # ---- age: only numbers that are clearly about the applicant's age
    ranges = set(re.findall(rf"{NUM}\s*(?:to|-|–|and)\s*{NUM}\s*years", low))
    ranges = {r for r in ranges if int(r[0]) < int(r[1]) and 3 <= int(r[0]) <= 90}
    lo = hi = None
    if len(ranges) == 1:                                   # several different ranges = depends on case: skip
        lo, hi = map(int, next(iter(ranges)))
    elif not ranges:
        near = r"(?:age|aged|old|years of age)"
        m = re.search(rf"(?:{near}[^.]{{0,40}}?)(?:not (?:be )?(?:below|less than|under)|at least|minimum|above|over|more than)\s*{NUM}\s*(?:years|yrs)", low) \
            or re.search(rf"{NUM}\s*(?:years|yrs)(?: of age)?\s*(?:or|and)\s*(?:above|more|older|over)", low) \
            or re.search(rf"(?:at least|minimum age(?: of)?|not (?:be )?(?:below|less than|under))\s*{NUM}\s*(?:years|yrs)\s*(?:of age|old)", low)
        if m:
            lo = int(m.group(1))
        m = re.search(rf"(?:{near}[^.]{{0,40}}?)(?:not (?:be )?(?:above|more than|exceed(?:ing)?|older than)|maximum|below|under|up to|upto|limit[^.]{{0,20}}?)\s*{NUM}\s*(?:years|yrs)", low) \
            or re.search(rf"{NUM}\s*(?:years|yrs)(?: of age)?\s*(?:or|and)\s*(?:below|less|younger|under)", low) \
            or re.search(rf"(?:maximum age(?: of)?|not (?:be )?(?:above|older than))\s*{NUM}\s*(?:years|yrs)", low)
        if m:
            hi = int(m.group(1))
    if lo and 3 <= lo <= 90:
        rule["min_age"] = lo
    if hi and 3 <= hi <= 100 and (not lo or hi >= lo):
        rule["max_age"] = hi

    # ---- income (monthly figures are converted to a year; two different limits = skip)
    figs = set()
    for m in re.finditer(r"(monthly |annual |family |household |yearly )?(?:income|salary)[^.]{0,80}?(?:not exceed|exceeding|less than|below|up to|upto|"
                         r"should not be more than|not more than|within)\s*(?:rs\.?|inr|₹)?\s*[\d,\.]+\s*(?:lakh|lac|crore)?(?:[^.]{0,25}?per annum)?", low):
        v = money(re.sub(r"^.*?(?:exceed|exceeding|less than|below|up to|upto|more than|within)", "", m.group(0)))
        if v and v >= 1000:
            if (m.group(1) or "").strip() == "monthly" and "per annum" not in m.group(0):
                v *= 12
            figs.add(v)
    if len(figs) == 1:
        rule["max_income"] = figs.pop()

    # ---- caste / category (skip when the text lists alternatives or says who is NOT eligible)
    castes = []
    if re.search(r"\bscheduled castes?\b|(?<![a-z]\.)\bsc\b(?!\.)", pos): castes.append("sc")
    if re.search(r"\bscheduled tribes?\b|(?<![a-z]\.)\bst\b(?!\.)|\btribal\b", pos): castes.append("st")
    if re.search(r"\bobc\b|other backward|backward class", pos): castes.append("obc")
    if castes and not disj and not RELAX.search(low) and re.search(r"belong|categor|caste|tribe|tribal|community|\bsc\b|\bst\b|\bobc\b", pos):
        rule["castes"] = sorted(set(castes))

    # ---- gender
    female = re.search(r"\b(women|woman|female|girl|girls|ladies|lady)\b", low) or \
        re.search(r"(?:must|should) be a widow|\bunmarried women\b", low)
    if female and not disj and not RELAX.search(low) and not re.search(r"\bmales?\b|\bboy", low):
        rule["gender"] = "female"

    # ---- disability, BPL, residence
    if re.search(r"disab|divyang|differently.abled|handicap|\bpwd\b|visually|blind|\bdeaf\b|locomotor", low) \
            and not re.search(r"(?:child|son|daughter|dependent|member)[^.]{0,40}(?:disab|mentally)", low):
        rule["disability"] = True
    if not disj and re.search(r"below poverty line|\bbpl\b|economically weaker|\bews\b|antyodaya|\baay\b", low):
        rule["bpl"] = True
    rural = re.search(r"\brural\b", low)
    urban = re.search(r"\burban\b", low)
    if rural and not urban and not re.search(r"\burban\b", low):
        rule["residence"] = "rural"
    elif urban and not rural and not re.search(r"\brural\b|can be", low):
        rule["residence"] = "urban"

    # ---- occupation: text only (tags were too noisy), and not from 'must not be ...' sentences
    hits = [o for o, kws in OCC.items() if any(k in pos for k in kws)]
    if not hits and not pos.strip():
        blob = " ".join(tags).lower()
        hits = [o for o, kws in OCC.items() if any(k in blob for k in kws)]
    if hits and not disj:
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