"""
Step 8b: learn, from the TRAIN split only, which words in each language mean what.

    python src/build_lexicon.py

For every field value (female, rural, farmer, BPL ...) it keeps the words that appear almost only in
queries where that value is true. No dictionary is typed by hand, so it works for all 15 languages.
Writes data/lexicon/lexicon.json (used automatically by src/read_query.py) and a readable report.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TOK = re.compile(r"[^\s\.,;:!?()\"'।|/]+")
FIELDS = {"gender": ["male", "female"], "residence": ["rural", "urban"],
          "occupation": ["farmer", "student", "worker", "entrepreneur", "unemployed", "vendor"],
          "disability": [True], "bpl": [True]}
MIN_COUNT, MIN_PRECISION = 4, 0.95
STATE_WORDS = set("andhra arunachal pradesh assam bihar chhattisgarh goa gujarat haryana himachal jharkhand karnataka kerala madhya maharashtra manipur meghalaya mizoram nagaland odisha punjab rajasthan sikkim tamil nadu telangana tripura uttar uttarakhand west bengal delhi jammu kashmir ladakh puducherry chandigarh andaman nicobar lakshadweep".split())


def tokens(text):
    return set(TOK.findall(text.lower()))


def main():
    rows = [json.loads(l) for l in open("data/splits/train.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["row_type"] in ("standard", "named", "vague") and isinstance(r["true_profile"], dict)
            ]
    print(len(rows), "training queries (all languages; English keeps generic words like 'card' out)")
    toks = [tokens(" ".join(r["turns"]) if r["n_turns"] > 1 else r["query"]) for r in rows]
    total = Counter(t for ts in toks for t in ts)
    lex, report = {}, []
    for field, values in FIELDS.items():
        lex[field] = {}
        for v in values:
            with_v = Counter()
            for r, ts in zip(rows, toks):
                if r["true_profile"][field] == v and (field in ("disability", "bpl") or field not in r["missing_fields"]):
                    with_v.update(ts)
            good = [t for t, c in with_v.items()
                    if c >= MIN_COUNT and c / total[t] >= MIN_PRECISION and len(t) > 1 and not t.isdigit() and t not in STATE_WORDS
                    and t not in {"want", "up", "set", "open", "start", "as", "work", "daily", "land", "home", "person", "certificate", "currently", "right", "now", "sector"}]
            good.sort(key=lambda t: -with_v[t])
            lex[field][str(v).lower()] = good[:60]
            report.append(f"{field}={v}: {' '.join(good[:25])}")
    # words that mean 'lakh': a number followed by a word, where number x 100000 is the family income
    lakh = Counter()
    for r in rows:
        inc = r["true_profile"]["income"]
        if r["language"] == "english" or not inc or "income" in r["missing_fields"]:
            continue
        for n, w in re.findall(r"(\d+(?:\.\d+)?)\s+([^\s\d\.,;:!?()]+)", r["query"].lower()):
            if abs(float(n) * 100000 - inc) < 1:
                lakh[w] += 1
    lex["_lakh"] = {"words": [w for w, c in lakh.items() if c >= 2]}
    report.append("lakh words: " + " ".join(lex["_lakh"]["words"]))
    Path("data/lexicon").mkdir(parents=True, exist_ok=True)
    json.dump(lex, open("data/lexicon/lexicon.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    Path("data/lexicon/report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()