"""
Step 4a: build one clean text "card" per scheme (the thing the search index will hold).

Run from the repo root:
    python src/build_cards.py

Reads  : data/registry/scheme_registry.csv
         data/en/schemes_list.jsonl     (brief info for every scheme)
         data/en/schemes_detail.jsonl   (eligibility / benefits, where scraped)
Writes : data/cards/scheme_cards.jsonl  (one line per scheme: slug, name, state, level,
                                         split, has_detail, text)
"""
import argparse
import csv
import html
import json
import re
from pathlib import Path

# max characters kept from each section (keeps cards a sensible length for the models)
LIMITS = {"summary": 700, "description": 900, "eligibility": 900, "benefits": 600}


def clean(text):
    """HTML entities, markdown, <br> and extra whitespace -> plain text."""
    if not text:
        return ""
    t = html.unescape(str(text))
    t = re.sub(r"<[^>]+>", " ", t)                       # html tags like <br>
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)       # [label](url) -> label
    t = re.sub(r"[*_`#>]+", " ", t)                      # markdown marks
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.M)       # "1. " list numbers
    t = re.sub(r"\s+", " ", t).strip()
    return t


def cut(text, n):
    text = clean(text)
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def read_jsonl(path):
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def labels(items):
    """[{'label': 'X'}, ...] or ['X', ...] -> ['X', ...]"""
    out = []
    for i in items or []:
        out.append(i.get("label") if isinstance(i, dict) else i)
    return [x for x in out if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    data = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data
    (out_dir / "cards").mkdir(parents=True, exist_ok=True)

    reg = list(csv.DictReader(open(data / "registry" / "scheme_registry.csv",
                                   encoding="utf-8-sig")))
    lst = {x["slug"]: x for x in read_jsonl(data / "en" / "schemes_list.jsonl")}
    det = {}
    for d in read_jsonl(data / "en" / "schemes_detail.jsonl"):
        det[d["slug"]] = d.get("en") or {}

    n_detail = 0
    with open(out_dir / "cards" / "scheme_cards.jsonl", "w", encoding="utf-8",
              newline="\n") as f:
        for r in reg:
            slug = r["slug"]
            li = lst.get(slug, {})
            d = det.get(slug)
            name = r["name"] or li.get("schemeName", "")
            state = r["state"] or "All"
            level = r["level"]
            cat = [c for c in r["category"].split("|") if c]
            tags = [t for t in r["tags"].split("|") if t]
            parts = [f"Scheme: {name}"]
            short = li.get("schemeShortTitle")
            if short and short.lower() != name.lower():
                parts.append(f"Also called: {short}")
            parts.append(f"State: {state} ({level})")
            if cat:
                parts.append("Category: " + ", ".join(cat))
            if tags:
                parts.append("Tags: " + ", ".join(tags))
            brief = li.get("briefDescription", "")
            if d:
                n_detail += 1
                bd = d.get("basicDetails", {})
                sub = labels(bd.get("schemeSubCategory"))
                if sub:
                    parts.append("Sub-category: " + ", ".join(sub))
                ben = labels(bd.get("targetBeneficiaries"))
                if ben:
                    parts.append("For: " + ", ".join(ben))
                sc = d.get("schemeContent", {})
                brief = sc.get("briefDescription") or brief
                parts.append("Summary: " + cut(brief, LIMITS["summary"]))
                desc = cut(sc.get("detailedDescription_md"), LIMITS["description"])
                if desc and desc[:120] != cut(brief, 120)[:120]:
                    parts.append("Details: " + desc)
                elig = cut((d.get("eligibilityCriteria") or {}).get("eligibilityDescription_md"),
                           LIMITS["eligibility"])
                if elig:
                    parts.append("Eligibility: " + elig)
                benefits = cut(sc.get("benefits_md"), LIMITS["benefits"])
                if benefits:
                    parts.append("Benefits: " + benefits)
            elif brief:
                parts.append("Summary: " + cut(brief, LIMITS["summary"]))
            card = {
                "slug": slug, "name": name, "state": state, "level": level,
                "split": r["split"], "has_detail": int(bool(d)),
                "text": "\n".join(parts),
            }
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
    print(f"cards written: {len(reg)}  (with eligibility/benefits detail: {n_detail})")


if __name__ == "__main__":
    main()