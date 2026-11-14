"""
Step 2 of JANORA: build the scheme registry and freeze a scheme-ID split.

Run from the repo root (E:\\marg):
    python src/make_split.py

Reads  : data/en/schemes_list.jsonl, data/rag_schemes.json,
         data/rag_train.jsonl, data/rag_test.jsonl
Writes : data/registry/scheme_registry.csv
         data/registry/scheme_split_v1.json   <- the frozen split (never edit by hand)
         data/splits/train.jsonl, val.jsonl, test.jsonl   (all queries, pooled and re-split)
         data/registry/split_report.txt

Rules
  * One scheme = one slug (duplicates in the list are merged).
  * Every scheme gets exactly one split: train / val / test (70/15/15).
  * Schemes that have queries and schemes that don't are split separately,
    so each split has a fair share of both.
  * Every query inherits the split of its target scheme.
    Rows with no target scheme (off-topic) are assigned by hashing the row id.
  * If scheme_split_v1.json already exists it is NEVER regenerated
    (only new schemes are added) - that is what "frozen" means.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
RATIOS = (0.70, 0.15, 0.15)          # train, val, test
SPLITS = ("train", "val", "test")


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_registry(data_dir):
    """Merge the scraped list with rag_schemes.json into one record per slug."""
    reg = {}

    for x in read_jsonl(data_dir / "en" / "schemes_list.jsonl"):
        slug = x.get("slug")
        if not slug or slug in reg:
            continue
        reg[slug] = {
            "slug": slug,
            "name": x.get("schemeName", ""),
            "level": (x.get("level") or "").lower(),
            "state": "|".join(x.get("beneficiaryState") or []),
            "category": "|".join(x.get("schemeCategory") or []),
            "tags": "|".join(x.get("tags") or []),
            "close_date": x.get("schemeCloseDate") or "",
            "in_list": 1,
            "in_rag": 0,
        }

    rag = json.load(open(data_dir / "rag_schemes.json", encoding="utf-8"))
    added_from_rag = 0
    for s in rag:
        slug = s["id"]
        if slug not in reg:                      # scheme missing from the scraped list
            reg[slug] = {
                "slug": slug,
                "name": s.get("name", ""),
                "level": (s.get("level") or "").lower(),
                "state": s.get("state") or "",
                "category": s.get("category") or "",
                "tags": "|".join(s.get("topics") or []),
                "close_date": "",
                "in_list": 0,
                "in_rag": 1,
            }
            added_from_rag += 1
        else:
            reg[slug]["in_rag"] = 1
    return reg, added_from_rag


def assign_splits(slugs, seed):
    """Deterministic 70/15/15 split of a list of slugs."""
    slugs = sorted(slugs)
    random.Random(seed).shuffle(slugs)
    n = len(slugs)
    n_train = round(n * RATIOS[0])
    n_val = round(n * RATIOS[1])
    out = {}
    for i, s in enumerate(slugs):
        out[s] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


def hash_split(key):
    """Stable split for rows that have no scheme (e.g. off-topic)."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default=None, help="default: same as --data-dir")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    (out_dir / "registry").mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    # ---- 1. registry ------------------------------------------------------
    reg, added = build_registry(data_dir)

    # ---- 2. queries (pooled) ---------------------------------------------
    queries = []
    for src in ("train", "test"):
        for r in read_jsonl(data_dir / f"rag_{src}.jsonl"):
            r["source_file"] = src
            r["uid"] = f"{src}-{r['id']}"
            queries.append(r)
    with_q = {q["target_slug"] for q in queries if q.get("target_slug")}
    missing_targets = sorted(s for s in with_q if s not in reg)
    for s in missing_targets:                      # keep the pipeline running
        reg[s] = {"slug": s, "name": "", "level": "", "state": "", "category": "",
                  "tags": "", "close_date": "", "in_list": 0, "in_rag": 0}

    for slug, rec in reg.items():
        rec["has_queries"] = int(slug in with_q)

    # ---- 3. frozen scheme split ------------------------------------------
    split_path = out_dir / "registry" / "scheme_split_v1.json"
    if split_path.exists():
        split = json.load(open(split_path, encoding="utf-8"))
        new = [s for s in reg if s not in split]
        if new:                                    # only NEW schemes get assigned
            split.update(assign_splits(new, SEED + 1))
        print(f"Existing frozen split found; {len(new)} new schemes added.")
    else:
        with_group = [s for s in reg if reg[s]["has_queries"]]
        no_group = [s for s in reg if not reg[s]["has_queries"]]
        split = {}
        split.update(assign_splits(with_group, SEED))
        split.update(assign_splits(no_group, SEED))
        print("Created a new frozen split.")
    for slug in reg:
        reg[slug]["split"] = split[slug]
    json.dump(dict(sorted(split.items())), open(split_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)

    # ---- 4. assign every query to a split --------------------------------
    by_split = defaultdict(list)
    for q in queries:
        t = q.get("target_slug")
        q["split"] = split[t] if t else hash_split(q["uid"])
        by_split[q["split"]].append(q)
    for name in SPLITS:
        write_jsonl(out_dir / "splits" / f"{name}.jsonl", by_split[name])

    # ---- 5. registry csv ---------------------------------------------------
    cols = ["slug", "name", "level", "state", "category", "tags", "close_date",
            "in_list", "in_rag", "has_queries", "split"]
    with open(out_dir / "registry" / "scheme_registry.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for slug in sorted(reg):
            w.writerow({c: reg[c] if False else reg[slug][c] for c in cols})

    # ---- 6. checks + report ------------------------------------------------
    lines = []
    lines.append(f"Schemes in registry: {len(reg)} "
                 f"(added from rag_schemes.json because not in list: {added})")
    lines.append(f"Schemes with at least one query: {len(with_q)}")
    if missing_targets:
        lines.append(f"WARNING: {len(missing_targets)} query targets not in any scheme file: "
                     f"{missing_targets[:10]}")
    lines.append("")
    lines.append(f"{'split':6} {'schemes':>8} {'w/ queries':>11} {'queries':>8}")
    for name in SPLITS:
        sch = [s for s in reg if reg[s]["split"] == name]
        lines.append(f"{name:6} {len(sch):8d} {sum(reg[s]['has_queries'] for s in sch):11d} "
                     f"{len(by_split[name]):8d}")
    # leakage test: no scheme may have queries in more than one split
    seen = defaultdict(set)
    for q in queries:
        if q.get("target_slug"):
            seen[q["target_slug"]].add(q["split"])
    leaks = [s for s, v in seen.items() if len(v) > 1]
    lines.append("")
    lines.append(f"LEAKAGE CHECK (schemes whose queries sit in >1 split): {len(leaks)}  "
                 f"{'OK' if not leaks else 'FAIL'}")
    for name in SPLITS:
        langs = Counter(q["language"] for q in by_split[name])
        lines.append(f"{name} languages: {dict(langs)}")
        types = Counter(q["row_type"] for q in by_split[name])
        lines.append(f"{name} row types: {dict(types)}")
    sha = hashlib.sha256(split_path.read_bytes()).hexdigest()[:16]
    lines.append("")
    lines.append(f"scheme_split_v1.json sha256[:16] = {sha}  (note it down; it must never change)")
    report = "\n".join(lines)
    (out_dir / "registry" / "split_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()