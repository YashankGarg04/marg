"""Score src/read_query.py on a split.   python src/eval_reader.py [--split val] [--show 15]"""
import argparse, json, sys, random
from collections import defaultdict, Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_query import read_query

FIELDS = ["age", "gender", "state", "caste", "income", "occupation", "disability", "bpl", "residence"]
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="val")
ap.add_argument("--show", type=int, default=0, help="print N wrong examples per field")
ap.add_argument("--language", default="")
args = ap.parse_args()
rows = [json.loads(l) for l in open(Path("data/splits") / f"{args.split}.jsonl", encoding="utf-8")]
rows = [r for r in rows if r["row_type"] in ("standard", "named", "vague") and isinstance(r["true_profile"], dict)]
if args.language:
    rows = [r for r in rows if r["language"] == args.language]
print(f"{len(rows)} queries")


def score(group_rows):
    st = {f: Counter() for f in FIELDS}
    for r in group_rows:
        got = read_query(r["turns"] if r["n_turns"] > 1 else r["query"])
        tp = r["true_profile"]
        for f in FIELDS:
            stated = f not in r["missing_fields"]
            if f in ("disability", "bpl"):                    # only 'yes' is ever stated
                truth = tp[f] is True
                st[f]["should"] += truth
                st[f]["said"] += f in got
                st[f]["right"] += f in got and truth
                st[f]["wrong"] += f in got and not truth
                continue
            truth = tp[f]
            st[f]["should"] += stated and truth is not None
            if f in got:
                st[f]["said"] += 1
                ok = got[f] == truth
                st[f]["right"] += ok and stated
                st[f]["wrong"] += (not ok) and stated
                st[f]["invented"] += not stated and not ok
    return st


def show(title, st):
    print(f"\n{title}")
    print(f"{'field':11}{'stated':>7}{'said':>6}{'right':>6}{'wrong':>6}{'found %':>9}{'wrong %':>9}")
    for f in FIELDS:
        s = st[f]
        print(f"{f:11}{s['should']:7}{s['said']:6}{s['right']:6}{s['wrong']:6}"
              f"{100 * s['right'] / max(1, s['should']):9.1f}{100 * s['wrong'] / max(1, s['said']):9.1f}")


show("ALL", score(rows))
groups = defaultdict(list)
for r in rows:
    groups[r["language"]].append(r)
print("\nfound % by language / wrong % (all fields together)")
for lang, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    st = score(g)
    should = sum(s["should"] for s in st.values()); right = sum(s["right"] for s in st.values())
    said = sum(s["said"] for s in st.values()); wrong = sum(s["wrong"] for s in st.values())
    print(f"  {lang:11} n={len(g):4}  found {100 * right / max(1, should):5.1f}%   wrong {100 * wrong / max(1, said):5.1f}%")
noise = defaultdict(list)
for r in rows:
    noise[r["noise_level"]].append(r)
print("\nby noise (English only)")
for k, g in noise.items():
    g = [r for r in g if r["language"] == "english"]
    st = score(g)
    should = sum(s["should"] for s in st.values()); right = sum(s["right"] for s in st.values())
    said = sum(s["said"] for s in st.values()); wrong = sum(s["wrong"] for s in st.values())
    print(f"  {k:7} n={len(g):4}  found {100 * right / max(1, should):5.1f}%   wrong {100 * wrong / max(1, said):5.1f}%")
if args.show:
    rng = random.Random(1)
    bad = defaultdict(list)
    for r in rows:
        got = read_query(r["turns"] if r["n_turns"] > 1 else r["query"])
        for f in FIELDS:
            if f in got and f not in r["missing_fields"] and f not in ("disability", "bpl") and got[f] != r["true_profile"][f]:
                bad[f].append((r["query"][:230], got[f], r["true_profile"][f]))
    for f, lst in bad.items():
        print(f"\n--- wrong {f} ({len(lst)})")
        for q, g, t in rng.sample(lst, min(args.show, len(lst))):
            print(f"  got={g!r} true={t!r} | {q}")