"""
Step 7a: test the eligibility engine on your queries.

    python src/eval_eligibility.py                 # validation split
    python src/eval_eligibility.py --split test    # only at the very end

Three experiments (all use the true_profile of each query and the rules in data/rag_schemes.json):
  A. Full profile       -> does the engine reproduce the dataset's eligible lists exactly?
  B. Hide one detail    -> how many schemes become "unknown", and does a truly eligible
                           scheme ever get a wrong "no"?   (it must never)
  C. Ask questions      -> start knowing only the state, ask the most useful question each
                           time; how fast does the list of possible schemes shrink?

Writes results/eligibility_<split>.json
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eligibility import (FIELDS, MAYBE, NO, YES, best_questions,  # noqa: E402
                         load_schemes, scheme_status)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def statuses(rules, profile):
    return {slug: scheme_status(rule, profile) for slug, rule in rules.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--sample", type=int, default=300, help="queries used in experiments B and C")
    args = ap.parse_args()
    data = Path(args.data_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    schemes = load_schemes(data / "rag_schemes.json")
    rules = {s: v["rule"] for s, v in schemes.items()}
    rows = [r for r in read_jsonl(data / "splits" / f"{args.split}.jsonl")
            if r["row_type"] == "standard" and isinstance(r["true_profile"], dict)]
    print(f"split={args.split}: {len(rows)} queries with a profile, {len(rules)} schemes with rules")
    result = {"split": args.split, "queries": len(rows), "schemes_with_rules": len(rules)}

    # ---------------------------------------------------------------- A
    exact_total = exact_cat = 0
    tp = fp = fn = 0
    for r in rows:
        st = statuses(rules, r["true_profile"])
        yes = {s for s, v in st.items() if v == YES}
        exact_total += len(yes) == r["n_eligible_total"]
        yes_cat = {s for s in yes if schemes[s]["category"] == r["category"]}
        gold = set(r["eligible_same_category_slugs"])
        exact_cat += yes_cat == gold
        tp += len(yes_cat & gold)
        fp += len(yes_cat - gold)
        fn += len(gold - yes_cat)
    n = len(rows)
    result["A_full_profile"] = {
        "eligible_count_matches": exact_total / n, "same_category_list_matches": exact_cat / n,
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
    print("\nA. Full profile")
    print(f"   eligible count matches the dataset      : {exact_total}/{n}")
    print(f"   same-category list matches exactly      : {exact_cat}/{n}")
    print(f"   precision {tp / max(1, tp + fp):.3f}   recall {tp / max(1, tp + fn):.3f}")

    # ---------------------------------------------------------------- B
    rng = random.Random(42)
    sample = rng.sample(rows, min(args.sample, len(rows)))
    print(f"\nB. Hide one detail ({len(sample)} queries, average over the {len(rules)} schemes)")
    print(f"   {'hidden':12} {'yes':>7} {'unknown':>8} {'no':>7}   truly-eligible kept   wrong yes")
    B = {}
    for hidden in [None] + FIELDS:
        sums = Counter()
        kept = total_elig = wrong_yes = 0
        for r in sample:
            full = r["true_profile"]
            part = dict(full)
            if hidden:
                part[hidden] = None
            truth = statuses(rules, full)
            st = statuses(rules, part)
            for s, v in st.items():
                sums[v] += 1
                if truth[s] == YES:
                    total_elig += 1
                    kept += v in (YES, MAYBE)
                if v == YES and truth[s] != YES:
                    wrong_yes += 1
        k = len(sample)
        name = hidden or "(nothing)"
        B[name] = {"yes": sums[YES] / k, "unknown": sums[MAYBE] / k, "no": sums[NO] / k,
                   "truly_eligible_kept": kept / max(1, total_elig), "wrong_yes": wrong_yes}
        print(f"   {name:12} {sums[YES] / k:7.1f} {sums[MAYBE] / k:8.1f} {sums[NO] / k:7.1f}"
              f"   {kept / max(1, total_elig):8.1%}            {wrong_yes}")
    result["B_hide_one"] = B

    # ---------------------------------------------------------------- C
    print(f"\nC. Ask the most useful question each time (start: only the state is known)")
    steps = defaultdict(lambda: Counter())
    first_q = Counter()
    for r in sample:
        full = r["true_profile"]
        known = {"state": full["state"]}
        for step in range(0, 6):
            st = statuses(rules, known)
            c = Counter(st.values())
            steps[step]["yes"] += c[YES]
            steps[step]["unknown"] += c[MAYBE]
            steps[step]["no"] += c[NO]
            q = best_questions(rules, known)
            if step == 0 and q:
                first_q[q[0][0]] += 1
            if not q:
                for later in range(step + 1, 6):          # nothing left to ask
                    steps[later]["yes"] += c[YES]
                    steps[later]["unknown"] += c[MAYBE]
                    steps[later]["no"] += c[NO]
                break
            known[q[0][0]] = full[q[0][0]]
    k = len(sample)
    print(f"   {'questions asked':16} {'yes':>7} {'unknown':>8} {'no':>7}   (average number of schemes)")
    C = {}
    for step in range(0, 6):
        s = steps[step]
        C[step] = {"yes": s["yes"] / k, "unknown": s["unknown"] / k, "no": s["no"] / k}
        print(f"   {step:<16} {s['yes'] / k:7.1f} {s['unknown'] / k:8.1f} {s['no'] / k:7.1f}")
    result["C_questions"] = C
    result["C_first_question"] = dict(first_q)
    print("   first question asked:", dict(first_q.most_common()))

    json.dump(result, open(Path(args.out_dir) / f"eligibility_{args.split}.json", "w",
                           encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()