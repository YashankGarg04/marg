"""
Step 9b: does adding the eligibility check help?   python src/eval_pipeline.py [--split val] [--n 600]

Same search for every row (BM25, so it runs anywhere); what changes is what we do with the results:
  search only        : show the top 3 search results
  + query details    : drop schemes that the details in the message rule out
  + form + query     : same, but the person also filled the in-app form (profile known up front)
Measured against the person's TRUE profile and the 1,082 hand-made rules:
  target@3      the scheme the question was written about is in the 3 shown
  any_eligible@3  at least one shown scheme is really one the person qualifies for
  ineligible    share of shown schemes (with known rules) that the person does NOT qualify for
  wrongly dropped  a scheme the person truly qualifies for was removed (must stay ~0)
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eligibility import NO, YES, scheme_status  # noqa: E402
from pipeline import Marg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--language", default="")
    args = ap.parse_args()
    marg = Marg()
    rows = [json.loads(l) for l in open(f"data/splits/{args.split}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["row_type"] in ("standard", "named") and isinstance(r["true_profile"], dict)
            and r["target_slug"] in marg.names and (not args.language or r["language"] == args.language)]
    random.Random(0).shuffle(rows)
    rows = rows[:args.n]
    print(f"{len(rows)} queries ({args.split})")
    gold = marg.known
    modes = ["search only", "+ query details", "+ form + query"]
    stat = {m: dict(t3=0, anyel=0, shown=0, known=0, bad=0, drop=0, droppable=0, yes=0) for m in modes}
    for r in rows:
        text = r["turns"] if r["n_turns"] > 1 else r["query"]
        truth = r["true_profile"]
        cand = marg.searcher.top(" ".join(text) if isinstance(text, list) else text, 30)
        for mode in modes:
            if mode == "search only":
                shown = cand[:3]
            else:
                form = truth if mode == "+ form + query" else None
                shown = [x["slug"] for x in marg.answer(text, form=form)["results"]]
            s = stat[mode]
            s["t3"] += r["target_slug"] in shown
            known_shown = [x for x in shown if x in gold]
            s["shown"] += len(shown)
            s["known"] += len(known_shown)
            s["bad"] += sum(scheme_status(marg.rules[x], truth) == NO for x in known_shown)
            s["anyel"] += any(scheme_status(marg.rules.get(x), truth) == YES for x in shown)
            s["yes"] += sum(scheme_status(marg.rules.get(x), truth) == YES for x in shown)
            if mode != "search only":                              # a truly eligible scheme removed from the candidates?
                prof = None
                from eligibility import ProfileManager
                from read_query import read_query
                pm = ProfileManager()
                if form:
                    pm.set_defaults(form)
                pm.update(read_query(text), turn=1)
                prof = pm.current()
                for x in cand:
                    if x in gold and scheme_status(marg.rules[x], truth) == YES:
                        s["droppable"] += 1
                        s["drop"] += scheme_status(marg.rules[x], prof) == NO
    n = len(rows)
    print(f"\n{'':20}{'target@3':>10}{'any eligible@3':>16}{'ineligible shown':>18}{'wrongly dropped':>17}")
    out = {}
    for m in modes:
        s = stat[m]
        row = {"target@3": s["t3"] / n, "any_eligible@3": s["anyel"] / n,
               "ineligible_share": s["bad"] / max(1, s["known"]),
               "wrongly_dropped": s["drop"] / max(1, s["droppable"]) if m != "search only" else None}
        out[m] = row
        wd = f"{row['wrongly_dropped']:.1%}" if row["wrongly_dropped"] is not None else "-"
        print(f"{m:20}{row['target@3']:10.3f}{row['any_eligible@3']:16.3f}{row['ineligible_share']:18.1%}{wd:>17}")
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open(f"results/pipeline_{args.split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()