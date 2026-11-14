"""
Print result files side by side.

    python src/compare_results.py results/retrieval_val_bm25.json results/retrieval_val_dense.json ^
                                  results/retrieval_val_hybrid.json results/retrieval_val_hybrid_ft.json
"""
import json
import sys
from pathlib import Path

paths = [Path(p) for p in sys.argv[1:]]
if not paths:
    raise SystemExit(__doc__)
res = {p.stem.replace("retrieval_", ""): json.load(open(p, encoding="utf-8")) for p in paths}
names = list(res)
w = max(len(n) for n in names) + 2


def row(label, getter):
    cells = []
    for n in names:
        v = getter(res[n])
        cells.append(f"{v:{w}.3f}" if v is not None else " " * (w - 1) + "-")
    print(f"{label:22}" + "".join(cells))


print(f"{'':22}" + "".join(f"{n:>{w}}" for n in names))
for metric in ("target@1", "target@3", "target@10", "target@30", "any_rel@3", "MRR"):
    row(metric + " (overall)", lambda r, m=metric: r["overall"][m])
print("\ntarget@3 by group")
groups = []
for section in ("by_row_type", "by_noise", "by_language"):
    for g in res[names[0]][section]:
        groups.append((section, g))
for section, g in groups:
    row(f"{g}", lambda r, s=section, g=g: r[s].get(g, {}).get("target@3"))