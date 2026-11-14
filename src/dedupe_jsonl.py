"""Keep one row per slug in a .jsonl file (the LAST copy wins).
Usage:  python src/dedupe_jsonl.py data/en/schemes_detail.jsonl
        python src/dedupe_jsonl.py data/en/schemes_list.jsonl
"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = {}
n = 0
with path.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                      # half-written last line after a crash
        n += 1
        rows[r.get("slug")] = r
with path.open("w", encoding="utf-8", newline="\n") as f:
    for r in rows.values():
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{path}: {n} rows -> {len(rows)} unique slugs")