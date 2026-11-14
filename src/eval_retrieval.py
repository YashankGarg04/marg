"""
Step 4b: retrieval baselines - BM25, dense (BGE-M3) and hybrid - scored on your split.

Run from the repo root (validation split; the test split is for the very end):
    python src/eval_retrieval.py --methods bm25
    python src/eval_retrieval.py --methods bm25 dense hybrid        # needs sentence-transformers

Dense needs:  pip install sentence-transformers
              (first run downloads BAAI/bge-m3, ~2.3 GB; use a GPU such as Colab/Kaggle if you can,
               on CPU it can take 30-60 minutes. Embeddings are cached in data/cache/.)

Writes: results/retrieval_<split>_<method>.json  and prints a summary table.

How it is scored
  * The search index holds ALL schemes (every card), so the system has to find the needle
    among ~5,000 schemes, not only among the split's schemes.
  * Queries scored: only those of the chosen split with row_type standard / named.
  * target hit@k : is the scheme the query was written about (target_slug) in the top k?
  * any-relevant : is ANY scheme from relevant_slugs (schemes the person is eligible for in
                   that category) in the top k?
  * MRR          : 1 / rank of the target scheme.
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

KS = (1, 3, 10)
_SPLIT_RE = re.compile(r"[\s\.,;:!?()\[\]{}\"'“”‘’/\\|\-–—_@#%&*+=<>~`^]+")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def tokenize(text):
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


# --------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.dl.mean())
        self.post = defaultdict(list)                    # term -> [(doc, tf)]
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post[t].append((i, c))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.post.items()}

    def scores(self, query_tokens):
        s = np.zeros(self.N, dtype=np.float32)
        for t in set(query_tokens):
            if t not in self.post:
                continue
            idf = self.idf[t]
            for i, tf in self.post[t]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg)
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s


# -------------------------------------------------------------------- dense
def dense_scores(cards, queries, cache_dir, model_name, batch):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("Dense search needs:  pip install sentence-transformers")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = model_name.split("/")[-1]
    model = None

    def encode(texts, name, max_len):
        nonlocal model
        path = cache_dir / f"{tag}_{name}_{len(texts)}.npy"
        if path.exists():
            return np.load(path)
        if model is None:
            model = SentenceTransformer(model_name)
        model.max_seq_length = max_len
        print(f"encoding {len(texts)} {name} ...")
        emb = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                           show_progress_bar=True)
        np.save(path, emb)
        return emb

    card_emb = encode([c["text"] for c in cards], "cards", 512)
    q_emb = encode([q["query"] for q in queries], "queries", 128)
    return q_emb @ card_emb.T                            # cosine similarity matrix


# -------------------------------------------------------------------- metrics
def rrf(rank_lists, k=60):
    """Reciprocal rank fusion of several score arrays -> one score array."""
    fused = np.zeros_like(rank_lists[0])
    for s in rank_lists:
        order = np.argsort(-s)
        r = np.empty_like(order)
        r[order] = np.arange(len(order))
        fused += 1.0 / (k + r + 1)
    return fused


def evaluate(score_rows, queries, slug_index, slugs):
    per = []
    for sc, q in zip(score_rows, queries):
        order = np.argsort(-sc)[:100]
        ranked = [slugs[i] for i in order]
        tgt = q["target_slug"]
        rel = set(q["relevant_slugs"])
        rank = ranked.index(tgt) + 1 if tgt in ranked else None
        any_rank = next((i + 1 for i, s in enumerate(ranked) if s in rel), None)
        per.append({
            "language": q["language"], "noise_level": q["noise_level"],
            "row_type": q["row_type"], "rank": rank, "any_rank": any_rank,
        })
    return per


def summarize(per, key=None):
    groups = defaultdict(list)
    for p in per:
        groups[p[key] if key else "ALL"].append(p)
    out = {}
    for g, rows in groups.items():
        n = len(rows)
        m = {"n": n}
        for k in KS:
            m[f"target@{k}"] = round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n, 3)
        m["any_rel@3"] = round(sum(1 for r in rows if r["any_rank"] and r["any_rank"] <= 3) / n, 3)
        m["MRR"] = round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3)
        out[g] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--methods", nargs="+", default=["bm25"],
                    choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N queries")
    args = ap.parse_args()
    data = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cards = read_jsonl(data / "cards" / "scheme_cards.jsonl")
    slugs = [c["slug"] for c in cards]
    slug_index = {s: i for i, s in enumerate(slugs)}
    queries = [q for q in read_jsonl(data / "splits" / f"{args.split}.jsonl")
               if q["row_type"] in ("standard", "named") and q["target_slug"] in slug_index]
    if args.limit:
        queries = queries[:args.limit]
    print(f"split={args.split}: {len(queries)} queries, {len(cards)} schemes in the index")

    methods = set(args.methods)
    bm = dense = None
    if methods & {"bm25", "hybrid"}:
        bm25 = BM25([tokenize(c["text"]) for c in cards])
        bm = np.stack([bm25.scores(tokenize(q["query"])) for q in queries])
    if methods & {"dense", "hybrid"}:
        dense = dense_scores(cards, queries, data / "cache" / args.split,
                             args.model, args.batch)

    for m in args.methods:
        if m == "bm25":
            rows = bm
        elif m == "dense":
            rows = dense
        else:
            rows = np.stack([rrf([bm[i], dense[i]]) for i in range(len(queries))])
        per = evaluate(rows, queries, slug_index, slugs)
        result = {"method": m, "split": args.split, "overall": summarize(per)["ALL"],
                  "by_language": summarize(per, "language"),
                  "by_noise": summarize(per, "noise_level"),
                  "by_row_type": summarize(per, "row_type")}
        json.dump(result, open(out / f"retrieval_{args.split}_{m}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n===== {m.upper()}  ({args.split}) =====")
        print("overall   ", result["overall"])
        for title, key in (("language", "by_language"), ("noise", "by_noise"),
                           ("row_type", "by_row_type")):
            print(f"-- by {title}")
            for g, v in sorted(result[key].items()):
                print(f"  {g:10} n={v['n']:4d}  target@1={v['target@1']:.3f}  "
                      f"@3={v['target@3']:.3f}  @10={v['target@10']:.3f}  "
                      f"any_rel@3={v['any_rel@3']:.3f}  MRR={v['MRR']:.3f}")


if __name__ == "__main__":
    main()