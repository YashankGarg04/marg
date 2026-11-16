"""
Step 5b (part 1): does a RERANKER improve the top results? (no training yet)

Pipeline:  fine-tuned retriever (+ a little BM25)  ->  top-K candidates
           ->  cross-encoder reads (query, scheme card) pairs and re-scores them
           ->  the best 3-4 are the answer.

Run on Colab (GPU), from the repo root, after the retriever is trained:
    python src/rerank_eval.py --model models/marg-retriever --bm25-weight 0.1 --topk 30 --tag ft
    python src/rerank_eval.py --model models/marg-retriever --bm25-weight 0.1 --topk 30 --reranker dummy   # no-GPU glue test

Needs:  pip install sentence-transformers   (reranker: BAAI/bge-reranker-v2-m3, ~2.3 GB download)
Writes: results/retrieval_val_rerank<_tag>.json   (same format as the other result files)
Prints: how often the right scheme is inside the top-K candidates (the ceiling for any reranker).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_retrieval import (BM25, dense_scores, evaluate, read_jsonl, rrf,  # noqa: E402
                            summarize, tokenize)


def dummy_scorer(pairs):
    """Word-overlap scorer, only to test the pipeline without a model."""
    out = []
    for q, d in pairs:
        qt, dt = set(tokenize(q)), set(tokenize(d))
        out.append(len(qt & dt) / (len(qt) + 1))
    return np.array(out, dtype=np.float32)


def fuse_scores(shape, cand, sc, w, k=60):
    """Blend of two opinions inside each query's candidate list:
    1/(k+rank by first stage) + w/(k+rank by reranker).  cand rows are already best-first for the first stage."""
    out = np.full(shape, -1e9, dtype=np.float32)
    for i in range(len(cand)):
        r2 = np.empty(len(cand[i]), dtype=np.int64)
        r2[np.argsort(-sc[i])] = np.arange(len(cand[i]))
        out[i, cand[i]] = 1.0 / (k + np.arange(len(cand[i])) + 1) + w / (k + r2 + 1)
    return out


def load_scorer(name, batch, fp16):
    if name == "dummy":
        return dummy_scorer
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise SystemExit("pip install sentence-transformers")
    ce = CrossEncoder(name, max_length=512)
    if fp16:
        try:
            ce.model.half()
        except Exception as e:                                    # noqa: BLE001
            print("fp16 not available, continuing in fp32:", e)
    return lambda pairs: np.array(ce.predict(pairs, batch_size=batch, show_progress_bar=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--model", default="models/marg-retriever", help="the fine-tuned retriever")
    ap.add_argument("--bm25-weight", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--fuse", nargs="+", type=float, default=[],
                    help="also score blends of first-stage rank and reranker rank, e.g. --fuse 0.25 0.5 1 2 "
                         "(weight of the reranker; 1 = equal). Saved as <name>_f<weight>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    data, out = Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cards = read_jsonl(data / "cards" / "scheme_cards.jsonl")
    slugs = [c["slug"] for c in cards]
    slug_index = {s: i for i, s in enumerate(slugs)}
    queries = [q for q in read_jsonl(data / "splits" / f"{args.split}.jsonl")
               if q["row_type"] in ("standard", "named") and q["target_slug"] in slug_index]
    if args.limit:
        queries = queries[:args.limit]
    print(f"split={args.split}: {len(queries)} queries, {len(cards)} schemes")

    # 1. first-stage candidates
    bm25 = BM25([tokenize(c["text"]) for c in cards])
    bm = np.stack([bm25.scores(tokenize(q["query"])) for q in queries])
    dense = dense_scores(cards, queries, data / "cache" / args.split, args.model, 16)
    first = np.stack([rrf([bm[i], dense[i]], [args.bm25_weight, 1.0]) for i in range(len(queries))])
    cand = np.argsort(-first, axis=1)[:, :args.topk]

    # ceiling: the right scheme must be among the candidates for a reranker to find it
    hit = np.mean([slug_index[q["target_slug"]] in cand[i] for i, q in enumerate(queries)])
    print(f"first stage: right scheme inside the top {args.topk} candidates for {hit:.1%} of queries")

    # 2. rerank
    scorer = load_scorer(args.reranker, args.batch, not args.no_fp16)
    pairs = [(q["query"], cards[j]["text"]) for i, q in enumerate(queries) for j in cand[i]]
    print(f"scoring {len(pairs)} (query, scheme) pairs with {args.reranker} ...")
    sc = np.asarray(scorer(pairs), dtype=np.float32).reshape(len(queries), args.topk)

    final = np.full(first.shape, -1e9, dtype=np.float32)
    for i in range(len(queries)):
        final[i, cand[i]] = sc[i]
    fused_finals = {w: fuse_scores(first.shape, cand, sc, w) for w in args.fuse}

    for w, fin in fused_finals.items():
        per_w = evaluate(fin, queries, slug_index, slugs)
        o = summarize(per_w)["ALL"]
        print(f"blend, reranker weight {w:g}: target@1={o['target@1']:.3f}  @3={o['target@3']:.3f}  "
              f"@10={o['target@10']:.3f}  MRR={o['MRR']:.3f}")
        nm = "rerank" + (f"_{args.tag}" if args.tag else "") + f"_f{w:g}"
        json.dump({"method": nm, "split": args.split, "overall": o, "by_language": summarize(per_w, "language"),
                   "by_noise": summarize(per_w, "noise_level"), "by_row_type": summarize(per_w, "row_type")},
                  open(out / f"retrieval_{args.split}_{nm}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    per = evaluate(final, queries, slug_index, slugs)
    name = "rerank" + (f"_{args.tag}" if args.tag else "")
    result = {"method": name, "split": args.split, "retriever": args.model,
              "bm25_weight": args.bm25_weight, "topk": args.topk, "reranker": args.reranker,
              "candidate_recall": float(hit), "overall": summarize(per)["ALL"],
              "by_language": summarize(per, "language"),
              "by_noise": summarize(per, "noise_level"),
              "by_row_type": summarize(per, "row_type")}
    json.dump(result, open(out / f"retrieval_{args.split}_{name}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n===== RERANK =====")
    print("overall", result["overall"])
    for title, key in (("language", "by_language"), ("noise", "by_noise"), ("row_type", "by_row_type")):
        print(f"-- by {title}")
        for g, v in sorted(result[key].items()):
            print(f"  {g:10} n={v['n']:4d}  target@1={v['target@1']:.3f}  @3={v['target@3']:.3f}  "
                  f"@10={v['target@10']:.3f}  MRR={v['MRR']:.3f}")


if __name__ == "__main__":
    main()