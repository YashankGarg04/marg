"""
Step 10: fine-tune the cross-encoder reranker (BAAI/bge-reranker-v2-m3) on your TRAIN split.

Why: the off-the-shelf reranker made search worse (target@3 0.658 -> 0.620) because it was never taught what a
good answer looks like for these queries and cards. Here it sees, for each training query, the right scheme
plus 7 look-alike wrong schemes, and learns to put the right one first (softmax over the group of 8).

Run on Colab (T4), from the repo root, after the retriever is trained:
    python src/train_reranker.py --retriever models/marg-retriever            # ~30-45 min
    python src/train_reranker.py --dry-run --pool bm25                        # no GPU: only builds and describes the data
Then score it on the full index (same table as before):
    python src/rerank_eval.py --model models/marg-retriever --bm25-weight 0.1 --topk 30 --reranker models/marg-reranker --tag ftrr

Leak rules: only TRAIN queries and TRAIN scheme cards are used for training. Hard negatives come from the tuned
retriever's own top results and never include the target nor any scheme the person is eligible for
(relevant_slugs / eligible_same_category_slugs), because those are also valid answers. A small slice of the
VALIDATION split (cards of val schemes only) picks the best epoch; the test split is never touched.
"""
import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_retrieval import BM25, read_jsonl, tokenize  # noqa: E402

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load(data, split, cards_by_split):
    rows = [q for q in read_jsonl(Path(data) / "splits" / f"{split}.jsonl")
            if q["row_type"] in ("standard", "named") and q["target_slug"] in cards_by_split]
    return rows


def pool_bm25(queries, cards, k):
    bm = BM25([tokenize(c["text"]) for c in cards])
    return [list(np.argsort(-bm.scores(tokenize(q["query"])))[:k]) for q in queries]


def pool_dense(queries, cards, retriever, k, bs=64):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(retriever)
    st.max_seq_length = 256
    D = st.encode([c["text"] for c in cards], batch_size=bs, normalize_embeddings=True, show_progress_bar=True)
    Q = st.encode([q["query"] for q in queries], batch_size=bs, normalize_embeddings=True, show_progress_bar=True)
    sims = Q @ D.T
    out = [list(np.argsort(-row)[:k]) for row in sims]
    del st
    return out


def make_groups(queries, cards, pools, group, rng, take_from=15):
    """[positive, neg1 ... neg(group-1)] per query, as card indices."""
    idx = {c["slug"]: i for i, c in enumerate(cards)}
    groups = []
    for q, pool in zip(queries, pools):
        banned = {q["target_slug"]} | set(q.get("relevant_slugs") or []) | set(q.get("eligible_same_category_slugs") or [])
        negs = [j for j in pool if cards[j]["slug"] not in banned][:take_from]
        rng.shuffle(negs)
        negs = negs[:group - 1]
        while len(negs) < group - 1:                                  # rare: pool too small -> random fill
            j = rng.randrange(len(cards))
            if cards[j]["slug"] not in banned and j not in negs:
                negs.append(j)
        groups.append([idx[q["target_slug"]]] + negs)
    return groups


def freeze_lower(model, n):
    frozen = 0
    for name, p in model.named_parameters():
        low = name.lower()
        if "embeddings" in low or any(f"encoder.layer.{i}." in low for i in range(n)):
            p.requires_grad = False
            frozen += 1
    return frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--retriever", default="models/marg-retriever")
    ap.add_argument("--base", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", default="models/marg-reranker")
    ap.add_argument("--pool", default="dense", choices=["dense", "bm25"], help="where hard negatives come from")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--group", type=int, default=8, help="1 right + (group-1) wrong schemes per query")
    ap.add_argument("--qps", type=int, default=4, help="queries per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--freeze-layers", type=int, default=8)
    ap.add_argument("--val-n", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N training queries")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    cards = read_jsonl(Path(args.data_dir) / "cards" / "scheme_cards.jsonl")
    tr_cards = [c for c in cards if c["split"] == "train"]
    va_cards = [c for c in cards if c["split"] == "val"]
    qtr = load(args.data_dir, "train", {c["slug"] for c in tr_cards})
    qva = load(args.data_dir, "val", {c["slug"] for c in va_cards})
    rng.shuffle(qva)
    qva = qva[:args.val_n]
    if args.limit:
        qtr = qtr[:args.limit]
    print(f"train queries={len(qtr)} over {len(tr_cards)} train cards | val subset={len(qva)} over {len(va_cards)} val cards")

    pool_fn = (lambda qs, cs, k: pool_bm25(qs, cs, k)) if args.pool == "bm25" else \
              (lambda qs, cs, k: pool_dense(qs, cs, args.retriever, k))
    tr_pool = pool_fn(qtr, tr_cards, 30)
    groups = make_groups(qtr, tr_cards, tr_pool, args.group, rng)
    if args.dry_run:
        g = groups[0]
        print("\nexample query:", qtr[0]["query"][:160])
        print("  right  :", tr_cards[g[0]]["name"])
        for j in g[1:4]:
            print("  wrong  :", tr_cards[j]["name"])
        hit = np.mean([q["target_slug"] in [tr_cards[j]["slug"] for j in p[:3]] for q, p in zip(qtr, tr_pool)])
        print(f"\nfirst stage ({args.pool}) finds the right train scheme in its top 3 for {hit:.1%} of train queries")
        print(f"{len(groups)} groups of {args.group}  ->  {math.ceil(len(groups) / args.qps)} steps per epoch")
        return

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("No GPU found. Use a Colab T4 (Runtime > Change runtime type).")
    va_pool = pool_fn(qva, va_cards, 30)
    va_gold = [q["target_slug"] for q in qva]
    va_cand = [[va_cards[j]["slug"] for j in p] for p in va_pool]
    base_r3 = float(np.mean([g in c[:3] for g, c in zip(va_gold, va_cand)]))
    print(f"first stage alone on the val subset: R@3 = {base_r3:.3f}")
    text = {c["slug"]: c["text"] for c in cards}

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base).to(dev)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    print("frozen tensors:", freeze_lower(model, args.freeze_layers))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.0f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(groups) / args.qps)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda")

    def score(qs, ts):
        enc = tok(qs, ts, truncation="only_second", max_length=args.max_len, padding=True, return_tensors="pt").to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(**enc).logits.float().squeeze(-1)

    def val_r3():
        model.eval()
        hits = []
        with torch.no_grad():
            for q, cand, gold in zip(qva, va_cand, va_gold):
                s = torch.cat([score([q["query"]] * len(c), [text[x] for x in c]) for c in [cand[:30]]]).cpu().numpy()
                hits.append(gold in [cand[j] for j in np.argsort(-s)[:3]])
        model.train()
        return float(np.mean(hits))

    best, step = -1.0, 0
    for ep in range(args.epochs):
        model.train()
        order = list(range(len(groups)))
        rng.shuffle(order)
        run = 0.0
        for b in range(0, len(order), args.qps):
            ids = order[b:b + args.qps]
            qs = [qtr[i]["query"] for i in ids for _ in range(args.group)]
            ts = [tr_cards[j]["text"] for i in ids for j in groups[i]]
            logits = score(qs, ts).view(len(ids), args.group)
            loss = F.cross_entropy(logits, torch.zeros(len(ids), dtype=torch.long, device=dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run += loss.item()
            if step % 100 == 0:
                print(f"  epoch {ep + 1} step {step}/{steps}  loss {run / 100:.4f}")
                run = 0.0
        r3 = val_r3()
        print(f"epoch {ep + 1}: reranked val R@3 = {r3:.3f}   (first stage alone {base_r3:.3f})")
        if r3 > best:
            best = r3
            Path(args.out).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
            print("  saved ->", args.out)
    print("best reranked val R@3:", round(best, 4))


if __name__ == "__main__":
    main()