"""
Step 5a: fine-tune the retriever (BGE-M3) on your TRAIN queries.

What it learns: a query (any language) should land close to the card of the scheme it was
written about, and far from look-alike schemes (hard negatives).

Run from the repo root, on a GPU machine (Colab / Kaggle):
    pip install sentence-transformers
    python src/train_retriever.py --dry-run                        # no GPU needed: checks the data
    python src/train_retriever.py --limit 160 --epochs 1 --out models/smoke   # 2-minute smoke test
    python src/train_retriever.py --out models/marg-retriever               # the real run

Then score it exactly like the baselines:
    python src/eval_retrieval.py --methods dense hybrid --model models/marg-retriever --tag ft

Training data
  * only rows of the TRAIN split with row_type standard or named and a known target scheme
  * positive  = the target scheme's card
  * hard negative = a look-alike scheme from the TRAIN schemes only (never val/test schemes):
      English queries : a BM25 near-miss (rank 3..20 for that query, not the target)
      other languages : a scheme with the same category (and state if possible)
  * in-batch negatives too (the other positives in the batch), with batches built so the same
    scheme never appears twice in one batch (it would count as its own negative).

If you run out of GPU memory: --batch 8 --max-seq 192, and keep --grad-ckpt on.
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_retrieval import BM25, read_jsonl, tokenize  # noqa: E402


# ------------------------------------------------------------------ data
def build_examples(data, seed, neg_rank_min, neg_rank_max, limit):
    rng = random.Random(seed)
    cards = {c["slug"]: c for c in read_jsonl(data / "cards" / "scheme_cards.jsonl")}
    reg = {r["slug"]: r for r in csv.DictReader(
        open(data / "registry" / "scheme_registry.csv", encoding="utf-8-sig"))}
    train_slugs = [s for s in cards if reg[s]["split"] == "train"]
    train_set = set(train_slugs)

    by_cat = defaultdict(list)           # (category, state) and category -> train slugs
    by_cat_only = defaultdict(list)
    for s in train_slugs:
        by_cat[(reg[s]["category"], reg[s]["state"])].append(s)
        by_cat_only[reg[s]["category"]].append(s)

    bm25 = BM25([tokenize(cards[s]["text"]) for s in train_slugs])

    def category_negative(pos):
        r = reg[pos]
        for pool in (by_cat.get((r["category"], r["state"]), []),
                     by_cat_only.get(r["category"], []), train_slugs):
            options = [s for s in pool if s != pos]
            if options:
                return rng.choice(options)

    rows = [q for q in read_jsonl(data / "splits" / "train.jsonl")
            if q["row_type"] in ("standard", "named")
            and q["target_slug"] in train_set]
    seen, examples = set(), []
    for q in rows:
        key = (q["query"], q["target_slug"])
        if key in seen:
            continue
        seen.add(key)
        pos = q["target_slug"]
        neg = None
        if q["language"] == "english":
            sc = bm25.scores(tokenize(q["query"]))
            top = [train_slugs[i] for i in np.argsort(-sc)[:neg_rank_max]]
            cand = [s for s in top[neg_rank_min - 1:] if s != pos]
            if cand:
                neg = rng.choice(cand)
        if neg is None:
            neg = category_negative(pos)
        examples.append({"query": q["query"], "pos": pos, "neg": neg,
                         "language": q["language"]})
    rng.shuffle(examples)
    if limit:
        examples = examples[:limit]
    texts = {s: cards[s]["text"] for s in cards}
    return examples, texts


def make_batches(examples, batch_size, rng):
    """Greedy batches where no scheme appears twice and no negative equals another positive."""
    pending = examples[:]
    rng.shuffle(pending)
    batches = []
    while pending:
        batch, rest, pos_set, neg_set = [], [], set(), set()
        for ex in pending:
            ok = (len(batch) < batch_size and ex["pos"] not in pos_set
                  and ex["pos"] not in neg_set and ex["neg"] not in pos_set)
            if ok:
                batch.append(ex)
                pos_set.add(ex["pos"])
                neg_set.add(ex["neg"])
            else:
                rest.append(ex)
        if not batch:                       # safety: never loop forever
            batch, rest = [rest[0]], rest[1:]
        batches.append(batch)
        pending = rest
    return [b for b in batches if len(b) >= max(2, batch_size // 4)]


# ------------------------------------------------------------------ training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--out", default="models/marg-retriever")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--scale", type=float, default=20.0, help="similarity scale (1/temperature)")
    ap.add_argument("--max-seq", type=int, default=256, help="max tokens of a scheme card")
    ap.add_argument("--max-q", type=int, default=96, help="max tokens of a query")
    ap.add_argument("--neg-rank-min", type=int, default=3)
    ap.add_argument("--neg-rank-max", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only N examples")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="only build and describe the data")
    ap.add_argument("--allow-cpu", action="store_true", help="run on CPU (only for a tiny test)")
    args = ap.parse_args()

    t0 = time.time()
    data = Path(args.data_dir)
    examples, texts = build_examples(data, args.seed, args.neg_rank_min,
                                     args.neg_rank_max, args.limit)
    rng = random.Random(args.seed)
    batches = make_batches(examples, args.batch, rng)
    langs = defaultdict(int)
    for e in examples:
        langs[e["language"]] += 1
    print(f"training examples: {len(examples)}   batches/epoch: {len(batches)} "
          f"(batch {args.batch})   built in {time.time() - t0:.0f}s")
    print("by language:", dict(sorted(langs.items())))
    ex = examples[0]
    print("\nexample query   :", ex["query"][:140])
    print("positive card   :", texts[ex["pos"]].split("\n")[0])
    print("hard negative   :", texts[ex["neg"]].split("\n")[0])
    if args.dry_run:
        return

    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise SystemExit(
            "No usable GPU for PyTorch (torch %s, cuda available: False).\n"
            "Check:  nvidia-smi   and   python -c \"import torch; print(torch.__version__)\"\n"
            "A version ending in '+cpu' means the CPU-only PyTorch is installed.\n"
            "Easiest fix: run this step on Google Colab or Kaggle (free GPU).\n"
            "Add --allow-cpu only to test the code on a handful of examples." % torch.__version__)
    if device == "cuda":
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}  ({gb:.1f} GB memory)")
    model = SentenceTransformer(args.model, device=device)
    if not args.no_grad_ckpt:
        try:
            model[0].auto_model.gradient_checkpointing_enable()
        except Exception as e:                                   # noqa: BLE001
            print("gradient checkpointing not available:", e)
    use_fp16 = device == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(batches)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))

    def embed(strings, max_len):
        model.max_seq_length = max_len
        tok = model.preprocess(strings) if hasattr(model, "preprocess") else model.tokenize(strings)
        feats = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in tok.items()}
        return F.normalize(model(feats)["sentence_embedding"], dim=-1)

    step, running = 0, []
    for epoch in range(args.epochs):
        model.train()
        for batch in make_batches(examples, args.batch, random.Random(args.seed + epoch)):
            q_texts = [e["query"] for e in batch]
            d_texts = [texts[e["pos"]] for e in batch] + [texts[e["neg"]] for e in batch]
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_fp16):
                q = embed(q_texts, args.max_q)
                d = embed(d_texts, args.max_seq)
                scores = (q @ d.T).float() * args.scale
                loss = F.cross_entropy(scores, torch.arange(len(batch), device=device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running.append(loss.item())
            if step % 25 == 0 or step == 1:
                print(f"epoch {epoch + 1}/{args.epochs}  step {step}/{total}  "
                      f"loss {np.mean(running[-25:]):.4f}  ({time.time() - t0:.0f}s)")

    model.eval()
    model.max_seq_length = 512
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    json.dump({"base_model": args.model, "examples": len(examples), "epochs": args.epochs,
               "batch": args.batch, "lr": args.lr, "scale": args.scale,
               "max_seq": args.max_seq, "seed": args.seed, "final_loss": float(np.mean(running[-25:]))},
              open(Path(args.out) / "marg_train_config.json", "w"), indent=1)
    print(f"\nsaved to {args.out}  (do NOT commit this folder to git)")


if __name__ == "__main__":
    main()