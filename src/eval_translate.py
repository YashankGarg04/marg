"""
Step 11: does translating non-English queries to English before searching help?

    python src/eval_translate.py --model models/marg-retriever            # Colab GPU, ~10 min
    python src/eval_translate.py --translator dummy --no-dense            # laptop: glue test only

For every non-English validation query it measures three ways of searching (fine-tuned search + 0.1 BM25):
  original   : search with the query as written
  translated : translate to English first (NLLB-200, 600M), then search
  both       : fuse the two result lists
Details (age, state, income ...) are always read from the ORIGINAL text by src/read_query.py; the translation is only
used to find schemes, because the scheme cards are in English.
Writes data/translations/val_en.jsonl (reusable) and results/translate_val.json.
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_retrieval import BM25, dense_scores, evaluate, read_jsonl, rrf, summarize, tokenize  # noqa: E402

NLLB = {"assamese": "asm_Beng", "bengali": "ben_Beng", "gujarati": "guj_Gujr", "hindi": "hin_Deva",
        "hinglish": "hin_Deva", "kannada": "kan_Knda", "malayalam": "mal_Mlym", "marathi": "mar_Deva",
        "nepali": "npi_Deva", "odia": "ory_Orya", "punjabi": "pan_Guru", "tamil": "tam_Taml",
        "telugu": "tel_Telu", "urdu": "urd_Arab"}


def translate_all(rows, translator, out_path, batch=24):
    cache = {}
    if out_path.exists():
        cache = {r["query"]: r["english"] for r in read_jsonl(out_path)}
    todo = defaultdict(list)
    for r in rows:
        if r["query"] not in cache:
            todo[r["language"]].append(r["query"])
    if todo and translator == "dummy":
        for qs in todo.values():
            cache.update({q: q for q in qs})
    elif todo:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = "facebook/nllb-200-distilled-600M"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = (model.half() if dev == "cuda" else model).to(dev).eval()
        bos = tok.convert_tokens_to_ids("eng_Latn")
        for lang, qs in todo.items():
            tok.src_lang = NLLB[lang]
            qs = sorted(qs, key=len)
            print(f"translating {len(qs)} {lang} queries ...")
            for i in range(0, len(qs), batch):
                chunk = qs[i:i + batch]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=160).to(dev)
                with torch.no_grad():
                    out = model.generate(**enc, forced_bos_token_id=bos, num_beams=4, max_new_tokens=160)
                cache.update(zip(chunk, tok.batch_decode(out, skip_special_tokens=True)))
        del model
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q, e in cache.items():
            f.write(json.dumps({"query": q, "english": e}, ensure_ascii=False) + "\n")
    return cache


def first_stage(cards, queries, cache_dir, args):
    bm25 = BM25([tokenize(c["text"]) for c in cards])
    bm = np.stack([bm25.scores(tokenize(q["query"])) for q in queries])
    if args.no_dense:
        return bm
    dense = dense_scores(cards, queries, cache_dir, args.model, 16)
    return np.stack([rrf([bm[i], dense[i]], [args.bm25_weight, 1.0]) for i in range(len(queries))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--model", default="models/marg-retriever")
    ap.add_argument("--bm25-weight", type=float, default=0.1)
    ap.add_argument("--translator", default="nllb", choices=["nllb", "dummy"])
    ap.add_argument("--translations", default="", help="use this ready-made file of translations (from src/make_translations.py)")
    ap.add_argument("--tag", default="", help="suffix for the results file, e.g. it2")
    ap.add_argument("--no-dense", action="store_true", help="BM25 only (laptop glue test)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    data = Path(args.data_dir)
    cards = read_jsonl(data / "cards" / "scheme_cards.jsonl")
    slugs = [c["slug"] for c in cards]
    slug_index = {s: i for i, s in enumerate(slugs)}
    rows = [q for q in read_jsonl(data / "splits" / f"{args.split}.jsonl")
            if q["row_type"] in ("standard", "named") and q["target_slug"] in slug_index and q["language"] != "english"]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} non-English {args.split} queries")
    tr_path = Path(args.translations) if args.translations else data / "translations" / f"{args.split}_en.jsonl"
    tr = translate_all(rows, "dummy" if args.translations else args.translator, tr_path)
    rows_tr = [dict(r, query=tr[r["query"]]) for r in rows]

    cdir = data / "cache"
    dirs = {}
    for tag, rs in (("orig", rows), ("tr", rows_tr)):
        d = cdir / f"{args.split}_nonen_{tag}" if tag == "orig" else cdir / f"{args.split}_nonen_tr{('_' + args.tag) if args.tag else ''}"
        d.mkdir(parents=True, exist_ok=True)
        for f in (cdir / args.split).glob("*_cards_*.npy"):            # reuse card embeddings
            if not (d / f.name).exists():
                shutil.copy(f, d / f.name)
        dirs[tag] = d
    A = first_stage(cards, rows, dirs["orig"], args)
    B = first_stage(cards, rows_tr, dirs["tr"], args)
    C = np.stack([rrf([A[i], B[i]], [1.0, 1.0]) for i in range(len(rows))])
    results = {}
    print(f"\n{'':12}{'target@1':>9}{'@3':>7}{'@10':>7}{'@30':>7}{'MRR':>7}{'any_rel@3':>11}")
    for name, S in (("original", A), ("translated", B), ("both", C)):
        per = evaluate(S, rows, slug_index, slugs)
        o = summarize(per)["ALL"]
        results[name] = {"overall": o, "by_language": summarize(per, "language")}
        print(f"{name:12}{o['target@1']:9.3f}{o['target@3']:7.3f}{o['target@10']:7.3f}{o['target@30']:7.3f}"
              f"{o['MRR']:7.3f}{o['any_rel@3']:11.3f}")
    print(f"\ntarget@3 by language   {'original':>10}{'translated':>12}{'both':>8}")
    for lang in sorted(results["original"]["by_language"]):
        v = [results[k]["by_language"][lang] for k in ("original", "translated", "both")]
        print(f"  {lang:10} n={v[0]['n']:3d}  {v[0]['target@3']:10.3f}{v[1]['target@3']:12.3f}{v[2]['target@3']:8.3f}")
    Path("results").mkdir(exist_ok=True)
    json.dump(results, open(f"results/translate_{args.split}{('_' + args.tag) if args.tag else ''}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()