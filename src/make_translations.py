"""
Translate the non-English queries to English with your friend's translation layer (Yogesh's repo:
IndicTrans2 or NLLB, language ID, number protection, Hinglish transliteration) and save them in the same
format that src/eval_translate.py reads.

Run in its OWN Colab session (his layer needs transformers < 5, our reranker/retriever need a newer one):
    !git clone https://github.com/YogeshBijalwan2211/marg.git /content/yogesh
    !pip install -q "transformers<5" sentencepiece sacrebleu IndicTransToolkit
    (IndicTrans2 models are gated: accept the terms on Hugging Face for ai4bharat/indictrans2-indic-en-dist-200M, then
     from huggingface_hub import login; login())
    !python src/make_translations.py --layer-repo /content/yogesh --backend indictrans2 --split val

Output: data/translations/<split>_en_<backend>.jsonl   ({"query": original, "english": translation})
Then, in the normal session:   python src/eval_translate.py --translations data/translations/val_en_indictrans2.jsonl --tag it2
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-repo", required=True, help="folder of Yogesh's repo (contains janora/translation)")
    ap.add_argument("--backend", default="indictrans2", choices=["indictrans2", "nllb", "fake"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, args.layer_repo)
    from janora.translation import Glossary, TranslationConfig, TranslationLayer
    from janora.translation.langid import label_to_code

    glossary_path = Path(args.layer_repo) / "data" / "registry" / "glossary.json"
    glossary = Glossary.from_json(str(glossary_path)) if glossary_path.exists() else Glossary()
    if args.backend == "fake":                                   # glue test without models
        class Fake:
            name = "fake"
            def translate(self, sentences, src, tgt):            # noqa: E306
                return sentences
        layer = TranslationLayer(TranslationConfig(backend="nllb"), glossary, backend=Fake())
    else:
        layer = TranslationLayer(TranslationConfig(backend=args.backend), glossary)

    rows = [json.loads(l) for l in open(Path(args.data_dir) / "splits" / f"{args.split}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["row_type"] in ("standard", "named") and r["language"] != "english"]
    if args.limit:
        rows = rows[:args.limit]
    out_path = Path(args.data_dir) / "translations" / f"{args.split}_en_{args.backend}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done, issues = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            q = r["query"]
            try:
                res = layer.to_english(q, forced_lang=label_to_code(r["language"], q))
                english = res.english or q
                issues += bool(res.issues)
            except Exception as e:                                # noqa: BLE001  keep going, fall back to the original text
                print("failed:", repr(e)[:100])
                english = q
            f.write(json.dumps({"query": q, "english": english}, ensure_ascii=False) + "\n")
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(rows)}")
    print(f"wrote {out_path}  ({done} queries, {issues} with quality-gate issues)")


if __name__ == "__main__":
    main()