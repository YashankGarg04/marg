"""
Step 9: the whole MARG chain in one place.

    message (any language)  ->  read details  ->  merge with the form  ->  search  ->
    eligibility check  ->  top 3 with reasons and "what to tell me next"

    python src/pipeline.py "I am a 24 year old woman from Bihar, family income 2 lakh, need a scholarship"

The search part is BM25 here so it runs on any laptop; src/eval_pipeline.py shows how much a better
searcher (the fine-tuned one on Colab) adds on top.  Everything else is the same.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eligibility import (FAILED, MAYBE, MET, NO, UNKNOWN, UNVERIFIED, YES, ProfileManager,  # noqa: E402
                         describe, explain, missing_for, scheme_status)
from eval_retrieval import BM25, read_jsonl, tokenize  # noqa: E402
from read_query import read_query  # noqa: E402


SAFE_EXTRA = ("castes", "gender", "disability", "bpl")        # extractor fields precise enough to add to a hand-made rule


def load_rules(data="data", mode="benchmark"):
    """slug -> rule.
    benchmark: the dataset's 1,082 hand-made rules win as they are (this is what the scores are measured against).
    real     : same, plus caste / gender / disability / BPL conditions the extractor found that the hand-made rule lacks
               (e.g. an ST-only scheme whose rule forgot the caste).
    The other ~3,500 schemes always come from the extractor + the scheme's own state."""
    data = Path(data)
    known = {s["id"]: s["rule"] for s in json.load(open(data / "rag_schemes.json", encoding="utf-8"))}
    extracted = json.load(open(data / "rules_extracted.json", encoding="utf-8")) if (data / "rules_extracted.json").exists() else {}
    state = {}
    with open(data / "registry" / "scheme_registry.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["level"] == "state" and r["state"] and r["state"] != "All":
                state[r["slug"]] = r["state"].lower()
    rules = {k: dict(v) for k, v in known.items()}
    if mode == "real":
        for slug in known:
            for k in SAFE_EXTRA:
                if k in extracted.get(slug, {}) and k not in rules[slug]:
                    rules[slug][k] = extracted[slug][k]
    for slug, rule in extracted.items():
        if slug not in rules:
            rules[slug] = dict(rule, **({"state": state[slug]} if slug in state else {}))
    return rules, set(known)


class Searcher:
    def __init__(self, data="data"):
        self.cards = read_jsonl(Path(data) / "cards" / "scheme_cards.jsonl")
        self.slugs = [c["slug"] for c in self.cards]
        self.bm25 = BM25([tokenize(c["text"]) for c in self.cards])

    def top(self, text, k=30):
        import numpy as np
        s = self.bm25.scores(tokenize(text))
        order = np.argsort(-s)[:k]
        return [self.slugs[i] for i in order]


class DenseSearcher:
    """Fine-tuned BGE-M3 + a little BM25 (weighted rank fusion, BM25 weight 0.1 = the best setting we measured).
    Card embeddings are read from data/cache (made by eval_retrieval.py on Colab); queries are encoded on the fly."""

    def __init__(self, model_dir, data="data", bm25_weight=0.1, emb_path=None):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self.np = np
        self.base = Searcher(data)
        self.cards, self.slugs = self.base.cards, self.base.slugs
        tag = Path(model_dir).name
        paths = [Path(emb_path)] if emb_path else sorted(Path(data, "cache").glob(f"*/{tag}_cards_{len(self.cards)}.npy"))
        if not paths:
            raise SystemExit(f"No card embeddings for {tag} in {data}/cache. Run on Colab first:\n"
                             f"  python src/eval_retrieval.py --methods dense --model {model_dir} --tag ft")
        self.cards_emb = np.load(paths[0])
        self.model = SentenceTransformer(str(model_dir))
        self.model.max_seq_length = 128
        self.w = bm25_weight
        self.cache = {}

    def prepare(self, texts, batch=32):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            emb = self.model.encode(todo, batch_size=batch, normalize_embeddings=True, show_progress_bar=True)
            self.cache.update(zip(todo, emb))

    def top(self, text, k=30):
        from eval_retrieval import rrf
        np = self.np
        self.prepare([text])
        dense = self.cards_emb @ self.cache[text]
        bm = self.base.bm25.scores(tokenize(text))
        fused = rrf([bm, dense], [self.w, 1.0])
        return [self.slugs[i] for i in np.argsort(-fused)[:k]]


class Marg:
    def __init__(self, data="data", searcher=None, mode="benchmark"):
        self.mode = mode
        self.rules, self.known = load_rules(data, mode)
        self.searcher = searcher or Searcher(data)
        self.names = {c["slug"]: c["name"] for c in self.searcher.cards}

    def answer(self, message, form=None, k_search=30, k_show=3, profile=None, order="status"):
        """message: text or list of turns. form: dict of defaults. Returns profile, results, questions."""
        pm = profile or ProfileManager()
        if form:
            pm.set_defaults(form)
        found = read_query(message)
        change = pm.update(found, turn=1)
        prof = pm.current()
        text = " ".join(message) if isinstance(message, (list, tuple)) else message
        cand = self.searcher.top(text, k_search)
        out = {"yes": [], "unknown": [], "unverified": []}
        for slug in cand:
            st = scheme_status(self.rules.get(slug), prof)
            if st == NO:
                continue
            key = {YES: "yes", MAYBE: "unknown", UNVERIFIED: "unverified"}[st]
            out[key].append(slug)
        if order == "search":      # keep the search order, only remove schemes the person is ruled out of
            keep = set(out["yes"] + out["unknown"] + out["unverified"])
            ranked = [x for x in cand if x in keep]
        else:                      # confirmed matches first, then ones that need a detail, then unchecked
            ranked = out["yes"] + out["unknown"] + out["unverified"]
        results = []
        for slug in ranked[:k_show]:
            rule = self.rules.get(slug)
            status = scheme_status(rule, prof)
            lines = explain(rule, prof) if rule else []
            results.append({"slug": slug, "name": self.names[slug], "status": status,
                            "why": [t for s, t in lines if s == MET], "check": [t for s, t in lines if s == UNKNOWN],
                            "need": missing_for(rule, prof) if rule else []})
        ask = {}
        for r in results:
            for f in r["need"]:
                ask[f] = ask.get(f, 0) + 1
        return {"profile": prof, "read_from_message": found, "needs_confirmation": change["needs_confirmation"],
                "results": results, "ask_next": [f for f, _ in sorted(ask.items(), key=lambda kv: -kv[1])][:2]}


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or "I am a 24 year old woman from Bihar, family income 2 lakh, need a scholarship"
    res = Marg().answer(msg)
    print("details read:", res["read_from_message"])
    for r in res["results"]:
        print(f"\n[{r['status']}] {r['name']} ({r['slug']})")
        for t in r["why"]:
            print("   ok:", t)
        for t in r["check"]:
            print("   to check:", t)
    print("\nask next:", res["ask_next"])