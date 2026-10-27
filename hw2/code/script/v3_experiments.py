"""Controlled experiment registry. Each parent -> child changes one method family."""
from common import ROOT, read_json


def experiments():
    base = read_json(ROOT / "configs/v3/inference.json")
    rows = {}

    def add(key, name, parent=None, **changes):
        config = dict(rows[parent]["config"] if parent else base)
        config.update(changes)
        rows[key] = {"name": name, "parent": parent, "config": config}

    add("E00", "Unigram BM25 (same tokenizer/stemming)", strategy="bm25",
        bm25_bigrams=False, turn_weights=False, title_boost=1.0)
    add("E01", "Weighted phrase BM25", "E00", bm25_bigrams=True, turn_weights=True, title_boost=3.0)
    add("E02", "Dense BGE base", strategy="dense", dense_model="models/bge-base-en-v1.5",
        document_views=1, recent_query_weight=0.0)
    add("E03", "Dense BGE large", "E02", dense_model="models/bge-large-en-v1.5")
    add("E04", "BM25 + dense weighted RRF", "E03", strategy="hybrid")
    add("E05", "Full + recent dialogue", "E04", recent_query_weight=0.5)
    add("E06", "Head + title/tail product views", "E05", document_views=2)
    add("E07", "Base cross-encoder", "E06", strategy="hybrid_rerank", reranker_model="models/bge-reranker-base")
    add("E08", "V2-M3 cross-encoder", "E07", reranker_model="models/bge-reranker-v2-m3")
    add("E09", "V2-M3 LoRA, four random negatives", "E08", reranker_model="models/v3-reranker-random")
    add("E10", "V2-M3 LoRA, three hard + one random", "E08", reranker_model="models/v3-reranker-hard")
    for key, mode in [("E09", "random"), ("E10", "hard")]:
        manifest = ROOT / f"runs/v3/{mode}/manifest.json"
        if manifest.exists():
            rows[key]["config"]["reranker_model"] = read_json(manifest)["inference_config"]["reranker_model"]
    add("B40", "V2-M3 budget 40", "E08", rerank_k=40)
    add("B120", "V2-M3 budget 120", "E08", rerank_k=120)
    return rows


if __name__ == "__main__":
    import json
    print(json.dumps(experiments(), indent=2))
