"""Shared experiment I/O; inference remains inside retrieval_utils.py."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def query_text(row):
    from eval_utils import build_eval_query
    return build_eval_query(row["conversation"])


def model_fingerprint(path):
    digest = hashlib.sha256()
    for file in sorted(Path(path).iterdir()):
        if file.is_file() and file.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}:
            digest.update(file.name.encode())
            with file.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def short_query(tokenizer, text, limit):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > limit:
        tokens = tokens[:limit // 2] + tokens[-(limit - limit // 2):]
    return tokenizer.decode(tokens, skip_special_tokens=True)


def exact_metrics(rows, rankings, k=10):
    if not rows:
        raise ValueError("Cannot score an empty split")
    reciprocal, hits = [], []
    for row in rows:
        pids = rankings[row["dialogue_id"]][:k]
        rank = pids.index(row["target"]) + 1 if row["target"] in pids else 0
        reciprocal.append(1 / rank if rank else 0.0)
        hits.append(float(rank > 0))
    return {"queries": len(rows), f"exact_recall@{k}": sum(hits) / len(rows),
            f"exact_mrr@{k}": sum(reciprocal) / len(rows)}


def verify_train_manifest():
    """Fail before any training if prepared data changed or is not from official train."""
    manifest = read_json(ROOT / "prepared/manifest.json")
    for relative, expected in manifest["sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Data changed: {relative}. Re-prepare and re-mine explicitly.")
    original = {row["dialogue_id"]: row for row in read_jsonl(ROOT / "data/train/dialogues.jsonl")}
    train = read_jsonl(ROOT / "prepared/train.jsonl")
    dev = read_jsonl(ROOT / "prepared/validation.jsonl")
    for row in train + dev:
        if original.get(row["dialogue_id"]) != row:
            raise ValueError("Prepared examples must exactly match released training examples")
    if {r["target"] for r in train} & {r["target"] for r in dev}:
        raise ValueError("Training/validation target leakage")
    return train, dev, manifest


def run_retrieval(rows, config_path, strategy, k=10, metadata_path=None):
    from eval_utils import load_products
    from retrieval_utils import retrieve_and_rerank
    os.environ["HW2_CONFIG"] = str(config_path)
    os.environ["HW2_STRATEGY"] = strategy
    if metadata_path:
        os.environ["HW2_RUN_METADATA"] = str(metadata_path)
    else:
        os.environ.pop("HW2_RUN_METADATA", None)
    # Only this official training pool is available to training utilities.
    passages = load_products(str(ROOT / "data/train/products.jsonl"))
    queries = [{"qid": row["dialogue_id"], "query": query_text(row)} for row in rows]
    return retrieve_and_rerank(queries, passages, top_k=k)
