"""Fixed-source train-only split. Never accepts a public/private data path."""
import random
from common import ROOT, read_json, read_jsonl, write_json, write_jsonl, sha256


def main():
    output = ROOT / "prepared"
    if (output / "manifest.json").exists():
        raise SystemExit("prepared/manifest.json already exists; preserve it for reproducibility.")
    cfg = read_json(ROOT / "configs/training.json")
    rows = read_jsonl(ROOT / "data/train/dialogues.jsonl")
    if len({r["dialogue_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training dialogue IDs")
    products = {r["parent_asin"] for r in read_jsonl(ROOT / "data/train/products.jsonl")}
    if not all(r["target"] in products for r in rows):
        raise ValueError("Training target absent from training product pool")
    targets = sorted({row["target"] for row in rows})
    random.Random(cfg["seed"]).shuffle(targets)
    count = max(1, int(len(targets) * cfg["validation_fraction"]))
    validation_targets = set(targets[:count])
    train = [r for r in rows if r["target"] not in validation_targets]
    dev = [r for r in rows if r["target"] in validation_targets]
    if not train or not dev:
        raise ValueError("Both train and validation must be nonempty")
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", dev)
    paths = ["data/train/dialogues.jsonl", "data/train/products.jsonl",
             "prepared/train.jsonl", "prepared/validation.jsonl"]
    write_json(output / "manifest.json", {
        "source": "official train only", "split_by": "target product", "seed": cfg["seed"],
        "train_count": len(train), "validation_count": len(dev),
        "sha256": {p: sha256(ROOT / p) for p in paths},
    })
    print(f"Prepared {len(train)} train / {len(dev)} validation; disjoint target products.")


if __name__ == "__main__":
    main()
