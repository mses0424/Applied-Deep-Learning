"""Time the unchanged TA retrieval entrypoint and record output provenance."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, read_json, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogues", default="data/public/dialogues.jsonl")
    parser.add_argument("--products", default="data/public/products.jsonl")
    parser.add_argument("--config", default="configs/selected.json")
    parser.add_argument("--strategy", choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="results/v2/public.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    if Path(args.output).exists():
        raise SystemExit("Output exists. Choose a new path to preserve experiment history.")
    cfg = Path(args.config).resolve()
    config = read_json(cfg)
    metadata = Path(args.output).with_suffix(".run.json")
    env = dict(os.environ, HW2_CONFIG=str(cfg), HW2_STRATEGY=args.strategy or config["strategy"],
               HW2_RUN_METADATA=str(metadata.resolve()), TOKENIZERS_PARALLELISM="false")
    start = time.perf_counter()
    subprocess.run([sys.executable, "retrieval.py", "--dialogues", args.dialogues,
                    "--products", args.products, "--output", args.output,
                    "--top-k", str(args.top_k)], env=env, check=True)
    receipt = read_json(metadata)
    receipt.update({"wall_seconds": time.perf_counter() - start, "prediction_sha256": sha256(args.output),
                    "dialogues_sha256": sha256(args.dialogues), "products_sha256": sha256(args.products),
                    "config_sha256": sha256(cfg), "retrieval_source_sha256": sha256("retrieval_utils.py"),
                    "top_k": args.top_k, "python": sys.version})
    write_json(metadata, receipt)
    print(f"Saved predictions and {metadata}. A cold run on the TA GPU is needed for the final time limit.")


if __name__ == "__main__":
    main()
