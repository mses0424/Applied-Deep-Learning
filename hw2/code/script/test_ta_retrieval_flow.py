"""Short isolated smoke test of the packaged TA retrieval.py command contract."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from common import ROOT, read_json, read_jsonl


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    selected_source = ROOT / "models/selected-listwise-reranker"
    if not selected_source.is_dir():
        selected_source = ROOT / "models/v4-reranker-listwise"
    dense_source = ROOT / "models/e5-base-v2"
    for path in (selected_source, dense_source):
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path)

    cpu_smoke = os.environ.get("HW2_SMOKE_DEVICE") == "cpu"
    dialogue_limit = 2 if cpu_smoke else 8
    product_limit = 64 if cpu_smoke else 512
    dialogues = read_jsonl(ROOT / "data/train/dialogues.jsonl")[:dialogue_limit]
    all_products = read_jsonl(ROOT / "data/train/products.jsonl")
    targets = {row["target"] for row in dialogues}
    products = []
    included = set()
    for row in all_products:
        pid = row["parent_asin"]
        if pid in targets or len(products) < product_limit:
            if pid not in included:
                products.append(row)
                included.add(pid)
        if len(products) >= product_limit and targets <= included:
            break
    if not targets <= included:
        raise ValueError("Smoke fixture omitted a target product")

    with tempfile.TemporaryDirectory(prefix="hw2_ta_flow_", dir="/tmp") as temporary:
        stage = Path(temporary)
        for name in ("retrieval.py", "retrieval_utils.py", "eval_utils.py"):
            shutil.copy2(ROOT / name, stage / name)
        (stage / "configs").mkdir()
        config = read_json(ROOT / "configs/v4/selected.json")
        config["dense_model"] = "models/e5-base-v2"
        config["reranker_model"] = "models/selected-listwise-reranker"
        config["cache_dir"] = "cache/retrieval"
        if cpu_smoke:
            # Preserve hybrid retrieval and cross-encoder reranking while
            # shrinking only the disposable fixture for a fast CPU fallback.
            config.update(retrieve_k=32, rerank_k=16,
                          lexical_quota=12, dense_quota=4)
        (stage / "configs/inference.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        (stage / "models").mkdir()
        (stage / "models/e5-base-v2").symlink_to(dense_source, target_is_directory=True)
        (stage / "models/selected-listwise-reranker").symlink_to(
            selected_source, target_is_directory=True
        )
        write_jsonl(stage / "dialogues.jsonl", dialogues)
        write_jsonl(stage / "products.jsonl", products)

        env = dict(os.environ)
        for name in list(env):
            if name.startswith("HW2_"):
                env.pop(name)
        # Keep the default test identical to the GPU-based TA command, while
        # allowing an explicit CPU smoke test when the local CUDA driver is
        # temporarily unavailable.
        if os.environ.get("HW2_SMOKE_DEVICE"):
            env["HW2_DEVICE"] = os.environ["HW2_SMOKE_DEVICE"]
        command = [sys.executable, "retrieval.py", "--dialogues", "dialogues.jsonl",
                   "--products", "products.jsonl", "--output", "output.json"]
        subprocess.run(command, cwd=stage, env=env, check=True)
        payload = json.loads((stage / "output.json").read_text(encoding="utf-8"))
        valid = {row["parent_asin"] for row in products}
        expected_qids = {row["dialogue_id"] for row in dialogues}
        if payload.get("top_k") != 10 or len(payload.get("results", [])) != len(dialogues):
            raise ValueError("Invalid output size/top_k")
        if {row["qid"] for row in payload["results"]} != expected_qids:
            raise ValueError("Output query IDs differ from input")
        for row in payload["results"]:
            ids = row["ranked_pids"]
            if len(ids) != 10 or len(set(ids)) != 10 or not set(ids) <= valid:
                raise ValueError(f"Invalid ranking for {row['qid']}")
    mode = "reduced CPU fixture" if cpu_smoke else "default GPU configuration"
    print(f"TA retrieval.py CLI smoke: PASS ({mode}, semantic model path, valid Top-10)")


if __name__ == "__main__":
    main()
