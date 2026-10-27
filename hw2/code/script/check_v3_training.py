"""Validate completed LoRA evidence before skipping training on a repeated run."""
import argparse
from common import ROOT, read_json, sha256, model_fingerprint, verify_train_manifest


def check(mode):
    verify_train_manifest()
    manifest = read_json(ROOT / f"runs/v3/{mode}/manifest.json")
    cfg = read_json(ROOT / "configs/v3/training.json")
    if mode == "random":
        cfg["random_negatives"] += cfg["hard_negatives"]
        cfg["hard_negatives"] = 0
    checks = [manifest["training_config"] == cfg,
              manifest["training_source_sha256"] == sha256(ROOT / "code/script/train_reranker.py"),
              manifest["checkpoint_fingerprint"] == model_fingerprint(ROOT / manifest["inference_config"]["reranker_model"]),
              manifest["pretrained_fingerprint"] == model_fingerprint(ROOT / "models/bge-reranker-v2-m3"),
              manifest["mining"] == read_json(ROOT / "prepared/v3/mining.json"),
              manifest["mining"]["candidates_sha256"] == sha256(ROOT / "prepared/v3/candidates.json"),
              manifest["mining"]["manifest_sha256"] == sha256(ROOT / "prepared/manifest.json"),
              manifest["mining"]["retrieval_source_sha256"] == sha256(ROOT / "retrieval_utils.py"),
              manifest["mining"]["config"] == read_json(ROOT / "configs/v3/inference.json")]
    if not all(checks):
        raise ValueError("Training evidence changed; do not reuse stale checkpoints")
    print(f"Verified completed {mode} LoRA training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negatives", choices=["random", "hard"], required=True)
    check(parser.parse_args().negatives)
