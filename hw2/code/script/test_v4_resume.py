"""Tiny-model GPU integration test of training/export/resume; no homework model is changed."""
import gc
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch
from common import ROOT, read_json, write_json, write_jsonl, sha256


def main():
    import torch
    from transformers import BertConfig, BertForSequenceClassification, AutoTokenizer
    import train_v4
    import v4_common
    if not torch.cuda.is_available():
        raise SystemExit("This short integration test needs CUDA")
    real = ROOT
    with tempfile.TemporaryDirectory(prefix="hw2-v4-resume-test-") as directory:
        root = Path(directory)
        for name in ["retrieval_utils.py", "code/script/common.py", "code/script/v4_common.py",
                     "code/script/v4_pipeline.py", "code/script/train_v4.py"]:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(real / name, root / name)
        model_dir = root / "models/tiny"
        tokenizer = AutoTokenizer.from_pretrained(real / "models/bge-base-en-v1.5", local_files_only=True)
        model = BertForSequenceClassification(BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=16,
                    num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=1))
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        del model
        cfg = read_json(real / "configs/v4/training.json")
        cfg.update(initial_model="models/tiny", epochs=1, min_epochs=1, patience=1, query_batch_size=2,
                   gradient_accumulation_steps=2, max_length=32, query_max_length=8, lora_r=2, lora_alpha=4)
        inference = {**v4_common.candidate_configs()["R3_bge_union"], "dense_model": "models/tiny",
                     "reranker_model": "models/tiny", "rerank_k": 10, "rerank_max_length": 32, "query_max_length": 8}
        rows = [{"dialogue_id": f"q{i}", "target": f"p{i}",
                 "conversation": [{"role": "user", "content": "I need shampoo"}]} for i in range(7)]
        products = [{"parent_asin": f"p{i}", "title": f"Product {i}", "description": "Hair shampoo"} for i in range(80)]
        write_jsonl(root / "data/train/products.jsonl", products)
        write_json(root / "configs/v4/training.json", cfg)
        write_json(root / "configs/v4/candidates_selected.json", inference)
        write_json(root / "prepared/manifest.json", {"fixture": True})
        write_json(root / "prepared/v4/candidates.json", {r["dialogue_id"]: [p["parent_asin"] for p in products] for r in rows})
        with patch.object(train_v4, "ROOT", root), patch.object(v4_common, "ROOT", root), \
             patch.object(train_v4, "verify_train_manifest", return_value=(rows[:5], rows[5:], {"fixture": True})), \
             patch.object(sys, "argv", ["train_v4.py", "--mode", "pairwise"]), \
             patch.dict(os.environ, {"HW2_DEVICE": "cuda:0", "HW2_V4_REPRODUCE": "0"}):
            write_json(root / "prepared/v4/mining.json", {
                "manifest_sha256": sha256(root / "prepared/manifest.json"),
                "candidates_sha256": sha256(root / "prepared/v4/candidates.json"),
                "signature": v4_common.signature(inference)})
            train_v4.main()
            train_v4.verify_complete("pairwise")
            manifest = root / "runs/v4/pairwise/manifest.json"
            before = read_json(manifest)
            from safetensors.torch import load_file
            weights_before = load_file(str(root / before["checkpoint"] / "model.safetensors"))
            assert before["trained_epochs"] == 1
            history = read_json(root / "runs/v4/pairwise/history.json")
            assert history[0]["learning_probe"]["lora_b_l2_norm"] == 0
            assert history[1]["learning_probe"]["lora_b_l2_norm"] > 0
            assert history[1]["mean_grad_norm_before_clip"] > 0
            assert history[1]["validation"]["retrieval_misses"] == 0
            assert history[1]["learning_probe"]["identical_swapped_query_fraction"] == 1
            # Remove only the disposable fixture completion receipt to simulate
            # interruption after its epoch checkpoint but before final export.
            manifest.unlink()
            train_v4.main()
            after = read_json(manifest)
            weights_after = load_file(str(root / after["checkpoint"] / "model.safetensors"))
            assert weights_before.keys() == weights_after.keys()
            for name in weights_before:
                torch.testing.assert_close(weights_before[name], weights_after[name], rtol=0, atol=0)
            assert before["history_sha256"] == after["history_sha256"]
            train_v4.verify_complete("pairwise")
    gc.collect()
    torch.cuda.empty_cache()
    print("Tiny-model v4 training, partial accumulation, checkpoint resume and merged export: PASS")


if __name__ == "__main__":
    main()
