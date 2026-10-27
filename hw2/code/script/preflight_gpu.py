"""Small train-only execution/throughput check; never saves trained weights.

Full training and full-corpus retrieval are deliberately not run here.
"""
import gc
import math
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from common import ROOT, read_json, read_jsonl, write_json, query_text, short_query


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", action="store_true")
    args = parser.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
    from eval_utils import load_products
    from retrieval_utils import retrieve_and_rerank
    from retrieval import save_submission
    from evaluate import load_submission

    os.chdir(ROOT)
    output = ROOT / ("results/preflight_v3" if args.v3 else "results/preflight")
    output.mkdir(parents=True, exist_ok=True)
    config_dir = ROOT / ("configs/v3" if args.v3 else "configs")
    cfg = read_json(config_dir / "inference.json")
    training = read_json(config_dir / "training.json")
    manifest = read_json("prepared/manifest.json")
    rows = read_jsonl("prepared/train.jsonl")[:8]
    products = load_products("data/train/products.jsonl")
    ids = list(products)[:192]
    ids += [r["target"] for r in rows if r["target"] not in ids]
    subset = {pid: products[pid] for pid in ids}
    queries = [{"qid": r["dialogue_id"], "query": query_text(r)} for r in rows]
    os.environ["HW2_CONFIG"] = str(config_dir / "inference.json")
    os.environ["HW2_RUN_METADATA"] = str(output / "retrieval.run.json")
    os.environ["HW2_CACHE_DIR"] = "cache/preflight"
    os.environ["HW2_STRATEGY"] = "hybrid_rerank"
    ranked = retrieve_and_rerank(queries, subset)
    save_submission(output / "retrieval.json", ranked, top_k=10)
    load_submission(output / "retrieval.json", queries, set(subset))
    summary = {"purpose": "small train-only compatibility test, not a score claim",
               "retrieval_contract": "PASS", "gpu": torch.cuda.get_device_name(0),
               "python": __import__("sys").version, "torch": torch.__version__}
    write_json(output / "benchmark.json", summary)
    print("End-to-end retrieval and official contract: PASS", flush=True)
    texts = [products[pid] for pid in ids[:96]]
    device = "cuda:0"
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["dense_model"], local_files_only=True)
    model = AutoModel.from_pretrained(ROOT / cfg["dense_model"], local_files_only=True,
                                      dtype=torch.float16).to(device).eval()
    # Padding to production maximum provides a conservative compute estimate.
    dense_batch = cfg["encode_batch_size"]
    dense_input = tokenizer(texts[:dense_batch], padding="max_length", max_length=384,
                            truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        model(**dense_input)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(8):
            encoded = model(**dense_input).last_hidden_state
            assert torch.isfinite(encoded).all()
        torch.cuda.synchronize()
    summary["dense_products_per_second_at_384_tokens"] = (8 * dense_batch) / (time.perf_counter() - start)
    del model, encoded, dense_input, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["reranker_model"], local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROOT / cfg["reranker_model"], local_files_only=True).to(device)
    if args.v3:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=training["lora_r"], lora_alpha=training["lora_alpha"],
                               lora_dropout=training["lora_dropout"], target_modules=training["target_modules"],
                               modules_to_save=training["modules_to_save"], bias="none"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    query = short_query(tokenizer, queries[0]["query"], training["query_max_length"])
    groups = training["query_batch_size"]
    pairs = tokenizer([query] * (groups * 5), texts[:groups * 5], padding="max_length", truncation="only_second",
                       max_length=384, return_tensors="pt").to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    durations = []
    successful_updates = 0
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**pairs).logits.view(groups, 5).float()
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(groups, dtype=torch.long, device=device),
                                                      label_smoothing=training["label_smoothing"])
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        # GradScaler may legitimately skip early steps while calibrating scale.
        successful_updates += int(torch.isfinite(norm))
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    assert successful_updates > 0, "No finite optimizer updates during the smoke test"
    summary["finite_optimizer_updates"] = successful_updates
    summary["training_microbatch_seconds_at_384_tokens"] = sum(durations[1:]) / 3
    summary["training_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    summary["forward_backward_optimizer"] = "PASS (4 disposable in-memory steps, no checkpoint)"
    model.eval()
    inference_batch = cfg["rerank_batch_size"]
    small_pairs = {key: value[:inference_batch] for key, value in pairs.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        model(**small_pairs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            logits = model(**small_pairs).logits
            assert torch.isfinite(logits).all()
        torch.cuda.synchronize()
    actual_batch = len(small_pairs["input_ids"])
    summary["rerank_pairs_per_second_at_384_tokens"] = (10 * actual_batch) / (time.perf_counter() - start)
    dense_rate = summary["dense_products_per_second_at_384_tokens"]
    rerank_rate = summary["rerank_pairs_per_second_at_384_tokens"]
    summary["estimated_compute_seconds"] = {
        "mining_dense_50000_plus_queries": (50000 * cfg.get("document_views", 1) + 2 * 4830) / dense_rate,
        "training_two_epochs": 2 * math.ceil(manifest["train_count"] / groups) * summary["training_microbatch_seconds_at_384_tokens"],
        "three_validation_passes": 3 * manifest["validation_count"] * cfg["rerank_k"] / rerank_rate,
        "public_retrieval_cold": (75000 * cfg.get("document_views", 1) + 2000) / dense_rate + 1000 * cfg["rerank_k"] / rerank_rate,
    }
    summary["estimate_limits"] = "Compute extrapolations at max padding; exclude CPU indexing, tokenizer, I/O and contention. Not a TA-hardware guarantee."
    write_json(output / "benchmark.json", summary)
    print(__import__("json").dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
