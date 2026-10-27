"""Real-model v4 GPU short test only; never exports trained weights."""
import gc
import os
import random
import time
from common import ROOT, read_json, write_json, verify_train_manifest, query_text, short_query
from v4_common import candidate_configs, title_key, sample_group, ranking_loss


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model
    from eval_utils import load_products
    from retrieval_utils import retrieve_and_rerank
    from retrieval import save_submission
    from evaluate import load_submission
    train, _, _ = verify_train_manifest()
    cfg = read_json(ROOT / "configs/v4/training.json")
    output = ROOT / "results/preflight_v4"
    inference = {**candidate_configs()["R4_e5_union"], "strategy": "hybrid_rerank"}
    write_json(output / "config.json", inference)
    for name in list(os.environ):
        if name.startswith("HW2_"): os.environ.pop(name)
    os.environ.update(HW2_CONFIG=str(output / "config.json"), HW2_DEVICE="cuda:0",
                      HW2_CACHE_DIR=str(ROOT / "cache/preflight_v4"), HW2_RUN_METADATA=str(output / "retrieval.run.json"))
    products = load_products(str(ROOT / "data/train/products.jsonl"))
    rows = train[:8]
    ids = list(dict.fromkeys([r["target"] for r in rows] + sorted(products)[:192]))
    subset = {p: products[p] for p in ids}
    queries = [{"qid": r["dialogue_id"], "query": query_text(r)} for r in rows]
    predictions = retrieve_and_rerank(queries, subset)
    save_submission(output / "retrieval.json", predictions, top_k=10)
    load_submission(output / "retrieval.json", queries, set(subset))
    print("V4 E5 + protected union + 512-token reranker contract PASS", flush=True)
    device = "cuda:0"
    torch.manual_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["initial_model"], local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / cfg["initial_model"],
                                                               local_files_only=True, dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                           lora_dropout=cfg["lora_dropout"], target_modules=cfg["target_modules"],
                           modules_to_save=cfg["modules_to_save"], bias="none"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    batch = rows[:cfg["query_batch_size"]]
    titles = {p: title_key(products[p]) for p in products}
    candidates = {r["dialogue_id"]: ids for r in rows}
    texts_q, texts_p = [], []
    group = 1 + cfg["hard_negatives"] + cfg["medium_negatives"] + cfg["random_negatives"]
    rng = random.Random(cfg["seed"])
    for row in batch:
        selected = sample_group(row, candidates, sorted(products), titles, rng, 1, cfg)
        texts_q.extend([short_query(tokenizer, query_text(row), cfg["query_max_length"])] * group)
        texts_p.extend(products[p] for p in selected)
    inputs = tokenizer(texts_q, texts_p, padding="max_length", truncation="only_second",
                       max_length=cfg["max_length"], return_tensors="pt").to(device)
    durations, updates = [], 0
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**inputs).logits.view(len(batch), group).float()
            loss, _, _ = ranking_loss(logits, cfg["label_smoothing"], cfg["pairwise_weight"])
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite preflight loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
        old = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updates += int(scaler.get_scale() >= old)
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-started)
    if not updates: raise FloatingPointError("No successful optimizer update")
    result = {"purpose": "short train-only compatibility test, no saved training weights", "status": "PASS",
              "gpu": torch.cuda.get_device_name(0), "config": cfg, "finite_updates": updates,
              "pairs_per_microbatch": len(batch)*group, "tokens_per_pair": cfg["max_length"],
              "microbatch_seconds": sum(durations[1:])/3,
              "training_peak_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30,
              "training_peak_reserved_gib": torch.cuda.max_memory_reserved(device)/2**30}
    write_json(output / "benchmark.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
