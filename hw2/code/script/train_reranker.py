"""Fine-tune a cross-encoder with train-only hard/random negatives and validation selection."""
from __future__ import annotations

import math
import os
import random
import time

from common import (ROOT, read_json, write_json, sha256, verify_train_manifest, model_fingerprint,
                    query_text, short_query, exact_metrics)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", action="store_true", help="LoRA on V2-M3 instead of base full fine-tuning")
    parser.add_argument("--negatives", choices=["random", "hard"], default="hard")
    args = parser.parse_args()
    if not args.v3 and args.negatives != "hard":
        parser.error("--negatives random is a v3 experiment")
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from tqdm import tqdm
    from eval_utils import load_products

    train, dev, manifest = verify_train_manifest()
    prepared = ROOT / ("prepared/v3" if args.v3 else "prepared")
    config_dir = ROOT / ("configs/v3" if args.v3 else "configs")
    mining = read_json(prepared / "mining.json")
    if mining["manifest_sha256"] != sha256(ROOT / "prepared/manifest.json"):
        raise ValueError("Candidate manifest mismatch")
    if mining["candidates_sha256"] != sha256(prepared / "candidates.json"):
        raise ValueError("Candidate file changed after mining")
    cfg = read_json(config_dir / "training.json")
    inference = read_json(config_dir / "inference.json")
    if args.v3:
        if mining["config"] != inference or mining["retrieval_source_sha256"] != sha256(ROOT / "retrieval_utils.py"):
            raise ValueError("Mining source/config changed; use a fresh experiment copy")
        if args.negatives == "random":
            cfg["random_negatives"] += cfg["hard_negatives"]
            cfg["hard_negatives"] = 0
    if (cfg["max_length"], cfg["query_max_length"]) != (inference["rerank_max_length"], inference["query_max_length"]):
        raise ValueError("Train/inference token budgets must agree")
    output = ROOT / (f"runs/v3/{args.negatives}" if args.v3 else "runs/training")
    if output.exists():
        raise SystemExit(f"{output} exists. Use a separate project copy for another experiment.")
    checkpoint_relative = os.environ.get("HW2_TRAIN_OUTPUT", "models/reranker-trained")
    if checkpoint_relative not in {"models/reranker-trained", "models/reranker-reproduced"}:
        raise ValueError("HW2_TRAIN_OUTPUT must name models/reranker-trained or models/reranker-reproduced")
    if args.v3:
        checkpoint_relative = f"models/v3-reranker-{args.negatives}"
        if (ROOT / checkpoint_relative).exists():
            checkpoint_relative += "-reproduced"
    checkpoint = ROOT / checkpoint_relative
    if checkpoint.exists():
        raise SystemExit(f"{checkpoint} exists; refusing to replace existing weights")
    output.mkdir(parents=True)
    candidates = read_json(prepared / "candidates.json")
    passages = load_products(str(ROOT / "data/train/products.jsonl"))
    pids = sorted(passages)
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = os.environ.get("HW2_DEVICE", "cuda:0")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("Training requires CUDA. Check .venv and CUDA_VISIBLE_DEVICES.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Reproduction always starts from the original pretrained checkpoint, even
    # when the packaged inference config points at a previously trained model.
    pretrained_dir = ROOT / ("models/bge-reranker-v2-m3" if args.v3 else "models/bge-reranker-base")
    pretrained_hash = model_fingerprint(pretrained_dir)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_dir, local_files_only=True,
    ).to(device)
    if args.v3:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"], modules_to_save=cfg["modules_to_save"],
            bias="none"))
        model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    parameter_counts = {"trainable": sum(p.numel() for p in trainable),
                        "total": sum(p.numel() for p in model.parameters())}
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_queries = {r["dialogue_id"]: short_query(tokenizer, query_text(r), cfg["query_max_length"])
                     for r in train + dev}
    group_size = 1 + cfg["hard_negatives"] + cfg["random_negatives"]
    micro_batch = cfg["query_batch_size"]
    accumulate = cfg["gradient_accumulation_steps"]
    micro_steps = math.ceil(len(train) / micro_batch)
    updates_per_epoch = math.ceil(micro_steps / accumulate)
    total_steps = updates_per_epoch * cfg["epochs"]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg["warmup_ratio"]), total_steps)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024 if args.v3 else 65536)
    history = []
    started = time.perf_counter()

    def validate():
        model.eval()
        rankings = {}
        with torch.inference_mode():
            for row in tqdm(dev, desc="Train-validation (no public data)"):
                ids = candidates[row["dialogue_id"]][:inference["rerank_k"]]
                scores = []
                for offset in range(0, len(ids), inference["rerank_batch_size"]):
                    batch = ids[offset:offset + inference["rerank_batch_size"]]
                    inputs = tokenizer([train_queries[row["dialogue_id"]]] * len(batch),
                                       [passages[pid] for pid in batch], padding=True,
                                       truncation="only_second", max_length=cfg["max_length"],
                                       return_tensors="pt").to(device)
                    with torch.autocast("cuda", dtype=torch.float16):
                        scores.extend(model(**inputs).logits.flatten().float().cpu().tolist())
                # Stable tie-break preserves the hybrid candidate order.
                order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
                rankings[row["dialogue_id"]] = [ids[i] for i in order]
        return exact_metrics(dev, rankings)

    metrics = validate()
    best_key = (metrics["exact_mrr@10"], metrics["exact_recall@10"])
    baseline_key = best_key
    if args.v3:
        # Compare best TRAINED epoch with E08 later; do not silently substitute
        # pretrained weights when fine-tuning makes the result worse.
        best_key = (-1.0, -1.0)
    selected_epoch = 0
    history.append({"epoch": 0, "checkpoint": "pretrained", "validation": metrics})
    write_json(output / "history.json", history)
    print("Pretrained validation:", metrics, flush=True)
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_started = time.perf_counter()
        updates, skipped_updates = 0, 0
        rng = random.Random(seed + epoch)
        shuffled = list(train)
        rng.shuffle(shuffled)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, offset in enumerate(tqdm(range(0, len(shuffled), micro_batch), desc=f"Epoch {epoch}")):
            batch = shuffled[offset:offset + micro_batch]
            query_batch, product_batch = [], []
            for row in batch:
                gold = row["target"]
                # Retrieved negatives are weak labels: alternatives may also be
                # valid. Avoid the first few high-ranked items and use smoothing.
                pool = [pid for pid in candidates[row["dialogue_id"]][5:] if pid != gold]
                hard = rng.sample(pool, min(cfg["hard_negatives"], len(pool)))
                selected = [gold] + hard
                while len(selected) < group_size:
                    candidate = rng.choice(pids)
                    if candidate not in selected:
                        selected.append(candidate)
                query_batch.extend([train_queries[row["dialogue_id"]]] * group_size)
                product_batch.extend(passages[pid] for pid in selected)
            inputs = tokenizer(query_batch, product_batch, padding=True, truncation="only_second",
                               max_length=cfg["max_length"], return_tensors="pt").to(device)
            # Last partial accumulation window uses its actual number of steps.
            window_start = (step // accumulate) * accumulate
            window_size = min(accumulate, micro_steps - window_start)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**inputs).logits.view(len(batch), group_size).float()
                loss = F.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device),
                                       label_smoothing=cfg["label_smoothing"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; stop and inspect precision/inputs")
            scaler.scale(loss / window_size).backward()
            losses.append(loss.item())
            if (step + 1) % accumulate == 0 or step + 1 == micro_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"])
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    updates += 1
                else:
                    skipped_updates += 1
                optimizer.zero_grad(set_to_none=True)
        metrics = validate()
        key = (metrics["exact_mrr@10"], metrics["exact_recall@10"])
        if key > best_key:
            best_key, selected_epoch = key, epoch
            save_path = output / "adapter" if args.v3 else checkpoint
            model.save_pretrained(save_path, safe_serialization=True)
            tokenizer.save_pretrained(save_path)
        if updates == 0:
            raise FloatingPointError("No successful optimizer updates; do not report this as trained")
        history.append({"epoch": epoch, "mean_loss": sum(losses) / len(losses), "validation": metrics,
                        "updates": updates, "overflow_skips": skipped_updates,
                        "epoch_seconds": time.perf_counter() - epoch_started})
        write_json(output / "history.json", history)
        print(f"Epoch {epoch} validation:", metrics, flush=True)
    selected_config = dict(inference)
    selected_config["reranker_model"] = "models/bge-reranker-base"
    if selected_epoch:
        selected_config["reranker_model"] = checkpoint_relative
    if args.v3:
        # Export standalone weights: inference has no PEFT dependency and uses
        # exactly the best adapter checkpoint, not the final epoch in memory.
        import gc
        from peft import PeftModel
        del model, optimizer, scheduler, trainable
        gc.collect()
        torch.cuda.empty_cache()
        base = AutoModelForSequenceClassification.from_pretrained(pretrained_dir, local_files_only=True)
        merged = PeftModel.from_pretrained(base, output / "adapter").merge_and_unload(safe_merge=True)
        merged.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
    else:
        write_json(ROOT / "configs/selected.json", selected_config)
    write_json(output / "manifest.json", {
        "training_config": cfg, "inference_config": selected_config, "seed": seed,
        "source_manifest": manifest, "mining": mining, "selected_epoch": selected_epoch,
        "selection": "train-validation exact MRR@10, then exact Recall@10; " +
                     ("best trained epoch; pretrained compared separately" if args.v3 else "pretrained eligible"),
        "beats_pretrained": best_key > baseline_key, "parameter_counts": parameter_counts,
        "pretrained_fingerprint": pretrained_hash,
        "training_source_sha256": sha256(__file__),
        "checkpoint_fingerprint": model_fingerprint(checkpoint) if checkpoint.exists() else pretrained_hash,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "torch_version": torch.__version__,
    })
    print(f"Selected epoch={selected_epoch}; saved {checkpoint_relative}. Public performance is still unmeasured.")


if __name__ == "__main__":
    main()
