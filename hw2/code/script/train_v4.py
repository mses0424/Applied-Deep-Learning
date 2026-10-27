"""Larger-batch LoRA with curriculum, controlled losses, early stopping and epoch resume."""
import argparse
import gc
import hashlib
import math
import os
import random
import time
from common import (ROOT, read_json, write_json, sha256, model_fingerprint, verify_train_manifest,
                    short_query, query_text, exact_metrics)
from v4_common import signature, inference_config, sample_group, title_key, ranking_loss


def specification(mode):
    cfg = read_json(ROOT / "configs/v4/training.json")
    if mode == "listwise": cfg["pairwise_weight"] = 0.0
    mining = read_json(ROOT / "prepared/v4/mining.json")
    inference = inference_config()
    if mining["manifest_sha256"] != sha256(ROOT / "prepared/manifest.json"):
        raise ValueError("Train split changed")
    if mining["candidates_sha256"] != sha256(ROOT / "prepared/v4/candidates.json"):
        raise ValueError("Mined candidates changed")
    if mining["signature"] != signature(read_json(ROOT / "configs/v4/candidates_selected.json")):
        raise ValueError("Mining code/model/config changed")
    if (cfg["max_length"], cfg["query_max_length"]) != (inference["rerank_max_length"], inference["query_max_length"]):
        raise ValueError("Training and inference token budgets differ")
    spec = {"training": signature(cfg), "inference": inference,
            "mining_sha256": sha256(ROOT / "prepared/v4/mining.json")}
    return cfg, inference, spec


def verify_complete(mode):
    verify_train_manifest()
    _, _, spec = specification(mode)
    manifest = read_json(ROOT / f"runs/v4/{mode}/manifest.json")
    if manifest["specification"] != spec:
        raise ValueError(f"Stale completed v4 training: {mode}")
    if manifest["checkpoint_fingerprint"] != model_fingerprint(ROOT / manifest["checkpoint"]):
        raise ValueError("Trained checkpoint changed")
    if manifest["history_sha256"] != sha256(ROOT / f"runs/v4/{mode}/history.json"):
        raise ValueError("Training history changed")
    print("Verified complete v4", mode, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["listwise", "pairwise"], required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from peft import (LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict,
                      set_peft_model_state_dict)
    from eval_utils import load_products
    from tqdm import tqdm

    train, dev, _ = verify_train_manifest()
    cfg, inference, spec = specification(args.mode)
    output = ROOT / f"runs/v4/{args.mode}"
    checkpoint_relative = f"models/v4-reranker-{args.mode}" + ("-reproduced" if os.environ.get("HW2_V4_REPRODUCE") == "1" else "")
    checkpoint = ROOT / checkpoint_relative
    if (output / "manifest.json").exists():
        verify_complete(args.mode)
        return
    if (output / "specification.json").exists():
        if read_json(output / "specification.json") != spec:
            raise ValueError("Experiment changed; do not resume under different settings")
    elif checkpoint.exists() or output.exists():
        raise ValueError("Unowned output exists; refusing to overwrite")
    write_json(output / "specification.json", spec)
    device = os.environ.get("HW2_DEVICE", "cuda:0")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_device(torch.device(device))
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(device)
    base_dir = ROOT / cfg["initial_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(base_dir, local_files_only=True,
                                                               dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                           lora_dropout=cfg["lora_dropout"], target_modules=cfg["target_modules"],
                           modules_to_save=cfg["modules_to_save"], bias="none"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    parameter_counts = {"trainable": sum(p.numel() for p in params), "total": sum(p.numel() for p in model.parameters())}
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    micro, accum = cfg["query_batch_size"], cfg["gradient_accumulation_steps"]
    micro_steps = math.ceil(len(train) / micro)
    steps = math.ceil(micro_steps / accum) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * cfg["warmup_ratio"]), steps)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    passages = load_products(str(ROOT / "data/train/products.jsonl"))
    pids = sorted(passages)
    titles = {pid: title_key(passages[pid]) for pid in pids}
    candidates = read_json(ROOT / "prepared/v4/candidates.json")
    queries = {r["dialogue_id"]: short_query(tokenizer, query_text(r), cfg["query_max_length"]) for r in train + dev}
    group = 1 + cfg["hard_negatives"] + cfg["medium_negatives"] + cfg["random_negatives"]

    # Fixed, distinct-target TRAIN-only probes. Unlike resampled training loss,
    # these examples/negatives remain identical across epochs and both losses.
    probe_rows, seen_targets = [], set()
    for row in sorted(train, key=lambda r: hashlib.sha256((str(cfg["seed"])+r["dialogue_id"]).encode()).hexdigest()):
        if row["target"] in seen_targets: continue
        seen_targets.add(row["target"])
        probe_rows.append(row)
        if len(probe_rows) >= cfg["learning_probe_size"]: break
    probe_rng = random.Random(cfg["seed"] + 7907)
    probe_groups = [sample_group(r, candidates, pids, titles, probe_rng, 2, cfg) for r in probe_rows]
    probe_definition = {"source": "train only; deterministic distinct-target sample; not used for selection",
        "groups": [{"qid": r["dialogue_id"], "positive_and_negatives": ids,
                    "swapped_query_qid": probe_rows[(i+1) % len(probe_rows)]["dialogue_id"]}
                   for i, (r, ids) in enumerate(zip(probe_rows, probe_groups))]}
    if (output / "learning_probe.json").exists() and read_json(output / "learning_probe.json") != probe_definition:
        raise ValueError("Fixed learning probe changed; do not compare different probes across epochs")
    write_json(output / "learning_probe.json", probe_definition)

    def score_pairs(query, ids):
        scores = []
        for offset in range(0, len(ids), inference["rerank_batch_size"]):
            batch = ids[offset:offset+inference["rerank_batch_size"]]
            inputs = tokenizer([query] * len(batch), [passages[i] for i in batch],
                               padding=True, truncation="only_second", max_length=cfg["max_length"], return_tensors="pt").to(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                scores.extend(model(**inputs).logits.flatten().float().cpu().tolist())
        return scores

    def learning_probe():
        model.eval()
        summaries = {}
        for swapped in (False, True):
            nll, margins, reciprocal, strict_wins = [], [], [], []
            for i, ids in enumerate(probe_groups):
                row = probe_rows[(i+1) % len(probe_rows)] if swapped else probe_rows[i]
                scores = torch.tensor(score_pairs(queries[row["dialogue_id"]], ids))
                nll.append(float(-scores.log_softmax(0)[0]))
                margin = float(scores[0] - scores[1:].max())
                margins.append(margin)
                strict_wins.append(float(margin > 0))
                # Pessimistic ties avoid credit just because gold is at index 0.
                reciprocal.append(1.0 / (1 + int((scores[1:] >= scores[0]).sum())))
            summaries["swapped_query" if swapped else "correct_query"] = {
                "fixed_group_nll": sum(nll)/len(nll), "gold_minus_hardest_negative": sum(margins)/len(margins),
                "strict_top1": sum(strict_wins)/len(strict_wins), "group_mrr_pessimistic_ties": sum(reciprocal)/len(reciprocal)}
        summaries["queries"] = len(probe_rows)
        summaries["query_swap_top1_drop"] = summaries["correct_query"]["strict_top1"] - summaries["swapped_query"]["strict_top1"]
        summaries["identical_swapped_query_fraction"] = sum(
            queries[r["dialogue_id"]] == queries[probe_rows[(i+1) % len(probe_rows)]["dialogue_id"]]
            for i, r in enumerate(probe_rows)) / len(probe_rows)
        summaries["lora_b_l2_norm"] = math.sqrt(sum(float(p.detach().float().square().sum())
            for name, p in model.named_parameters() if "lora_B" in name))
        return summaries

    def validate():
        model.eval()
        rankings = {}
        covered, ranking_misses, margins = 0, 0, []
        for row in tqdm(dev, desc="v4 train-validation", mininterval=5):
            ids = candidates[row["dialogue_id"]][:inference["rerank_k"]]
            scores = score_pairs(queries[row["dialogue_id"]], ids)
            order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
            rankings[row["dialogue_id"]] = [ids[i] for i in order]
            if row["target"] in ids:
                covered += 1
                gold_index = ids.index(row["target"])
                negatives = [s for i, s in enumerate(scores) if i != gold_index]
                if negatives: margins.append(scores[gold_index] - max(negatives))
                ranking_misses += int(row["target"] not in rankings[row["dialogue_id"]][:10])
        metrics = exact_metrics(dev, rankings)
        metrics.update(candidate_covered=covered, retrieval_misses=len(dev)-covered,
                       covered_but_not_top10=ranking_misses,
                       conditional_recall10=(covered-ranking_misses)/covered if covered else None,
                       conditional_mrr10=metrics["exact_mrr@10"]*len(dev)/covered if covered else None,
                       covered_gold_minus_hardest_negative=sum(margins)/len(margins) if margins else None)
        return metrics

    history, best, best_epoch, stale, first_epoch, elapsed_prior = [], (-1.0, -1.0), 0, 0, 1, 0.0
    resume = output / "resume.pt"
    if resume.exists():
        # Load only this project's checksum-verified locally produced checkpoint.
        if sha256(resume) != read_json(output / "resume.receipt.json")["sha256"]:
            raise ValueError("Resume checkpoint is incomplete or changed")
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["specification"] != spec: raise ValueError("Resume settings mismatch")
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, best, best_epoch, stale = state["history"], tuple(state["best"]), state["best_epoch"], state["stale"]
        first_epoch, elapsed_prior = state["epoch"] + 1, state["elapsed_seconds"]
        del state
        print(f"Resume after completed epoch {first_epoch-1}; any interrupted epoch restarts", flush=True)
    else:
        history.append({"epoch": 0, "validation": validate(), "checkpoint": "original pretrained",
                        "learning_probe": learning_probe()})
        write_json(output / "history.json", history)
    start = time.perf_counter()
    for epoch in range(first_epoch, cfg["epochs"] + 1):
        if epoch > cfg["min_epochs"] and stale >= cfg["patience"]: break
        epoch_start = time.perf_counter()
        rng = random.Random(cfg["seed"] + epoch)
        shuffled = list(train)
        rng.shuffle(shuffled)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = [0.0, 0.0, 0.0]
        updates, skips = 0, 0
        grad_sum, finite_grad_steps = 0.0, 0
        for step in tqdm(range(micro_steps), desc=f"v4 {args.mode} epoch {epoch}/{cfg['epochs']}", mininterval=5):
            rows = shuffled[step*micro:(step+1)*micro]
            texts_q, texts_p = [], []
            for row in rows:
                selected = sample_group(row, candidates, pids, titles, rng, epoch, cfg)
                texts_q.extend([queries[row["dialogue_id"]]] * group)
                texts_p.extend(passages[pid] for pid in selected)
            inputs = tokenizer(texts_q, texts_p, padding=True, truncation="only_second", max_length=cfg["max_length"],
                               return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**inputs).logits.view(len(rows), group).float()
                loss, ce, pair = ranking_loss(logits, cfg["label_smoothing"], cfg["pairwise_weight"])
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss")
            window_start = (step // accum) * accum * micro
            window_queries = min(accum * micro, len(shuffled) - window_start)
            scaler.scale(loss * len(rows) / window_queries).backward()
            for i, value in enumerate((loss, ce, pair)): sums[i] += value.item() * len(rows)
            if (step+1) % accum == 0 or step+1 == micro_steps:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                if torch.isfinite(grad_norm):
                    grad_sum += float(grad_norm)
                    finite_grad_steps += 1
                old = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old:
                    scheduler.step()
                    updates += 1
                else: skips += 1
                optimizer.zero_grad(set_to_none=True)
        if updates == 0: raise FloatingPointError("No finite optimizer update this epoch")
        metrics = validate()
        probe_metrics = learning_probe()
        key = (metrics["exact_mrr@10"], metrics["exact_recall@10"])
        if key > best:
            best, best_epoch, stale = key, epoch, 0
            model.save_pretrained(output / f"adapter_epoch_{epoch}", safe_serialization=True)
            tokenizer.save_pretrained(output / f"adapter_epoch_{epoch}")
        else: stale += 1
        baseline = history[0]
        learning_signals = {
            "fixed_train_nll_change": probe_metrics["correct_query"]["fixed_group_nll"] - baseline["learning_probe"]["correct_query"]["fixed_group_nll"],
            "validation_mrr_change": metrics["exact_mrr@10"] - baseline["validation"]["exact_mrr@10"],
            "validation_recall_change": metrics["exact_recall@10"] - baseline["validation"]["exact_recall@10"],
            "query_dependence_warning": "Query swapping is a diagnostic, not a proof of semantic understanding or generalization",
        }
        history.append({"epoch": epoch, "validation": metrics, "loss": sums[0]/len(train),
                        "learning_probe": probe_metrics,
                        "learning_signals": learning_signals,
                        "mean_grad_norm_before_clip": grad_sum/finite_grad_steps if finite_grad_steps else None,
                        "ce": sums[1]/len(train), "pairwise": sums[2]/len(train), "updates": updates,
                        "overflow_skips": skips, "epoch_seconds": time.perf_counter()-epoch_start,
                        "stale_epochs": stale, "lr": scheduler.get_last_lr()[0]})
        state = {"specification": spec, "epoch": epoch, "history": history, "best": best,
                 "best_epoch": best_epoch, "stale": stale, "elapsed_seconds": elapsed_prior+time.perf_counter()-start,
                 "adapter": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()},
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        torch.save(state, output / "resume.tmp")
        (output / "resume.tmp").replace(resume)
        write_json(output / "resume.receipt.json", {"sha256": sha256(resume)})
        write_json(output / "history.json", history)
        del state
        print(f"Epoch {epoch}: {metrics}; best={best_epoch}; patience={stale}/{cfg['patience']}", flush=True)
    peak = torch.cuda.max_memory_allocated(device)/2**30
    del model, optimizer, scheduler, params
    if "logits" in locals():
        del logits, loss, ce, pair, inputs
    gc.collect()
    torch.cuda.empty_cache()
    base = AutoModelForSequenceClassification.from_pretrained(base_dir, local_files_only=True, dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, output / f"adapter_epoch_{best_epoch}").merge_and_unload(safe_merge=True)
    merged.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    write_json(output / "manifest.json", {"specification": spec, "selected_epoch": best_epoch,
               "checkpoint": checkpoint_relative,
               "trained_epochs": history[-1]["epoch"], "selection": "best trained epoch by validation MRR then Recall",
               "parameter_counts": parameter_counts, "peak_allocated_gib": peak,
               "runtime_seconds": elapsed_prior+time.perf_counter()-start,
               "checkpoint_fingerprint": model_fingerprint(checkpoint), "history_sha256": sha256(output / "history.json")})
    print("V4 TRAINING COMPLETE:", args.mode, "selected epoch", best_epoch, flush=True)


if __name__ == "__main__":
    main()
