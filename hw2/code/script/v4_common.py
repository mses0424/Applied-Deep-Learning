"""Shared v4 controls; training uses only released train data, never public labels."""
import re
from common import ROOT, read_json, write_json, sha256, model_fingerprint


def candidate_configs():
    base = read_json(ROOT / "configs/v3/inference.json")
    base.update(strategy="hybrid", dense_model="models/bge-base-en-v1.5", document_views=1,
                recent_query_weight=0.0, lexical_weight=3.0, dense_weight=1.0,
                retrieve_k=400, rerank_k=128, rerank_max_length=512, query_max_length=192,
                dense_max_length=512, encode_batch_size=8, rerank_batch_size=4,
                candidate_policy="protected_union", lexical_quota=96, dense_quota=32,
                rerank_blend=0.0)
    configs = {
        "R1_lexical": {**base, "lexical_quota": 128, "dense_quota": 0},
        "R2_weighted_rrf": {**base, "candidate_policy": "rrf"},
        "R3_bge_union": dict(base),
        "R4_e5_union": {**base, "dense_model": "models/e5-base-v2", "dense_pooling": "mean",
                        "query_prefix": "query: ", "passage_prefix": "passage: "},
    }
    configs["R5_e5_user"] = {**configs["R4_e5_union"], "dense_query_mode": "user"}
    return configs


def signature(config, models=True):
    files = ["retrieval_utils.py", "code/script/common.py", "code/script/v4_common.py",
             "code/script/v4_pipeline.py", "code/script/train_v4.py", "prepared/manifest.json"]
    value = {"config": config, "hashes": {p: sha256(ROOT / p) for p in files}}
    if models:
        value["models"] = {k: model_fingerprint(ROOT / config[k]) for k in
                           ("dense_model", "reranker_model", "initial_model") if k in config}
    return value


def inference_config(model=None):
    cfg = read_json(ROOT / "configs/v4/candidates_selected.json")
    cfg["strategy"] = "hybrid_rerank"
    if model:
        cfg["reranker_model"] = model
    return cfg


def title_key(text):
    return re.sub(r"[^a-z0-9]+", " ", text.split("\n", 1)[0].lower()).strip()


def sample_group(row, candidates, pids, titles, rng, epoch, cfg):
    gold = row["target"]
    chosen = [gold]

    def valid(pid):
        return pid not in chosen and (not titles[gold] or titles[pid] != titles[gold])

    ranks = candidates[row["dialogue_id"]]
    # Epoch 1 emphasises less ambiguous, easier retrieved negatives; later
    # epochs increase difficulty. Ranks 1-5 remain excluded from hard pools.
    hard_pool = ranks[32:160] if epoch == 1 else ranks[5:64]
    medium_pool = ranks[64:256]
    for pool, count in [(hard_pool, cfg["hard_negatives"]), (medium_pool, cfg["medium_negatives"])]:
        eligible = list(dict.fromkeys(pid for pid in pool if valid(pid)))
        chosen.extend(rng.sample(eligible, min(count, len(eligible))))
    group_size = 1 + cfg["hard_negatives"] + cfg["medium_negatives"] + cfg["random_negatives"]
    attempts = 0
    while len(chosen) < group_size:
        pid = rng.choice(pids)
        if valid(pid):
            chosen.append(pid)
        attempts += 1
        if attempts > 10000:
            raise ValueError("Too few distinct-title non-target products to construct a training group")
    return chosen


def ranking_loss(logits, smoothing, pairwise_weight):
    import torch
    import torch.nn.functional as F
    ce = F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device),
                         label_smoothing=smoothing)
    pair = F.softplus(logits[:, 1:] - logits[:, :1]).mean()
    return ce + pairwise_weight * pair, ce, pair
