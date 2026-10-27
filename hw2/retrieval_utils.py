#!/usr/bin/env python3
"""Student-editable retrieval and reranking implementation."""

from __future__ import annotations

from typing import Any


DEFAULT_TOP_K = 10


def retrieve_and_rerank(
    queries: list[dict[str, Any]],
    passages: dict[str, str],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Return one final ranked PID list per query."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(passages) < top_k:
        raise ValueError(
            f"product collection has {len(passages)} items, fewer than top_k={top_k}"
        )

    # ==================== STUDENT CODE START ====================
    # Only modify the code inside this block.
    #
    # Implement retrieval and optional reranking here. Return one dictionary
    # per query in the following form:
    # {
    #     "qid": query["qid"],
    #     "ranked_pids": [pid_1, pid_2, ..., pid_top_k],
    # }
    # This is a query-aware BM25 implementation.  Building postings only for
    # terms that occur in this batch of queries keeps memory use modest even
    # for the 100k-product private collection.
    import math
    import re
    from collections import Counter, defaultdict
    import gc
    import hashlib
    import json
    import os
    from pathlib import Path
    import time

    import numpy as np

    if not queries:
        return []
    started = time.perf_counter()
    root = Path(__file__).resolve().parent
    config_path = Path(os.environ.get("HW2_CONFIG", root / "configs/inference.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["device"] = os.environ.get("HW2_DEVICE", config["device"])
    config["cache_dir"] = os.environ.get("HW2_CACHE_DIR", config["cache_dir"])
    config["encode_batch_size"] = int(os.environ.get("HW2_ENCODE_BATCH", config["encode_batch_size"]))
    config["rerank_batch_size"] = int(os.environ.get("HW2_RERANK_BATCH", config["rerank_batch_size"]))
    strategy = os.environ.get("HW2_STRATEGY", config["strategy"])
    if strategy not in {"bm25", "dense", "hybrid", "hybrid_rerank"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    candidate_k = min(len(passages), max(top_k, int(config["retrieve_k"])))
    rerank_k = min(candidate_k, max(top_k, int(config["rerank_k"])))

    def local_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    token_re = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
    stopwords = {
        "a", "about", "all", "also", "am", "an", "and", "anything", "are",
        "as", "at", "be", "been", "but", "by", "can", "do", "does", "for",
        "from", "get", "got", "had", "has", "have", "having", "help", "i",
        "if", "in", "is", "it", "its", "just", "like", "me", "maybe", "more",
        "my", "need", "not", "of", "oh", "okay", "on", "one", "or", "other",
        "prefer", "probably", "really", "recommend", "right", "should", "so",
        "some", "something", "sounds", "sure", "than", "that", "the", "them",
        "then", "there", "these", "they", "think", "this", "to", "too", "try",
        "use", "used", "using", "want", "was", "way", "what", "when", "which",
        "with", "would", "yeah", "yes", "you", "your",
        # Speaker labels inserted by eval_utils.build_eval_query.
        "user", "assistant",
    }

    def normalize_token(token: str) -> str:
        token = token.strip("'-")
        if token.endswith("'s"):
            token = token[:-2]
        # Conservative stemming joins common catalogue/query variants without
        # damaging short product terms (e.g. gel, oil, lash).
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 6 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 5 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        return token

    def tokenize(text: str) -> list[str]:
        return [
            normalized
            for raw in token_re.findall(text.lower())
            if (normalized := normalize_token(raw))
            and normalized not in stopwords
            and len(normalized) > 1
        ]

    def features(text: str) -> list[str]:
        words = tokenize(text)
        # Phrases are especially important here: "body oil", "eye liner", and
        # "hair spray" share unigrams but have different functional roles.
        return words + ([f"@{left}_{right}" for left, right in zip(words, words[1:])]
                        if config.get("bm25_bigrams", True) else [])

    # Retain turns separately so that the latest user answer (which commonly
    # supplies the decisive constraint) can receive extra weight.
    query_counters: list[Counter[str]] = []
    query_vocabulary: set[str] = set()
    for query in queries:
        lines = str(query["query"]).splitlines()
        weighted: Counter[str] = Counter()
        for line_number, line in enumerate(lines):
            role, _, content = line.partition(":")
            line_features = features(content if content else line)
            distance_from_end = len(lines) - 1 - line_number
            # Synthetic dialogues often mention an old, unrelated purchase in
            # the opening turn.  The last question/answer pair identifies the
            # actual request and therefore receives the most weight.
            weight = 3.0 if distance_from_end == 0 else (
                2.0 if distance_from_end == 1 else 0.75
            )
            if role.strip().lower() == "user":
                weight *= 1.1
            if not config.get("turn_weights", True):
                weight = 1.0
            for term in line_features:
                weighted[term] += weight
        query_counters.append(weighted)
        query_vocabulary.update(weighted)

    # Sorted IDs make ties independent of JSONL order. IDs are used only as
    # identifiers, never as relevance features or a query-to-target lookup.
    pids = sorted(passages)
    postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    doc_lengths: list[int] = []
    for doc_id, pid in enumerate(pids):
        text = passages[pid]
        title, _, description = text.partition("\n")
        title_tokens = tokenize(title)
        body_tokens = tokenize(description)
        title_counts = Counter(features(title))
        body_counts = Counter(features(description))
        doc_lengths.append(len(title_tokens) + len(body_tokens))
        for term in query_vocabulary.intersection(title_counts.keys() | body_counts.keys()):
            # A title occurrence is far more diagnostic than boilerplate in a
            # long Amazon description, but body evidence still matters.
            weighted_tf = float(config.get("title_boost", 3.0)) * title_counts[term] + body_counts[term]
            postings[term].append((doc_id, weighted_tf))

    collection_size = len(pids)
    average_length = max(sum(doc_lengths) / collection_size, 1.0)
    k1, b = 1.5, 0.65
    length_norms = np.asarray([
        1.0 - b + b * document_length / average_length
        for document_length in doc_lengths
    ], dtype=np.float32)
    # Precompute term contributions once, not once per query. Compact arrays
    # also avoid millions of Python arithmetic operations at search time.
    impacts = {}
    for term, posting in postings.items():
        ids = np.fromiter((entry[0] for entry in posting), dtype=np.int32)
        tf = np.fromiter((entry[1] for entry in posting), dtype=np.float32)
        idf = math.log1p((collection_size - len(ids) + 0.5) / (len(ids) + 0.5))
        impacts[term] = (ids, idf * tf * (k1 + 1) / (tf + k1 * length_norms[ids]))
    del postings

    def best_ids(scores: np.ndarray, count: int) -> list[int]:
        # Stable deterministic ties, including all-zero queries.
        return np.argsort(-scores, kind="stable")[:count].tolist()

    lexical_rankings = []
    for query, term_weights in zip(queries, query_counters):
        scores = np.zeros(collection_size, dtype=np.float32)
        for term, query_weight in term_weights.items():
            if term not in impacts:
                continue
            ids, values = impacts[term]
            scores[ids] += values * (1.0 + math.log1p(query_weight))
        lexical_rankings.append(best_ids(scores, candidate_k))
    del impacts
    rankings = lexical_rankings
    timings = {"lexical_seconds": time.perf_counter() - started}
    cache_hit = False
    model_fingerprints = {}

    if strategy != "bm25":
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        from tqdm import tqdm

        device = os.environ.get("HW2_DEVICE", config["device"])
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; activate .venv or explicitly set HW2_DEVICE=cpu")
        torch.manual_seed(config["seed"])
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        if device.startswith("cuda"):
            # Initialise the CUDA context before resetting allocator statistics.
            # PyTorch 2.8 otherwise raises "Invalid device argument" on first use.
            torch.cuda.set_device(torch.device(device))
            torch.cuda.reset_peak_memory_stats(device)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        dense_dir = local_path(config["dense_model"])
        if not (dense_dir / "config.json").is_file():
            raise FileNotFoundError(f"Model missing: {dense_dir}. Run bash download.sh first.")

        def fingerprint(path: Path) -> str:
            digest = hashlib.sha256()
            for file in sorted(path.iterdir()):
                if file.is_file() and file.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}:
                    digest.update(file.name.encode())
                    with file.open("rb") as handle:
                        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                            digest.update(block)
            return digest.hexdigest()

        dense_hash = fingerprint(dense_dir)
        model_fingerprints["dense"] = dense_hash
        corpus_hash = hashlib.sha256()
        for pid in pids:
            corpus_hash.update(json.dumps([pid, passages[pid]], ensure_ascii=False).encode())
        cache_key = hashlib.sha256(json.dumps([
            corpus_hash.hexdigest(), dense_hash, config["dense_max_length"], "cls-normalized-v2",
            config.get("document_views", 1), config.get("tail_words", 180), str(dtype),
            config.get("dense_pooling", "cls"), config.get("passage_prefix", ""),
        ]).encode()).hexdigest()
        views = int(config.get("document_views", 1))
        if views not in (1, 2):
            raise ValueError("document_views must be 1 (head) or 2 (head + title/tail)")
        cache_dir = local_path(config["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        embedding_file = cache_dir / f"{cache_key}.npy"
        tokenizer = AutoTokenizer.from_pretrained(dense_dir, local_files_only=True)
        encoder = AutoModel.from_pretrained(dense_dir, local_files_only=True, dtype=dtype).to(device).eval()

        def encode(texts: list[str], description: str) -> np.ndarray:
            batch_size = int(os.environ.get("HW2_ENCODE_BATCH", config["encode_batch_size"]))
            if batch_size <= 0:
                raise ValueError("Encode batch size must be positive")
            output = np.empty((len(texts), encoder.config.hidden_size), dtype=np.float32)
            # Length ordering reduces padding without changing output order.
            order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
            for offset in tqdm(range(0, len(order), batch_size), desc=description):
                indices = order[offset:offset + batch_size]
                inputs = tokenizer([texts[i] for i in indices], padding=True, truncation=True,
                                   max_length=config["dense_max_length"], return_tensors="pt").to(device)
                with torch.inference_mode():
                    states = encoder(**inputs).last_hidden_state.float()
                    if config.get("dense_pooling", "cls") == "mean":
                        mask = inputs["attention_mask"].unsqueeze(-1)
                        hidden = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
                    elif config.get("dense_pooling", "cls") == "cls":
                        hidden = states[:, 0]
                    else:
                        raise ValueError("dense_pooling must be cls or mean")
                    vectors = torch.nn.functional.normalize(hidden, dim=-1)
                output[indices] = vectors.cpu().numpy()
            return output

        if embedding_file.exists():
            embeddings = np.load(embedding_file, allow_pickle=False)
            cache_hit = True
            if embeddings.shape != (len(pids) * views, encoder.config.hidden_size):
                raise ValueError(f"Invalid cache shape; remove only {embedding_file} and retry")
        else:
            texts = [passages[pid] for pid in pids]
            if views == 2:
                texts += [passages[pid].split("\n", 1)[0] + "\n" +
                          " ".join(passages[pid].split()[-int(config.get("tail_words", 180)):])
                          for pid in pids]
            embeddings = encode([config.get("passage_prefix", "") + t for t in texts], "Encode products").astype(np.float16)
            temporary = embedding_file.with_suffix(f".{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, embeddings.astype(np.float16), allow_pickle=False)
            temporary.replace(embedding_file)

        instruction = config.get("query_prefix", "Represent this sentence for searching relevant passages: ")
        full = [str(query["query"]) for query in queries]
        if config.get("dense_query_mode", "full") == "user":
            full = ["\n".join(line for line in text.splitlines() if line.partition(":")[0].lower().strip() == "user") or text
                    for text in full]
        recent = ["\n".join(text.splitlines()[-3:]) for text in full]
        recent_weight = float(config.get("recent_query_weight", 0.5))
        use_recent = strategy != "dense" and recent_weight > 0
        q_vectors = encode([instruction + text for text in full + (recent if use_recent else [])], "Encode dialogues")
        del encoder, tokenizer
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        corpus_tensor = torch.as_tensor(embeddings, dtype=dtype, device=device)
        rankings = []
        fusion_scores = []
        rrf_k = int(config["rrf_k"])
        # Only O(N*dimension + batch_queries*N) memory, no full Q-by-N matrix.
        for offset in range(0, len(queries), 16):
            stop = min(offset + 16, len(queries))
            a = torch.as_tensor(q_vectors[offset:stop], dtype=dtype, device=device)
            with torch.inference_mode():
                sim_full = (a @ corpus_tensor.T).reshape(stop-offset, views, len(pids)).amax(1).float().cpu().numpy()
                if use_recent:
                    b_vectors = torch.as_tensor(q_vectors[len(queries)+offset:len(queries)+stop], dtype=dtype, device=device)
                    sim_recent = (b_vectors @ corpus_tensor.T).reshape(stop-offset, views, len(pids)).amax(1).float().cpu().numpy()
            for row, qi in enumerate(range(offset, stop)):
                if strategy == "dense":
                    rankings.append(best_ids(sim_full[row], candidate_k))
                    continue
                fused = defaultdict(float)
                dense_ids = best_ids(sim_full[row], candidate_k)
                lists = [(lexical_rankings[qi], float(config.get("lexical_weight", 1.0))),
                         (dense_ids, float(config.get("dense_weight", 1.5)))]
                if use_recent:
                    lists.append((best_ids(sim_recent[row], candidate_k), recent_weight))
                for ids, weight in lists:
                    for rank, doc_id in enumerate(ids, 1):
                        fused[doc_id] += weight / (rrf_k + rank)
                selected = sorted(fused, key=lambda i: (-fused[i], pids[i]))[:candidate_k]
                if config.get("candidate_policy", "rrf") == "protected_union":
                    lex_n = int(config["lexical_quota"])
                    dense_n = int(config["dense_quota"])
                    if lex_n < 0 or dense_n < 0 or lex_n + dense_n > int(config["rerank_k"]):
                        raise ValueError("Protected source quotas must fit the rerank budget")
                    mandatory = set(lexical_rankings[qi][:lex_n] + dense_ids[:dense_n])
                    protected = sorted(mandatory, key=lambda i: (-fused[i], pids[i]))
                    selected = (protected + [i for i in selected if i not in mandatory])[:candidate_k]
                    # Keep all protected candidates, fill the remainder by RRF,
                    # then retain RRF ordering WITHIN the rerank shortlist.
                    budget = min(int(config["rerank_k"]), len(selected))
                    selected[:budget] = sorted(selected[:budget], key=lambda i: (-fused[i], pids[i]))
                rankings.append(selected)
                fusion_scores.append(fused)
        del corpus_tensor, embeddings, q_vectors
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        timings["dense_and_fusion_seconds"] = time.perf_counter() - started - timings["lexical_seconds"]

        if os.environ.get("HW2_CANDIDATES_OUTPUT"):
            trace = Path(os.environ["HW2_CANDIDATES_OUTPUT"])
            trace.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text(json.dumps({q["qid"]: [pids[i] for i in r]
                                         for q, r in zip(queries, rankings)}), encoding="utf-8")

        if strategy == "hybrid_rerank":
            rerank_started = time.perf_counter()
            reranker_dir = local_path(config["reranker_model"])
            if not (reranker_dir / "config.json").exists():
                raise FileNotFoundError(f"Missing reranker: {reranker_dir}")
            model_fingerprints["reranker"] = fingerprint(reranker_dir)
            tokenizer = AutoTokenizer.from_pretrained(reranker_dir, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                reranker_dir, local_files_only=True, dtype=dtype,
            ).to(device).eval()
            batch_size = int(os.environ.get("HW2_RERANK_BATCH", config["rerank_batch_size"]))
            if batch_size <= 0:
                raise ValueError("Reranker batch size must be positive")
            for qi in tqdm(range(len(queries)), desc="Rerank dialogues"):
                # Preserve the opening product request and latest constraints.
                # The same truncation is used during train-only fine-tuning.
                query_ids = tokenizer.encode(str(queries[qi]["query"]), add_special_tokens=False)
                limit = int(config["query_max_length"])
                if len(query_ids) > limit:
                    query_ids = query_ids[:limit // 2] + query_ids[-(limit - limit // 2):]
                query_text = tokenizer.decode(query_ids, skip_special_tokens=True)
                candidates = rankings[qi][:rerank_k]
                logits = []
                for offset in range(0, len(candidates), batch_size):
                    batch_ids = candidates[offset:offset + batch_size]
                    inputs = tokenizer([query_text] * len(batch_ids),
                                       [passages[pids[i]] for i in batch_ids], padding=True,
                                       truncation="only_second", max_length=config["rerank_max_length"],
                                       return_tensors="pt").to(device)
                    with torch.inference_mode():
                        logits.extend(model(**inputs).logits.view(-1).float().cpu().tolist())
                # Raw cross-encoder relevance determines order; fusion breaks ties.
                scored = sorted(zip(candidates, logits),
                                key=lambda pair: (-pair[1], -fusion_scores[qi][pair[0]], pids[pair[0]]))
                alpha = float(config.get("rerank_blend", 0.0))
                if not 0 <= alpha <= 1:
                    raise ValueError("rerank_blend must be in [0,1]")
                if alpha:
                    prior = {doc: rank for rank, doc in enumerate(candidates, 1)}
                    cross = {doc: rank for rank, (doc, _) in enumerate(scored, 1)}
                    scored.sort(key=lambda pair: (-(1-alpha)/(rrf_k+cross[pair[0]])
                                                   -alpha/(rrf_k+prior[pair[0]]), pids[pair[0]]))
                rankings[qi] = [doc_id for doc_id, _ in scored] + rankings[qi][rerank_k:]
            del model, tokenizer
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            timings["rerank_seconds"] = time.perf_counter() - rerank_started
        timings["peak_cuda_allocated_gib"] = (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.startswith("cuda") else None
        )
        timings["peak_cuda_reserved_gib"] = (
            torch.cuda.max_memory_reserved(device) / 2**30 if device.startswith("cuda") else None
        )

    results = [{"qid": query["qid"], "ranked_pids": [pids[i] for i in ranking[:top_k]]}
               for query, ranking in zip(queries, rankings)]
    timings["total_seconds"] = time.perf_counter() - started
    print(json.dumps({"strategy": strategy, "cache_hit": cache_hit, **timings}), flush=True)
    if os.environ.get("HW2_RUN_METADATA"):
        destination = Path(os.environ["HW2_RUN_METADATA"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"config": config, "strategy": strategy,
            "query_count": len(queries), "product_count": len(pids), "cache_hit": cache_hit,
            "model_fingerprints": model_fingerprints, "timings": timings}, indent=2), encoding="utf-8")
    # ===================== STUDENT CODE END =====================

    return results
