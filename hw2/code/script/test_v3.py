"""Fast CPU regression tests with tiny random models; no homework training or downloads."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from common import ROOT, read_json, write_json


class PipelineTests(unittest.TestCase):
    def test_registry_controls(self):
        from v3_experiments import experiments
        rows = experiments()
        for parent, child, expected in [("E02", "E03", {"dense_model"}),
                                         ("E04", "E05", {"recent_query_weight"}),
                                         ("E05", "E06", {"document_views"}),
                                         ("E07", "E08", {"reranker_model"})]:
            a, b = rows[parent]["config"], rows[child]["config"]
            self.assertEqual({k for k in a if a[k] != b[k]}, expected)

    def test_lora_backward_merge_reload(self):
        import torch
        from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification, AutoModelForSequenceClassification
        from peft import LoraConfig, get_peft_model, PeftModel
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        cfg = XLMRobertaConfig(vocab_size=40, hidden_size=16, num_hidden_layers=1,
                               num_attention_heads=2, intermediate_size=32, num_labels=1,
                               hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        with tempfile.TemporaryDirectory(prefix="hw2-lora-") as tmp:
            base = XLMRobertaForSequenceClassification(cfg)
            base.save_pretrained(Path(tmp) / "base")
            model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["query", "value"],
                                                   modules_to_save=["classifier"], bias="none"))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            inputs = {"input_ids": torch.randint(4, 40, (5, 12)), "attention_mask": torch.ones(5, 12, dtype=torch.long)}
            params = [p for p in model.parameters() if p.requires_grad]
            self.assertLess(sum(p.numel() for p in params), sum(p.numel() for p in model.parameters()))
            optimizer = torch.optim.AdamW(params, lr=1e-3)
            logits = model(**inputs).logits.view(1, 5)
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long), label_smoothing=0.1)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in params if p.grad is not None))
            optimizer.step()
            model.eval()
            expected = model(**inputs).logits.detach()
            model.save_pretrained(Path(tmp) / "adapter")
            fresh = XLMRobertaForSequenceClassification.from_pretrained(Path(tmp) / "base")
            merged = PeftModel.from_pretrained(fresh, Path(tmp) / "adapter").merge_and_unload(safe_merge=True).eval()
            torch.testing.assert_close(expected, merged(**inputs).logits, atol=1e-6, rtol=1e-4)
            merged.save_pretrained(Path(tmp) / "merged")
            reloaded = AutoModelForSequenceClassification.from_pretrained(Path(tmp) / "merged").eval()
            torch.testing.assert_close(expected, reloaded(**inputs).logits, atol=1e-6, rtol=1e-4)

    def test_all_inference_paths_and_cache(self):
        import torch
        from transformers import BertConfig, BertModel, BertForSequenceClassification, AutoTokenizer
        from retrieval_utils import retrieve_and_rerank
        from retrieval import save_submission
        from evaluate import load_submission
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="hw2-inference-") as tmp:
            folder = Path(tmp)
            tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/bge-base-en-v1.5", local_files_only=True)
            cfg = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=2, intermediate_size=32, num_labels=1)
            for name, model in [("dense", BertModel(cfg)), ("reranker", BertForSequenceClassification(cfg))]:
                model.save_pretrained(folder / name)
                tokenizer.save_pretrained(folder / name)
            config = read_json(ROOT / "configs/v3/inference.json")
            config.update(device="cpu", dense_model=str(folder / "dense"), reranker_model=str(folder / "reranker"),
                          cache_dir=str(folder / "cache"), dense_max_length=32, rerank_max_length=48,
                          query_max_length=16, retrieve_k=15, rerank_k=12)
            products = {f"p{i:02d}": f"Product {i}\nHair shampoo and conditioner for dry hair." for i in range(20)}
            queries = [{"qid": "q0", "query": "user: I need shampoo.\nuser: For dry hair."}]
            env = dict(os.environ)
            for key in list(env):
                if key.startswith("HW2_"):
                    del env[key]
            env.update(HW2_CONFIG=str(folder / "config.json"), HW2_RUN_METADATA=str(folder / "metadata.json"),
                       HW2_CANDIDATES_OUTPUT=str(folder / "candidates.json"))
            with patch.dict(os.environ, env, clear=True):
                for strategy in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
                    config["strategy"] = strategy
                    write_json(folder / "config.json", config)
                    first = retrieve_and_rerank(queries, products)
                    second = retrieve_and_rerank(queries, products)
                    self.assertEqual(first, second)
                    self.assertEqual(len(set(first[0]["ranked_pids"])), 10)
                    save_submission(folder / "predictions.json", first, top_k=10)
                    load_submission(folder / "predictions.json", queries, set(products))
                    if strategy != "bm25":
                        self.assertTrue(read_json(folder / "metadata.json")["cache_hit"])
                        self.assertEqual(len(read_json(folder / "candidates.json")["q0"]), 15)
                config.update(document_views=1, recent_query_weight=0, bm25_bigrams=False,
                              title_boost=1, turn_weights=False)
                write_json(folder / "config.json", config)
                retrieve_and_rerank(queries, products)
                self.assertFalse(read_json(folder / "metadata.json")["cache_hit"])
                self.assertEqual(retrieve_and_rerank([], products), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
