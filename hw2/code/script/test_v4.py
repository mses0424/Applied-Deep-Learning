"""Fast controls/loss/sampling and protected-union regression tests."""
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from common import ROOT, read_json, write_json
from v4_common import candidate_configs, ranking_loss, sample_group


class V4Tests(unittest.TestCase):
    def test_controlled_retriever_configs(self):
        c = candidate_configs()
        self.assertTrue(all(v["rerank_k"] == 128 for v in c.values()))
        self.assertEqual({k for k in c["R4_e5_union"] if c["R4_e5_union"][k] != c["R5_e5_user"].get(k)}, set())
        self.assertEqual(c["R5_e5_user"]["dense_query_mode"], "user")

    def test_negative_curriculum_distinctness_and_titles(self):
        cfg = read_json(ROOT / "configs/v4/training.json")
        pids = [f"p{i}" for i in range(400)]
        titles = {p: p for p in pids}
        titles["p33"] = titles["p0"]
        candidates = {"q": pids}
        for epoch in (1, 2, 8):
            group = sample_group({"dialogue_id": "q", "target": "p0"}, candidates, pids, titles,
                                 random.Random(2026), epoch, cfg)
            self.assertEqual(group[0], "p0")
            self.assertEqual(len(set(group)), 7)
            self.assertNotIn("p33", group)
            bounds = (32, 160) if epoch == 1 else (5, 64)
            self.assertTrue(all(bounds[0] <= int(p[1:]) < bounds[1] for p in group[1:5]))

    def test_pairwise_loss_and_gradient(self):
        import torch
        scores = torch.tensor([[0., 1., -1., 2., 3., -2., 0.]], requires_grad=True)
        loss, ce, pair = ranking_loss(scores, 0.1, 0.2)
        torch.testing.assert_close(loss, ce + .2*pair)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertLess(scores.grad[0, 0].item(), 0)

    def test_protected_union_and_mean_pooling(self):
        import torch
        from transformers import BertConfig, BertModel, AutoTokenizer
        from retrieval_utils import retrieve_and_rerank
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="hw2-v4-test-") as directory:
            tmp = Path(directory)
            tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/bge-base-en-v1.5", local_files_only=True)
            model = BertModel(BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=16, num_hidden_layers=1,
                                        num_attention_heads=2, intermediate_size=32))
            model.save_pretrained(tmp / "encoder")
            tokenizer.save_pretrained(tmp / "encoder")
            cfg = {**candidate_configs()["R4_e5_union"], "device": "cpu", "dense_model": str(tmp / "encoder"),
                   "cache_dir": str(tmp / "cache"), "rerank_k": 10, "retrieve_k": 20,
                   "lexical_quota": 7, "dense_quota": 3, "dense_max_length": 32}
            products = {f"p{i}": f"Product {i}\n" + ("shampoo dry hair " * (i+1)) for i in range(24)}
            queries = [{"qid": "q", "query": "user: dry hair shampoo"}]
            env = {k: v for k, v in os.environ.items() if not k.startswith("HW2_")}
            env.update(HW2_CONFIG=str(tmp / "config.json"), HW2_CANDIDATES_OUTPUT=str(tmp / "candidates.json"))
            with patch.dict(os.environ, env, clear=True):
                write_json(tmp / "config.json", {**cfg, "strategy": "bm25"})
                lexical = retrieve_and_rerank(queries, products)[0]["ranked_pids"]
                write_json(tmp / "config.json", {**cfg, "strategy": "dense"})
                dense = retrieve_and_rerank(queries, products)[0]["ranked_pids"]
                write_json(tmp / "config.json", cfg)
                actual = retrieve_and_rerank(queries, products)[0]["ranked_pids"]
                self.assertTrue(set(lexical[:7] + dense[:3]).issubset(actual))
                self.assertEqual(len(set(actual)), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
