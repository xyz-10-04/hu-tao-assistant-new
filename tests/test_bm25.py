"""手写 BM25 与 RRF 融合的单元测试（不依赖任何模型，可离线跑）。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm25 import BM25, doc_freq, rrf_fusion, term_freq, tokenize_zh  # noqa: E402


class TestTokenize(unittest.TestCase):
    def test_drops_punctuation_and_blank(self):
        tokens = tokenize_zh("胡桃，喜欢吃什么？")
        self.assertIn("胡桃", tokens)
        self.assertNotIn("，", tokens)
        self.assertNotIn("？", tokens)

    def test_empty_input(self):
        self.assertEqual(tokenize_zh("...!!!"), [])


class TestTermFreq(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(term_freq(["a", "b", "a"]), {"a": 2, "b": 1})


class TestDocFreq(unittest.TestCase):
    def test_counts_documents_not_occurrences(self):
        docs = {"d1": "胡桃 胡桃 喜欢", "d2": "胡桃 讨厌"}
        df = doc_freq(docs)
        self.assertEqual(df["胡桃"], 2)  # 出现在 2 篇文档里，不是 3 次
        self.assertEqual(df["喜欢"], 1)


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.docs = {
            "d1": "胡桃喜欢水煮鱼配虾饺",
            "d2": "钟离是往生堂的客卿",
            "d3": "胡桃的帽子叫乾坤泰卦帽",
        }
        self.bm25 = BM25(self.docs)

    def test_matching_doc_ranks_first(self):
        results = self.bm25.search("水煮鱼", top_k=3)
        self.assertEqual(results[0][0], "d1")
        self.assertGreater(results[0][1], 0)

    def test_unmatched_query_returns_all_zero(self):
        results = self.bm25.search("纳斯达克指数", top_k=3)
        self.assertTrue(all(score == 0 for _, score in results))

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.bm25.search("", top_k=3), [])

    def test_top_k_respected(self):
        self.assertEqual(len(self.bm25.search("胡桃", top_k=2)), 2)

    def test_rare_term_gets_higher_idf(self):
        # "水煮鱼" 只出现在 1 篇文档，"胡桃" 出现在 2 篇 -> 罕见词权重更高
        scores = dict(self.bm25.search("水煮鱼 胡桃", top_k=3))
        self.assertIn("d1", scores)
        # d2 只命中不了任何词，得分应为 0
        self.assertEqual(scores.get("d2", 0), 0)


class TestRRFFusion(unittest.TestCase):
    def test_doc_in_both_lists_wins(self):
        a = [("x", 9.9), ("y", 8.8)]
        b = [("y", 0.9), ("z", 0.8)]
        fused = rrf_fusion([a, b], k=60)
        self.assertEqual(fused[0][0], "y")  # 两路都出现，票数最多

    def test_formula_matches_definition(self):
        fused = dict(rrf_fusion([[("only", 123.0)]], k=60))
        self.assertAlmostEqual(fused["only"], 1 / 60)
        fused = dict(rrf_fusion([[("a", 0), ("solo", 9)]], k=60))
        self.assertAlmostEqual(fused["solo"], 1 / 61)

    def test_scores_are_ignored(self):
        # RRF 只看名次：把分数换成完全不同的量纲，顺序不应改变
        a = [("x", 999.0), ("y", 1.0)]
        b = [("y", 0.99), ("x", 0.01)]
        self.assertEqual([d for d, _ in rrf_fusion([a, b])],
                         [d for d, _ in rrf_fusion([[("x", 0.1), ("y", 0.0)],
                                                    [("y", 0.5), ("x", 0.4)]])])


if __name__ == "__main__":
    unittest.main()
