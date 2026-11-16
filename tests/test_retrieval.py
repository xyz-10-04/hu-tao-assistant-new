"""检索链路的契约测试。

需要本地 BGE 模型与 FAISS 索引，缺失时自动跳过（CI 里不跑）。
"""
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

MODEL_READY = os.path.isdir(os.path.join(BASE_DIR, "models", "bge-small-zh-v1.5"))
INDEX_READY = os.path.isfile(os.path.join(BASE_DIR, "faiss_index", "index.faiss"))


@unittest.skipUnless(MODEL_READY and INDEX_READY, "本地模型或索引缺失，跳过检索测试")
class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rag_build

        cls.rag = rag_build

    def test_return_count(self):
        results = self.rag.search_hu_tao("胡桃喜欢吃什么", k=5, return_ids=True)
        self.assertLessEqual(len(results), 5)
        self.assertGreater(len(results), 0)

    def test_ids_exist_in_docs_dict(self):
        """检索返回的 doc_id 必须能在当前知识库切分里找到，否则拼 prompt 会 KeyError。"""
        _, docs_dict, _ = self.rag._get_store(None)
        results = self.rag.search_hu_tao("胡桃喜欢吃什么", k=8, return_ids=True)
        for doc_id, _score in results:
            self.assertIn(doc_id, docs_dict)

    def test_deterministic(self):
        a = self.rag.search_hu_tao("往生堂的含义", k=3, return_ids=True)
        b = self.rag.search_hu_tao("往生堂的含义", k=3, return_ids=True)
        self.assertEqual([x[0] for x in a], [x[0] for x in b])

    def test_empty_query(self):
        self.assertEqual(self.rag.search_hu_tao("   ", k=3), [])

    def test_doc_id_is_stable_across_batches(self):
        """回归：doc_id 必须只由「文件名 + 文件内序号」决定，与切分批次无关。"""
        full = self.rag.split_documents(self.rag.load_documents())
        one_cat = self.rag.split_documents(self.rag.load_documents("角色评价"))
        full_map = {c.metadata["doc_id"]: c.page_content for c in full}
        for chunk in one_cat:
            doc_id = chunk.metadata["doc_id"]
            self.assertIn(doc_id, full_map, f"{doc_id} 在分类切分里存在但全量切分里没有")
            self.assertEqual(full_map[doc_id], chunk.page_content)

    def test_known_gold_hit(self):
        """已知答案的样本：问"胡桃喜欢吃什么"必须召回水煮鱼那条台词。"""
        results = self.rag.search_hu_tao("胡桃喜欢吃什么", k=3)
        self.assertTrue(any("水煮鱼" in text for text in results))


if __name__ == "__main__":
    unittest.main()
