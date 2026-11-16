import jieba
import re
import math

def tokenize_zh(text: str) -> list[str]:
    result = []
    tokens = jieba.lcut(text)
    for t in tokens:
        t = t.strip()
        if re.match(r'^[\W_]+$', t):
            continue
        if not t:
            continue
        result.append(t)
    return result

def term_freq(tokens: list[str]) -> dict[str, int]:
    result = {}
    for t in tokens:
        if t in result:        
            result[t] += 1        
        else:
            result[t] = 1      
    return result


def doc_freq(docs: dict[str, str]) -> dict[str, int]:
    result = {}
    for text in docs.values():
        tokens = tokenize_zh(text)
        uniques_tokens = set(tokens)
        for t in uniques_tokens:
            if t in result:
                result[t] += 1
            else:
                result[t] = 1
    return result



class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1 
        self.b = b
        self.N = len(self.docs)
        self.tokenized = {doc_id: tokenize_zh(text) for doc_id, text in docs.items()}
        self.doc_len = {doc_id: len(self.tokenized[doc_id]) for doc_id in docs}
        self.df = doc_freq(docs) 
        self.avg_len = sum(self.doc_len.values()) / (self.N or 1)
        self.tf = {doc_id: term_freq(tokens) for doc_id, tokens in self.tokenized.items()}


    def _score(self, query_tokens: list[str], doc_id: str) -> float:
        """算一篇文档对 query 的 BM25 得分。"""
        score = 0.0
        for q in query_tokens:
            tf = self.tf[doc_id].get(q, 0)
            if tf == 0:
                continue
            idf = math.log((self.N - self.df[q] + 0.5) / (self.df[q] + 0.5) + 1)
            tf_component = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avg_len)
                )
            score += idf * tf_component

        return score


    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not query or not self.docs:
            return []
        query_tokens = tokenize_zh(query)
        result = [(doc_id, self._score(query_tokens, doc_id)) for doc_id in self.docs]
        top = sorted(result, key=lambda x: x[1], reverse = True)
        return top[:top_k]


def rrf_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：只用名次融合多路检索结果。

    score(d) = Σ 1 / (k + rank_i(d))

    注意两点：
    1. 输入元组里的**分数会被完全忽略**，只取它的名次。这是 RRF 的设计前提——
       BM25 分数无上界、余弦相似度挤在高分区，两者量纲不可比；而"第几名"是无量纲的，
       跨检索器天然可比。所以调用方传什么分数都不影响结果（本项目向量路传的就是 0.0）。
    2. `k` 是钝化系数：k 越大，第 1 名和第 5 名的权重差距越小，融合越"民主"。
       60 是原论文的经验默认值。注意 rank 从 0 开始，所以 **k 必须 >= 1**——
       k=0 时首名权重是 1/(0+0)，会直接除零。
    """
    if k < 1:
        raise ValueError("k 必须 >= 1（rank 从 0 开始，k=0 时首名权重为 1/0）")

    scores: dict[str, float] = {}
    for ranked_list in result_lists:
        # 用 _ 显式标记「第二项存在但故意不用」，避免被误读成漏用了分数
        for rank, (doc_id, _unused_score) in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


