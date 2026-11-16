"""检索链路可视化：把 BM25 → 向量 → RRF → 精排 每一步的中间结果打出来。

为什么需要它：
    1. 别人 clone 仓库后，不用配 API Key 就能验证"检索确实在工作"；
    2. 自己 debug 时能看清是哪一步把正确答案弄丢了；
    3. 面试前跑一遍，能把"检索四步"从背的变成看到的。

用法：
    python scripts/demo_retrieval.py "胡桃喜欢吃什么"
    python scripts/demo_retrieval.py "行秋怎么看待你" --category character
    python scripts/demo_retrieval.py "往生堂的含义" --top-k 3
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bm25 import rrf_fusion  # noqa: E402
import rag_build  # noqa: E402


def preview(doc_dict, doc_id, width=58):
    return doc_dict[doc_id].replace("\n", " ")[:width]


def main():
    parser = argparse.ArgumentParser(description="打印检索链路的每一步中间结果")
    parser.add_argument("query", help="用户问题")
    parser.add_argument("--category", default=None,
                        choices=[None, "character", "story", "misc"],
                        help="限定检索范围（默认检索全部知识）")
    parser.add_argument("--top-k", type=int, default=5, help="最终返回条数")
    args = parser.parse_args()

    print("=" * 92)
    print(f"问题：{args.query}")
    print(f"范围：{args.category or '全部知识'}   最终返回：{args.top_k} 条")
    print("=" * 92)

    t0 = time.time()
    vectordb, docs_dict, bm25 = rag_build._get_store(args.category)
    load_cost = time.time() - t0
    print(f"\n[0] 索引与 BM25 就绪：{len(docs_dict)} 个语义块，耗时 {load_cost:.2f}s"
          f"（首次运行含模型加载）")

    # ---------- ① BM25 ----------
    t = time.time()
    bm25_results = bm25.search(args.query, top_k=rag_build.BM25_TOP_K)
    bm25_cost = time.time() - t
    print(f"\n[1] BM25 关键词召回：top {rag_build.BM25_TOP_K}，耗时 {bm25_cost*1000:.1f}ms")
    for i, (doc_id, score) in enumerate(bm25_results, 1):
        print(f"    {i:2d}. {score:7.3f}  {doc_id:22s} {preview(docs_dict, doc_id)}")

    # ---------- ② 向量 ----------
    t = time.time()
    faiss_docs = vectordb.as_retriever(
        search_kwargs={"k": rag_build.FAISS_TOP_K}
    ).invoke(args.query)
    faiss_cost = time.time() - t
    faiss_results = [(d.metadata["doc_id"], 0.0) for d in faiss_docs]
    print(f"\n[2] BGE 向量召回：top {rag_build.FAISS_TOP_K}，耗时 {faiss_cost*1000:.1f}ms")
    for i, (doc_id, _) in enumerate(faiss_results, 1):
        print(f"    {i:2d}. {'(分数交给距离)':>7s}  {doc_id:22s} {preview(docs_dict, doc_id)}")

    # ---------- ③ RRF ----------
    t = time.time()
    fused = rrf_fusion([faiss_results, bm25_results])[:rag_build.RERANK_CANDIDATES]
    rrf_cost = time.time() - t
    bm25_ids = {d for d, _ in bm25_results}
    faiss_ids = {d for d, _ in faiss_results}
    print(f"\n[3] RRF 融合：取前 {rag_build.RERANK_CANDIDATES} 送精排，耗时 {rrf_cost*1000:.1f}ms")
    print(f"    {'#':>3s} {'RRF分':>9s}  来源       doc_id")
    for i, (doc_id, score) in enumerate(fused, 1):
        sources = []
        if doc_id in bm25_ids:
            sources.append("BM25")
        if doc_id in faiss_ids:
            sources.append("向量")
        print(f"    {i:3d} {score:9.6f}  {'+'.join(sources):8s}  {doc_id:22s} "
              f"{preview(docs_dict, doc_id, 40)}")

    # ---------- ④ 精排 ----------
    t = time.time()
    pairs = [[args.query, docs_dict[d]] for d, _ in fused]
    scores = rag_build.get_reranker().predict(pairs)
    rerank_cost = time.time() - t
    reranked = sorted(((d, float(s)) for (d, _), s in zip(fused, scores)),
                      key=lambda x: -x[1])
    print(f"\n[4] BGE 精排（Cross-Encoder）：{len(fused)} 条候选，耗时 {rerank_cost*1000:.0f}ms")
    for i, (doc_id, score) in enumerate(reranked[:args.top_k], 1):
        print(f"    {i:2d}. {score:7.3f}  {doc_id:22s} {preview(docs_dict, doc_id)}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 92)
    print("汇总")
    print("=" * 92)
    print(f"  BM25 {bm25_cost*1000:6.1f}ms  →  向量 {faiss_cost*1000:6.1f}ms  "
          f"→  RRF {rrf_cost*1000:5.1f}ms  →  精排 {rerank_cost*1000:6.0f}ms")
    total = bm25_cost + faiss_cost + rrf_cost + rerank_cost
    print(f"  检索总耗时（不含模型加载）：{total*1000:.0f}ms，其中精排占 "
          f"{rerank_cost/total:.0%}")
    print(f"\n  最终送进 prompt 的上下文（{args.top_k} 条）：")
    for doc_id, score in reranked[:args.top_k]:
        print(f"    · [{score:.2f}] {docs_dict[doc_id][:70]}")

    top1 = reranked[0][1] if reranked else 0.0
    print(f"\n  提示：Top-1 精排分 {top1:.2f}。实测阈值 0.5 可过滤全部域外问题，"
          f"域内仅丢 1 条寒暄题（见 eval/param_ablation.md）。")


if __name__ == "__main__":
    main()
