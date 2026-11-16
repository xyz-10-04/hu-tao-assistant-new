"""消融实验：各检索组件对 Recall 的贡献。

用法：
    python eval/run_ablation.py

输出 4 组对照（全量索引、同一份 20 条评测集）：
    bm25    —— 只用关键词倒排
    faiss   —— 只用 BGE 向量
    rrf     —— 两路 RRF 融合，不精排
    hybrid  —— 融合 + BGE-Reranker 精排（线上默认）
"""

import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from rag_build import search_hu_tao  # noqa: E402

CONFIGS = [
    ("bm25", "只用 BM25"),
    ("faiss", "只用向量"),
    ("rrf", "RRF 融合（无精排）"),
    ("hybrid", "RRF + BGE 精排（线上）"),
]


def evaluate(retriever):
    with open(os.path.join(BASE_DIR, "eval", "questions.jsonl"), encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    hits = {1: 0, 3: 0, 5: 0}
    times = []
    for q in questions:
        started = time.time()
        results = search_hu_tao(q["question"], k=5, return_ids=True, retriever=retriever)
        times.append(time.time() - started)
        ids = [doc_id for doc_id, _ in results]
        for n in (1, 3, 5):
            if any(g in ids[:n] for g in q["expected_doc_ids"]):
                hits[n] += 1
    total = len(questions)
    return {n: hits[n] / total for n in (1, 3, 5)}, sum(times) / total


if __name__ == "__main__":
    print("消融实验（全量索引 / 20 条评测集 / k=5）")
    print("=" * 72)
    print(f"{'配置':22s} {'Recall@1':>10s} {'Recall@3':>10s} {'Recall@5':>10s} {'平均延迟':>10s}")
    print("-" * 72)
    rows = []
    for retriever, label in CONFIGS:
        metrics, latency = evaluate(retriever)
        rows.append((label, retriever, metrics, latency))
        print(f"{label:22s} {metrics[1]:>9.0%} {metrics[3]:>9.0%} {metrics[5]:>9.0%} {latency:>9.3f}s")
    print("=" * 72)

    # 落一份 markdown，方便直接贴进 README
    out = os.path.join(BASE_DIR, "eval", "ablation.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 检索消融实验\n\n")
        f.write("| 配置 | Recall@1 | Recall@3 | Recall@5 | 平均延迟 |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for label, retriever, m, lat in rows:
            f.write(f"| {label} | {m[1]:.0%} | {m[3]:.0%} | {m[5]:.0%} | {lat:.3f}s |\n")
    print(f"已写入 {out}")
