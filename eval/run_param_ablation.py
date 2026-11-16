"""参数消融：验证「参数为什么是这个值」。

对应求职规划/04-代码自检清单.md 的 L3 层。回答三个问题：
    1. BM25_TOP_K / FAISS_TOP_K / RERANK_CANDIDATES 各自对 Recall 的影响有多大？
    2. 相关性阈值 min_score 该定多少？域内问题和域外问题能不能被分开？
    3. 去掉 chunk_overlap 会怎样？

用法：python eval/run_param_ablation.py
"""

import json
import os
import statistics
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import rag_build  # noqa: E402
from rag_build import search_hu_tao  # noqa: E402

OUT_OF_DOMAIN = [
    "今天纳斯达克指数多少",
    "帮我写一个快速排序",
    "深圳明天天气怎么样",
    "信用卡怎么申请",
    "讲个笑话",
]


def load_questions():
    path = os.path.join(BASE_DIR, "eval", "questions.jsonl")
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def recall(questions, k=5, min_score=None):
    hits = {1: 0, 3: 0, 5: 0}
    empty = 0
    for q in questions:
        results = search_hu_tao(q["question"], k=k, return_ids=True, min_score=min_score)
        ids = [d for d, _ in results]
        if not ids:
            empty += 1
        for n in (1, 3, 5):
            if any(g in ids[:n] for g in q["expected_doc_ids"]):
                hits[n] += 1
    n = len(questions)
    return {i: hits[i] / n for i in (1, 3, 5)}, empty / n


def sweep_bm25(questions):
    print("\n【1】BM25_TOP_K 的影响（其余参数不变）")
    print(f"  {'BM25_TOP_K':>12s} {'Recall@1':>10s} {'Recall@3':>10s} {'Recall@5':>10s}")
    rows = []
    original = rag_build.BM25_TOP_K
    for value in (3, 5, 10, 20):
        rag_build.BM25_TOP_K = value
        m, _ = recall(questions)
        rows.append((value, m))
        print(f"  {value:>12d} {m[1]:>9.0%} {m[3]:>9.0%} {m[5]:>9.0%}")
    rag_build.BM25_TOP_K = original
    return rows


def sweep_faiss(questions):
    print("\n【2】FAISS_TOP_K 的影响")
    print(f"  {'FAISS_TOP_K':>12s} {'Recall@1':>10s} {'Recall@3':>10s} {'Recall@5':>10s}")
    rows = []
    original = rag_build.FAISS_TOP_K
    for value in (1, 3, 10):
        rag_build.FAISS_TOP_K = value
        m, _ = recall(questions)
        rows.append((value, m))
        print(f"  {value:>12d} {m[1]:>9.0%} {m[3]:>9.0%} {m[5]:>9.0%}")
    rag_build.FAISS_TOP_K = original
    return rows


def sweep_rerank(questions):
    print("\n【3】RERANK_CANDIDATES 的影响（精排池大小）")
    print(f"  {'RERANK_CANDIDATES':>18s} {'Recall@1':>10s} {'Recall@3':>10s} {'Recall@5':>10s}")
    rows = []
    original = rag_build.RERANK_CANDIDATES
    for value in (3, 5, 10):
        rag_build.RERANK_CANDIDATES = value
        m, _ = recall(questions)
        rows.append((value, m))
        print(f"  {value:>18d} {m[1]:>9.0%} {m[3]:>9.0%} {m[5]:>9.0%}")
    rag_build.RERANK_CANDIDATES = original
    return rows


def sweep_threshold(questions):
    print("\n【4】相关性阈值 min_score 的影响（Recall 与拒答率）")
    print(f"  {'min_score':>10s} {'Recall@1':>10s} {'Recall@3':>10s} {'拒答率':>10s}")
    rows = []
    for value in (None, -8.0, -4.0, -2.0, 0.0, 2.0):
        m, empty = recall(questions, min_score=value)
        rows.append((value, m, empty))
        label = "不设" if value is None else f"{value:+.1f}"
        print(f"  {label:>10s} {m[1]:>9.0%} {m[3]:>9.0%} {empty:>9.0%}")
    return rows


def score_distribution(questions):
    """域内 vs 域外问题的 Top-1 精排分数分布，用来决定阈值定在哪。"""
    print("\n【5】域内 / 域外问题的 Top-1 精排分数分布")

    def top1(q):
        results = search_hu_tao(q, k=1, return_ids=True)
        return results[0][1] if results else None

    inside_pairs = [(q["id"], top1(q["question"])) for q in questions]
    outside = [top1(q) for q in OUT_OF_DOMAIN]
    inside = [s for _, s in inside_pairs]

    def brief(values, name):
        values = [v for v in values if v is not None]
        print(f"  {name:14s} min={min(values):7.2f}  中位数={statistics.median(values):7.2f}  "
              f"max={max(values):7.2f}")
        return values

    brief(inside, "域内（20 条）")
    brief(outside, "域外（5 条）")

    print("\n【6】阈值取舍：不同 min_score 下，域内丢几题、域外滤掉几题")
    print(f"  {'阈值':>6s} {'域内保留':>10s} {'域内丢失':>10s} {'域外过滤':>10s}  丢失的题")
    rows = []
    for t in (0.3, 0.4, 0.5, 0.6):
        kept = [(qid, s) for qid, s in inside_pairs if s is not None and s >= t]
        lost = [qid for qid, s in inside_pairs if s is not None and s < t]
        filtered_out = sum(1 for s in outside if s is not None and s < t)
        rows.append((t, len(kept), len(lost), filtered_out, lost))
        print(f"  {t:>6.1f} {len(kept):>8d}/{len(questions)} {len(lost):>10d} "
              f"{filtered_out:>8d}/{len(OUT_OF_DOMAIN)}  {','.join(lost)}")
    return inside, outside, rows


def main():
    questions = load_questions()
    print("=" * 80)
    print("参数消融：用数据回答「这个参数为什么是这个值」")
    print("=" * 80)

    bm25_rows = sweep_bm25(questions)
    faiss_rows = sweep_faiss(questions)
    rerank_rows = sweep_rerank(questions)
    thr_rows = sweep_threshold(questions)
    inside, outside, thr_tradeoff = score_distribution(questions)

    out = os.path.join(BASE_DIR, "eval", "param_ablation.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 参数消融：为什么是这些值\n\n")
        f.write("> 目的：把「我觉得差不多」换成「我测过」。\n\n")

        f.write("## BM25_TOP_K\n\n| 值 | Recall@1 | Recall@3 | Recall@5 |\n|---|---|---|---|\n")
        for v, m in bm25_rows:
            f.write(f"| {v} | {m[1]:.0%} | {m[3]:.0%} | {m[5]:.0%} |\n")

        f.write("\n## FAISS_TOP_K\n\n| 值 | Recall@1 | Recall@3 | Recall@5 |\n|---|---|---|---|\n")
        for v, m in faiss_rows:
            f.write(f"| {v} | {m[1]:.0%} | {m[3]:.0%} | {m[5]:.0%} |\n")

        f.write("\n## RERANK_CANDIDATES（精排池）\n\n| 值 | Recall@1 | Recall@3 | Recall@5 |\n|---|---|---|---|\n")
        for v, m in rerank_rows:
            f.write(f"| {v} | {m[1]:.0%} | {m[3]:.0%} | {m[5]:.0%} |\n")

        f.write("\n## min_score（相关性阈值）\n\n| 阈值 | Recall@1 | Recall@3 | 拒答率 |\n|---|---|---|---|\n")
        for v, m, e in thr_rows:
            label = "不设" if v is None else f"{v:+.1f}"
            f.write(f"| {label} | {m[1]:.0%} | {m[3]:.0%} | {e:.0%} |\n")

        f.write("\n## 域内 / 域外分数分布\n\n")
        f.write(f"- 域内（20 条）：min={min(inside):.2f}，中位数={statistics.median(inside):.2f}\n")
        f.write(f"- 域外（5 条） ：min={min(outside):.2f}，中位数={statistics.median(outside):.2f}\n")

        f.write("\n## 阈值取舍\n\n| 阈值 | 域内保留 | 域内丢失 | 域外过滤 | 丢失的题 |\n|---|---|---|---|---|\n")
        for t, kept, lost, filtered, lost_ids in thr_tradeoff:
            f.write(f"| {t:.1f} | {kept}/20 | {lost} | {filtered}/5 | {','.join(lost_ids) or '-'} |\n")

        f.write("\n## 注意：20 条样本的统计局限\n\n")
        f.write("20 条评测集里，**1 道题的差异就是 5 个百分点**。"
                "所以上表里 85% 和 80% 的差距只是 1 题波动，不能当成「参数改进了」。"
                "只有差距超过 2 题（10 个百分点）才值得当成信号。\n")
        f.write(f"- 域内最低分 {min(inside):.2f} / 域外最高分 {max(outside):.2f}\n")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
