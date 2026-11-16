"""检索评测：Recall@1 / @3 / @5。

修复记录（2026-11-16）：
    原实现 `is_hit(retrieved_ids, expected_ids, k)` 里截断的是 `expected_ids[:k]`，
    而 `retrieved_ids` 已经是 Top-5，导致 "Recall@1" 实际等于 "Top-5 命中"，
    Recall@1 被高估（旧口径 90%，真实 80%）。现在显式截断 retrieved。

用法：
    python eval/run_retrieval_eval.py                 # 全量 + 分类 两种模式都跑
    python eval/run_retrieval_eval.py --mode global   # 只跑生产链路（全量索引）
    python eval/run_retrieval_eval.py --min-score 0   # 试相关性阈值
"""

import argparse
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from rag_build import search_hu_tao  # noqa: E402

QUESTIONS_PATH = os.path.join(BASE_DIR, "eval", "questions.jsonl")


def recall_at(retrieved, gold, k):
    """前 k 条里是否命中任意一条 gold。"""
    return any(doc_id in retrieved[:k] for doc_id in gold)


def run(mode, min_score=None, k=5):
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    hits = {1: 0, 3: 0, 5: 0}
    latencies = []
    failures = []

    for q in questions:
        category = q["category"] if mode == "category" else None
        started = time.time()
        results = search_hu_tao(
            q["question"], k=k, category=category,
            return_ids=True, min_score=min_score,
        )
        latencies.append(time.time() - started)

        retrieved = [doc_id for doc_id, _ in results]
        gold = q["expected_doc_ids"]
        ok = {}
        for n in (1, 3, 5):
            ok[n] = recall_at(retrieved, gold, n)
            if ok[n]:
                hits[n] += 1
        if not ok[3]:
            failures.append((q["id"], q["question"], gold, retrieved))

    total = len(questions)
    print(f"\n{'=' * 74}")
    title = "分类检索（category 路由，评测专用）" if mode == "category" \
        else "全量检索（生产链路 search_hu_tao_knowledge 实际走的路径）"
    print(f"模式: {mode}  |  {title}")
    if min_score is not None:
        print(f"相关性阈值: {min_score}")
    print(f"{'=' * 74}")
    print(f"  题目数      : {total}")
    print(f"  Recall@1    : {hits[1] / total:.2%}  ({hits[1]}/{total})")
    print(f"  Recall@3    : {hits[3] / total:.2%}  ({hits[3]}/{total})")
    print(f"  Recall@5    : {hits[5] / total:.2%}  ({hits[5]}/{total})")
    print(f"  平均延迟    : {sum(latencies) / total:.3f}s"
          f"  (min {min(latencies):.3f}s / max {max(latencies):.3f}s)")

    if failures:
        print(f"\n  Recall@3 未命中 {len(failures)} 条：")
        for qid, question, gold, got in failures:
            print(f"    ❌ {qid} {question}")
            print(f"      期望: {gold}")
            print(f"      实际: {got}")

    return {"mode": mode, "recall@1": hits[1] / total,
            "recall@3": hits[3] / total, "recall@5": hits[5] / total,
            "questions": total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["global", "category", "both"], default="both")
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    modes = ["global", "category"] if args.mode == "both" else [args.mode]
    summary = [run(m, args.min_score) for m in modes]

    print(f"\n{'=' * 74}")
    print("汇总")
    print(f"{'=' * 74}")
    print(f"{'模式':10s} {'Recall@1':>10s} {'Recall@3':>10s} {'Recall@5':>10s}")
    for s in summary:
        print(f"{s['mode']:10s} {s['recall@1']:>10.2%} {s['recall@3']:>10.2%} {s['recall@5']:>10.2%}")
