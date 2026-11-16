"""生成侧评测：跑完整的 Agent 问答链路，自动给出一版粗评，并生成人工评分表。

为什么需要它：
    检索指标的 Recall@k 只说明"正确材料被找出来了"，不说明"模型答对了"。
    本项目此前只有检索评测，这是最大的短板。这里补一个便宜的自动代理指标：
    回答是否覆盖了金标准文档里的实体（名词/专有名词），以及是否出现拒答。

用法：
    python eval/run_answer_eval.py            # 跑全部 20 条（会消耗 API 额度）
    python eval/run_answer_eval.py --limit 3  # 先试 3 条
"""

import argparse
import json
import os
import re
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import jieba.posseg as pseg  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from rag_build import _get_store  # noqa: E402

REFUSAL_PATTERNS = ["未找到", "没有找到", "不知道", "不清楚", "无法回答", "知识库中没有", "我不了解"]


def entities_of(text):
    """抽取名词类实体，作为"回答里应该出现"的锚点。"""
    words = set()
    for word, flag in pseg.cut(text):
        word = word.strip()
        if len(word) < 2:
            continue
        if re.match(r"^[\W_]+$", word):
            continue
        if flag.startswith("n") or flag in ("nz", "nr", "ns", "vn"):
            words.add(word)
    return words


def main(limit=None):
    from main import agent

    with open(os.path.join(BASE_DIR, "eval", "questions.jsonl"), encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    if limit:
        questions = questions[:limit]

    _, docs_dict, _ = _get_store(None)

    records = []
    refused = 0
    for i, q in enumerate(questions, 1):
        gold_text = "\n".join(docs_dict.get(g, "") for g in q["expected_doc_ids"])
        gold_entities = entities_of(gold_text)

        started = time.time()
        try:
            result = agent.invoke({"messages": [HumanMessage(content=q["question"])]})
            reply = result["messages"][-1].content
            if not isinstance(reply, str):
                reply = str(reply)
            error = ""
        except Exception as exc:  # noqa: BLE001
            reply, error = "", f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - started

        covered = sorted(e for e in gold_entities if e in reply)
        coverage = len(covered) / len(gold_entities) if gold_entities else 0.0
        is_refusal = any(p in reply for p in REFUSAL_PATTERNS)
        refused += int(is_refusal)

        records.append({
            "id": q["id"],
            "question": q["question"],
            "gold_doc_ids": q["expected_doc_ids"],
            "gold_entities": sorted(gold_entities),
            "covered_entities": covered,
            "entity_coverage": round(coverage, 3),
            "is_refusal": is_refusal,
            "latency_s": round(elapsed, 2),
            "answer_len": len(reply),
            "reply": reply,
            "error": error,
            # 留给人工打分
            "human_score": "",
            "human_note": "",
        })
        print(f"  [{i}/{len(questions)}] {q['id']} 覆盖率={coverage:.0%} "
              f"拒答={is_refusal} 延迟={elapsed:.1f}s")

    out_jsonl = os.path.join(BASE_DIR, "eval", "answers.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_cov = sum(r["entity_coverage"] for r in records) / len(records)
    avg_lat = sum(r["latency_s"] for r in records) / len(records)
    perfect = sum(1 for r in records if r["entity_coverage"] >= 0.6)
    errors = [r for r in records if r["error"]]

    # 人工评分表
    review = os.path.join(BASE_DIR, "eval", "answers_review.md")
    with open(review, "w", encoding="utf-8") as f:
        f.write("# 生成质量人工评分表\n\n")
        f.write("> 打分标准：3 = 完全正确且有依据；2 = 大部分正确但有细节偏差；1 = 答错或答偏；0 = 拒答/报错\n")
        f.write("> 在最后一列填 0-3，回填到 answers.jsonl 的 human_score 字段也能统计。\n\n")
        f.write("| id | 问题 | 实体覆盖率 | 延迟 | 回答（截断） | 打分 | 备注 |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for r in records:
            answer = r["reply"].replace("\n", " ").replace("|", "｜")[:80]
            f.write(f"| {r['id']} | {r['question']} | {r['entity_coverage']:.0%} | "
                    f"{r['latency_s']}s | {answer} |  |  |\n")

    print("\n" + "=" * 70)
    print("生成侧自动粗评（实体覆盖率是代理指标，不等于正确率）")
    print("=" * 70)
    print(f"  样本数          : {len(records)}")
    print(f"  平均实体覆盖率  : {avg_cov:.1%}")
    print(f"  覆盖率 ≥60% 的  : {perfect}/{len(records)}")
    print(f"  拒答数          : {refused}")
    print(f"  报错数          : {len(errors)}")
    print(f"  平均响应延迟    : {avg_lat:.2f}s")
    print(f"\n  明细：{out_jsonl}")
    print(f"  人工评分表：{review}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    main(parser.parse_args().limit)
