"""后处理 answers.jsonl：修正拒答检测的正则，重算统计并重写人工评分表。

不需要重新调用 API。
"""
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(BASE_DIR, "eval", "answers.jsonl")

# 旧版子串匹配会把原台词里的"你不知道吗"误判成拒答；
# 这里收紧成"第一人称不知道 / 明确的无法回答"。
REFUSAL_RE = re.compile(
    r"(我(也)?(并)?不(太)?(知道|清楚|了解))|(无法回答)|(知识库(中|里)?(并)?(没有|未收录))|(未找到相关)|(没有相关(信息|资料))"
)

# 寒暄/短问句：本来就不含实体，覆盖率指标对它们不适用
SMALLTALK = {"q006", "q008", "q009", "q010"}


def main():
    records = [json.loads(line) for line in open(PATH, encoding="utf-8") if line.strip()]

    for r in records:
        r["is_refusal"] = bool(REFUSAL_RE.search(r["reply"]))

    scored = [r for r in records if not r["error"]]
    factual = [r for r in scored if r["id"] not in SMALLTALK]

    def stat(rows, name):
        cov = sum(r["entity_coverage"] for r in rows) / len(rows)
        ge60 = sum(1 for r in rows if r["entity_coverage"] >= 0.6)
        print(f"  {name:26s} n={len(rows):2d}  平均覆盖率={cov:5.1%}  ≥60% 的={ge60}/{len(rows)}")

    print("=" * 74)
    print("生成侧统计（代理指标：金标准文档实体在回答中的覆盖率）")
    print("=" * 74)
    stat(scored, "全部样本")
    stat(factual, "排除寒暄类（q006/8/9/10）")
    print(f"  拒答数（修正检测后）      {sum(1 for r in records if r['is_refusal'])}")
    print(f"  报错数                    {sum(1 for r in records if r['error'])}")
    print(f"  平均延迟                  {sum(r['latency_s'] for r in scored)/len(scored):.2f}s")

    review = os.path.join(BASE_DIR, "eval", "answers_review.md")
    with open(review, "w", encoding="utf-8") as f:
        f.write("# 生成质量人工评分表\n\n")
        f.write("> 打分：3 = 完全正确且有依据；2 = 大部分正确但有细节偏差；1 = 答错或答偏；0 = 拒答/报错\n")
        f.write("> 把分数填到 `answers.jsonl` 的 `human_score` 字段后，可统计真实准确率。\n\n")
        f.write("## 自动代理指标（不是准确率）\n\n")
        f.write(f"- 全部 20 条：平均实体覆盖率 "
                f"{sum(r['entity_coverage'] for r in scored)/len(scored):.1%}\n")
        f.write(f"- 排除 4 条寒暄类：平均实体覆盖率 "
                f"{sum(r['entity_coverage'] for r in factual)/len(factual):.1%}\n")
        f.write(f"- 平均响应延迟 {sum(r['latency_s'] for r in scored)/len(scored):.2f}s\n\n")
        f.write("**局限**：实体覆盖率只说明「金标准里的名词有没有在回答里出现」，"
                "不说明回答对不对；寒暄类问题没有实体，指标不适用。所以必须人工打分。\n\n")
        f.write("## 逐条评分\n\n")
        f.write("| id | 问题 | 覆盖率 | 延迟 | 回答（截断） | 打分 | 备注 |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for r in records:
            answer = r["reply"].replace("\n", " ").replace("|", "｜")[:70]
            f.write(f"| {r['id']} | {r['question']} | {r['entity_coverage']:.0%} | "
                    f"{r['latency_s']}s | {answer} |  |  |\n")

    with open(PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  已更新 {PATH}")
    print(f"  已重写 {review}")


if __name__ == "__main__":
    main()
