"""切分策略消融实验：复现"统一切分" vs "长短文本分开切"。

为什么需要重做：
    历史记录里"chunk_size=150 时 Recall@3 掉到 30%"这个数字，是按当次的 doc_id 标注算出来的；
    换一套切分，doc_id 全变，评测集直接失效。所以这里换一个与切分无关的口径：

        命中 = 检索返回的某一块，是否「完整覆盖」了标准答案所需的内容（词级覆盖 >= 70%）

这个口径衡量的是"能不能用一块把答案答出来"，正好对应切分腰斩导致的失败。

用法：python eval/run_chunk_ablation.py
"""

import json
import os
import re
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import jieba  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

import rag_build  # noqa: E402
from bm25 import BM25, rrf_fusion  # noqa: E402

CONFIGS = [
    ("统一 150 / overlap 30", dict(chunk_size=150, chunk_overlap=30, custom=False)),
    ("统一 250 / overlap 50", dict(chunk_size=250, chunk_overlap=50, custom=False)),
    ("统一 300 / overlap 50", dict(chunk_size=300, chunk_overlap=50, custom=False)),
    ("长短分开（线上方案）", dict(chunk_size=300, chunk_overlap=50, custom=True)),
]

SENT_END = "。！？…”』」\n"


def words_of(text):
    return {w for w in jieba.lcut(text) if len(w.strip()) >= 2 and not re.match(r"^[\W_]+$", w)}


def split_uniform(docs, chunk_size, chunk_overlap):
    """历史方案：所有文档统一用 RecursiveCharacterTextSplitter。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = []
    for doc in docs:
        chunks.extend(splitter.split_documents([doc]))
    return chunks


def stats(chunks, label):
    lens = [len(c.page_content) for c in chunks]
    long_chunks = [c for c in chunks if len(c.page_content) > 80]
    broken = [c for c in long_chunks if c.page_content.strip()[-1:] not in SENT_END]
    broken_rate = len(broken) / len(long_chunks) if long_chunks else 0
    print(f"  {label:24s} 块数={len(chunks):4d}  平均长度={sum(lens)/len(lens):6.1f}  "
          f"最长={max(lens):4d}  长块结尾不完整率={broken_rate:.0%}")
    return {"chunks": len(chunks), "avg_len": sum(lens) / len(lens),
            "max_len": max(lens), "broken_rate": broken_rate}


def evaluate(chunks, questions, gold_chunks, k=5):
    from langchain_community.vectorstores import FAISS

    docs_dict = {f"i{i}": c.page_content for i, c in enumerate(chunks)}
    vectordb = FAISS.from_documents(
        documents=[Document(page_content=t, metadata={"doc_id": did})
                   for did, t in docs_dict.items()],
        embedding=rag_build.get_embedding_model(),
    )
    bm25 = BM25(docs_dict)
    reranker = rag_build.get_reranker()
    gold_words = [words_of(t) for t in gold_chunks]

    hits = {1: 0, 3: 0, 5: 0}
    densities = []
    for q in questions:
        faiss_docs = vectordb.as_retriever(search_kwargs={"k": 3}).invoke(q["question"])
        faiss_res = [(d.metadata["doc_id"], 0.0) for d in faiss_docs]
        fused = rrf_fusion([faiss_res, bm25.search(q["question"], top_k=10)])[:10]
        scores = reranker.predict([[q["question"], docs_dict[did]] for did, _ in fused])
        ranked = [did for (did, _), _s in sorted(zip(fused, scores), key=lambda x: -x[1])]

        # 每条 gold 需要一块完整覆盖；任一 gold 被覆盖即算该题命中
        def covered(doc_text):
            w = words_of(doc_text)
            return any(len(g & w) / len(g) >= 0.7 for g in gold_words if g)

        for n in (1, 3, 5):
            if any(covered(docs_dict[did]) for did in ranked[:n]):
                hits[n] += 1

        # 答案密度：Top-1 块里，与标准答案重叠的词占该块总词数的比例。
        # 块越大越容易"覆盖"答案，但密度会被无关内容稀释 —— 这是切分粒度真正的代价。
        top_text = docs_dict[ranked[0]]
        top_words = words_of(top_text)
        best = max((len(g & top_words) / len(top_words) for g in gold_words if g), default=0)
        densities.append(best)

    total = len(questions)
    result = {n: hits[n] / total for n in (1, 3, 5)}
    result["density"] = sum(densities) / total
    return result


def main():
    # 标准答案所需的内容：取「线上方案」下当次评测集 gold 对应的块正文
    _, gold_map, _ = rag_build._get_store(None)
    questions = [json.loads(line) for line in
                 open(os.path.join(BASE_DIR, "eval", "questions.jsonl"), encoding="utf-8")
                 if line.strip()]
    gold_chunks = [gold_map[g] for q in questions for g in q["expected_doc_ids"] if g in gold_map]

    docs = rag_build.load_documents()
    print("切分策略消融（命中定义：单块完整覆盖标准答案 >= 70% 的词）")
    print("=" * 96)
    rows = []
    for label, cfg in CONFIGS:
        if cfg["custom"]:
            # 临时把 split_documents 换成"统一策略"，只为生成对照
            chunks = split_uniform(docs, cfg["chunk_size"], cfg["chunk_overlap"])
        else:
            chunks = split_uniform(docs, cfg["chunk_size"], cfg["chunk_overlap"])
        st = stats(chunks, label)
        metrics = evaluate(chunks, questions, gold_chunks)
        rows.append((label, st, metrics))
        print(f"    -> Recall@1 {metrics[1]:.0%}  Recall@3 {metrics[3]:.0%}  "
              f"Recall@5 {metrics[5]:.0%}  Top-1答案密度 {metrics['density']:.1%}")

    # 线上方案（长短分开）单独用真正的 split_documents
    online_chunks = rag_build.split_documents(docs)
    st = stats(online_chunks, "长短分开（线上方案）")
    metrics = evaluate(online_chunks, questions, gold_chunks)
    rows[-1] = ("长短分开（线上方案）", st, metrics)
    print(f"    -> Recall@1 {metrics[1]:.0%}  Recall@3 {metrics[3]:.0%}  "
          f"Recall@5 {metrics[5]:.0%}  Top-1答案密度 {metrics['density']:.1%}")

    out = os.path.join(BASE_DIR, "eval", "chunk_ablation.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 切分策略消融\n\n")
        f.write("> 命中口径与切分无关：检索返回的某一块是否完整覆盖标准答案 70% 以上的词。\n")
        f.write("> 答案密度：Top-1 块里与标准答案重叠的词占该块总词数的比例，用来衡量「块里混了多少无关内容」。\n\n")
        f.write("| 切分策略 | 块数 | 平均长度 | 长块结尾不完整率 | Recall@1 | Recall@3 | Recall@5 | Top-1 答案密度 |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for label, st, m in rows:
            f.write(f"| {label} | {st['chunks']} | {st['avg_len']:.0f} | "
                    f"{st['broken_rate']:.0%} | {m[1]:.0%} | {m[3]:.0%} | {m[5]:.0%} | "
                    f"{m['density']:.1%} |\n")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
