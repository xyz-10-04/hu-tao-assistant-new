"""知识库构建与混合检索。

改动说明（2026-11-16 审计修复）：
1. doc_id 改为「文件名 + 文件内序号」，不再依赖批量切分的全局递增序号，
   同一块内容在任何切分上下文下都得到同一个 doc_id（原来是 _25 这种名字会串到别的文件）。
2. 模型与索引改为懒加载 + 进程内缓存：import 时不再加载 1.3G 的 BGE 模型，
   启动从 70s+ 降到秒级；同一分类的 FAISS/BM25 只构建一次。
3. 路径改为基于本文件位置，不再依赖当前工作目录（uvicorn 从别的目录启动也不会挂）。
4. 新增 min_score 阈值，可过滤掉与问题无关的召回，避免无关上下文污染 prompt。
"""

import os
import threading

from bm25 import BM25, rrf_fusion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BGE_EMBEDDING_DIR = os.path.join(MODELS_DIR, "bge-small-zh-v1.5", "snapshots", "master")
BGE_RERANKER_DIR = os.path.join(MODELS_DIR, "bge-reranker-base", "snapshots", "master")

CATEGORY_MAP = {
    "character": "角色评价",
    "story": "背景故事",
    "misc": "其他台词",
}

# 检索参数
FAISS_TOP_K = 3        # 向量路召回条数
BM25_TOP_K = 10        # 关键词路召回条数
RERANK_CANDIDATES = 10  # 送进 Cross-Encoder 精排的候选数


# --------------------------------------------------------------------------- #
# 模型的懒加载：模块被 import 时不做任何重活
# --------------------------------------------------------------------------- #
_lock = threading.RLock()  # 可重入：_get_store 持锁期间还会触发模型加载
_embedding_model = None
_reranker = None
_store_cache = {}


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _lock:
            if _embedding_model is None:
                # 延迟到真正要用时才 import：torch 一进来就是 10s 起步，
                # 放在模块顶层会让 uvicorn 每次重启都卡半分钟
                from langchain_huggingface import HuggingFaceEmbeddings

                _embedding_model = HuggingFaceEmbeddings(
                    model_name=BGE_EMBEDDING_DIR,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},  # BGE 必须归一化
                )
    return _embedding_model


def get_reranker():
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                _reranker = CrossEncoder(BGE_RERANKER_DIR)
    return _reranker


def __getattr__(name):
    """兼容旧代码里的 rag_build.embedding_model / rag_build.reranker 写法。"""
    if name == "embedding_model":
        return get_embedding_model()
    if name == "reranker":
        return get_reranker()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def clear_cache():
    """测试用：清掉索引/BM25 缓存。"""
    _store_cache.clear()


# --------------------------------------------------------------------------- #
# 文档加载与切分
# --------------------------------------------------------------------------- #
def load_documents(sub_dir=None, verbose=False):
    # 延迟 import：langchain 的 loader 会拖进一堆重依赖，启动阶段用不到
    from langchain_community.document_loaders import TextLoader

    data_dir = os.path.join(KNOWLEDGE_DIR, sub_dir) if sub_dir else KNOWLEDGE_DIR
    if not os.path.isdir(data_dir):
        return []

    docs = []
    if sub_dir:
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
    else:
        files = []
        for root, _dirs, names in os.walk(data_dir):
            files.extend(os.path.join(root, n) for n in names if n.endswith(".txt"))

    # 排序保证不同机器、不同文件系统下顺序一致
    for path in sorted(files):
        if verbose:
            print(f"加载文件: {os.path.basename(path)}")
        docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def split_documents(docs):
    """切分文档并写入稳定的 doc_id。

    doc_id 规则：`{文件名去掉后缀}_{该文件内的块序号}`
    —— 只取决于这个文件自己的内容，与本次切分是否包含别的文件无关。
    """
    chunks = []
    for doc in docs:
        source = doc.metadata["source"]

        if "voice_lines" in source or "评价" in source:
            # 台词 / 角色评价：一条就是一块，严格按空行切，避免混入多条
            from langchain_core.documents import Document

            paragraphs = [p.strip() for p in doc.page_content.split("\n\n") if p.strip()]
            chunks.extend(
                Document(page_content=p, metadata=doc.metadata.copy()) for p in paragraphs
            )
        else:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            )
            chunks.extend(text_splitter.split_documents([doc]))

    counters = {}
    for chunk in chunks:
        stem = os.path.splitext(os.path.basename(chunk.metadata["source"]))[0]
        idx = counters.get(stem, 0)
        counters[stem] = idx + 1
        chunk.metadata["doc_id"] = f"{stem}_{idx}"

    return chunks


def build_vectorstore(chunks, index_path):
    from langchain_community.vectorstores import FAISS

    vectordb = FAISS.from_documents(documents=chunks, embedding=get_embedding_model())
    vectordb.save_local(index_path)
    return vectordb


# --------------------------------------------------------------------------- #
# 检索：FAISS + BM25 召回 → RRF 融合 → BGE 精排
# --------------------------------------------------------------------------- #
def index_path(category=None):
    return os.path.join(BASE_DIR, f"faiss_index_{category}" if category else "faiss_index")


def _get_store(category):
    """按分类加载/缓存在 FAISS 索引与 BM25，避免每次查询都重建。"""
    if category in _store_cache:
        return _store_cache[category]

    # 模型加载（可能十几秒）放在锁外面，避免持锁期间阻塞其它线程
    embedding = get_embedding_model()

    with _lock:
        if category in _store_cache:
            return _store_cache[category]

        idx_dir = index_path(category)
        chunks = split_documents(load_documents(CATEGORY_MAP.get(category) if category else None))
        docs_dict = {c.metadata["doc_id"]: c.page_content for c in chunks}

        if os.path.exists(os.path.join(idx_dir, "index.faiss")):
            from langchain_community.vectorstores import FAISS

            vectordb = FAISS.load_local(
                idx_dir,
                embedding,
                allow_dangerous_deserialization=True,
            )
            stale = {d.metadata.get("doc_id") for d in vectordb.docstore._dict.values()} - set(docs_dict)
            if stale:
                raise RuntimeError(
                    f"索引 {idx_dir} 与当前知识库不一致（{len(stale)} 个 doc_id 找不到），"
                    f"请先运行 python scripts/rebuild_index.py 重建索引"
                )
        else:
            vectordb = build_vectorstore(chunks, idx_dir)

        bm25 = BM25(docs_dict)
        _store_cache[category] = (vectordb, docs_dict, bm25)
        return _store_cache[category]


def search_hu_tao(query, k=10, category=None, return_ids=False, min_score=None, retriever="hybrid"):
    """检索胡桃知识库。

    参数
    ----
    query       : 用户问题
    k           : 最终返回条数
    category    : character / story / misc，None 表示检索全部知识
    return_ids  : 返回 [(doc_id, score)] 而不是纯文本，供评测使用
    min_score   : 精排分数下限，低于该分数的召回会被丢弃（None 表示不过滤）
    retriever   : 消融实验用，默认 "hybrid"
                  bm25   —— 只用关键词倒排
                  faiss  —— 只用向量语义
                  rrf    —— 两路 RRF 融合后直接返回（不做精排）
                  hybrid —— RRF 融合 + BGE 精排（线上默认）
    """
    if not query or not query.strip():
        return []

    vectordb, docs_dict, bm25 = _get_store(category)

    if retriever == "bm25":
        fused = bm25.search(query, top_k=max(k, RERANK_CANDIDATES))
        return _to_result(fused[:k], docs_dict, return_ids)

    if retriever == "faiss":
        docs = vectordb.as_retriever(
            search_kwargs={"k": max(k, RERANK_CANDIDATES)}
        ).invoke(query)
        fused = [(doc.metadata["doc_id"], 0.0) for doc in docs]
        return _to_result(fused[:k], docs_dict, return_ids)

    faiss_docs = vectordb.as_retriever(search_kwargs={"k": FAISS_TOP_K}).invoke(query)
    faiss_results = [(doc.metadata["doc_id"], 0.0) for doc in faiss_docs]

    # RRF 只使用排名，所以两路的原始分数不参与融合（也就不需要归一化）
    fused = rrf_fusion([faiss_results, bm25.search(query, top_k=BM25_TOP_K)])
    fused = fused[:RERANK_CANDIDATES]

    if retriever == "rrf":
        return _to_result(fused[:k], docs_dict, return_ids)

    pairs = [[query, docs_dict[doc_id]] for doc_id, _ in fused]
    rerank_scores = get_reranker().predict(pairs)

    reranked = sorted(
        ((doc_id, float(score)) for (doc_id, _), score in zip(fused, rerank_scores)),
        key=lambda x: x[1],
        reverse=True,
    )

    if min_score is not None:
        reranked = [item for item in reranked if item[1] >= min_score]

    return _to_result(reranked[:k], docs_dict, return_ids)


def _to_result(items, docs_dict, return_ids):
    if return_ids:
        return list(items)
    return [docs_dict[doc_id] for doc_id, _ in items]
