"""重建全部 FAISS 索引。

用法：
    python scripts/rebuild_index.py

修改 knowledge/ 下的任何文档后都需要跑一次，否则检索会报
「索引与当前知识库不一致」——这是故意加的护栏，避免静默返回错内容。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_build import (  # noqa: E402
    CATEGORY_MAP,
    build_vectorstore,
    index_path,
    load_documents,
    split_documents,
)


def rebuild(category=None):
    sub_dir = CATEGORY_MAP.get(category) if category else None
    chunks = split_documents(load_documents(sub_dir))
    path = index_path(category)
    build_vectorstore(chunks, path)
    print(f"  {path:26s} 分类={category or '全量':10s} 块数={len(chunks):4d}  -> 已写入")
    return len(chunks)


if __name__ == "__main__":
    print("开始重建索引：")
    total = rebuild(None)
    for cat in CATEGORY_MAP:
        rebuild(cat)
    print(f"完成，全量索引共 {total} 块。")
