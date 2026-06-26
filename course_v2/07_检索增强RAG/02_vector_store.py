"""
【07 检索增强RAG / 02】FAISS 向量库 —— 把向量存起来、快速检索、落盘复用
==================================================================
〔07/01〕手算了相似度，但文档一多就慢。向量库（Vector Store）专门解决这件事：
批量存向量 + 快速找最近邻，还能把索引保存到磁盘，下次直接加载不用重新向量化。

新概念（只有这一个）：
  FAISS 向量库
    FAISS.from_texts(texts, embeddings)   把一批文本向量化并建索引
    .similarity_search(query, k=...)      检索最相关的 k 条
    .save_local(路径) / FAISS.load_local(路径, embeddings)  落盘 / 加载

为什么重要：
  这是 RAG 的「数据库」。本课建好的索引会保存到本课目录下的 faiss_index/，
  〔07/06〕〔07/07〕〔07/08〕都会直接加载它，不必重复向量化。
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import get_embeddings   # noqa: E402

load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "faiss_index")     # 索引落盘位置，后续课复用
embeddings = get_embeddings()


# ── 1. 准备「知识块」：从知识库文本按段落切几条 ────────────────────────────
# 真实切分用专门的 splitter（〔07/04〕），这里先按空行简单切，聚焦向量库本身
with open(os.path.join(HERE, "assets", "knowledge.txt"), encoding="utf-8") as f:
    chunks = [seg.strip() for seg in f.read().split("\n\n") if seg.strip()]
print(f"知识块数量：{len(chunks)}")


# ── 2. 建索引 + 落盘 ───────────────────────────────────────────────────────
vectorstore = FAISS.from_texts(chunks, embeddings)     # 批量向量化并建索引
vectorstore.save_local(INDEX_DIR)                      # 保存到磁盘
print(f"索引已保存到：{INDEX_DIR}")


if __name__ == "__main__":
    # ── 3. 加载并检索 ──────────────────────────────────────────────────────
    loaded = FAISS.load_local(
        INDEX_DIR, embeddings,
        allow_dangerous_deserialization=True,   # 加载本地 pickle 需显式允许（自己存的，可信）
    )

    query = "RAG 是用来解决什么问题的？"
    print(f"\n=== 检索：{query} ===")
    docs = loaded.similarity_search(query, k=2)        # 取最相关的 2 条
    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] {d.page_content[:60]}...")

    # 也能拿到「相似度分数」（FAISS 用 L2 距离，越小越相似）
    print("\n=== 带分数的检索 ===")
    for d, score in loaded.similarity_search_with_score(query, k=2):
        print(f"  距离 {score:.0f}  ←  {d.page_content[:40]}...")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  知识块文本
        │ FAISS.from_texts(texts, embeddings)   ← 批量向量化 + 建索引
        ▼
   FAISS 向量库 ──save_local──► 磁盘 faiss_index/
        │
   query ─►similarity_search(query, k=2)──► 最相关的 k 条文档

★ 核心规律：
  向量库 = 批量存向量 + 快速最近邻检索 + 可落盘复用。
  from_texts 建库、similarity_search 检索、save_local/load_local 持久化，是三件套。
  load_local 要传 allow_dangerous_deserialization=True（索引含 pickle，确认来源可信再加载）。

  注意 FAISS 返回的 score 是 L2 距离：越小越相似（不是越大越相似）。
  这个量级因 embedding 模型而异，做分数过滤时必须实测校准（〔07/06〕）。
"""
