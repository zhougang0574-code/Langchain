"""
【07 检索增强RAG / 01】Embedding 与相似度 —— 先看懂「向量为什么能检索」
==================================================================
RAG 的第一块基石是 Embedding（向量化）：把文本变成一串数字。本课不碰向量库、
不碰检索框架，只用最朴素的方式建立直觉——语义相近的句子，向量也相近。

新概念（只有这一个）：
  Embedding —— 把一段文本映射成一个高维向量（一串浮点数）。
    embed_query(text)      把一条文本变成向量
    余弦相似度              衡量两个向量「方向」有多接近（越接近 1 越相似）

为什么先讲这个：
  看懂「问题向量和文档向量谁更近」，就理解了后面向量库检索的全部原理——
  向量库只是把这件事做得又快又大规模而已（〔07/02〕）。
"""

import os
import sys

from dotenv import load_dotenv

# 把 course_v2 目录加进路径，好导入共享的 _common（Embeddings 工厂）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import get_embeddings   # noqa: E402

load_dotenv()

embeddings = get_embeddings()        # 阿里百炼 text-embedding-v1（见 _common.py）


# ── 1. 把文本变成向量 ──────────────────────────────────────────────────────
def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：点积 / (模长 × 模长)。不依赖 numpy，纯 Python 写清楚原理。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


if __name__ == "__main__":
    text = "LangChain 是一个构建 LLM 应用的框架"
    vec = embeddings.embed_query(text)
    print("=== 一段文本被映射成向量 ===")
    print(f"文本：{text}")
    print(f"向量维度：{len(vec)}，前 5 维：{[round(x, 4) for x in vec[:5]]}")

    # ── 2. 比较「问题」和几个「候选文档」的相似度 ──────────────────────────
    query = "什么是 LangChain？"
    candidates = [
        "LangChain 是用于构建大语言模型应用的开源框架。",   # 高度相关
        "RAG 是检索增强生成技术。",                          # 有点相关
        "今天北京的天气晴朗，气温 25 度。",                  # 完全无关
    ]
    q_vec = embeddings.embed_query(query)

    print(f"\n=== 问题：{query} ===")
    print("（相似度越高 = 语义越接近）")
    for doc in candidates:
        sim = cosine(q_vec, embeddings.embed_query(doc))
        print(f"  相似度 {sim:.3f}  ←  {doc}")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  "什么是 LangChain？"            候选文档们
        │ embed_query                │ embed_query
        ▼                            ▼
     问题向量  ──余弦相似度比较──►  各文档向量
                    │
                    ▼
        相似度最高的就是「最相关」的文档

★ 核心规律：
  Embedding 把语义编码成向量，方向相近 = 语义相近。检索的本质就是
  「把问题向量化，找最近的文档向量」——这正是 RAG「按语义检索」的根基。

  本课手算相似度只为看清原理。真实场景文档成千上万，挨个算太慢——
  所以需要专门的向量库来加速，这是下一课〔07/02〕FAISS 的事。
  （Embeddings 类抽到了 _common.py 复用，避免每个 RAG 文件重写一遍。）
"""
