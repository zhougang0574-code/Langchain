"""
【07 检索增强RAG / 06】检索策略 —— MMR 去冗余 + 分数过滤〔进阶〕
==========================================================
〔07/05〕用的是最朴素的 top-k 检索：只看「和问题最像的 k 条」。这有两个毛病：
① 若知识库里有几块内容雷同，top-k 可能全挤在同一个点上，浪费名额；
② 哪怕问题和知识库完全无关，它也会硬塞 k 条回来。本课用两招改进。

新概念（只有这一个，两种策略）：
  ① MMR（最大边际相关）：as_retriever(search_type="mmr")
     每选一条都兼顾「和问题相关」+「和已选的不重复」，结果更多样、不冗余。
  ② 分数过滤：similarity_search_with_score + 阈值
     按相似度分数把「不够相关」的结果丢掉，宁可不答也不乱答。

前置：本课加载〔07/02〕保存的 faiss_index/，所以请先跑过 02。
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import get_embeddings   # noqa: E402

load_dotenv()
HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(HERE, "faiss_index")
embeddings = get_embeddings()

vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


# ── 1. MMR：相关性 + 多样性 ────────────────────────────────────────────────
def demo_mmr():
    print("=== MMR 检索（去冗余）===")
    mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,            # 最终返回 3 条
            "fetch_k": 8,      # 先按相关性召回 8 条候选，再从中挑「既相关又互不重复」的 3 条
            "lambda_mult": 0.5,  # 0=最大多样性，1=纯相关性（退化为普通 top-k）
        },
    )
    for d in mmr.invoke("LangChain 的核心组件"):
        print("  -", d.page_content[:40].strip(), "...")


# ── 2. 分数过滤：不相关就不返回 ────────────────────────────────────────────
def filtered_search(query: str, threshold: float) -> list[Document]:
    """只保留 L2 距离小于阈值的结果（FAISS 距离越小越相似）。"""
    hits = vectorstore.similarity_search_with_score(query, k=4)
    print(f"  原始距离：{[round(float(s)) for _, s in hits]}")
    keep = [d for d, score in hits if score < threshold]
    print(f"  过滤后保留 {len(keep)} 条（阈值={threshold}）")
    return keep


if __name__ == "__main__":
    demo_mmr()

    print("\n=== 分数过滤 ===")
    # 阈值需按实际数据校准：text-embedding-v1 实测——相关查询 L2 距离约 7000~11000，
    # 无关查询约 16000+，所以取 12000 作分界（换 embedding 模型必须重新实测，不能照搬）
    print("[相关查询]")
    filtered_search("什么是 RAG", threshold=12000)
    print("[无关查询 —— 应被全部过滤]")
    filtered_search("意大利面怎么做", threshold=12000)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  普通 top-k：只看「与问题的相关性」→ 可能 3 条都讲同一件事（冗余）

  MMR：每一步选择都权衡两件事
    ① 与问题的相关性（越高越好）
    ② 与「已选文档」的差异性（越不同越好）
    lambda_mult 调两者权重：小→多样性优先，大→相关性优先

★ 核心规律：
  MMR 解决「检索结果扎堆重复」，用 fetch_k 先多召回、再按「相关+不重复」精选 k 条。
  分数过滤解决「无关也硬答」，给相似度设阈值，不够相关就不返回。

  关键坑：FAISS 分数是 L2 距离，量级随 embedding 模型而变，阈值必须用自己的数据实测，
  绝不能照搬别处的数字（这是 archive 里那课特意标注的踩坑点）。
"""
