"""
【07 检索增强RAG / 07】MultiQueryRetriever —— 用 LLM 多角度改写提升召回〔进阶〕
========================================================================
〔07/05〕〔07/06〕都只用「原问题」去检索。但一个问题可能有多种问法，而用户的
具体措辞未必和知识库里的措辞接近，导致漏检。MultiQuery 让 LLM 把问题改写成
几个不同角度的子问题，分别检索再合并去重，召回更全。

新概念（只有这一个）：
  MultiQueryRetriever.from_llm(retriever, llm)
    - 内部用 llm 把原问题扩写成多个子查询
    - 每个子查询各自检索，结果合并去重
    - 对外仍是一个 retriever：.invoke(问题) → 合并后的 Document 列表

前置：加载〔07/02〕保存的 faiss_index/，请先跑过 02。
"""

import os
import sys
import logging

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import get_embeddings   # noqa: E402

load_dotenv()
HERE = os.path.dirname(os.path.abspath(__file__))
embeddings = get_embeddings()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"), temperature=0,
)

vectorstore = FAISS.load_local(
    os.path.join(HERE, "faiss_index"), embeddings, allow_dangerous_deserialization=True
)


# ── 1. 包一层 MultiQueryRetriever ──────────────────────────────────────────
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    llm=llm,
)

# 打开日志，能看到 LLM 自动生成了哪几个子查询
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


if __name__ == "__main__":
    query = "LangChain 怎么存和查私有知识？"

    # 不靠日志，也能直接调内部 llm_chain 看生成的子查询，更直观
    print("=== LLM 把原问题扩写成的子查询 ===")
    generated = multi_retriever.llm_chain.invoke({"question": query})
    for q in generated:
        print("  -", q)

    print(f"\n=== 合并去重后的检索结果 ===")
    docs = multi_retriever.invoke(query)
    print(f"共 {len(docs)} 条不重复文档（单一查询 k=2 最多 2 条，多角度后更全）")
    for d in docs:
        print("  -", d.page_content[:40].strip(), "...")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  原问题 "怎么存和查私有知识？"
        │ LLM 扩写
        ▼
  子查询1 "LangChain 如何存储向量？"
  子查询2 "怎样检索文档？"
  子查询3 "FAISS 向量库怎么用？"
        │ 各自检索
        ▼
  结果合并去重 → 更全的文档集合

★ 核心规律：
  MultiQueryRetriever 用「LLM 多角度改写 + 合并去重」对抗「用户措辞和库里措辞不一致」
  导致的漏检，提升召回率。对外仍是普通 retriever，可无缝替换〔07/05〕里的 retriever。

  代价是每次检索多花几次 LLM 调用。可直接 .llm_chain.invoke() 看它到底生成了哪些子查询，
  调试时别只看「最终返回几条」，要看「改写得合不合理」。
"""
