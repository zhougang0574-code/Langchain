"""
【07 检索增强RAG / 09】上下文压缩与重排序 —— 把检索结果再精炼一遍〔进阶〕
====================================================================
〔07/05〕的检索器返回的是「整块」文档，里面常夹杂和问题无关的句子；MMR/MultiQuery
（〔06〕〔07〕）改善的是「召回哪些块」，但块内部仍可能很啰嗦。检索到之后、喂给 LLM 之前，
再精炼一道，能提升答案质量、省 token。这一步叫「后处理」，两种主流做法：压缩与重排序。

新概念（只有这一个）：
  ContextualCompressionRetriever(base_retriever, base_compressor)
    - 包在普通检索器外面：先正常检索，再用 compressor 对结果「二次加工」
    - 两类常用 compressor：
        LLMChainExtractor —— 压缩：用 LLM 把每块里「和问题相关的句子」抽出来，丢掉无关部分
        LLMListwiseRerank —— 重排序：用 LLM 给候选块按相关性重排，只留最相关的前几条

它和前面检索课的关系：
  MMR/MultiQuery 管「召回」，本课管「召回之后的精炼」。对外仍是一个 retriever，
  可无缝替换〔07/05〕RAG 链里的 retriever。

前置：加载〔07/02〕保存的 faiss_index/，请先跑过 02。
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor, LLMListwiseRerank

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import get_embeddings   # noqa: E402

load_dotenv()
HERE = os.path.dirname(os.path.abspath(__file__))

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"), temperature=0,
)
base_retriever = FAISS.load_local(
    os.path.join(HERE, "faiss_index"), get_embeddings(), allow_dangerous_deserialization=True
).as_retriever(search_kwargs={"k": 4})


if __name__ == "__main__":
    query = "RAG 解决什么问题？"

    print("=== 普通检索：返回整块，含无关句子 ===")
    for d in base_retriever.invoke(query):
        print(f"  ({len(d.page_content)}字) {d.page_content[:50].strip()}...")

    # ── 1. 压缩：LLMChainExtractor 抽出每块里和问题相关的句子 ──────────────
    print("\n=== 上下文压缩（LLMChainExtractor）：每块只留相关句子 ===")
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever, base_compressor=compressor
    )
    for d in compression_retriever.invoke(query):
        print(f"  ({len(d.page_content)}字) {d.page_content[:60].strip()}...")

    # ── 2. 重排序：LLMListwiseRerank 按相关性重排，只留前 2 条 ─────────────
    print("\n=== 重排序（LLMListwiseRerank）：召回 4 条 → 精排留最相关 2 条 ===")
    reranker = LLMListwiseRerank.from_llm(llm, top_n=2)
    rerank_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever, base_compressor=reranker
    )
    for d in rerank_retriever.invoke(query):
        print(f"  - {d.page_content[:50].strip()}...")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  base_retriever 检索出 4 块（整块、含噪声）
        │ ContextualCompressionRetriever 二次加工
        ▼
  压缩 LLMChainExtractor：每块抽出相关句子 → 块变短、噪声少、省 token
  重排 LLMListwiseRerank ：对 4 块按相关性重排 → 只留最相关的 top_n

★ 核心规律：
  检索后处理 = 召回之后再精炼。压缩（抽相关句）让上下文更干净、更省 token；
  重排序（精排留前几）让最相关的排在前、剔除滥竽充数。两者都用
  ContextualCompressionRetriever 包装，对外仍是普通 retriever，可直接替换 RAG 链里的检索器。

  代价：每次检索多花 LLM 调用（压缩/重排都要过一遍 LLM）。
  生产里若追求低延迟，重排常改用专门的交叉编码器 rerank 模型（如各家的 rerank API）替代 LLM 重排。
  优先级口诀：先把「召回」做好（数据/分块/MMR），再用压缩/重排锦上添花。
"""
