"""
【07 检索增强RAG / 05】完整 RAG 链 —— 把前四课串成「检索增强问答」
==============================================================
前四课分别学了 RAG 的四个零件：向量化〔01〕、向量库〔02〕、加载器〔03〕、切分〔04〕。
本课把它们 + 〔04 LCEL〕的链拼成一条完整的 RAG 链：问一句，先检索再作答。

新概念（只有这一个）：
  retriever —— 向量库的「检索器」视角：vectorstore.as_retriever()。
    它本身是个 Runnable，输入问题字符串、输出相关 Document 列表，可直接进 LCEL 链。

完整 RAG 链的结构（复习〔04/04〕的 assign/parallel）：
  {context: 检索并拼接, question: 原样透传} | prompt | llm | parser
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import get_embeddings   # noqa: E402

load_dotenv()
HERE = os.path.dirname(os.path.abspath(__file__))

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"), temperature=0,
)
embeddings = get_embeddings()


# ── 1. 加载 → 切分 → 建库 → 取 retriever（前四课的串联）───────────────────
docs = TextLoader(os.path.join(HERE, "assets", "knowledge.txt"), encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20).split_documents(docs)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})   # 每次检索取 3 块


# ── 2. 把检索到的多个 Document 拼成一段上下文文本 ──────────────────────────
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ── 3. 组装 RAG 链 ─────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是问答助手，只能依据下面的「上下文」回答；上下文没有就说不知道。\n\n上下文：\n{context}"),
    ("human", "{question}"),
])

# {context: 检索+拼接, question: 原样透传}  ← 复习〔04/04〕：用 dict 同时备好两个字段
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    for q in ["LangChain 是谁创建的？", "RAG 解决什么问题？", "今天股市怎么样？"]:
        print(f"\nQ: {q}")
        print("A:", rag_chain.invoke(q))
    print("\n（最后一问知识库里没有，应回答「不知道」而不是瞎编——这正是 RAG 的可控性）")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  问题字符串
        │
        ├─ context :  retriever（检索3块）→ format_docs（拼成一段文本）
        └─ question:  RunnablePassthrough（原样透传）
        │  合成 {"context": ..., "question": ...}
        ▼
     prompt（把上下文和问题塞进模板）
        ▼  llm → parser
     基于私有知识的回答

★ 核心规律：
  RAG 链 = 「检索出 context」+「透传 question」并成一个 dict → prompt → llm → parser。
  retriever 是 Runnable，能直接进 LCEL；format_docs 把多个 Document 拼成一段文本。

  prompt 里「只依据上下文、没有就说不知道」这句很关键：它把模型约束在私有知识范围内，
  是 RAG「可控、可溯源、少幻觉」的来源。进阶检索策略见〔07/06〕〔07/07〕。
"""
