"""
第8课：RAG 进阶 —— 对话式 RAG、MultiQueryRetriever、查漏补缺

学习要点：
1. 对话式 RAG —— 结合 Memory，让 RAG 支持多轮追问
2. 问题重写（Contextualize） —— 把"它是什么"还原为"LangChain 是什么"
3. MultiQueryRetriever —— 用 LLM 生成多个查询角度，提升召回率
4. MMR 检索策略 —— 最大边际相关，减少结果冗余
5. 文档评分过滤 —— 只使用高置信度的检索结果
6. 整体 RAG 查漏：常见问题和最佳实践
"""

import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiQueryRetriever

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


class AliyunEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


embeddings = AliyunEmbeddings(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model="text-embedding-v1",
)

vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ─────────────────────────────────────────────
# 1. 对话式 RAG 的核心问题
# ─────────────────────────────────────────────
# 问题：用户追问"它有什么优点？"时，"它"指什么 LLM 不知道
# 解决：在检索前，用历史对话把问题还原为完整、独立的问题
#
# 问题重写 prompt：把带指代词的问题改写为完整问题
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "根据对话历史和最新用户问题，将问题改写为不依赖历史的独立问题。"
     "如果问题已经完整，直接返回原问题，不要作答。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 问题改写链：有历史时改写，无历史时原样返回
contextualize_chain = contextualize_prompt | llm | StrOutputParser()

print("=== 问题重写演示 ===")
from langchain_core.messages import HumanMessage, AIMessage
fake_history = [
    HumanMessage(content="LangChain 是什么？"),
    AIMessage(content="LangChain 是一个用于构建 LLM 应用的开源框架。"),
]
rewritten = contextualize_chain.invoke({
    "history": fake_history,
    "input": "它是什么时候创建的？",
})
print("改写前: 它是什么时候创建的？")
print("改写后:", rewritten)


# ─────────────────────────────────────────────
# 2. 完整对话式 RAG 链
# ─────────────────────────────────────────────
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个问答助手，根据上下文回答问题。如果不知道就说不知道。\n\n上下文：\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

def contextualize_and_retrieve(input_dict: dict) -> list[Document]:
    """有历史时先改写问题再检索，无历史时直接检索。"""
    history = input_dict.get("history", [])
    question = input_dict["input"]
    if history:
        # 有历史：改写问题再检索
        standalone = contextualize_chain.invoke({"history": history, "input": question})
    else:
        standalone = question
    return retriever.invoke(standalone)

conversational_rag_chain = (
    RunnablePassthrough.assign(
        # 先检索（可能改写问题），把文档格式化后赋给 context
        context=RunnableLambda(contextualize_and_retrieve) | RunnableLambda(format_docs)
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

# 包装记忆
store: dict = {}
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_with_memory = RunnableWithMessageHistory(
    conversational_rag_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "rag_session_1"}}

print("\n=== 对话式 RAG 多轮问答 ===")
q1 = conversational_rag_with_memory.invoke({"input": "LangChain 是什么？"}, config=config)
print("Q1: LangChain 是什么？")
print("A1:", q1)

q2 = conversational_rag_with_memory.invoke({"input": "它有哪些核心组件？"}, config=config)
print("\nQ2: 它有哪些核心组件？")  # "它"依赖上文
print("A2:", q2)

q3 = conversational_rag_with_memory.invoke({"input": "LangGraph 和它有什么区别？"}, config=config)
print("\nQ3: LangGraph 和它有什么区别？")
print("A3:", q3)


# ─────────────────────────────────────────────
# 3. MultiQueryRetriever —— 多角度检索提升召回
# ─────────────────────────────────────────────
# 原理：用 LLM 把一个问题改写成多个角度的子问题，分别检索，去重合并
# 解决单一查询角度导致遗漏相关文档的问题
print("\n=== MultiQueryRetriever ===")
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    llm=llm,
)

# 开启日志可以看到生成的多个查询
import logging
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

docs = multi_retriever.invoke("介绍一下 LangChain 的向量存储方案")
print(f"MultiQuery 检索到 {len(docs)} 个不重复文档（单查询只有2个）")
for doc in docs:
    print(f"  - {doc.page_content[:60]}...")


# ─────────────────────────────────────────────
# 4. MMR 检索策略 —— 减少结果冗余
# ─────────────────────────────────────────────
# MMR（Maximal Marginal Relevance）：在相关性和多样性之间取平衡
# lambda_mult 越小 → 多样性优先；越大 → 相关性优先
print("\n=== MMR 检索 ===")
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # 最终返回 3 个
        "fetch_k": 10,    # 先召回 10 个候选
        "lambda_mult": 0.5,  # 0=最大多样性，1=纯相关性
    }
)
docs = mmr_retriever.invoke("LangChain 的组件")
print(f"MMR 返回 {len(docs)} 个多样性文档")


# ─────────────────────────────────────────────
# 5. 带分数过滤 —— 只用高置信度结果
# ─────────────────────────────────────────────
# 解决：即使没有相关内容，普通检索也会强行返回 k 个结果
print("\n=== 分数过滤 ===")
def filtered_retriever(query: str, threshold: float = 12000.0) -> list[Document]:
    """只返回相似度分数低于阈值的文档（FAISS L2 距离，越小越相似）。
    text-embedding-v1 实测：相关查询 ~4000-10000，无关查询 ~18000+。
    threshold=12000 可以过滤掉与知识库完全无关的查询。
    """
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)
    print(f"  原始分数: {[round(float(s), 0) for _, s in docs_with_scores]}")
    filtered = [doc for doc, score in docs_with_scores if score < threshold]
    print(f"  过滤后 {len(filtered)} 个（threshold={threshold}）")
    return filtered

print("  [相关查询]")
filtered_retriever("LangChain 的核心组件")
print("  [无关查询——应被过滤]")
filtered_retriever("今天股市行情怎么样")


if __name__ == "__main__":
    pass


"""
执行流程图（对话式 RAG）：

用户追问（含指代词）+ 对话历史
         │
         │ contextualize_chain
         ▼
   独立完整问题（"LangChain 是什么时候创建的？"）
         │
         │ retriever.invoke()
         ▼
   相关文档 [doc1, doc2, doc3]
         │
         │ format_docs()
         ▼
   上下文字符串
         │
         ▼
   QA Prompt（含历史 + 上下文 + 当前问题）
         │
         │ LLM 生成
         ▼
   基于私有知识 + 对话历史的回答
         │
         │ RunnableWithMessageHistory 自动追加本轮到历史
         ▼
   历史更新，等待下一轮


核心知识点 ★：

★ 对话式 RAG 的关键：检索前先"问题重写"，把指代词还原为完整问题
★ MultiQueryRetriever 用多个查询角度提升召回率，适合问题表达方式多样的场景
★ MMR 减少检索结果冗余，当知识库有大量相似内容时效果明显
★ 带分数过滤解决"宁可不答也不乱答"的问题，提升 RAG 可信度
★ 对话式 RAG = 第5课 Memory + 第7课 RAG 的组合，核心是 contextualize_chain
★ RAG 优化优先级：数据质量 > 分块策略 > 检索策略 > Prompt 设计 > LLM 选择
"""
