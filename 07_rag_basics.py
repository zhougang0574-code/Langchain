"""
第7课：RAG 基础（检索增强生成）

学习要点：
1. RAG 解决什么问题：LLM 训练数据有截止日期，无法回答私有/最新知识
2. Document —— LangChain 的文档对象：page_content + metadata
3. RecursiveCharacterTextSplitter —— 递归分割文本，保留语义完整性
4. OpenAIEmbeddings —— 把文本转为向量（接阿里百炼）
5. FAISS —— 本地向量数据库，存储和检索向量
6. 相似度检索：similarity_search / similarity_search_with_score
7. 完整 RAG 链：检索 + 注入 Prompt + LLM 生成
"""

import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# 阿里百炼 Embedding：OpenAIEmbeddings 内部会把文本转成 token id 再发送，
# 但阿里百炼只接受原始字符串，所以实现一个轻量的自定义 Embeddings 类
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


# ─────────────────────────────────────────────
# 1. 准备原始文档（模拟私有知识库）
# ─────────────────────────────────────────────
raw_text = """
LangChain 是一个用于构建大语言模型应用的开源框架，由 Harrison Chase 于2022年10月创建。
它提供了一套标准化的接口，让开发者可以轻松连接 LLM、数据源、工具和记忆组件。

LangChain 的核心组件包括：
- LCEL（LangChain Expression Language）：用 | 管道符连接各组件，构建可组合的链
- Prompt Template：结构化管理提示词，支持变量插值和 partial 绑定
- Output Parser：将 LLM 输出解析为结构化数据（str/JSON/Pydantic）
- Memory：管理对话历史，支持多会话隔离
- Tools & Agent：让 LLM 具备调用外部工具的能力
- RAG：结合向量检索，让 LLM 回答私有知识

LangGraph 是 LangChain 的姊妹项目，用于构建有状态的多步骤 Agent 工作流。
与 AgentExecutor 相比，LangGraph 提供更细粒度的控制，支持条件分支、循环和人工介入。

向量数据库是 RAG 的核心基础设施。常用的有：
- FAISS：Facebook 开源，纯本地，适合开发和中小规模生产
- Chroma：轻量级，支持持久化，开发者友好
- Pinecone：云托管，高可用，适合大规模生产
- Weaviate：支持混合搜索（向量 + 关键词）

Embedding 模型把文本转换为高维向量，语义相似的文本向量距离近。
阿里百炼提供 text-embedding-v1/v2/v3 等向量模型，通过 OpenAI 兼容接口调用。
"""


# ─────────────────────────────────────────────
# 2. 文本分割
# ─────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # 每块最大字符数
    chunk_overlap=30,     # 相邻块重叠字符数，避免语义在边界被截断
    # 递归分割优先级：先按段落(\n\n)，再按换行(\n)，再按句子，最后按字符
    separators=["\n\n", "\n", "。", "，", " ", ""],
)

chunks = splitter.create_documents([raw_text])
print("=== 文本分割 ===")
print(f"原文长度: {len(raw_text)} 字符，分割后: {len(chunks)} 块")
for i, chunk in enumerate(chunks):
    print(f"[{i}] ({len(chunk.page_content)}字) {chunk.page_content[:60]}...")


# ─────────────────────────────────────────────
# 3. 构建向量数据库
# ─────────────────────────────────────────────
print("\n=== 构建 FAISS 向量库 ===")
# from_documents 内部：① 调用 embeddings.embed_documents() 把每块文本转为向量
#                      ② 存入 FAISS 索引
vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"向量库构建完成，共 {vectorstore.index.ntotal} 条向量")


# ─────────────────────────────────────────────
# 4. 相似度检索
# ─────────────────────────────────────────────
print("\n=== 相似度检索 ===")
query = "LangChain 有哪些核心组件？"

# 返回最相关的 k 个文档
docs = vectorstore.similarity_search(query, k=3)
for i, doc in enumerate(docs):
    print(f"[{i}] {doc.page_content[:80]}...")

# 带分数的检索（分数越低越相似，L2 距离）
print("\n--- 带相似度分数 ---")
docs_with_scores = vectorstore.similarity_search_with_score(query, k=2)
for doc, score in docs_with_scores:
    print(f"score={score:.4f} | {doc.page_content[:60]}...")


# ─────────────────────────────────────────────
# 5. 转为 Retriever 对象
# ─────────────────────────────────────────────
# as_retriever() 把 vectorstore 包装成 Runnable，可以接入 LCEL 链
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # 检索 top-3
)
print("\n=== Retriever ===")
docs = retriever.invoke("向量数据库有哪些选择？")
print(f"检索到 {len(docs)} 个文档块")


# ─────────────────────────────────────────────
# 6. 完整 RAG 链
# ─────────────────────────────────────────────
# 把检索到的文档列表拼接成字符串，注入 prompt
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个问答助手。根据以下检索到的上下文回答问题。"
     "如果上下文中没有相关信息，就说不知道，不要编造。\n\n"
     "上下文：\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {
        # retriever 检索文档，format_docs 把列表转字符串
        "context": retriever | RunnableLambda(format_docs),
        # RunnablePassthrough 原样传递 question
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

print("\n=== RAG 完整链 ===")
questions = [
    "LangChain 是什么时候创建的？",
    "LangGraph 和 AgentExecutor 有什么区别？",
    "FAISS 适合什么场景？",
]
for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {rag_chain.invoke(q)}")


# ─────────────────────────────────────────────
# 7. 向量库持久化与加载
# ─────────────────────────────────────────────
# 保存到本地目录
vectorstore.save_local("faiss_index")
print("\n=== 持久化 ===")
print("已保存到 ./faiss_index/")

# 从本地加载（下次启动不需要重新 embed）
loaded_vs = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,  # FAISS 加载需要此参数
)
print(f"加载成功，共 {loaded_vs.index.ntotal} 条向量")


if __name__ == "__main__":
    pass


"""
执行流程图：

原始文档（长文本）
      │
      │ RecursiveCharacterTextSplitter
      ▼
[chunk_0, chunk_1, ..., chunk_N]  ← 每块 ~200 字
      │
      │ OpenAIEmbeddings.embed_documents()
      ▼
[vec_0, vec_1, ..., vec_N]  ← 每条向量维度 1536
      │
      │ FAISS.from_documents()
      ▼
   FAISS 向量索引（本地）

查询时：
用户问题
      │
      │ embed_query() → 查询向量
      │ similarity_search() → 找最近邻
      ▼
[相关 chunk_i, chunk_j, chunk_k]
      │
      │ format_docs() → 拼接字符串
      ▼
   Prompt（含上下文 + 问题）
      │
      │ LLM 生成
      ▼
   基于私有知识的回答


核心知识点 ★：

★ chunk_overlap 很重要：避免重要信息刚好在块边界被切断，通常设 chunk_size 的 10-20%
★ similarity_search_with_score 的分数是 L2 距离，越小越相似（不是 0-1 的相似度）
★ as_retriever() 把 vectorstore 变成 Runnable，才能接入 LCEL 链
★ RAG 链的标准结构：{"context": retriever|format, "question": passthrough} | prompt | llm
★ temperature=0 在 RAG 中很重要：减少 LLM 在上下文基础上"发挥"的概率
★ 向量库持久化：save_local / load_local，生产环境必须做，避免每次启动重新 embed
★ RAG 效果取决于：分块质量 > embedding 模型质量 > 检索策略 > prompt 设计
"""
