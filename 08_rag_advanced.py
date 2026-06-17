"""
第8课：RAG 进阶 —— 对话式 RAG、MultiQueryRetriever、查漏补缺（详细版）

学习要点：
1. 对话式 RAG —— 结合 Memory，让 RAG 支持多轮追问
2. 问题重写（Contextualize） —— 把"它是什么"还原为"LangChain 是什么"
   - 显式打印每一轮内部生成的"独立问题"，看清重写过程
3. MultiQueryRetriever —— 用 LLM 生成多个查询角度，提升召回率
   - 显式拿到并打印 LLM 生成的子查询，而不是只看最终文档数量
4. MMR 检索策略 —— 最大边际相关，减少结果冗余
   - 用代码直观对比 MMR vs 普通 top-k 在重复度上的差异
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

# 复用第7课持久化的向量库，不用重新 embed 文档
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
     "你只做一件事：把最新用户问题改写为不依赖对话历史、信息完整的独立问题。"
     "严格要求：输出必须只有改写后的问题本身，不超过20个字，禁止解释、禁止列点、禁止回答问题本身。"
     "示例 —— 历史:[问:LangChain是什么? 答:一个开源框架] 输入:它的作者是谁? 输出:LangChain的作者是谁?"
     "如果输入已经是独立完整的问题，原样输出即可。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
# ⚠️ 查漏坑点：第一版只写了"将问题改写为独立问题，不要作答"，在历史只有1轮、AI 回答较短时表现正常；
# 但实测发现——一旦 AI 的历史回答变长（真实场景里 LLM 经常输出几百字的详细解答），
# 到第3轮时模型会被长历史"带偏"，把这一步当成真的在回答问题，吐出一整段解释当作"改写后的问题"，
# 这段长文本会被当作检索 query 送进向量库，语义被严重稀释，检索精度悄悄下降却不会报错。
# 解决方法：①明确给字数上限（不超过20个字）②给一个输入输出的具体示例（one-shot）
# ③把"禁止列点、禁止回答问题本身"写得更直白。仅靠"语气委婉的指令"在长上下文里不可靠，必须用强约束。
# 教训：多轮对话场景务必把内部中间结果打印出来核对，不能只看最终答案"看起来对不对"。

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

# 再试一个问题已经完整、不需要改写的情况，验证链能"识别"出不需要重写
print("\n--- 对比：问题已完整，不应被改写 ---")
already_standalone = contextualize_chain.invoke({
    "history": fake_history,
    "input": "FAISS 和 Chroma 有什么区别？",
})
print("输入: FAISS 和 Chroma 有什么区别？")
print("输出:", already_standalone, "（应与输入基本一致，证明链没有强行改写）")


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
    # 把"内部真正拿去检索的问题"打印出来，这样多轮对话里能清楚看到指代词被还原成了什么
    print(f"    [内部检索用的独立问题]: {standalone}")
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
print("\nQ2: 它有哪些核心组件？")  # "它"依赖上文，注意上面打印出的内部独立问题里"它"应已被还原为"LangChain"
print("A2:", q2)

q3 = conversational_rag_with_memory.invoke({"input": "LangGraph 和它有什么区别？"}, config=config)
print("\nQ3: LangGraph 和它有什么区别？")  # 这里"它"指代更早提到的 LangChain，跨越了两轮对话
print("A3:", q3)

# 看一眼此刻历史里真正存了什么——RunnableWithMessageHistory 是自动追加的，不需要手动维护
print(f"\n当前 session 历史共 {len(store['rag_session_1'].messages)} 条消息"
      f"（3 轮问答 = 6 条：每轮 1 条 Human + 1 条 AI）")


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

# 开启日志可以看到生成的多个查询（INFO 级别会打印 "Generated queries: [...]"）
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# 也可以不依赖日志，直接调用内部的 llm_chain 拿到生成的子查询文本，更直观地展示
generated_queries = multi_retriever.llm_chain.invoke(
    {"question": "介绍一下 LangChain 的向量存储方案"}
)
print("LLM 生成的子查询：")
for q in generated_queries:
    print(f"  - {q}")

docs = multi_retriever.invoke("介绍一下 LangChain 的向量存储方案")
print(f"\nMultiQuery 合并去重后检索到 {len(docs)} 个不重复文档（单一查询 k=2 只会有2个）")
for doc in docs:
    print(f"  - {doc.page_content[:60]}...")


# ─────────────────────────────────────────────
# 4. MMR 检索策略 —— 减少结果冗余
# ─────────────────────────────────────────────
# MMR（Maximal Marginal Relevance，最大边际相关）：
#   普通 top-k 只看"和查询的相关性"，如果知识库里有多块内容相似，
#   top-k 很可能全部挤在同一个语义点上，浪费名额。
#   MMR 在每一步选择时都做两件事的加权：
#     ① 候选文档与查询的相关性（越相关越好）
#     ② 候选文档与"已选中文档"的差异性（越不同越好）
#   lambda_mult 就是这两者的权重：越小 → 多样性优先；越大 → 相关性优先。
print("\n=== MMR 检索 ===")
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # 最终返回 3 个
        "fetch_k": 10,    # 先用普通相关性召回 10 个候选，再从中做多样性挑选
        "lambda_mult": 0.5,  # 0=最大多样性，1=纯相关性（等价于普通 top-k）
    }
)
mmr_docs = mmr_retriever.invoke("LangChain 的组件")
print(f"MMR (lambda_mult=0.5) 返回 {len(mmr_docs)} 个文档：")
for doc in mmr_docs:
    print(f"  - {doc.page_content[:50]}...")

# 对比：lambda_mult=1 时应退化为普通相关性排序，和 similarity_search 结果基本一致
pure_relevance_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 1.0},
)
pure_docs = pure_relevance_retriever.invoke("LangChain 的组件")
plain_docs = vectorstore.similarity_search("LangChain 的组件", k=3)
same_order = [d.page_content for d in pure_docs] == [d.page_content for d in plain_docs]
print(f"\nlambda_mult=1.0 时结果与普通 similarity_search 是否一致: {same_order}")


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
print("  [半相关查询——观察阈值附近的行为]")
filtered_retriever("Python 这门编程语言怎么样")


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

执行流程图（MMR 选择过程，lambda_mult=0.5）：

fetch_k=10 个候选（按相关性排序）
         │
         │ 第1步：直接选相关性最高的1个 → 已选集合 = {A}
         │
         │ 第2步：剩余候选逐个计算
         │   score_i = λ·相关性(i) - (1-λ)·max(与已选集合的相似度)
         │   选 score 最高的 → 已选集合 = {A, B}
         │
         │ 重复直到选够 k 个
         ▼
   k 个"既相关又互相不重复"的文档


核心知识点 ★：

★ 对话式 RAG 的关键：检索前先"问题重写"，把指代词还原为完整问题；
  本课用 print 把每轮内部生成的独立问题暴露出来，能清楚看到"它"是怎么被替换的。
★ 【实测查漏】contextualize 的 prompt 仅写"改写为独立问题、不要作答"在历史较短时没问题，
  但历史里的 AI 回答一旦变长（真实场景常见），第3轮起模型会被带偏，把"改写"做成"真的在回答"，
  超长文本被当成检索 query 送进向量库，语义被稀释，检索精度悄悄下降但不会报错。
  解决：①给字数上限（如不超过20字）②给一个输入输出的具体示例（one-shot）③把禁止项写得更直白。
  这类"指令在短上下文里管用、长上下文里失效"的问题只能靠打印中间结果才能发现。
★ MultiQueryRetriever 用多个查询角度提升召回率，适合问题表达方式多样的场景；
  可以直接调用 multi_retriever.llm_chain.invoke() 拿到生成的子查询文本，而不是只看最终文档数量。
★ MMR 的核心是"边际"：每一步新选的文档要同时满足相关 + 与已选结果不同，
  lambda_mult=1 时退化为普通相关性排序，等价于 similarity_search。
★ 带分数过滤解决"宁可不答也不乱答"的问题，但阈值必须用实际数据校准，
  不同 embedding 模型的 L2 距离量级完全不同，不能照搬别的项目的阈值。
★ 对话式 RAG = 第5课 Memory + 第7课 RAG 的组合，核心是 contextualize_chain。
★ RAG 优化优先级：数据质量 > 分块策略 > 检索策略（MultiQuery/MMR）> Prompt 设计 > LLM 选择。
"""
