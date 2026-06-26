"""
【07 检索增强RAG / 08】对话式 RAG —— 多轮追问 + 问题重写（含实测踩坑）〔进阶〕
========================================================================
〔07/05〕的 RAG 是「一问一答」，不记得上文。但真实对话里用户会追问：
「它是谁创建的？」——「它」指什么，得结合上一轮才知道。难点在于：检索器只认
当前这句话的字面，「它有什么优点」直接拿去检索几乎检不到东西。

新概念（只有这一个）：
  问题重写（contextualize）—— 检索前先用 LLM 把「带指代词的追问」+「对话历史」
  改写成一个「不依赖上下文、信息完整的独立问题」，再拿这个独立问题去检索。
  组合：〔05 记忆〕的历史 + 本课的重写 + 〔07/05〕的 RAG 链。

前置：加载〔07/02〕保存的 faiss_index/，请先跑过 02。
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import get_embeddings   # noqa: E402

load_dotenv()
HERE = os.path.dirname(os.path.abspath(__file__))
embeddings = get_embeddings()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"), temperature=0,
)
retriever = FAISS.load_local(
    os.path.join(HERE, "faiss_index"), embeddings, allow_dangerous_deserialization=True
).as_retriever(search_kwargs={"k": 3})


# ── 1. 问题重写链：把带指代词的追问改成独立问题 ────────────────────────────
# ⚠️ 实测踩坑：prompt 只写「改写成独立问题、不要作答」时，历史一长，模型会被带偏，
#    把「改写」做成「真的在回答」，吐一大段话当 query，语义被稀释、检索悄悄变差还不报错。
#    解法：① 给字数上限 ② 给一个输入输出示例 ③ 把禁止项写直白。
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你只做一件事：把用户最新问题改写为不依赖对话历史、信息完整的独立问题。"
     "严格要求：只输出改写后的问题本身，不超过 25 字，禁止解释、禁止作答。"
     "示例 —— 历史:[问:LangChain是什么? 答:一个框架] 输入:它是谁创建的? 输出:LangChain 是谁创建的?"
     "若输入本就完整，原样输出。"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
contextualize_chain = contextualize_prompt | llm | StrOutputParser()


def contextualize_and_retrieve(d: dict) -> list[Document]:
    """有历史就先改写问题再检索；并把「实际拿去检索的独立问题」打印出来核对。"""
    history = d.get("history", [])
    standalone = contextualize_chain.invoke({"history": history, "input": d["input"]}) if history else d["input"]
    print(f"    [内部检索用的独立问题]：{standalone}")
    return retriever.invoke(standalone)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(x.page_content for x in docs)


# ── 2. 完整对话式 RAG 链 ───────────────────────────────────────────────────
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "依据上下文回答，不知道就说不知道。\n上下文：\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
conv_rag = (
    RunnablePassthrough.assign(
        context=RunnableLambda(contextualize_and_retrieve) | RunnableLambda(format_docs)
    )
    | qa_prompt | llm | StrOutputParser()
)

# ── 3. 套上记忆（复习〔05/02〕）───────────────────────────────────────────
store: dict = {}
def get_history(sid: str):
    return store.setdefault(sid, InMemoryChatMessageHistory())

conv_rag_mem = RunnableWithMessageHistory(
    conv_rag, get_history, input_messages_key="input", history_messages_key="history"
)


if __name__ == "__main__":
    cfg = {"configurable": {"session_id": "s1"}}
    print("Q1: LangChain 是什么？")
    print("A1:", conv_rag_mem.invoke({"input": "LangChain 是什么？"}, config=cfg))

    print("\nQ2: 它是谁创建的？        （「它」依赖上一轮）")
    print("A2:", conv_rag_mem.invoke({"input": "它是谁创建的？"}, config=cfg))

    print("\nQ3: 它和 LangGraph 啥区别？（跨轮指代）")
    print("A3:", conv_rag_mem.invoke({"input": "它和 LangGraph 有什么区别？"}, config=cfg))


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  追问"它是谁创建的?" + 历史
        │ contextualize_chain（改写）
        ▼
  独立问题"LangChain 是谁创建的?"
        │ retriever 检索
        ▼
  相关文档 → qa_prompt（含上下文+历史+当前问题）→ llm → 回答
        │ RunnableWithMessageHistory 自动存回本轮
        ▼
  历史更新，等下一轮

★ 核心规律：
  对话式 RAG = 〔05 记忆〕+ 检索前「问题重写」+ 〔07/05〕RAG 链。重写是关键：
  指代词不还原，检索就会失准。

  实测教训：「改写指令」在短历史里好用、长历史里会失效（模型把改写做成作答）。
  必须用强约束（字数上限 + one-shot 示例 + 直白禁止项），并把「内部独立问题」打印出来核对——
  这类问题只看最终答案发现不了，必须看中间结果。
"""
