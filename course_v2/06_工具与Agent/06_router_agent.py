"""
【06 工具与Agent / 06】路由 Agent —— 用 LLM 当分类器分发到不同专家〔进阶〕
========================================================================
〔04/05〕的 RunnableBranch 靠写死的 Python 条件分支；〔06/04〕Agent 在「工具层」自己选工具。
本课的路由发生在更前面一层：先用一次 LLM 把请求「分类」，再分发给不同的专家链。

新概念（只有这一个）：
  Router（路由）模式 —— 先用 LLM（配 with_structured_output〔03/04〕）判断意图类别，
  再按类别 dispatch 到对应的专家处理。
  关键区别：路由 key 由模型「读懂语义」后决定（不同于〔04/05〕写死的 if 条件）。

为什么需要：
  一个助手要同时处理「数学 / 编程 / 闲聊」时，与其塞给一个全能 prompt，
  不如先分流到各自最擅长的链——每条专家链 prompt 更聚焦、更可控、也更省 token。
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# ── 1. 路由器：用结构化输出强制模型只回一个类别 ────────────────────────────
class Route(BaseModel):
    """对用户请求的意图分类。"""
    category: Literal["math", "code", "chat"] = Field(description="请求所属类别")

router = (
    ChatPromptTemplate.from_messages([
        ("system", "判断用户请求属于哪类：math=数学计算，code=编程问题，chat=日常闲聊。"),
        ("human", "{question}"),
    ])
    | llm.with_structured_output(Route)          # 〔03/04〕：直接拿到 Route 对象
)


# ── 2. 三条专家链：各自的 system prompt 更聚焦 ──────────────────────────────
def expert(system_prompt: str):
    return (
        ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
        | llm | StrOutputParser()
    )

experts = {
    "math": expert("你是数学老师，分步骤算出答案，给出最终数值。"),
    "code": expert("你是资深工程师，用简洁代码 + 一句话说明回答编程问题。"),
    "chat": expert("你是亲切的聊天伙伴，轻松自然地回应。"),
}


if __name__ == "__main__":
    for question in ["帮我算一下 18 乘以 24", "Python 怎么反转一个列表？", "今天有点累，聊两句吧"]:
        route = router.invoke({"question": question})       # → Route(category=...)
        answer = experts[route.category].invoke({"question": question})  # 按类别 dispatch
        print(f"\n[问题] {question}")
        print(f"[路由→ {route.category}] {answer}")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  question
        │  router（LLM + with_structured_output）读懂语义，给出 category
        ▼
  Route(category="math" | "code" | "chat")
        │  experts[category]  ← 按类别 dispatch 到对应专家链
        ▼
  专家链（各自聚焦的 system prompt）→ answer

★ 核心规律：
  路由 = 先用 LLM 当「分类器」（结构化输出锁定取值范围）得到路由 key，再分发到专家处理。
  vs〔04/05〕RunnableBranch：那里条件是写死的 Python 判断；这里 key 由模型理解语义后产生。

  这是「单 Agent + 分流」的轻量做法。当各专家本身也是带工具的 Agent、还要互相移交（handoff）、
  共享状态时，就升级为「多 Agent / supervisor」——那部分放在 LangGraph 课程，本课不展开。
"""
