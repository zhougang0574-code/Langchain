"""
【09 工程化与可靠性 / 04】何时不必上 LangGraph —— RunnableBranch 的边界
==================================================================
〔04/05〕学过 RunnableBranch。本课从「工程选型」角度再看它一次：很多人一遇到
「要分情况处理」就想上 LangGraph，但若只是「if-else 选条链」，纯 LCEL 就够，
上 LangGraph 反而增加复杂度。本课讲清这条边界。

新概念（不是新 API，是一条工程判断准则）：
  纯「选哪条链」（无循环、无需在多步间保存状态）→ 用 RunnableBranch，留在 LangChain。
  需要循环、需要持久化状态、需要人工干预/回放 → 才上 LangGraph。

为什么单列一课：
  选型是工程能力的一部分。把「够用就好」的判断显式化，避免无脑过度设计。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"), temperature=0,
)
parser = StrOutputParser()

# ── 一个轻量分流：客服路由到不同回复风格 ──────────────────────────────────
tech = ChatPromptTemplate.from_template("用专业技术口吻回答：{q}") | llm | parser
sales = ChatPromptTemplate.from_template("用热情销售口吻回答：{q}") | llm | parser
general = ChatPromptTemplate.from_template("用简洁中性口吻回答：{q}") | llm | parser

route = RunnableBranch(
    (lambda x: any(k in x["q"] for k in ["报错", "bug", "怎么实现"]), tech),
    (lambda x: any(k in x["q"] for k in ["价格", "购买", "优惠"]), sales),
    general,
)


if __name__ == "__main__":
    for q in ["这个接口报错了怎么办", "有什么优惠活动吗", "你们公司在哪"]:
        print(f"\n[{q}]")
        print(route.invoke({"q": q}))


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  需求：「按输入选一条处理链」
        │
        ├─ 无循环、无跨步状态 ──► RunnableBranch（本课）：留在 LangChain，轻量
        └─ 要循环 / 要持久化状态 / 要人工干预 ──► LangGraph：有状态图编排

★ 核心规律：
  不是「一有分支就上 LangGraph」。纯 if-else 选链用 RunnableBranch 就够，更简单。
  LangGraph 的价值在「循环 + 状态持久化 + 中断/回放」——没用到这些就不必引入。

  这条边界呼应〔06/03〕：create_agent 这类「循环调工具」的场景底层才用 LangGraph。
  工程上的好品味，是用「刚好够用」的工具，而不是最重的工具。
"""
