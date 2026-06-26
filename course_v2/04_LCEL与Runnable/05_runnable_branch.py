"""
【04 LCEL与Runnable / 05】RunnableBranch —— 条件分支，按输入走不同的链
==================================================================
前面的链都是「固定路径」。但有时要按输入内容走不同处理：数学问题走计算链、
翻译请求走翻译链、其它走闲聊链。这就是 if-else，但要在链里表达。

新概念（只有这一个）：
  RunnableBranch(
      (条件1, 链1),
      (条件2, 链2),
      默认链,                       # 最后一个不带条件，都不满足时走它
  )
  按顺序判断条件，第一个为 True 的分支被执行；都不满足走默认链。

什么时候用它：
  纯粹的「if-else 选一条链」用 RunnableBranch 就够，不必动用更重的 LangGraph。
  （需要循环、需要在多步之间保存状态时，才轮到 LangGraph 出场——见〔09/04〕。）
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)
parser = StrOutputParser()


# ── 1. 三条不同用途的子链 ──────────────────────────────────────────────────
translate_chain = ChatPromptTemplate.from_template("把这句翻译成英文：{query}") | llm | parser
math_chain = ChatPromptTemplate.from_template("只输出这道题的计算结果：{query}") | llm | parser
chat_chain = ChatPromptTemplate.from_template("友好地回应：{query}") | llm | parser


# ── 2. 判断条件（输入是 {"query": "..."}）──────────────────────────────────
def is_translate(x: dict) -> bool:
    return "翻译" in x["query"]


def is_math(x: dict) -> bool:
    return any(ch.isdigit() for ch in x["query"])


# ── 3. 组装分支：按顺序判断，命中即走对应链 ────────────────────────────────
branch = RunnableBranch(
    (is_translate, translate_chain),
    (is_math, math_chain),
    chat_chain,                     # 默认分支
)


if __name__ == "__main__":
    for q in ["帮我翻译：今天天气不错", "计算 23 乘以 17 等于多少", "你今天心情怎么样"]:
        print(f"\n[输入] {q}")
        print("[输出]", branch.invoke({"query": q}))


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  输入 ──► is_translate? ──是──► 翻译链
              │否
              ▼
           is_math? ──是──► 计算链
              │否
              ▼
           默认：闲聊链

★ 核心规律：
  RunnableBranch 是 LCEL 内置的「if-elif-else 选链」：按顺序匹配条件，
  第一个 True 的分支执行，全不中走最后的默认链。

  它只做「选哪条链」，不做循环、不存中间状态——这类轻量分支不必上 LangGraph。
  这条「何时该用 LangGraph」的边界，〔09/04〕会再强调一次。
"""
