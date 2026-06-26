"""
【04 LCEL与Runnable / 03】RunnableParallel —— 多路并发，结果合并成 dict
==================================================================
〔04/01〕〔04/02〕的链都是「一条直线」，一步接一步。但有时你想对同一个输入
「同时」跑几条独立的子链——比如同时让模型列优点和缺点，然后合并结果。

新概念（只有这一个）：
  RunnableParallel({"key1": 链1, "key2": 链2, ...})
    - 把同一个输入分发给每条子链，并发执行
    - 各子链结果按 key 合并成一个 dict 返回
  （在链里直接写一个 {"k": 子链} 字典，LangChain 会自动当成 RunnableParallel。）

为什么有用：
  ① 并发省时间（多路同时跑，不是排队）；
  ② 是「给下一步同时准备多个字段」的标准手段——〔07 RAG〕里同时备好
     context（检索结果）和 question（原问题）就靠它。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)
parser = StrOutputParser()


# ── 1. 两条独立子链，并发执行 ──────────────────────────────────────────────
pros = ChatPromptTemplate.from_template("用一句话说「{thing}」的一个优点") | llm | parser
cons = ChatPromptTemplate.from_template("用一句话说「{thing}」的一个缺点") | llm | parser

# 同一个输入 {"thing": ...} 分发给两条子链，结果合并成 {"优点":..., "缺点":...}
parallel = RunnableParallel({"优点": pros, "缺点": cons})


if __name__ == "__main__":
    import time

    print("=== RunnableParallel：并发跑两路 ===")
    t0 = time.time()
    result = parallel.invoke({"thing": "远程办公"})
    print(f"（两路并发，总耗时 {time.time() - t0:.1f}s，而不是两次相加）\n")
    print("优点：", result["优点"])
    print("缺点：", result["缺点"])
    print("\n返回类型：", type(result).__name__, "→ 一个 dict")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
              {"thing": "远程办公"}
               /                \\        ← 同一输入，分发两路
        pros 子链            cons 子链     ← 并发执行
               \\                /
        合并成 {"优点": r1, "缺点": r2}

★ 核心规律：
  RunnableParallel 把一个输入「扇出」到多条子链并发跑，结果按 key 合并成 dict。
  在 LCEL 里，一个写成 {"k": 子链, ...} 的字典会被自动当作 RunnableParallel。

  注意它会「重构」输出 dict——只保留你声明的 key，原始输入不会自动留下。
  要「保留原输入 + 追加新字段」，用下一课的 RunnablePassthrough.assign〔04/04〕。
"""
