"""
【04 LCEL与Runnable / 02】RunnableLambda —— 把任意函数接进链
=========================================================
〔04/01〕的链全是 LangChain 内置组件。但真实链里常需要插一段「自定义逻辑」：
预处理输入、改写模型输出、做个格式转换。这些是普通 Python 函数——
怎么让一个普通函数也能用 | 接进链？

新概念（只有这一个）：
  RunnableLambda(函数) —— 把普通函数包装成 Runnable，就能用 | 接进链。
    chain = prompt | llm | parser | RunnableLambda(我的函数)
  （在 | 链里直接写裸函数，LangChain 多数情况下也会自动帮你包成 RunnableLambda。）

附带一个常用小工具：
  itemgetter("key") —— 从输入 dict 里取一个字段，等价于 lambda x: x["key"]，但更简洁。
"""

import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)
parser = StrOutputParser()


# ── 1. 在链尾插一段自定义后处理 ────────────────────────────────────────────
def add_signature(text: str) -> str:
    return text + "\n\n—— 由 AI 助手生成 🤖"


prompt = ChatPromptTemplate.from_template("用一句话解释「{topic}」")
chain = prompt | llm | parser | RunnableLambda(add_signature)


# ── 2. itemgetter：从 dict 里抽字段，串进链 ────────────────────────────────
# 场景：输入是 dict，但下一步只需要其中一个字段
extract_chain = (
    RunnableLambda(lambda x: {"text": x["text"], "upper": x["text"].upper()})
    | itemgetter("upper")                       # 只取 upper 字段往下传
    | RunnableLambda(lambda s: f"大写结果：{s}")
)


if __name__ == "__main__":
    print("=== RunnableLambda 做后处理 ===")
    print(chain.invoke({"topic": "区块链"}))

    print("\n=== itemgetter 抽字段 ===")
    print(extract_chain.invoke({"text": "hello lcel"}))


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  prompt | llm | parser | RunnableLambda(add_signature)
                              │
                              ▼
                    自定义函数也成了链的一环

★ 核心规律：
  RunnableLambda 是「普通函数 ↔ 链」的转接头：任何 Python 函数都能接进 LCEL。
  函数的入参 = 上一步的输出，返回值 = 传给下一步的输入。

  itemgetter("k") 比 lambda x: x["k"] 更简洁，且对 LCEL 的类型推断更友好，
  在「从 dict 里挑字段」时很常用（〔07 RAG〕的链里会反复见到）。
"""
