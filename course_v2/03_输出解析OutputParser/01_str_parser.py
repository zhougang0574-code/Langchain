"""
【03 输出解析 / 01】StrOutputParser —— 从 AIMessage 取出纯文本
===========================================================
〔01/01〕里我们发现：llm.invoke() 返回的是 AIMessage 对象，要 .content 才是文本。
每次都手动 .content 很烦，而且在链里（〔04 LCEL〕）需要一个「能接在模型后面、
自动取文本」的标准组件。

新概念（只有这一个）：
  StrOutputParser —— 输入一个 AIMessage，输出它的 .content 字符串。
  它是最简单的「输出解析器」，也是 LCEL 链里最常见的收尾组件：model | StrOutputParser()

「输出解析器（OutputParser）」是什么？
  模型永远只会吐文本。解析器负责把这坨文本「整理成你要的形态」——
  最简单的是取出纯字符串（本课），复杂的能解析成 JSON / Pydantic 对象（后面几课）。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)

parser = StrOutputParser()


if __name__ == "__main__":
    response = llm.invoke("用一句话介绍 LangChain")

    print("=== 不用 parser：拿到的是 AIMessage ===")
    print(type(response).__name__, "→ 需要手动 .content")

    print("\n=== 用 parser：直接拿到 str ===")
    text = parser.invoke(response)     # 输入 AIMessage，输出 str
    print(type(text).__name__, "→", text)

    # 预告：下一域里 model | parser 就是把这两步串成一条链，自动完成
    print("\n=== 预告：LCEL 串联（〔04/01〕细讲）===")
    chain = llm | parser
    print(chain.invoke("用一句话介绍 Python"))


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  AIMessage(content="...")
        │  StrOutputParser().invoke()
        ▼
  "..."  (str)

★ 核心规律：
  OutputParser 是 LangChain 流水线的「最后一段」：把模型吐出的文本整理成目标形态。
  StrOutputParser 是其中最简单的一个——只取 .content。

  它最大的价值在链里：model | StrOutputParser() 让链的输出直接是字符串，
  而不是 AIMessage，下游处理更顺手。这是本课程后面几乎每条链的标准收尾。
"""
