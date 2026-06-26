"""
【08 流式与回调 / 01】stream —— 逐字输出的打字机效果
================================================
〔01/04〕提过 stream 是四种调用之一，当时只在模型上用。本课讲它在「整条链」上的
用法和价值：长回答不必干等十几秒，而是像打字机一样一点点冒出来，体验好得多。

新概念（只有这一个）：
  chain.stream(输入) —— 返回一个生成器，逐 chunk 产出。
    对 prompt|model|parser 链，parser 是 StrOutputParser 时，每个 chunk 直接是字符串片段。
    异步版是 chain.astream(...)，配合 async for 用。

为什么放在链上讲：
  流式是「整条链」的能力，不只是模型的。只要链的最后一步能增量产出，stream 就能用。
"""

import os
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"), model=os.getenv("MODEL"),
)
chain = ChatPromptTemplate.from_template("分 3 点介绍 {topic}") | llm | StrOutputParser()


# ── 1. 同步流式：for 循环逐片段打印 ────────────────────────────────────────
def demo_stream():
    print("=== stream（同步打字机）===")
    for chunk in chain.stream({"topic": "RAG 的好处"}):
        print(chunk, end="", flush=True)   # end="" + flush 才有「逐字冒出」的效果
    print()


# ── 2. 异步流式：async for ─────────────────────────────────────────────────
async def demo_astream():
    print("\n=== astream（异步流式，用于 FastAPI 等）===")
    async for chunk in chain.astream({"topic": "向量数据库"}):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    demo_stream()
    asyncio.run(demo_astream())


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  invoke：  [........等待整段生成........] 一次性返回   ← 用户干等
  stream：  片 段 一 点 一 点 冒 出 来                  ← 边生成边显示

★ 核心规律：
  stream 让链增量产出，是「打字机 UI」的基础；astream 是其异步版（Web 服务首选）。
  对 prompt|model|parser 链，StrOutputParser 会把模型的增量 chunk 直接吐成字符串片段。

  但 stream 只给你「最终文本片段」。若想知道链「内部」发生了什么（哪一步开始了、
  工具被调用了、token 级事件），需要更细的 astream_events——下一课〔08/02〕。
"""
