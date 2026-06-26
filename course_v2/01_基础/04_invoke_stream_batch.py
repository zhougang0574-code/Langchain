"""
【01 基础 / 04】四种调用方式 —— invoke / stream / batch / ainvoke
================================================================
前几课只用了 invoke（同步、等完整结果）。但模型对象其实统一支持四种调用方式，
对应四种不同场景。

新概念（只有这一个）：
  Runnable 接口的四个方法
    invoke   —— 同步，等完整响应（最常用）
    stream   —— 流式，逐 chunk 返回（前端打字机效果）
    batch    —— 批量，内部并发跑多条输入（比 for 循环 invoke 快）
    ainvoke  —— 异步，用于 async 环境（FastAPI、异步爬虫）

为什么重要：
  ChatOpenAI 实现了 Runnable 接口。LangChain 里「一切皆 Runnable」——
  prompt、parser、检索器、整条链，全都有这四个方法，用法完全一致。
  现在在模型上学会它，后面整条链都是同一套调用方式。
"""

import os
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0.7,
)

messages = [HumanMessage(content="用一句话介绍 Python")]


# ── 1. invoke：等完整结果 ──────────────────────────────────────────────────
def demo_invoke():
    print("=== invoke（同步，等完整）===")
    print(llm.invoke(messages).content)


# ── 2. stream：逐片段返回 ──────────────────────────────────────────────────
def demo_stream():
    print("\n=== stream（流式，逐 chunk）===")
    for chunk in llm.stream(messages):
        # 每个 chunk 也是消息对象，.content 是这一小段文本
        print(chunk.content, end="", flush=True)
    print()


# ── 3. batch：并发跑多条输入 ───────────────────────────────────────────────
def demo_batch():
    print("\n=== batch（批量，内部并发）===")
    questions = [
        [HumanMessage(content="1+1=?")],
        [HumanMessage(content="法国的首都是？")],
    ]
    # 传 list[输入]，返回 list[AIMessage]，顺序与输入一一对应
    for r in llm.batch(questions):
        print("-", r.content)


# ── 4. ainvoke：异步 ───────────────────────────────────────────────────────
async def demo_ainvoke():
    print("\n=== ainvoke（异步）===")
    response = await llm.ainvoke(messages)
    print(response.content)


if __name__ == "__main__":
    demo_invoke()
    demo_stream()
    demo_batch()
    asyncio.run(demo_ainvoke())   # async 方法要用 asyncio.run 驱动


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
                       ┌─ invoke()  → 等完整 ───────────→ AIMessage
  输入(messages) ──────┼─ stream()  → 逐 chunk ─────────→ AIMessage 片段 × N
                       ├─ batch()   → 并发多路 ──────────→ [AIMessage, ...]
                       └─ ainvoke() → 异步等待 ──────────→ AIMessage

★ 核心规律：
  这四个方法不是模型独有的——它们是 Runnable 接口的统一方法。
  〔04 LCEL〕里 prompt|model|parser 拼成的整条链，照样有 invoke/stream/batch/ainvoke，
  调用方式和这里一模一样。这就是 LangChain「组件可任意拼接」的根基。

  选哪个：默认 invoke；要打字机效果用 stream（〔08 流式〕细讲）；
  批量任务用 batch；异步服务（FastAPI）用 a 开头的版本。
"""
