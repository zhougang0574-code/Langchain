"""
【08 流式与回调 / 02】astream_events —— 看清链「内部」每一步的事件流
==============================================================
〔08/01〕的 stream 只给最终文本片段。但复杂链（RAG、Agent）里你常想知道：
检索什么时候开始的？模型在逐 token 产出什么？某一步的输入输出是什么？
astream_events 把链运行中的「所有事件」按时间流式吐出来。

新概念（只有这一个）：
  async for event in chain.astream_events(输入)
    每个 event 是一个 dict，关键字段：
      event["event"]  事件类型，如 on_chat_model_stream（模型逐 token）、
                      on_retriever_start/end、on_chain_start/end 等
      event["data"]   该事件的数据（chunk、输入、输出……）
      event["name"]   产生事件的组件名

用途：做细粒度 UI（显示「正在检索…」「正在生成…」）、调试链内部、token 级处理。
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
chain = ChatPromptTemplate.from_template("用一句话介绍 {topic}") | llm | StrOutputParser()


async def main():
    print("=== 事件流（只挑几类关键事件打印）===")
    token_buf = []
    async for event in chain.astream_events({"topic": "LangChain"}):
        kind = event["event"]

        if kind == "on_chain_start" and event["name"] == "RunnableSequence":
            print("[链开始]")
        elif kind == "on_chat_model_stream":
            # 模型逐 token 产出：累计起来，体现「token 级」粒度
            token_buf.append(event["data"]["chunk"].content)
        elif kind == "on_chat_model_end":
            print("[模型逐 token 完成] 拼起来 =", "".join(token_buf))
        elif kind == "on_chain_end" and event["name"] == "RunnableSequence":
            print("[链结束] 最终输出 =", event["data"]["output"])


if __name__ == "__main__":
    asyncio.run(main())


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  链运行
    │ astream_events 把每一步都变成事件吐出来
    ▼
  on_chain_start          ← 链开始
  on_prompt_end           ← 模板填好
  on_chat_model_stream ×N ← 模型逐 token（每个 token 一个事件）
  on_chat_model_end       ← 模型生成完毕
  on_parser_end / on_chain_end ← 解析、链结束

★ 核心规律：
  stream（〔08/01〕）= 只要最终文本片段；astream_events = 要链内部全部事件。
  按 event["event"] 的类型分流处理，就能做「正在检索/正在生成」这类细粒度 UI，
  也能在 token 级别介入（如敏感词流式过滤）。

  这是「可观测性」的流式版；下一课〔08/03〕的回调（Callback）是另一种可观测手段——
  不靠 async 事件流，而是注册「钩子函数」在固定时机被调用。
"""
