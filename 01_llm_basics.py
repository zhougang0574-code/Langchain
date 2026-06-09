"""
第1课：LLM 基础调用 & Messages 消息类型

学习要点：
1. ChatOpenAI 的初始化方式（兼容 OpenAI 格式接入阿里百炼）
2. 三种消息类型：SystemMessage / HumanMessage / AIMessage
3. 四种调用方式：invoke / stream / batch / ainvoke
4. 返回值 AIMessage 的结构：content / response_metadata / usage_metadata
5. StrOutputParser 的作用：从 AIMessage 中提取纯文本
6. 为什么 LangChain 把 LLM 封装成 Runnable（统一接口，支持链式调用）
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ChatOpenAI 兼容 OpenAI API 格式，因此可以直接对接阿里百炼等第三方服务
# base_url 指向阿里百炼的 compatible-mode 接口
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0.7,  # 控制随机性，0=确定性最强，1=最随机
)


# ─────────────────────────────────────────────
# 1. 消息类型
# ─────────────────────────────────────────────
# LangChain 用消息对象代替裸字符串，因为 Chat 模型本质是多轮对话
# System 设定角色/规则，Human 是用户输入，AI 是模型回复（用于历史记录）
messages = [
    SystemMessage(content="你是一个简洁的助手，每次回复不超过两句话。"),
    HumanMessage(content="什么是 LangChain？"),
]


# ─────────────────────────────────────────────
# 2. invoke —— 同步调用，等待完整响应后返回
# ─────────────────────────────────────────────
def demo_invoke():
    response = llm.invoke(messages)
    # response 是 AIMessage 对象，不是字符串
    print("=== invoke ===")
    print(type(response))               # <class 'langchain_core.messages.ai.AIMessage'>
    print("content:", response.content)
    # usage_metadata 记录 token 消耗，调试成本时有用
    print("tokens:", response.usage_metadata)


# ─────────────────────────────────────────────
# 3. stream —— 流式输出，逐 chunk 返回，适合前端打字机效果
# ─────────────────────────────────────────────
def demo_stream():
    print("\n=== stream ===")
    for chunk in llm.stream(messages):
        # chunk 也是 AIMessage，但 content 是片段
        print(chunk.content, end="", flush=True)
    print()


# ─────────────────────────────────────────────
# 4. batch —— 批量调用，内部并发执行，比循环 invoke 更快
# ─────────────────────────────────────────────
def demo_batch():
    print("\n=== batch ===")
    questions = [
        [HumanMessage(content="1+1=?")],
        [HumanMessage(content="Python 是什么？")],
    ]
    # batch 接受 list[input]，返回 list[AIMessage]
    responses = llm.batch(questions)
    for r in responses:
        print(r.content)


# ─────────────────────────────────────────────
# 5. ainvoke —— 异步调用，用于 async 环境（FastAPI、异步爬虫等）
# ─────────────────────────────────────────────
async def demo_ainvoke():
    print("\n=== ainvoke ===")
    response = await llm.ainvoke(messages)
    print(response.content)


# ─────────────────────────────────────────────
# 6. StrOutputParser —— 从 AIMessage 提取 content 字符串
# ─────────────────────────────────────────────
def demo_parser():
    print("\n=== StrOutputParser ===")
    parser = StrOutputParser()
    response = llm.invoke(messages)
    # parser.invoke 接收 AIMessage，返回纯字符串
    # 这是 LCEL 链式调用的基础：llm | parser
    text = parser.invoke(response)
    print(type(text), repr(text[:30]))


if __name__ == "__main__":
    demo_invoke()
    demo_stream()
    demo_batch()
    asyncio.run(demo_ainvoke())
    demo_parser()


"""
执行流程图：

用户输入 (str / Message)
        │
        ▼
  [messages 列表]
  SystemMessage("角色设定")
  HumanMessage("用户问题")
        │
        ├─── invoke()   ──→ 等待完整响应 ──→ AIMessage
        ├─── stream()   ──→ 逐 chunk 返回 ──→ AIMessage(片段) × N
        ├─── batch()    ──→ 并发多路请求 ──→ [AIMessage, AIMessage, ...]
        └─── ainvoke()  ──→ 异步等待     ──→ AIMessage
                                │
                                ▼
                        StrOutputParser
                                │
                                ▼
                           str (纯文本)


核心知识点 ★：

★ ChatOpenAI 实现了 Runnable 接口，因此支持 invoke/stream/batch/ainvoke 四种统一调用方式
★ 消息类型决定角色：System=规则，Human=用户，AI=历史回复（不是当前请求）
★ invoke 返回 AIMessage 对象，.content 才是文本内容
★ stream 适合流式 UI；batch 适合批量任务；ainvoke 适合异步服务
★ StrOutputParser 是 LCEL 链的第一个常用组件：llm | StrOutputParser()
★ temperature=0 → 确定性输出，适合结构化任务；temperature>0 → 更有创意
"""
