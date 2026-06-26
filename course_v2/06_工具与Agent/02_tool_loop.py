"""
【06 工具与Agent / 02】手动执行工具循环 —— 把结果用 ToolMessage 回灌
================================================================
〔06/01〕拿到了模型的 tool_calls（调用意图），但还没真正执行。本课手动走完整套
「执行工具 → 把结果告诉模型 → 模型给出最终回答」的循环，看清 Agent 的内部机制。

新概念（只有这一个）：
  ToolMessage(content=工具结果, tool_call_id=对应的调用id)
    - 工具执行完，结果要用 ToolMessage 包起来，连同之前的消息一起再发给模型
    - tool_call_id 把「结果」和「当初哪次调用」对上号

完整一轮的消息流：
  Human → AI(tool_calls) → ToolMessage(结果) → AI(最终回答)
这正是下一课 create_agent〔06/03〕在内部自动做的事。本课手搓一遍才理解它。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


@tool
def add(a: int, b: int) -> int:
    """计算两个整数相加的结果。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数相乘的结果。"""
    return a * b


tools = [add, multiply]
tools_by_name = {t.name: t for t in tools}      # 按名字找工具，方便执行
llm_with_tools = llm.bind_tools(tools)


if __name__ == "__main__":
    # 一个需要两步计算的问题：(3+5) 然后 ×2
    messages = [HumanMessage(content="先算 3 加 5，再把结果乘以 2")]

    # 循环：只要模型还想调工具，就执行并回灌，直到它给出最终回答
    for step in range(5):                       # 设上限防止意外死循环
        ai = llm_with_tools.invoke(messages)
        messages.append(ai)

        if not ai.tool_calls:
            # 没有 tool_calls 了 = 模型给出了最终回答，结束
            print("\n=== 最终回答 ===")
            print(ai.content)
            break

        # 有 tool_calls：逐个真正执行，把结果作为 ToolMessage 追加回消息列表
        for call in ai.tool_calls:
            tool = tools_by_name[call["name"]]
            result = tool.invoke(call["args"])
            print(f"  [执行] {call['name']}({call['args']}) = {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  messages = [Human("先算3+5再×2")]
        │  llm_with_tools.invoke(messages)
        ▼
  AI(tool_calls=[add(3,5)])  ──► 执行 add=8 ──► 追加 ToolMessage("8")
        │  再 invoke（带上结果）
        ▼
  AI(tool_calls=[multiply(8,2)]) ──► 执行=16 ──► 追加 ToolMessage("16")
        │  再 invoke
        ▼
  AI(content="结果是16")  ← 没有 tool_calls 了，循环结束

★ 核心规律：
  工具循环 = 反复「invoke → 若有 tool_calls 就执行并用 ToolMessage 回灌 → 再 invoke」，
  直到模型不再要求调工具。tool_call_id 负责把结果和对应的调用配对。

  这套循环就是 ReAct（推理-行动）的骨架。手写它很繁琐，所以下一课用
  create_agent〔06/03〕一行搞定——但你已经知道它内部在转什么了。
"""
