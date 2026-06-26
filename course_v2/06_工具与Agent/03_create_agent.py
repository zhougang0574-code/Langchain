"""
【06 工具与Agent / 03】create_agent —— 一行建好 ReAct Agent
=========================================================
〔06/02〕手写了「执行工具 → 回灌 → 再调用」的循环。这套循环非常通用，
LangChain 把它封装成了开箱即用的 Agent，不用你自己写循环。

新概念（只有这一个）：
  create_agent(model, tools, system_prompt) —— 直接得到一个会「自己跑工具循环」的 Agent。
    用法：agent.invoke({"messages": [{"role": "user", "content": "..."}]})
    返回的 result["messages"] 里能看到完整的推理-行动轨迹。

它和上一课的关系：
  create_agent 内部做的就是〔06/02〕那套循环（底层用 LangGraph 实现），
  你手写过一遍，现在用现成的，知其所以然。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# ── 1. 定义工具（和前两课一样）─────────────────────────────────────────────
@tool
def add(a: int, b: int) -> int:
    """计算两个整数相加。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数相乘。"""
    return a * b


# ── 2. 一行建 Agent ────────────────────────────────────────────────────────
agent = create_agent(
    llm,
    tools=[add, multiply],
    system_prompt="你是一个计算助手，需要计算时调用工具，不要自己心算。",
)


if __name__ == "__main__":
    # 输入是 messages 列表（OpenAI 风格的 role/content 字典也可以）
    result = agent.invoke({
        "messages": [{"role": "user", "content": "先算 12 加 8，再把结果乘以 5"}]
    })

    # result["messages"] 是完整轨迹：Human → AI(调add) → Tool → AI(调multiply) → Tool → AI(答)
    print("=== 完整推理-行动轨迹 ===")
    for m in result["messages"]:
        name = m.__class__.__name__
        if getattr(m, "tool_calls", None):
            print(f"[{name}] 调用工具：{[(c['name'], c['args']) for c in m.tool_calls]}")
        else:
            print(f"[{name}] {m.content}")

    print("\n=== 最终回答 ===")
    print(result["messages"][-1].content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  agent = create_agent(model, tools, system_prompt)
  agent.invoke({"messages": [user]})
        │  内部自动跑〔06/02〕那套「调用→执行→回灌」循环
        ▼
  result["messages"] = [Human, AI(调工具), Tool结果, AI(调工具), Tool结果, AI(最终答)]

★ 核心规律：
  create_agent 把工具循环封装成一行：传 model + tools + system_prompt 即得 ReAct Agent。
  输入用 {"messages": [...]}，输出 result["messages"] 是完整轨迹，最后一条是最终答案。

  它底层基于 LangGraph（你的另一个课程）。当 Agent 逻辑变复杂（多分支、要持久化、
  人工干预）时，就该直接用 LangGraph 自己编排——create_agent 是其最常用的预制件。
"""
