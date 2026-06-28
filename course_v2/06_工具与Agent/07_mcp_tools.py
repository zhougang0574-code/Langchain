"""
【06 工具与Agent / 07】MCP 工具 —— 接入外部标准协议的工具〔进阶〕
================================================================
前面所有工具都是本进程里手写的 @tool〔06/01〕。真实世界里，越来越多工具/数据源
以 MCP(Model Context Protocol，Anthropic 提出的开放标准)的形式作为独立服务发布。
本课用 langchain-mcp-adapters 把一个 MCP server 暴露的工具转成 LangChain 工具，
照常喂给 create_agent〔06/03〕——你已会的 Agent 用法一行都不用改。

新概念（只有这一个）：
  MCP 工具接入 —— MultiServerMCPClient 连接 MCP server，await get_tools() 取回工具列表，
  适配成标准 LangChain tools 后，create_agent 照常使用。
  关键认知：MCP 是「工具接入标准」、LangChain 是「应用框架」，两个层面，靠 adapter 桥接。

为什么需要：
  工具一旦标准化为 MCP，任何支持 MCP 的客户端(Claude Desktop、Claude Code、本课的 Agent)
  都能即插即用，不必为每个框架重写一遍工具——这正是「协议化」的价值。

依赖（可选，未装会有友好提示，不影响其它课）：pip install langchain-mcp-adapters mcp
配套的 MCP server 是同目录的 _mcp_math_server.py，会被自动以 stdio 子进程拉起。
"""

import os
import sys
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

# MCP 适配器是可选依赖：没装就只给安装提示，保证本文件能被其它课无痛跳过
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    _MCP_OK = True
except ImportError:
    _MCP_OK = False

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)

# 配套 server 的绝对路径；用当前解释器(sys.executable)把它当 stdio 子进程拉起
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mcp_math_server.py")


async def main():
    # 1. 连接 MCP server（可同时连多个，这里只连一个名为 math 的）
    client = MultiServerMCPClient({
        "math": {"command": sys.executable, "args": [SERVER], "transport": "stdio"},
    })

    # 2. 取回工具——返回的已经是标准 LangChain 工具，无需再包装
    tools = await client.get_tools()
    print("从 MCP server 取回的工具：", [t.name for t in tools])

    # 3. 照常建 Agent（和〔06/03〕完全一样，工具来源不同而已）
    agent = create_agent(llm, tools=tools, system_prompt="需要计算就调工具，不要心算。")
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "先算 7 加 8，再把结果乘以 3"}]
    })

    print("\n=== 最终回答 ===")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    if not _MCP_OK:
        print("未安装 MCP 适配器。请先运行：pip install langchain-mcp-adapters mcp")
    else:
        asyncio.run(main())


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  _mcp_math_server.py（独立进程，暴露 add/multiply）
        ▲  stdio
        │
  MultiServerMCPClient({"math": {stdio 拉起上面的 server}})
        │  await client.get_tools()
        ▼
  tools（已是标准 LangChain 工具）──► create_agent(llm, tools)〔06/03〕
        │  agent.ainvoke({"messages":[...]})
        ▼
  Agent 像用本地 @tool 一样调用这些「来自 MCP 的」工具，给出最终回答

★ 核心规律：
  MCP 接入 = 「连 server → get_tools() → 当普通 tools 喂给 create_agent」。
  Agent 侧用法零改动；变的只是工具的来源——从本进程的 @tool 变成外部标准服务。

  记住分层：MCP 解决「工具怎么标准化接入」，LangChain 解决「怎么编排模型与工具」，
  adapter 是两者之间的桥。理解这条边界，比记住具体 API 更重要。
"""
