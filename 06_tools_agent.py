"""
第6课：Tools & Agent

学习要点：
1. @tool 装饰器 —— 把普通函数变成 LangChain Tool
2. Tool 的三要素：name / description / function（description 决定 LLM 是否调用它）
3. bind_tools() —— 把工具绑定给 LLM，让 LLM 知道有哪些工具可用
4. tool_calls —— LLM 决定调用工具时，AIMessage 里携带的结构化调用信息
5. create_agent()（新版 LangChain 1.x）—— 底层由 LangGraph 驱动，替代旧版 AgentExecutor
6. Agent 的执行循环：推理 → 调用工具 → 观察 → 再推理

注意：LangChain 0.x 的 create_tool_calling_agent + AgentExecutor 在 1.x 中已移除，
      统一改为 create_agent()，底层是 LangGraph CompiledStateGraph。
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,  # Agent 任务用 0，让工具选择更稳定
)


# ─────────────────────────────────────────────
# 1. 用 @tool 定义工具
# ─────────────────────────────────────────────
# docstring 就是 description，LLM 读它来决定何时调用这个工具
# 参数类型注解会生成 JSON schema，告诉 LLM 应该传什么参数

@tool
def add(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数之积。"""
    return a * b

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。city 参数为城市名称，如'北京'、'上海'。"""
    weather_data = {
        "北京": "晴天，气温25°C",
        "上海": "多云，气温22°C",
        "广州": "阵雨，气温28°C",
    }
    return weather_data.get(city, f"暂无 {city} 的天气数据")

tools = [add, multiply, get_weather]

print("=== Tool 信息 ===")
for t in tools:
    print(f"名称: {t.name} | 描述: {t.description}")


# ─────────────────────────────────────────────
# 2. bind_tools()：让 LLM 知道工具存在
# ─────────────────────────────────────────────
llm_with_tools = llm.bind_tools(tools)

print("\n=== bind_tools 直接调用 ===")
msg = llm_with_tools.invoke([HumanMessage(content="35 乘以 7 等于多少？")])
print("content   :", msg.content)
# LLM 决定调用工具时 content 为空，tool_calls 有值
print("tool_calls:", msg.tool_calls)


# ─────────────────────────────────────────────
# 3. create_agent()（LangChain 1.x 新版 API）
# ─────────────────────────────────────────────
# 底层是 LangGraph CompiledStateGraph，替代旧版 AgentExecutor
# 返回的是图对象，用 .invoke() / .stream() 驱动执行
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个有用的助手，可以使用提供的工具来回答问题。",
)

print("\n=== Agent 执行（数学）===")
# 新 API：输入格式为 {"messages": [...]}
result = agent.invoke({
    "messages": [HumanMessage(content="先算 12 乘以 8，再加上 15，最终结果是多少？")]
})
# 最后一条消息是 AI 的最终回答
print("最终答案:", result["messages"][-1].content)

print("\n=== Agent 执行（天气）===")
result = agent.invoke({
    "messages": [HumanMessage(content="北京和上海今天天气怎么样？")]
})
print("最终答案:", result["messages"][-1].content)


# ─────────────────────────────────────────────
# 4. 查看 Agent 中间步骤（stream）
# ─────────────────────────────────────────────
print("\n=== Agent stream（逐步观察）===")
for step in agent.stream({
    "messages": [HumanMessage(content="3 加 4 的结果再乘以 2 是多少？")]
}):
    # 每个 step 是一个节点的输出 {"node_name": {"messages": [...]}}
    node_name = list(step.keys())[0]
    msgs = step[node_name].get("messages", [])
    for m in msgs:
        print(f"[{node_name}] {type(m).__name__}: {m.content or m.tool_calls}")


if __name__ == "__main__":
    pass


"""
执行流程图（ReAct 循环，LangGraph 驱动）：

用户问题 → {"messages": [HumanMessage(...)]}
         │
         ▼ LangGraph 节点循环
    ┌────────────────────────┐
    │  agent 节点（LLM 推理）│
    │  → 返回 tool_calls 或  │
    │    最终文本回答         │
    └─────────┬──────────────┘
              │ 有 tool_calls？
              │
       ┌──────┴──────┐
       │ Yes          │ No
       ▼              ▼
  tools 节点      END（输出最终答案）
  执行工具函数
  返回 ToolMessage
       │
       └──→ 回到 agent 节点继续推理

输出：{"messages": [HumanMessage, AIMessage(tool_calls), ToolMessage, AIMessage(最终答案)]}


⚠️  新旧版 API 对比：

旧版（LangChain 0.x）：
  create_tool_calling_agent(llm, tools, prompt)
  AgentExecutor(agent=agent, tools=tools, verbose=True)
  executor.invoke({"input": "..."})

新版（LangChain 1.x）：
  create_agent(model=llm, tools=tools, system_prompt="...")
  agent.invoke({"messages": [HumanMessage("...")]})


核心知识点 ★：

★ @tool 的 docstring = description，这是 LLM 决定调用哪个工具的依据，写清楚很重要
★ tool_calls 非空 → LLM 想调用工具；content 非空 → LLM 直接回答
★ LangChain 1.x 的 create_agent 底层是 LangGraph，输入输出格式是 {"messages": [...]}
★ 旧版 AgentExecutor 已移除，学新项目直接用 create_agent 或 LangGraph
★ temperature=0 让工具选择更确定，减少随机乱调工具的情况
★ stream() 可以看到每一步节点的执行结果，相当于旧版的 verbose=True
"""
