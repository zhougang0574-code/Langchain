"""
【06 工具与Agent / 01】@tool 与 bind_tools —— 让模型「会调用函数」
==============================================================
到这里模型只会「说话」。但很多任务需要它「动手」：查天气、算数、查数据库。
模型本身做不到这些，但它能「决定该调用哪个函数、传什么参数」——这就是工具调用。

新概念（只有这一个）：
  ① @tool 装饰器：把一个普通 Python 函数变成「工具」（带名字、描述、参数 schema）
  ② llm.bind_tools([工具列表])：把工具「绑」给模型，让它知道有哪些工具可用
  绑定后，模型回复的 AIMessage 里会多出 .tool_calls：它「想调用哪个工具、传什么参数」。

关键认知：
  模型不会真的执行函数！它只输出「调用意图」（tool_calls）。真正执行是下一课的事。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# ── 1. 用 @tool 定义两个工具 ───────────────────────────────────────────────
# docstring 很重要：模型靠它理解「这个工具是干嘛的、什么时候该用」
@tool
def add(a: int, b: int) -> int:
    """计算两个整数相加的结果。"""
    return a + b


@tool
def get_weather(city: str) -> str:
    """查询某个城市的天气。"""
    # 演示用，返回假数据；真实场景这里会调天气 API
    return f"{city}今天晴，25℃。"


# ── 2. 把工具绑给模型 ──────────────────────────────────────────────────────
llm_with_tools = llm.bind_tools([add, get_weather])


if __name__ == "__main__":
    # 问一个需要算数的问题
    resp = llm_with_tools.invoke("帮我算一下 23 加 19 等于几")
    print("=== 模型的 tool_calls（调用意图，不是结果）===")
    print(resp.tool_calls)
    # [{'name': 'add', 'args': {'a': 23, 'b': 19}, 'id': '...'}]

    print("\ncontent（通常为空，因为它选择了调工具而不是直接答）：", repr(resp.content))

    # 换一个该用天气工具的问题
    resp2 = llm_with_tools.invoke("北京天气怎么样？")
    print("\n=== 换个问题，模型自动选了 get_weather ===")
    print(resp2.tool_calls)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  "算 23 加 19"
        │  llm.bind_tools([add, get_weather]).invoke()
        ▼
  AIMessage(
    content="",                                  ← 没直接回答
    tool_calls=[{name:"add", args:{a:23,b:19}}]  ← 而是给出「调用意图」
  )

★ 核心规律：
  @tool 把函数变工具（靠 docstring + 类型注解生成给模型看的说明）；
  bind_tools 让模型「知道有这些工具」，它据此输出 tool_calls（选哪个 + 传什么参数）。

  注意：模型只「决定」不「执行」。拿到 tool_calls 后由你的代码去真正运行函数、
  再把结果喂回模型——这套「执行 + 回灌」循环是下一课〔06/02〕的内容。
"""
