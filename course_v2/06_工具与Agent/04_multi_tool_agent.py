"""
【06 工具与Agent / 04】多工具协作 —— Agent 自己决定调用顺序
========================================================
〔06/03〕的工具彼此独立。本课给一个更真实的例子：电商助手有「搜索商品」和
「查库存」两个工具，要回答「最受欢迎的无线耳机有货吗」必须先搜到商品 id、
再拿 id 去查库存——Agent 会自己安排这个顺序，不需要你写流程。

新概念（没有全新 API，是综合应用）：
  多个工具组合时，Agent 依据每个工具的 docstring 和「上一个工具的返回」，
  自主决定下一步调哪个、用什么参数。这就是 Agent 比固定链强的地方：动态编排。

本课重点：把工具写「好」——清晰的 docstring + 明确的返回，是 Agent 用对工具的前提。
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

# 模拟两张表：商品库（按热度）、库存库（按商品 id）
PRODUCTS = {
    "无线耳机": [
        {"id": "WH5", "name": "索尼 WH-1000XM5", "popularity": 95},
        {"id": "QC45", "name": "Bose QC45", "popularity": 88},
    ],
}
INVENTORY = {"WH5": 10, "QC45": 0}


# ── 1. 工具1：按品类搜商品，返回带 id 的列表 ───────────────────────────────
@tool
def search_products(category: str) -> str:
    """根据商品品类（如「无线耳机」）搜索商品，返回按热度排序的商品及其 id。"""
    items = PRODUCTS.get(category)
    if not items:
        return f"没有找到品类「{category}」的商品。"   # 兜底：永远有返回，别返回 None
    ranked = sorted(items, key=lambda x: x["popularity"], reverse=True)
    return "；".join(f"{p['name']}(id={p['id']}, 热度{p['popularity']})" for p in ranked)


# ── 2. 工具2：按商品 id 查库存 ─────────────────────────────────────────────
@tool
def check_inventory(product_id: str) -> str:
    """根据商品 id 查询库存数量。"""
    if product_id not in INVENTORY:
        return f"没有 id={product_id} 的库存记录。"
    n = INVENTORY[product_id]
    return f"id={product_id} 库存 {n} 件（{'有货' if n > 0 else '缺货'}）。"


agent = create_agent(
    llm,
    tools=[search_products, check_inventory],
    system_prompt="你是电商助手。需要先搜到商品 id，才能用 id 查库存。一步步来。",
)


if __name__ == "__main__":
    result = agent.invoke({
        "messages": [{"role": "user", "content": "最受欢迎的无线耳机有货吗？"}]
    })

    print("=== Agent 自主编排的调用轨迹 ===")
    for m in result["messages"]:
        if getattr(m, "tool_calls", None):
            print(f"[调用] {[(c['name'], c['args']) for c in m.tool_calls]}")
        elif m.__class__.__name__ == "ToolMessage":
            print(f"[工具返回] {m.content}")

    print("\n=== 最终回答 ===")
    print(result["messages"][-1].content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  "最受欢迎的无线耳机有货吗？"
        │  Agent 推理：得先知道是哪个商品
        ▼
  调 search_products("无线耳机") → "索尼XM5(id=WH5,热度95); ..."
        │  Agent 推理：拿到 id=WH5，去查库存
        ▼
  调 check_inventory("WH5") → "库存10件，有货"
        │
        ▼
  最终回答："最受欢迎的是索尼 WH-1000XM5，有货（10 件）。"

★ 核心规律：
  多工具下 Agent 会用「上一个工具的输出」决定下一步——调用顺序由它自主编排，不用你写流程。
  这是 Agent 相对固定链的核心优势：动态、自适应。

  工程要点：① 每个工具 docstring 必须清楚（Agent 靠它选工具）；
  ② 工具任何情况都要有合法返回（别返回 None，否则回灌会出问题——
     这是 archive 里速通课 Agent 的一个真实 bug，本课已规避）。
"""
