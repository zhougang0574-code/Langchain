"""
【01 基础 / 03】init_chat_model —— 统一入口，一行换 provider
==========================================================
〔01/01〕〔01/02〕都用 ChatOpenAI 这个「具体类」。但真实项目里你可能今天用
qwen-plus、明天想试 deepseek、本地调试想用 ollama——如果每换一个就改 import 和类名，
很麻烦。

新概念（只有这一个）：
  init_chat_model(model=..., model_provider=...) —— LangChain 的「模型工厂」。
  不用记每个厂商对应哪个类，用同一个函数 + 字符串参数就能建出模型对象。
  返回的对象和 ChatOpenAI 实例一样，照样 .invoke()，后面所有链都通用。

为什么有用：
  把「用哪个模型」从代码里抽成配置。换模型只改字符串，链路代码一行不动。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


# ── 1. 用 init_chat_model 建模型（等价于 ChatOpenAI，但更通用）─────────────
# model_provider="openai"：因为百炼走 OpenAI 兼容协议，所以 provider 填 openai。
# qwen-plus 不是 OpenAI 官方模型，LangChain 无法自动推断 provider，必须显式指定，
# 否则会报 "Unable to infer model provider"。
llm = init_chat_model(
    model=os.getenv("MODEL"),       # qwen-plus
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


# ── 2. 换个模型：只改 model 字符串，其余代码完全不变 ───────────────────────
# 百炼上同样兼容 deepseek-v3，切换成本就是改一个字符串
llm_deepseek = init_chat_model(
    model="deepseek-v3",
    model_provider="openai",
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


if __name__ == "__main__":
    q = "你是谁？用一句话回答"
    print("=== qwen-plus ===")
    print(llm.invoke(q).content)

    print("\n=== deepseek-v3（同一套代码，只换了 model 字符串）===")
    print(llm_deepseek.invoke(q).content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
ChatOpenAI vs init_chat_model：

  ChatOpenAI(...)                      ← 「具体类」，明确就是 OpenAI 协议
  init_chat_model(model, provider=...) ← 「工厂函数」，靠字符串选 provider

  两者返回的对象用法完全一样（都实现了 Runnable 接口）。
  本课程后面为聚焦讲解，多数仍直接写 ChatOpenAI；但你完全可以用 init_chat_model 平替。

★ 核心规律：
  接百炼这类「OpenAI 兼容」服务，provider 一律填 "openai" + base_url 指向其兼容接口。
  非 OpenAI 官方模型名（qwen/deepseek/...）必须显式传 model_provider，否则无法自动推断。

  init_chat_model 的价值：把「选哪个模型」变成配置项，方便多 provider 切换 / A-B 对比。
"""
