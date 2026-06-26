"""
【08 流式与回调 / 03】Callbacks —— 用钩子监控链的运行（耗时 / token / 调用）〔进阶〕
==========================================================================
〔08/02〕的 astream_events 是「拉」模型——你写 async for 主动取事件。回调是「推」
模型——你注册一个处理器，LangChain 在固定时机（模型开始/结束、出错等）自动回调它。
适合做监控、日志、统计成本，且不需要改链本身的代码。

新概念（只有这一个）：
  BaseCallbackHandler —— 继承它，重写若干 on_xxx 钩子方法。
    on_llm_start / on_llm_end / on_llm_error 等会在对应时机被自动调用。
  用法：invoke 时通过 config={"callbacks": [handler]} 挂上，或建模型时传 callbacks。

用途：统一统计 token 成本、记录每次调用耗时、把异常上报监控——一处编写，处处生效。
"""

import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()


# ── 1. 自定义回调：统计耗时和 token ────────────────────────────────────────
class StatsHandler(BaseCallbackHandler):
    def __init__(self):
        self.t0 = None
        self.total_tokens = 0
        self.calls = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        # 模型开始调用时记下起点
        self.t0 = time.time()

    def on_llm_end(self, response, **kwargs):
        # 模型返回时统计耗时与 token
        self.calls += 1
        elapsed = time.time() - self.t0
        usage = (response.llm_output or {}).get("token_usage", {})
        tokens = usage.get("total_tokens", 0)
        self.total_tokens += tokens
        print(f"  [回调] 第{self.calls}次调用：耗时 {elapsed:.2f}s，token {tokens}")


# ── 2. 挂上回调 ────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"), model=os.getenv("MODEL"),
)
stats = StatsHandler()


if __name__ == "__main__":
    print("=== 通过 config 挂回调，自动统计每次调用 ===")
    for q in ["1+1等于几", "法国首都是哪", "用一句话介绍 Python"]:
        ans = llm.invoke(q, config={"callbacks": [stats]})
        print(f"问：{q} → 答：{ans.content[:20]}...")

    print(f"\n=== 汇总：共 {stats.calls} 次调用，累计 {stats.total_tokens} tokens ===")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  invoke(config={"callbacks": [handler]})
        │  运行到固定时机自动「推」给你的钩子
        ▼
  on_llm_start  → 记起点
  on_llm_end    → 统计耗时/token
  on_llm_error  → 上报异常
  （还有 on_chain_start/end、on_tool_start/end 等更多钩子）

★ 核心规律：
  回调是「推」式可观测：注册一个 Handler，LangChain 在各阶段自动回调它，
  不用改链代码就能加监控/日志/成本统计，一处定义、随 config 处处挂载。

  对照：astream_events（〔08/02〕）是「拉」式、按需取事件，适合做实时 UI；
  callbacks 是「推」式、被动触发，适合做后台监控统计。两者按需选用。
"""
