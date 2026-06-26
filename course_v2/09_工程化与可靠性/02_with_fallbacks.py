"""
【09 工程化与可靠性 / 02】with_fallbacks —— 主方案失败自动降级到备用
==============================================================
〔09/01〕的重试解决「再试一次就好」的偶发失败。但若主模型彻底不可用（下线、
配额耗尽、网关故障），重试无意义——需要切换到另一个模型/链。

新概念（只有这一个）：
  runnable.with_fallbacks([备用1, 备用2, ...]) —— 主方案失败时，按顺序尝试备用方案，
  直到某个成功；对调用方透明（它感知不到中间的失败与切换）。

典型用法：
  主用便宜模型、失败降级到更稳的贵模型；或主供应商挂了切到备用供应商。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# ── 1. 一个「一定会失败」的主模型（用不存在的模型名制造故障）───────────────
broken = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
    model="not-a-real-model-xxx",
)

# ── 2. 一个正常的备用模型 ──────────────────────────────────────────────────
good = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"), model=os.getenv("MODEL"),
)

# ── 3. 给主模型挂上备用方案 ────────────────────────────────────────────────
llm_with_fallback = broken.with_fallbacks([good])


if __name__ == "__main__":
    print("=== with_fallbacks：主模型挂了自动切备用 ===")
    # 主模型(not-a-real-model)调用会失败，自动切到备用(qwen-plus)，调用方拿到正常结果
    result = llm_with_fallback.invoke("用一句话介绍 LangChain")
    print("主模型失败 → 自动降级到备用模型，结果：")
    print(result.content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  invoke ──► 主方案 ──► 失败
                         │
                         ▼  按顺序试备用列表
                    备用方案1 ──► 成功 ──► 返回（调用方感知不到中间失败）

★ 核心规律：
  with_fallbacks([...]) 解决「主方案彻底不可用」：传一串备用，按序尝试直到成功。
  和 with_retry 互补——重试扛「偶发抖动」，降级扛「整体故障」。

  两者可叠加：给链同时 with_retry（先就地重试几次）再 with_fallbacks（还不行就换备用），
  配合〔09/03〕缓存，构成一套基础的可靠性组合拳。
"""
