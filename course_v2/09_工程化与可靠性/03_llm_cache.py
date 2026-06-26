"""
【09 工程化与可靠性 / 03】set_llm_cache —— 相同输入命中缓存，省钱又省时
================================================================
同一个问题被问很多遍（FAQ、重复请求、调试时反复跑），每次都真调一次 LLM
既花钱又慢。缓存让「相同输入」第二次起直接返回上次的结果，不再调用模型。

新概念（只有这一个）：
  set_llm_cache(缓存对象) —— 全局开启 LLM 缓存。
    InMemoryCache  —— 存进程内存，重启即丢，适合开发调试
    （生产要重启后仍有效，换成 SQLiteCache 等基于文件/数据库的实现，接口一致）
  开启后，相同输入的第二次调用几乎是 0 延迟，且不消耗 token。

注意：缓存按「输入完全相同」命中。temperature>0 时同一问题本应每次不同，
用缓存会让它每次返回同一个答案——这点要清楚。
"""

import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

load_dotenv()

# ── 1. 全局开启内存缓存 ────────────────────────────────────────────────────
set_llm_cache(InMemoryCache())

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"), temperature=0,
)


if __name__ == "__main__":
    q = "用一句话解释什么是斐波那契数列"

    t0 = time.time()
    r1 = llm.invoke(q)          # 第一次：真正调用 LLM
    t1 = time.time()
    r2 = llm.invoke(q)          # 第二次：完全相同输入 → 命中缓存，不调用 LLM
    t2 = time.time()

    print("=== set_llm_cache 效果 ===")
    print(f"第一次耗时：{t1 - t0:.2f}s（真正调用了 LLM）")
    print(f"第二次耗时：{t2 - t1:.4f}s（命中缓存，几乎 0 延迟、0 token）")
    print(f"两次结果完全一致：{r1.content == r2.content}")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  invoke("问题A") ──► 缓存没有 ──► 真调 LLM ──► 存入缓存 ──► 返回
  invoke("问题A") ──► 缓存命中 ──► 直接返回缓存结果（不调 LLM、不花 token）

★ 核心规律：
  set_llm_cache 是性价比极高的优化：相同输入不重复花钱。InMemoryCache 适合开发
  （重启即丢）；生产要持久化就换 SQLiteCache 等，接口不变。

  代价/注意：命中条件是「输入完全相同」；对 temperature>0 的创意场景，缓存会抹掉随机性
  （每次返回同一答案）。按场景决定要不要开。
"""
