"""
第13课：工程化与可靠性 —— 让链从"能跑通"变成"扛得住"

为什么是这一课：前12课的链都是"理想情况"——网络一定通、模型一定不出错、
每次问同样的问题都重新调一次 API。真实生产环境里：网络会抖动、模型偶尔会超时报错、
同样的问题被问一百次没必要真调一百次 API（费钱又费时间）。这一课补上这些工程化能力。

学习要点：
1. .with_retry() —— 调用失败自动重试，不需要自己写 try/except 循环
2. .with_fallbacks() —— 主模型失败时自动切换到备用模型/备用链
3. set_llm_cache() —— 相同输入命中缓存，不重复调用 LLM，省钱又省时间
4. RunnableBranch —— 纯 LCEL 实现的轻量条件分支，不需要的话不用上升到 LangGraph

Python 小贴士（给新手）：
- 本课用一个"故意失败几次才成功"的函数来模拟网络抖动，这样不需要真的让网络出问题
  也能看到 retry 的效果；字典 attempt_counter 是用来"记住函数被调用了几次"的简单计数器
"""

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# ─────────────────────────────────────────────
# 1. .with_retry() —— 调用失败自动重试
# ─────────────────────────────────────────────
print("=== with_retry：自动重试 ===")
attempt_counter = {"n": 0}


def flaky_step(x: str) -> str:
    # 模拟"前两次失败，第三次才成功"，对应真实场景里的网络抖动/接口偶发超时
    attempt_counter["n"] += 1
    if attempt_counter["n"] < 3:
        raise ValueError(f"模拟第 {attempt_counter['n']} 次调用失败（网络抖动）")
    return f"第 {attempt_counter['n']} 次调用成功，输入是: {x}"


# stop_after_attempt=5：最多重试5次；不传的话默认也会重试，只是次数和等待策略不同
flaky_runnable = RunnableLambda(flaky_step).with_retry(stop_after_attempt=5)
result = flaky_runnable.invoke("hello")
print(f"结果: {result}")
print(f"总共尝试了 {attempt_counter['n']} 次才成功（中间失败的2次被自动重试吸收了，调用方完全不需要写 try/except）")


# ─────────────────────────────────────────────
# 2. .with_fallbacks() —— 主链失败时自动切换备用方案
# ─────────────────────────────────────────────
print("\n=== with_fallbacks：自动降级 ===")
# 故意用一个不存在的模型名，制造一个"一定会失败"的主模型
broken_llm = ChatOpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"), model="not-a-real-model")
# with_fallbacks 接收一个"备用方案列表"，主链失败时按顺序尝试，直到有一个成功
llm_with_fallback = broken_llm.with_fallbacks([llm])

result = llm_with_fallback.invoke("用一句话介绍 LangChain")
print(f"主模型(not-a-real-model)调用失败，自动切换到备用模型(qwen-plus)：\n{result.content}")


# ─────────────────────────────────────────────
# 3. set_llm_cache() —— 相同输入命中缓存
# ─────────────────────────────────────────────
print("\n=== set_llm_cache：缓存重复调用 ===")
# InMemoryCache：缓存存在进程内存里，进程一重启缓存就没了，适合开发调试
# 生产环境想要"重启后缓存依然有效"，可以换成 SQLiteCache（基于本地数据库文件持久化）
set_llm_cache(InMemoryCache())

question = "什么是斐波那契数列？用一句话回答"
t0 = time.time()
r1 = llm.invoke(question)
t1 = time.time()
r2 = llm.invoke(question)  # 完全相同的输入，应该直接命中缓存，不再真正调用 LLM
t2 = time.time()
print(f"第一次调用耗时 {t1 - t0:.2f}s（真正调用了 LLM）")
print(f"第二次调用耗时 {t2 - t1:.4f}s（命中缓存，几乎是0延迟）")
print(f"两次结果是否完全一致: {r1.content == r2.content}")


# ─────────────────────────────────────────────
# 4. RunnableBranch —— 纯 LCEL 的轻量条件分支
# ─────────────────────────────────────────────
# 第8/9课提到过"需要分支/循环就该上 LangGraph"，但如果只是"简单的 if-else 选择走哪条链"，
# 不涉及循环、不需要持久化状态，RunnableBranch 这种纯 LCEL 方案就足够了，不用上升到 LangGraph。
print("\n=== RunnableBranch：轻量条件分支 ===")


def is_math_question(x: dict) -> bool:
    return any(ch.isdigit() for ch in x["question"]) or "+" in x["question"] or "等于" in x["question"]


math_chain = RunnableLambda(lambda x: f"[数学问题处理] {x['question']}")
chat_chain = RunnableLambda(lambda x: f"[闲聊处理] {x['question']}")

# RunnableBranch((条件1, 分支1), (条件2, 分支2), ..., 默认分支)
# 按顺序检查每个条件，第一个返回 True 的分支会被执行；都不满足就走最后的默认分支
branch = RunnableBranch(
    (is_math_question, math_chain),
    chat_chain,  # 默认分支，没有配条件
)
print(branch.invoke({"question": "1+1等于几"}))
print(branch.invoke({"question": "你好，今天心情怎么样"}))


if __name__ == "__main__":
    pass


"""
执行流程图：

with_retry：
  调用 ──► 失败 ──► 等待一小段时间 ──► 重试 ──► 失败 ──► 重试 ──► 成功，返回结果
  （重试次数耗尽仍失败，才会把异常抛给调用方）

with_fallbacks：
  调用 ──► 主方案 ──► 失败
                       │
                       ▼
                  按顺序尝试备用方案列表中的下一个
                       │
                       ▼
                  备用方案成功 ──► 返回结果（调用方感知不到中间的失败）

set_llm_cache：
  调用("问题A") ──► 缓存里没有 ──► 真正调用LLM ──► 存入缓存 ──► 返回结果
  再次调用("问题A") ──► 缓存里有 ──► 直接返回缓存结果（不再调用LLM）

RunnableBranch：
  输入 ──► 条件1？ ──是──► 分支1
              │否
              ▼
           条件2？ ──是──► 分支2
              │否
              ▼
           默认分支


核心知识点 ★：

★ with_retry() 让"网络抖动/接口偶发失败"对调用方变得透明，不需要每个调用点都手写 try/except 循环。
  常用参数：stop_after_attempt（最多重试几次）。
★ with_fallbacks() 解决"主模型挂了怎么办"：传入一个备用方案列表，按顺序尝试，
  典型用法是"主用便宜模型，失败了降级到更贵但更稳的模型"，或者"主供应商挂了切到备用供应商"。
★ set_llm_cache() 是性价比极高的优化：相同输入不会重复花钱调用 LLM。
  InMemoryCache 适合开发调试（重启即丢）；生产环境要持久化就用 SQLiteCache 等基于文件/数据库的实现。
★ RunnableBranch 是 LCEL 内置的轻量分支方案：纯粹的"if-else 选择走哪条链"用它就够，
  不需要因为"有一个分支"就直接上 LangGraph——真正需要循环、需要在多步之间持久化状态时，
  才是 LangGraph 出场的时机（参考第8课关于二者分工的讨论）。
★ 这些工程化能力可以叠加使用：比如给一条 RAG 链同时加上 with_retry（扛网络抖动）、
  with_fallbacks（模型挂了降级）、外层套缓存（省重复调用），互不冲突。
"""
