"""
【04 LCEL与Runnable / 04】RunnablePassthrough 与 .assign —— 透传与追加字段
======================================================================
〔04/03〕的 RunnableParallel 会「重构」dict：只留你声明的 key，原输入丢了。
但很多时候你想「保留原输入，同时再加几个新字段」。这就是本课两个工具。

新概念（只有这一个，一体两面）：
  RunnablePassthrough()            —— 原样透传输入，什么都不改
  RunnablePassthrough.assign(k=…)  —— 在原 dict 基础上「追加/覆盖」字段，原有 key 全保留

和 RunnableParallel 的关键区别（最容易混的点）：
  RunnableParallel → 完全重构 dict，原 key 不保留
  .assign          → 在原 dict 上加字段，原 key 全保留
这个区别在 RAG 链里至关重要：你要保留 question，同时追加 context。
"""

import os

from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

load_dotenv()


# ── 1. Passthrough：原样透传 ───────────────────────────────────────────────
print("=== RunnablePassthrough：原样透传 ===")
print(RunnablePassthrough().invoke({"x": 1}))     # {'x': 1}，原封不动


# ── 2. .assign：保留原 dict，追加新字段 ────────────────────────────────────
print("\n=== .assign：原 key 保留 + 追加新字段 ===")
assign_chain = RunnablePassthrough.assign(
    upper=lambda d: d["text"].upper(),       # 新字段：大写
    length=lambda d: len(d["text"]),         # 新字段：长度
)
print(assign_chain.invoke({"text": "hello"}))
# {'text': 'hello', 'upper': 'HELLO', 'length': 5}  ← text 还在！


# ── 3. 对比 RunnableParallel：原 key 丢失 ──────────────────────────────────
print("\n=== 对比 RunnableParallel：原 text 不见了 ===")
parallel_chain = RunnableParallel({
    "upper": lambda d: d["text"].upper(),
})
print(parallel_chain.invoke({"text": "hello"}))
# {'upper': 'HELLO'}  ← 只剩声明的 key，text 没了


# ── 4. 经典组合：透传问题 + 追加「检索结果」（RAG 雏形）─────────────────────
print("\n=== RAG 雏形：question 透传，context 追加 ===")
rag_input = RunnableParallel({
    "question": RunnablePassthrough(),                          # 原样保留问题
    "context": RunnableLambda(lambda q: f"[关于「{q}」检索到的资料……]"),  # 追加检索结果
})
print(rag_input.invoke("什么是向量数据库"))


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  原始 dict {"text": "hello"}

   .assign(upper=..., length=...)            RunnableParallel({"upper": ...})
        │ 保留原 key + 追加                       │ 只留声明的 key
        ▼                                         ▼
  {"text", "upper", "length"}              {"upper"}

★ 核心规律：
  要「在原数据上加工、保留原字段」→ RunnablePassthrough.assign
  要「彻底重构出一个新 dict」      → RunnableParallel
  要「原样把输入往下传」          → RunnablePassthrough()

  〔07/05〕的 RAG 链就靠这套：用 Parallel/assign 同时备好 context 和 question，
  再喂给 prompt——这是 LCEL 里最高频的组合，务必分清三者区别。
"""
