"""
【09 工程化与可靠性 / 01】with_retry —— 调用失败自动重试
=====================================================
前面的链都假设「网络一定通、模型一定不出错」。但真实生产里网络会抖动、接口偶发超时。
每个调用点手写 try/except 重试循环既啰嗦又容易漏。

新概念（只有这一个）：
  runnable.with_retry(...) —— 返回一个「失败会自动重试」的新 Runnable。
    stop_after_attempt=N  最多尝试 N 次
  重试对调用方透明：中间失败被自动吸收，你不用写任何 try/except。

本课用一个「故意前两次失败、第三次成功」的函数模拟网络抖动，免得真把网络搞坏。
"""

from langchain_core.runnables import RunnableLambda

# ── 1. 模拟「偶发失败」的步骤 ──────────────────────────────────────────────
attempts = {"n": 0}


def flaky(x: str) -> str:
    attempts["n"] += 1
    if attempts["n"] < 3:
        # 前两次抛错，模拟网络抖动 / 接口偶发超时
        raise ValueError(f"第 {attempts['n']} 次调用失败（模拟抖动）")
    return f"第 {attempts['n']} 次成功，输入={x}"


# ── 2. 加上 with_retry，最多试 5 次 ────────────────────────────────────────
robust = RunnableLambda(flaky).with_retry(stop_after_attempt=5)


if __name__ == "__main__":
    print("=== with_retry：自动吸收中间失败 ===")
    result = robust.invoke("hello")            # 调用方完全不知道中间失败过两次
    print("结果：", result)
    print(f"实际尝试了 {attempts['n']} 次（前 2 次失败被自动重试，无需手写 try/except）")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  invoke ──► 失败 ──► 等一下重试 ──► 失败 ──► 重试 ──► 成功，返回
  （重试次数耗尽仍失败，才把异常抛给调用方）

★ 核心规律：
  with_retry() 让「偶发失败」对调用方透明，不必每个调用点手写重试循环。
  常用 stop_after_attempt 控制最多重试几次；它返回新 Runnable，可接进任何链。

  重试解决的是「同一个调用再试一次可能就好了」。但若是「这个模型彻底挂了」，
  重试多少次都没用——那要靠切换到备用模型，即下一课 with_fallbacks〔09/02〕。
"""
