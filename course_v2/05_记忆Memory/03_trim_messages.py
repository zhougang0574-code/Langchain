"""
【05 记忆Memory / 03】trim_messages —— 裁剪历史，控制 token〔进阶〕
=============================================================
〔05/02〕的历史会一直增长。但模型有上下文长度上限，而且 token 越多越慢越贵。
对话久了，必须「只保留最近一部分历史」——这就是裁剪。

新概念（只有这一个）：
  trim_messages(...) —— 按规则裁掉过长的历史，只留最近的若干条/若干 token。
    常用参数：
      max_tokens / token_counter  保留的预算上限与计数方式
      strategy="last"             保留「最近」的消息（最常用）
      include_system=True         始终保留开头的 System 设定

为什么重要：
  让「记忆」在长对话里可持续——既不超模型上限，又把钱花在最相关的近期上下文上。
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages

# ── 1. 造一段「很长」的历史 ────────────────────────────────────────────────
history = [
    SystemMessage(content="你是一个简洁的助手。"),
    HumanMessage(content="第1个问题：1+1=?"),
    AIMessage(content="2"),
    HumanMessage(content="第2个问题：2+2=?"),
    AIMessage(content="4"),
    HumanMessage(content="第3个问题：3+3=?"),
    AIMessage(content="6"),
    HumanMessage(content="第4个问题：4+4=?"),
    AIMessage(content="8"),
]


# ── 2. 按「消息条数」裁剪（用 len 当 token_counter 最直观）─────────────────
# token_counter=len → 把「条数」当预算；max_tokens=4 → 最多留 4 条
# strategy="last"   → 保留最近的；include_system=True → System 始终保留
trimmed = trim_messages(
    history,
    max_tokens=4,
    token_counter=len,
    strategy="last",
    include_system=True,
    start_on="human",      # 从 human 开始，保证裁剪后是合法的对话起点
)


if __name__ == "__main__":
    print(f"=== 原始历史 {len(history)} 条 ===")
    for m in history:
        print(f"  [{m.__class__.__name__}] {m.content}")

    print(f"\n=== 裁剪后 {len(trimmed)} 条（System 保留 + 最近几条）===")
    for m in trimmed:
        print(f"  [{m.__class__.__name__}] {m.content}")

    # 真实用法：在链里把历史先 trim 再喂给 prompt，控制每次调用的 token
    print("\n提示：实际项目里把 trim_messages 接在 history 和 prompt 之间，")
    print("      或在 RunnableWithMessageHistory 的链里先裁剪，避免历史无限膨胀。")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  完整历史（越来越长）
        │  trim_messages(max_tokens=预算, strategy="last", include_system=True)
        ▼
  [System] + [最近的若干条]   ← 旧的远期历史被丢掉

★ 核心规律：
  trim_messages 用「预算 + 策略」裁历史：strategy="last" 留最近、include_system 护住角色设定。
  token_counter 决定预算单位——用 len 按条数（演示直观），生产可传模型的 token 计数函数按真实 token 算。

  它和〔05/02〕配合：长对话里先裁剪再调用，让记忆「可持续」，不至于撑爆上下文或烧钱。
"""
