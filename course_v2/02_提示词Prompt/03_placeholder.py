"""
【02 提示词Prompt / 03】MessagesPlaceholder —— 给模板留一个「插历史」的槽
======================================================================
〔02/02〕的模板里，system 和 human 的条数是写死的。但对话历史长度是动态的：
可能 0 轮，也可能 10 轮。你没法在模板里写死「这里放 N 条消息」。

新概念（只有这一个）：
  MessagesPlaceholder(variable_name="history") —— 在模板中留一个「消息列表」占位槽。
  填值时传 history=[...一串消息...]，整段会被原样展开插入到这个位置。

为什么需要：
  这是「带记忆的对话」在 Prompt 层的接口。〔05 记忆Memory〕会自动往这个槽里
  灌入历史；本课先手动灌，看清它的作用。
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ── 1. 在 system 和当前问题之间，留一个 history 槽 ─────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁的助手。"),
    MessagesPlaceholder(variable_name="history"),   # ← 历史消息会插在这里
    ("human", "{input}"),                           # ← 当前这一轮的问题
])


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()
    llm = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=os.getenv("MODEL"),
    )

    # 手动准备一段历史（真实场景由 Memory 自动维护）
    history = [
        HumanMessage(content="我最喜欢的颜色是蓝色。"),
        AIMessage(content="好的，记住了，你喜欢蓝色。"),
    ]

    # 填值：history 槽收到列表，{input} 收到本轮问题
    filled = prompt.invoke({"history": history, "input": "我刚才说我喜欢什么颜色？"})
    print("=== 展开后的完整消息列表 ===")
    for m in filled.to_messages():
        print(f"  [{m.__class__.__name__}] {m.content}")

    print("\n=== 模型回答（能答对，说明历史被正确插入）===")
    print(llm.invoke(filled).content)

    print("\n=== 历史为空也能正常工作（传空列表即可）===")
    empty = prompt.invoke({"history": [], "input": "你好"})
    print(llm.invoke(empty).content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  ChatPromptTemplate:
    SystemMessage("你是一个简洁的助手。")
    MessagesPlaceholder("history")  ←── 填入 [Human("喜欢蓝色"), AI("记住了"), ...]
    HumanMessage("{input}")         ←── 填入本轮问题
        │  .invoke({"history": [...], "input": "..."})
        ▼
    [System, Human(历史), AI(历史), ..., Human(本轮)]   ← 一条完整对话

★ 核心规律：
  MessagesPlaceholder 是模板里「数量不定的一段消息」的占位符，
  variable_name 决定填值时用哪个 key 传这段历史。
  它是「带记忆对话」的标准结构：System → 历史槽 → 当前输入。

  〔05/02〕RunnableWithMessageHistory 会自动把历史塞进这个槽，
  你只管传 {input}，不用手动维护 history 列表。
"""
