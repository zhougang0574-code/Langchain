"""
【02 提示词Prompt / 02】ChatPromptTemplate —— 带角色的模板（日常主力）
==================================================================
〔02/01〕的 PromptTemplate 产出的是一坨纯文本。但聊天模型吃的是「消息列表」
（System/Human/AI，见〔01/02〕）。ChatPromptTemplate 就是「能填空的消息列表模板」。

新概念（只有这一个）：
  ChatPromptTemplate.from_messages([...]) —— 每个元素是一条「带占位符的消息」。
  填值后直接得到 messages 列表，可直接喂给模型。

本课额外展示「同一个模板的三种写法」，它们完全等价，挑顺手的用：
  ① ("system", "...")            元组写法（最简洁，最常用）
  ② {"role": "system", ...}      字典写法
  ③ SystemMessagePromptTemplate  消息模板类写法（最显式、最啰嗦）
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ── 1. 元组写法（推荐）──────────────────────────────────────────────────────
prompt_tuple = ChatPromptTemplate.from_messages([
    ("system", "你是一个{style}的翻译助手。"),
    ("human", "把这句话翻译成英文：{text}"),
])

# ── 2. 字典写法（等价）──────────────────────────────────────────────────────
prompt_dict = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "你是一个{style}的翻译助手。"},
    {"role": "human", "content": "把这句话翻译成英文：{text}"},
])

# ── 3. 消息模板类写法（等价，最显式）───────────────────────────────────────
prompt_class = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个{style}的翻译助手。"),
    HumanMessagePromptTemplate.from_template("把这句话翻译成英文：{text}"),
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

    # 填值后得到的是消息列表，可以打印出来看清楚结构
    msgs = prompt_tuple.invoke({"style": "正式严谨", "text": "今天过得怎么样"})
    print("=== 填值后的消息列表 ===")
    for m in msgs.to_messages():
        print(f"  [{m.__class__.__name__}] {m.content}")

    print("\n=== 三种写法结果一致 ===")
    for name, p in [("元组", prompt_tuple), ("字典", prompt_dict), ("类", prompt_class)]:
        filled = p.invoke({"style": "幽默口语", "text": "今天过得怎么样"})
        print(f"[{name}写法]", llm.invoke(filled).content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  ChatPromptTemplate.from_messages([
      ("system", "...{style}..."),
      ("human",  "...{text}..."),
  ])
        │  .invoke({"style": ..., "text": ...})
        ▼
  [ SystemMessage("..."), HumanMessage("...") ]   ← 直接可喂给模型

★ 核心规律：
  聊天模型场景一律用 ChatPromptTemplate（不是 PromptTemplate）。
  三种写法等价，元组 ("role", "模板") 最常用，本课程后面统一用它。

  这里的角色字符串只有 system/human；ai 角色和「插入整段历史」是下一课
  MessagesPlaceholder〔02/03〕要解决的事。
"""
