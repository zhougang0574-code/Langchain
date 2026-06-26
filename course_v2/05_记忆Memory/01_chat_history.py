"""
【05 记忆Memory / 01】ChatMessageHistory —— 手动维护一段对话历史
=============================================================
〔01/02〕讲过：模型本身无状态，所谓「记忆」就是每次把过往消息再发一遍。
〔02/03〕的 MessagesPlaceholder 给模板留好了「插历史」的槽。
本课把这两点接起来：用一个「历史容器」对象来存消息，手动维护多轮对话。

新概念（只有这一个）：
  InMemoryChatMessageHistory —— 一个存在内存里的「消息历史容器」。
    .add_user_message(...) / .add_ai_message(...)  往里加消息
    .messages                                      取出当前全部消息（用于回填模板）

本课故意「全手动」，让你看清记忆的机械原理；下一课〔05/02〕再自动化。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁的助手。"),
    MessagesPlaceholder(variable_name="history"),   # 历史插这里
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()


# ── 1. 建一个历史容器 ──────────────────────────────────────────────────────
history = InMemoryChatMessageHistory()


# ── 2. 封装一轮对话：取历史 → 调链 → 把这轮存回历史 ────────────────────────
def chat(user_input: str) -> str:
    # 把当前历史回填到模板的 history 槽
    answer = chain.invoke({"history": history.messages, "input": user_input})
    # 手动把「这一轮的用户输入 + AI 回答」追加进历史，供下一轮使用
    history.add_user_message(user_input)
    history.add_ai_message(answer)
    return answer


if __name__ == "__main__":
    print("Q1:", "我叫小明，今年28岁。")
    print("A1:", chat("我叫小明，今年28岁。"))

    print("\nQ2:", "我多大了？")          # 依赖上一轮，能答对说明历史生效
    print("A2:", chat("我多大了？"))

    print(f"\n当前历史共 {len(history.messages)} 条消息（2 轮 = 4 条）")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  每一轮：
    history.messages ──回填──► 模板 history 槽 ──► llm ──► 回答
                                                          │
    history.add_user_message / add_ai_message ◄──存回──────┘

★ 核心规律：
  「记忆」= 一个存消息的容器 + 每轮「读历史→调用→写历史」三步。
  本课手动写这三步，是为了让你看清记忆没有魔法，就是消息的存取。

  但每轮都手写「读/写历史」很啰嗦，还要自己管多个用户的会话。
  下一课 RunnableWithMessageHistory〔05/02〕把这套自动化，你只管传 input。
"""
