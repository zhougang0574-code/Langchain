"""
【05 记忆Memory / 02】RunnableWithMessageHistory —— 自动管理历史 + 多会话
======================================================================
〔05/01〕每轮都要手写「读历史→调链→写历史」，还得自己存历史。真实应用还有
个难题：同时服务很多用户，每个人的对话要分开存（不能串台）。

新概念（只有这一个）：
  RunnableWithMessageHistory(链, get_history函数, input_messages_key, history_messages_key)
    - 把一条普通链「包」成自动带记忆的链
    - 你只管 invoke({"input": ...})，它自动回填历史、自动把这轮存回去
    - 靠 session_id 区分不同会话：get_history(session_id) 决定这次用谁的历史

为什么是它：
  这是 LangChain 现在做对话记忆的标准方式（取代了旧的 ConversationChain/Memory 类）。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁的助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()


# ── 1. 用一个 dict 按 session_id 存各会话的历史 ────────────────────────────
store: dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    # 第一次见到某个 session_id 就新建一个历史容器，之后复用
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# ── 2. 把普通链包成「自动带记忆」的链 ──────────────────────────────────────
chat_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",       # invoke 时哪个 key 是本轮用户输入
    history_messages_key="history",   # 历史回填到模板的哪个槽（对应 MessagesPlaceholder）
)


if __name__ == "__main__":
    # 通过 config 指定 session_id；同一 id 共享历史
    cfg_ming = {"configurable": {"session_id": "user_xiaoming"}}

    print("=== 会话 user_xiaoming ===")
    print("A1:", chat_with_memory.invoke({"input": "我叫小明，喜欢篮球。"}, config=cfg_ming))
    # 第二轮不用手动传历史，它自动回填
    print("A2:", chat_with_memory.invoke({"input": "我喜欢什么运动？"}, config=cfg_ming))

    # 换一个 session_id：历史完全独立，不会串台
    cfg_other = {"configurable": {"session_id": "user_other"}}
    print("\n=== 会话 user_other（独立历史）===")
    print("A1:", chat_with_memory.invoke({"input": "我喜欢什么运动？"}, config=cfg_other))
    print("    ↑ 它不知道——因为这是另一个人的会话")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  invoke({"input": ...}, config={"configurable": {"session_id": "A"}})
        │
        │ ① get_history("A") 取出 A 的历史，自动填进 history 槽
        │ ② 跑链得到回答
        │ ③ 自动把本轮 input + 回答 存回 A 的历史
        ▼
     回答（下次 A 再来，历史已就绪）

★ 核心规律：
  RunnableWithMessageHistory 把〔05/01〕的「读/写历史」三步自动化，你只管传 input。
  session_id 是会话隔离的钥匙——get_history 按它返回对应历史，天然支持多用户。

  这里历史存在内存 dict（进程重启即丢）。要持久化，把 InMemoryChatMessageHistory
  换成数据库版（如 Redis/SQL 的 ChatMessageHistory）即可，接口完全一致。
"""
