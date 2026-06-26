"""
第5课：Memory & 对话历史管理

学习要点：
1. InMemoryChatMessageHistory —— 内存中存储对话历史（langchain_core 原生）
2. RunnableWithMessageHistory —— 把任意 LCEL 链包装成"有记忆"的链
3. session_id —— 区分不同用户/会话，每个 session 独立维护历史
4. get_session_history 函数 —— 告诉 RunnableWithMessageHistory 如何获取历史对象
5. input_messages_key / history_messages_key —— 指定输入和历史在 dict 中的 key
6. 手动读写历史：add_user_message / add_ai_message / messages
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
    temperature=0.7,
)


# ─────────────────────────────────────────────
# 1. InMemoryChatMessageHistory —— 手动操作历史
# ─────────────────────────────────────────────
history = InMemoryChatMessageHistory()

# 手动添加消息（模拟已有对话）
history.add_user_message("我叫小明，我喜欢编程")
history.add_ai_message("你好小明！编程是个很棒的爱好，你主要用哪种语言？")
history.add_user_message("我主要用 Python")
history.add_ai_message("Python 很适合！你在做什么方向的项目？")

print("=== 手动管理历史 ===")
for msg in history.messages:
    print(f"[{type(msg).__name__}] {msg.content}")


# ─────────────────────────────────────────────
# 2. RunnableWithMessageHistory —— 自动管理历史
# ─────────────────────────────────────────────
# 核心思路：
#   ① 定义带 MessagesPlaceholder 的 prompt（第2课的知识）
#   ② 用 RunnableWithMessageHistory 包装链
#   ③ 每次 invoke 时传 config={"configurable": {"session_id": "xxx"}}
#   → 框架自动：invoke 前读历史注入 prompt，invoke 后把本轮问答追加到历史

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手，能记住对话上下文。"),
    MessagesPlaceholder(variable_name="history"),  # 历史插槽
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# 每个 session_id 对应一个独立的历史对象
# 这里用 dict 模拟数据库，生产环境可换成 Redis / SQL
store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    # 不存在则创建新的，存在则返回已有的
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 包装成有记忆的链
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",      # chain 的输入 dict 里，用户消息的 key
    history_messages_key="history",  # prompt 里 MessagesPlaceholder 的 variable_name
)


# ─────────────────────────────────────────────
# 3. 多轮对话演示
# ─────────────────────────────────────────────
# config 指定 session_id，同一 session_id 共享同一段历史
config_user1 = {"configurable": {"session_id": "user_001"}}

print("\n=== 多轮对话（user_001）===")

r1 = chain_with_history.invoke({"input": "你好！我是大壮，我在学 LangChain"}, config=config_user1)
print("第1轮:", r1)

r2 = chain_with_history.invoke({"input": "你还记得我的名字吗？"}, config=config_user1)
print("第2轮:", r2)

r3 = chain_with_history.invoke({"input": "我正在学什么？"}, config=config_user1)
print("第3轮:", r3)

# 查看历史中积累的消息数
print(f"\nuser_001 历史消息数: {len(store['user_001'].messages)}")


# ─────────────────────────────────────────────
# 4. session 隔离验证
# ─────────────────────────────────────────────
config_user2 = {"configurable": {"session_id": "user_002"}}

print("\n=== session 隔离（user_002）===")
r = chain_with_history.invoke({"input": "你知道大壮是谁吗？"}, config=config_user2)
print(r)  # user_002 没有 user_001 的历史，所以不知道大壮


# ─────────────────────────────────────────────
# 5. 查看某 session 的完整历史
# ─────────────────────────────────────────────
print("\n=== user_001 完整历史 ===")
for msg in store["user_001"].messages:
    role = "用户" if msg.__class__.__name__ == "HumanMessage" else "AI"
    print(f"[{role}] {msg.content[:50]}...")


if __name__ == "__main__":
    pass


"""
执行流程图：

chain_with_history.invoke({"input": "..."}, config={"session_id": "abc"})
         │
         │ ① 调用 get_session_history("abc") 拿到历史对象
         ▼
   history.messages → [HumanMsg, AIMsg, HumanMsg, AIMsg, ...]
         │
         │ ② 注入 MessagesPlaceholder("history")
         ▼
   prompt 格式化：
   [SystemMessage]
   [历史消息 × N]
   [HumanMessage("...")]
         │
         │ ③ llm 推理
         ▼
      AIMessage
         │
         │ ④ StrOutputParser
         ▼
       str 响应
         │
         │ ⑤ 自动追加：history.add_user_message + add_ai_message
         ▼
   历史对象更新（下次调用自动包含本轮）


核心知识点 ★：

★ MessagesPlaceholder + RunnableWithMessageHistory 是 LangChain 记忆管理的标准组合
★ get_session_history 函数是扩展点：换成 Redis/DB 即可实现持久化记忆
★ config={"configurable": {"session_id": "..."}} 是触发记忆的必须参数，忘传就没有记忆
★ 历史自动追加：invoke 结束后框架自动把本轮 HumanMessage + AIMessage 写入历史
★ 不同 session_id 完全隔离，适合多用户场景
★ InMemoryChatMessageHistory 重启后丢失，生产用 RedisChatMessageHistory 等持久化方案
"""
