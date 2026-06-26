"""
【01 基础 / 02】消息类型 —— 用 System/Human/AI 代替裸字符串
========================================================
〔01/01〕直接传了一个字符串。但聊天模型的本质是「多轮对话」，
真实场景里需要区分「谁说的话」：是系统设定、用户提问，还是 AI 的历史回复。

新概念（只有这一个）：
  三种消息对象
    SystemMessage —— 设定角色 / 规则（对话开头放一条）
    HumanMessage  —— 用户说的话
    AIMessage     —— AI 说过的话（用于把历史回复喂回去，让模型「记得」上下文）

把 messages 列表传给 invoke，模型会按顺序理解整段对话。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0.7,
)


# ── 1. 用 System + Human 控制角色 ─────────────────────────────────────────
def demo_system_role():
    print("=== System 设定角色 ===")
    messages = [
        # System 像「岗前培训」：约束模型整段对话的风格
        SystemMessage(content="你是一个只会用文言文回答的助手。"),
        HumanMessage(content="今天天气真好"),
    ]
    print(llm.invoke(messages).content)


# ── 2. 手动拼一段「历史」让模型记住上下文 ──────────────────────────────────
def demo_history():
    print("\n=== 用 AIMessage 喂回历史，模型才能理解「它」指什么 ===")
    messages = [
        SystemMessage(content="你是一个简洁的助手，每次不超过一句话。"),
        HumanMessage(content="LangChain 是什么？"),
        # 把上一轮 AI 的回答作为 AIMessage 放进来——这就是「记忆」最底层的样子
        AIMessage(content="LangChain 是一个用于构建大语言模型应用的开源框架。"),
        HumanMessage(content="它是谁创建的？"),   # 「它」依赖上一轮的上下文
    ]
    print(llm.invoke(messages).content)


if __name__ == "__main__":
    demo_system_role()
    demo_history()


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
一段对话的消息结构：

  [ SystemMessage ]  ← 角色/规则，通常只在开头放一条
  [ HumanMessage  ]  ← 用户第 1 句
  [ AIMessage     ]  ← AI 第 1 句（历史回复）
  [ HumanMessage  ]  ← 用户第 2 句（可能含「它/这个」等指代词）
        │
        ▼  llm.invoke(messages)
     AIMessage      ← AI 第 2 句

★ 核心规律：
  模型本身是「无状态」的——它不会自动记得上一轮。
  所谓「记忆」，本质就是每次把过往的 Human/AI 消息一起再发一遍。
  〔05 记忆Memory〕整个域，做的就是「自动帮你维护并回填这段历史」这件事。

  传字符串 = 传 [HumanMessage(字符串)] 的简写；要设角色/带历史就必须用消息列表。
"""
