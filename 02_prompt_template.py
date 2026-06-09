"""
第2课：Prompt Template 提示词模板

学习要点：
1. ChatPromptTemplate.from_messages() —— 最常用的模板创建方式
2. 模板变量 {variable} 的占位与填充
3. partial() —— 预填充部分变量，生成新模板（常用于固定系统变量）
4. MessagesPlaceholder —— 插入消息列表，Memory / 多轮对话的关键
5. from_template() vs from_messages() 的区别
6. 模板 + LCEL：template | llm | parser 完整链路
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0.7,
)


# ─────────────────────────────────────────────
# 1. from_messages() —— 标准创建方式
# ─────────────────────────────────────────────
# 元组 ("role", "content") 是简写，等同于 SystemMessagePromptTemplate.from_template(...)
template = ChatPromptTemplate.from_messages([
    ("system", "你是一个专精 {domain} 的专家，回答要简洁专业。"),
    ("human", "请解释一下：{concept}"),
])

# format_messages() 只是格式化，不调用 LLM，返回消息列表
messages = template.format_messages(domain="Python", concept="GIL")
print("=== format_messages ===")
for m in messages:
    print(type(m).__name__, ":", m.content)


# ─────────────────────────────────────────────
# 2. LCEL 链路：template | llm | parser
# ─────────────────────────────────────────────
# template.invoke() 等同于 format_messages()，但返回 PromptValue 对象
# 接入 LCEL 后，template 负责格式化，llm 负责推理，parser 负责提取文本
chain = template | llm | StrOutputParser()

print("\n=== LCEL chain ===")
result = chain.invoke({"domain": "网络安全", "concept": "SQL 注入"})
print(result)


# ─────────────────────────────────────────────
# 3. partial() —— 预填充部分变量
# ─────────────────────────────────────────────
# 场景：domain 在整个应用里固定，concept 每次不同
# partial 返回一个新模板，已绑定的变量不再需要传
expert_template = template.partial(domain="机器学习")

print("\n=== partial ===")
# 现在只需传 concept，不需要传 domain
result = (expert_template | llm | StrOutputParser()).invoke({"concept": "过拟合"})
print(result)


# ─────────────────────────────────────────────
# 4. MessagesPlaceholder —— 多轮对话的核心
# ─────────────────────────────────────────────
# 作用：在模板中"留一个位置"，运行时插入完整的消息列表（历史记录）
# 没有它就无法在模板里动态塞入对话历史
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手。"),
    # variable_name 对应 invoke 时传入的 key
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 模拟多轮对话历史
history = [
    HumanMessage(content="我叫小明"),
    AIMessage(content="你好，小明！有什么可以帮你的？"),
]

print("\n=== MessagesPlaceholder ===")
result = (chat_template | llm | StrOutputParser()).invoke({
    "history": history,
    "input": "你还记得我叫什么名字吗？",
})
print(result)


# ─────────────────────────────────────────────
# 5. from_template() vs from_messages() 区别
# ─────────────────────────────────────────────
# from_template：只有一条 Human 消息，适合简单场景
simple = ChatPromptTemplate.from_template("用一句话解释 {concept}")
# from_messages：支持多角色，适合需要 System 或多轮对话的场景

print("\n=== from_template（简单场景）===")
result = (simple | llm | StrOutputParser()).invoke({"concept": "递归"})
print(result)


# ─────────────────────────────────────────────
# 6. 查看模板的输入变量（调试用）
# ─────────────────────────────────────────────
print("\n=== input_variables ===")
print("template.input_variables      :", template.input_variables)
print("chat_template.input_variables :", chat_template.input_variables)
# MessagesPlaceholder 的变量也会出现在 input_variables 里


if __name__ == "__main__":
    pass


"""
执行流程图：

① from_messages() / from_template()
   ──→ ChatPromptTemplate（含占位变量）
              │
              │ .partial(domain="xxx")
              ▼
        新 ChatPromptTemplate（部分变量已绑定）
              │
              │ LCEL: template | llm | parser
              ▼
   ┌──────────────────────────────────┐
   │  invoke({"concept": "...",       │
   │          "history": [...]}  )    │
   └──────────────┬───────────────────┘
                  │ template 格式化
                  ▼
         [SystemMessage, ...历史消息..., HumanMessage]
                  │ llm 推理
                  ▼
              AIMessage
                  │ StrOutputParser
                  ▼
               str (最终文本)


核心知识点 ★：

★ from_messages 接受元组 ("role", "content")，是最简洁的写法
★ format_messages() 只格式化不推理；invoke() 在 LCEL 链中才触发 LLM
★ partial() 生成绑定了部分变量的新模板，原模板不变（immutable）
★ MessagesPlaceholder 是多轮对话/Memory 的关键，运行时动态插入历史消息列表
★ input_variables 可以查看模板还缺哪些变量，调试时很有用
★ ChatPromptTemplate 本身也是 Runnable，支持 | 管道操作
"""
