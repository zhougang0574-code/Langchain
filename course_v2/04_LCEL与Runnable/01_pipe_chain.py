"""
【04 LCEL与Runnable / 01】管道 | —— 把 prompt、model、parser 串成一条链
==================================================================
前三个域我们分别学了零件：Prompt〔02〕、Model〔01〕、Parser〔03〕。
之前都是手动「填模板 → invoke 模型 → invoke 解析器」三步分开调。
本域学的 LCEL（LangChain Expression Language）就是把它们「粘」成一条链。

新概念（只有这一个）：
  管道操作符 |  ——  a | b 表示「a 的输出作为 b 的输入」，串成一个 RunnableSequence。
    chain = prompt | model | parser
  整条链本身又是一个 Runnable，照样有 invoke / stream / batch（见〔01/04〕）。

为什么叫「表达式语言」：
  你用 | 把组件「表达」成数据流，LangChain 负责按顺序执行、传递中间结果。
  这是 LangChain 的核心——后面所有复杂结构（RAG、Agent）都是 Runnable 拼出来的。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)


# ── 1. 三个零件 ────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("用一句话解释「{topic}」")
parser = StrOutputParser()

# ── 2. 用 | 串成一条链 ─────────────────────────────────────────────────────
# 数据流：dict → prompt → 消息列表 → model → AIMessage → parser → str
chain = prompt | llm | parser


if __name__ == "__main__":
    print("=== 链类型 ===")
    print(type(chain).__name__)               # RunnableSequence

    print("\n=== invoke：传变量 dict，直接拿到字符串 ===")
    print(chain.invoke({"topic": "梯度下降"}))

    # 整条链天然支持 stream（逐字输出），无需任何额外改造
    print("\n=== 整条链支持 stream ===")
    for chunk in chain.stream({"topic": "神经网络"}):
        print(chunk, end="", flush=True)
    print()

    # 也支持 batch（并发跑多个）
    print("\n=== 整条链支持 batch ===")
    for r in chain.batch([{"topic": "递归"}, {"topic": "并发"}]):
        print("-", r)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  {"topic": "..."}
        │ prompt   （填模板）
        ▼  PromptValue / 消息列表
        │ llm      （调模型）
        ▼  AIMessage
        │ parser   （取文本）
        ▼  str

★ 核心规律：
  a | b 创建 RunnableSequence，前一步的输出类型必须能当后一步的输入。
  prompt | model | parser 是 LangChain 最经典的链；整条链又是 Runnable，
  自动继承 invoke/stream/batch/ainvoke（〔01/04〕的四种调用对链同样成立）。

  接下来几课就是往这条链里插各种「特殊 Runnable」：函数〔04/02〕、并行〔04/03〕、
  透传/追加字段〔04/04〕、分支〔04/05〕。
"""
