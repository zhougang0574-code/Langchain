"""
第3课：LCEL 链式调用（LangChain Expression Language）

学习要点：
1. | 管道操作符的本质：创建 RunnableSequence，前一步输出是后一步输入
2. RunnablePassthrough —— 透传输入，常用于"保留原始值"
3. RunnablePassthrough.assign() —— 在输入 dict 上追加新字段
4. RunnableParallel —— 多路并发执行，结果合并为 dict
5. RunnableLambda —— 把任意函数包装成 Runnable
6. .bind() —— 给 Runnable 预绑定参数（常用于给 LLM 绑定 stop/tools）
7. 链的 stream / batch / invoke 统一性：整条链都支持
"""

import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0.7,
)


# ─────────────────────────────────────────────
# 1. | 管道的本质
# ─────────────────────────────────────────────
# a | b 等价于 RunnableSequence(first=a, last=b)
# 每一步的输出类型必须匹配下一步的输入类型
prompt = ChatPromptTemplate.from_template("用一句话解释 {topic}")
parser = StrOutputParser()

chain = prompt | llm | parser
# 查看链的类型
print("=== 链类型 ===")
print(type(chain).__name__)  # RunnableSequence
print(chain.first)           # 第一步: ChatPromptTemplate
print(chain.last)            # 最后一步: StrOutputParser


# ─────────────────────────────────────────────
# 2. RunnablePassthrough —— 透传
# ─────────────────────────────────────────────
# 场景：RAG 链需要同时把"问题"和"检索结果"传给 prompt
# 用 RunnablePassthrough 保留原始输入，避免被中间步骤丢弃
print("\n=== RunnablePassthrough ===")
passthrough_chain = (
    RunnableParallel({
        # question 原样透传
        "question": RunnablePassthrough(),
        # context 经过处理（这里用 lambda 模拟检索）
        "context": RunnableLambda(lambda q: f"[检索到的文档：{q} 相关内容]"),
    })
)
result = passthrough_chain.invoke("什么是向量数据库")
print(result)
# {'question': '什么是向量数据库', 'context': '[检索到的文档：...]'}


# ─────────────────────────────────────────────
# 3. RunnablePassthrough.assign() —— 在 dict 上追加字段
# ─────────────────────────────────────────────
# 与 RunnableParallel 的区别：
#   - RunnableParallel：完全重构 dict，原有 key 不会自动保留
#   - assign()：在原有 dict 基础上追加/覆盖字段，原有 key 全部保留
print("\n=== assign ===")
assign_chain = RunnablePassthrough.assign(
    # 追加 upper 字段：把 input["text"] 转大写
    upper=lambda x: x["text"].upper(),
    length=lambda x: len(x["text"]),
)
print(assign_chain.invoke({"text": "hello langchain"}))
# {'text': 'hello langchain', 'upper': 'HELLO LANGCHAIN', 'length': 15}


# ─────────────────────────────────────────────
# 4. RunnableParallel —— 并发执行多路
# ─────────────────────────────────────────────
# 每个 value 是独立的 Runnable，并发执行后结果合并
print("\n=== RunnableParallel（并发两路 LLM）===")
parallel_chain = RunnableParallel({
    "pros": ChatPromptTemplate.from_template("列出 {thing} 的3个优点，每点一行") | llm | parser,
    "cons": ChatPromptTemplate.from_template("列出 {thing} 的3个缺点，每点一行") | llm | parser,
})
result = parallel_chain.invoke({"thing": "远程工作"})
print("优点:\n", result["pros"])
print("缺点:\n", result["cons"])


# ─────────────────────────────────────────────
# 5. RunnableLambda —— 把函数变成 Runnable
# ─────────────────────────────────────────────
# 任何需要在链中插入自定义逻辑的地方都用它
def add_emoji(text: str) -> str:
    return text + " 🤖"

# 直接用 lambda 也可以，效果相同
lambda_chain = prompt | llm | parser | RunnableLambda(add_emoji)

print("\n=== RunnableLambda ===")
print(lambda_chain.invoke({"topic": "量子计算"}))


# ─────────────────────────────────────────────
# 6. itemgetter —— 从 dict 提取单个字段（比 lambda 更简洁）
# ─────────────────────────────────────────────
# 场景：输入是 dict，但下一步只需要其中某个字段
print("\n=== itemgetter ===")
extract_chain = (
    RunnablePassthrough.assign(upper=lambda x: x["text"].upper())
    # itemgetter("upper") 等价于 lambda x: x["upper"]
    | itemgetter("upper")
    | RunnableLambda(lambda s: f"提取到: {s}")
)
print(extract_chain.invoke({"text": "lcel is powerful"}))


# ─────────────────────────────────────────────
# 7. .bind() —— 预绑定 Runnable 参数
# ─────────────────────────────────────────────
# 场景：给 LLM 绑定 stop 词，或绑定 temperature 覆盖默认值
print("\n=== .bind() ===")
# 绑定 stop，遇到"。"就停止生成（控制输出长度）
bound_llm = llm.bind(stop=["。"])
bound_chain = prompt | bound_llm | parser
result = bound_chain.invoke({"topic": "黑洞"})
print(repr(result))  # 可以观察到输出在第一个句号前截断


# ─────────────────────────────────────────────
# 8. 整条链都支持 stream
# ─────────────────────────────────────────────
print("\n=== 链的 stream ===")
for chunk in chain.stream({"topic": "神经网络"}):
    print(chunk, end="", flush=True)
print()


if __name__ == "__main__":
    pass


"""
执行流程图：

     输入（dict 或 str）
          │
          ▼
  ┌───────────────────────────────────────┐
  │         RunnableSequence（|链）        │
  │                                       │
  │  step1: ChatPromptTemplate            │
  │    └─→ PromptValue（消息列表）         │
  │  step2: ChatOpenAI                    │
  │    └─→ AIMessage                      │
  │  step3: StrOutputParser               │
  │    └─→ str                            │
  └───────────────────────────────────────┘
          │
          ▼
       最终输出

RunnableParallel 示意：
          输入
         /    \
    branch1  branch2     ← 并发执行
         \    /
      合并为 dict
          │
          ▼
    {"key1": r1, "key2": r2}

RunnablePassthrough.assign 示意：
   原始 dict
   {"text": "hello"}
          │
          │ + upper="HELLO"
          │ + length=5
          ▼
   {"text": "hello", "upper": "HELLO", "length": 5}


核心知识点 ★：

★ a | b 创建 RunnableSequence，前一步输出必须匹配后一步的输入类型
★ RunnableParallel：完全重构输出 dict，并发执行所有分支
★ RunnablePassthrough：原样透传输入；常和 RunnableParallel 配合保留原始值
★ assign()：在原 dict 上追加字段，原有 key 自动保留（RunnableParallel 不会）
★ RunnableLambda：任意 Python 函数都能接入 LCEL 链
★ itemgetter("key") 比 lambda x: x["key"] 更简洁，且对 LCEL 类型推断更友好
★ .bind() 不调用 Runnable，只是返回一个"预设了参数"的新 Runnable
★ 整条 LCEL 链天然支持 invoke / stream / batch / ainvoke，无需额外处理
"""
