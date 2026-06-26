"""
【02 提示词Prompt / 04】partial —— 预先填好一部分变量
===================================================
〔02/01〕里每次 invoke 都要把所有变量传齐。但有些变量在「建模板时」就固定了
（比如系统语言、当前日期、输出格式说明），每次都传很啰嗦，也容易漏。

新概念（只有这一个）：
  prompt.partial(变量=值) —— 预先固定一部分变量，返回一个「还差剩余变量」的新模板。
  之后 invoke 只需传剩下的变量。

两种典型用法：
  ① 固定一个常量值（partial(language="中文")）
  ② 绑定一个「每次调用时动态求值」的函数（partial(date=今天日期函数)）
"""

from datetime import datetime

from langchain_core.prompts import PromptTemplate

# ── 1. 用常量预填 ───────────────────────────────────────────────────────────
base = PromptTemplate.from_template("用{language}回答：{question}")

# 把 language 固定成「文言文」，得到一个只差 question 的新模板
classical = base.partial(language="文言文")
print("=== partial 后还需要的变量 ===")
print(classical.input_variables)                       # ['question']
print(classical.invoke({"question": "什么是云计算"}).to_string())


# ── 2. 用函数预填「动态值」──────────────────────────────────────────────────
# partial 的值可以是一个无参函数：每次 invoke 时才调用它求当前值
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


dated = PromptTemplate.from_template(
    "当前时间是 {time}。请回答：{question}"
).partial(time=now_str)   # 注意传的是函数本身 now_str，不是 now_str()

print("\n=== 动态 partial：time 每次 invoke 自动取当前时刻 ===")
print(dated.invoke({"question": "现在几点？"}).to_string())


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
    print("\n=== 实跑 ===")
    print(llm.invoke(classical.invoke({"question": "什么是递归"})).content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  完整模板 {language}{question}
        │  .partial(language="文言文")
        ▼
  新模板（只差 {question}）
        │  .invoke({"question": ...})
        ▼
  最终文本

★ 核心规律：
  partial 把「建模板时就能定的变量」提前固定，调用点只关心「真正变化的变量」。
  partial 的值可以是常量，也可以是无参函数——传函数时每次 invoke 都会重新求值，
  适合「当前时间 / 随机 id」这类动态默认值。

  这与〔03 输出解析〕配合极常见：把 parser 的 format_instructions 用 partial 预填进模板。
"""
