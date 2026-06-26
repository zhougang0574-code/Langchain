"""
【04 LCEL与Runnable / 06】.bind() 与 with_config —— 给链预设参数〔进阶〕
==================================================================
〔04/05〕之前，链的每个组件参数都在「建链时」写死。但有时你想给链里的某个
Runnable 预先「绑」一个调用参数（比如给模型绑 stop 词），或给整条链打个运行配置。

新概念（只有这一个，两个相关方法）：
  runnable.bind(参数=值)   —— 返回一个「预设了该参数」的新 Runnable（不立即执行）
  runnable.with_config(...) —— 给运行附加配置（如 run_name / tags，便于调试和追踪）

典型场景：
  .bind(stop=[...])  给模型绑停止词，控制输出在某处截断；
  with_config(run_name="...") 给链命名，在回调/日志〔08/03〕里好认。
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
    temperature=0,
)
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template("列出 {topic} 的要点")


# ── 1. .bind(stop=...)：给模型预绑停止词 ───────────────────────────────────
# 绑定 stop=["2."]：模型生成到出现 "2." 时就停下——于是只会输出第 1 点
bound_llm = llm.bind(stop=["2."])
bound_chain = prompt | bound_llm | parser


# ── 2. with_config：给运行附加可观测信息 ──────────────────────────────────
named_chain = (prompt | llm | parser).with_config(
    run_name="要点生成链",          # 在回调/LangSmith 里显示这个名字
    tags=["demo", "lcel"],
)


if __name__ == "__main__":
    print("=== 普通链：完整输出 ===")
    print((prompt | llm | parser).invoke({"topic": "学好 LangChain"}))

    print("\n=== .bind(stop=['2.'])：遇到 '2.' 截断，只剩第 1 点 ===")
    print(bound_chain.invoke({"topic": "学好 LangChain"}))

    print("\n=== with_config(run_name=...)：功能不变，多了可观测标签 ===")
    print(named_chain.invoke({"topic": "学好 LangChain"}))


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  llm                       llm.bind(stop=["2."])
   │ 每次调用都要传 stop  →   │ stop 已经预绑好，调用时不用再传
   ▼                          ▼
  普通 Runnable             预设参数的新 Runnable

★ 核心规律：
  .bind() 不执行任何东西，只是返回一个「预设了某些调用参数」的新 Runnable，
  方便把固定参数（stop / 工具列表等）提前绑好，让链更干净。
  〔06/01〕里 llm.bind_tools(...) 就是同一思路：把工具预绑到模型上。

  with_config 不改变结果，只附加运行期元信息（名字/标签），是调试与追踪的抓手，
  配合〔08/03〕的回调能看清「链里每一步叫什么、跑了多久」。
"""
