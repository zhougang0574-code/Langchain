"""
【10 评估Evaluation / 01】用 LLM 当裁判 —— Criteria 打分 + RAG 忠实度检测
====================================================================
前面学会了搭各种链。但「搭出来效果好不好」光靠人眼看几个例子不靠谱：改个 prompt、
换个 chunk_size，到底变好还是变差？本域学「用 LLM 当裁判」给回答打分，把
「感觉还行」变成可量化的结论。

新概念（只有这一个）：
  load_evaluator(类型, criteria=..., llm=...) —— 加载一个「评估器」给回答打分。
    "criteria"          按某个标准（简洁性、相关性…）打分，不需要参考答案
    "labeled_criteria"  按自定义标准 + 参考资料打分（如检测 RAG 是否编造）
  evaluate_strings(input=, prediction=, reference=) 返回 {value, score, reasoning}

本课重点演示 RAG 最该查的一项：忠实度（答案有没有编造参考资料之外的内容）。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.evaluation import load_evaluator

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"), temperature=0,
)


if __name__ == "__main__":
    # ── 1. 内置标准：简洁性（不需要参考答案）──────────────────────────────
    print("=== Criteria：按「简洁性」打分 ===")
    concise = load_evaluator("criteria", criteria="conciseness", llm=llm)
    r = concise.evaluate_strings(
        input="什么是 LangChain？",
        prediction="LangChain 是一个用于构建大语言模型应用的开源框架。",
    )
    print(f"判定 value={r['value']}  score={r['score']}")
    print(f"理由：{r['reasoning'][:60]}...")

    # ── 2. 自定义标准 + 参考资料：忠实度（RAG 幻觉检测）─────────────────────
    # 带参考资料的自定义标准必须用 "labeled_criteria"（不是 "criteria"）
    print("\n=== Labeled Criteria：忠实度（有没有编造）===")
    faithful = load_evaluator(
        "labeled_criteria",
        criteria={"faithfulness": "答案中每条信息是否都能在参考资料中找到依据，不得包含参考资料之外编造的内容"},
        llm=llm,
    )
    check = faithful.evaluate_strings(
        input="LangChain 是谁创建的？",
        prediction="LangChain 由 Harrison Chase 创建，他之前在谷歌工作。",   # 「在谷歌工作」是编的
        reference="LangChain 是一个开源框架，由 Harrison Chase 于 2022 年 10 月创建。",
    )
    print(f"判定 value={check['value']}（应为 N：'之前在谷歌工作'参考资料里没有）")
    print(f"理由：{check['reasoning'][:80]}...")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  Criteria（无参考答案）：
    (问题, 预测答案) ──► LLM 按标准（如简洁性）打分 ──► Y/N + 理由

  Labeled Criteria（有参考资料）：
    (问题, 预测答案, 参考资料) ──► LLM 对照参考资料检查 ──► Y/N + 理由
    （常用于 RAG 忠实度：预测答案里有没有参考资料之外编造的内容）

★ 核心规律：
  「LLM-as-judge」是评估 LLM 应用最实用的手段：主观维度（简洁/礼貌）用 criteria；
  要对照资料判断「是否编造」用 labeled_criteria（忘了带 labeled 就拿不到 reference）。

  忠实度是 RAG 的核心质检项——RAG 比普通问答更容易「看着通顺其实编了一句」。
  下一课〔10/02〕把评估器用进「回归测试」，让迭代有可量化的依据。
"""
