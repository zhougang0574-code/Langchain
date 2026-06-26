"""
【10 评估Evaluation / 02】QA 评估 + 回归测试 —— 让迭代有可量化依据
=============================================================
〔10/01〕给单条回答打分。本课把它用到工程闭环：固定一组「问题+标准答案」当测试集，
批量跑链、批量打分、统计通过率。以后改了 prompt / chunk_size，重跑一遍对比通过率，
就知道改动到底让效果变好还是变差——这就是 LLM 应用的「回归测试」。

新概念（只有这一个）：
  load_evaluator("qa", llm=...) —— QA 评估器，对照标准答案判断预测答案是否正确。
    evaluate_strings(input=问题, prediction=实际答案, reference=标准答案) → CORRECT/INCORRECT
    它判断「语义层面对不对」，对「方向对、细节略出入」较宽容，不是逐字匹配。

把它套在一个测试集上循环跑，就得到通过率——可复现、可量化的迭代依据。
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_classic.evaluation import load_evaluator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import get_embeddings   # noqa: E402

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"), temperature=0,
)

# ── 复用〔07/02〕建好的向量库，拼一条 RAG 链作为被测对象 ───────────────────
RAG_INDEX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "07_检索增强RAG", "faiss_index"
)
retriever = FAISS.load_local(
    RAG_INDEX, get_embeddings(), allow_dangerous_deserialization=True
).as_retriever(search_kwargs={"k": 3})


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_messages([
        ("system", "依据上下文回答，不知道就说不知道。\n上下文：{context}"),
        ("human", "{question}"),
    ])
    | llm | StrOutputParser()
)


if __name__ == "__main__":
    qa_eval = load_evaluator("qa", llm=llm)

    # 固定测试集：每条 = 问题 + 标准答案应包含的事实
    # 真实项目里这份测试集会越攒越多：线上每发现一个答得不好的案例就加进来
    test_set = [
        {"q": "LangChain 是谁创建的？", "ref": "由 Harrison Chase 创建。"},
        {"q": "RAG 解决什么问题？", "ref": "解决 LLM 知识有截止日期、无法回答私有知识的问题。"},
        {"q": "FAISS 是谁开源的？", "ref": "FAISS 由 Facebook（Meta）开源。"},
    ]

    print("=== RAG 链回归测试 ===")
    passed = 0
    for case in test_set:
        answer = rag_chain.invoke(case["q"])
        grade = qa_eval.evaluate_strings(input=case["q"], prediction=answer, reference=case["ref"])
        ok = grade["value"] == "CORRECT"
        passed += ok
        print(f"\nQ: {case['q']}")
        print(f"A: {answer[:50]}...")
        print(f"判定: {grade['value']} {'✅' if ok else '❌'}")

    print(f"\n通过率：{passed}/{len(test_set)}")
    print("以后改了 prompt / chunk_size，重跑这个脚本对比通过率，就知道改动是好是坏。")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  固定测试集 [{问题, 标准答案}, ...]
        │ 对每条：rag_chain.invoke(问题) → 实际答案
        ▼
  qa_evaluator 逐条打分（CORRECT / INCORRECT）
        ▼
  统计通过率 ──► 改 prompt/参数后重跑，对比通过率变化

★ 核心规律：
  把 QA 评估器套在固定测试集上循环，就是 LLM 应用的「回归测试」：把「感觉还行」
  变成「通过率 2/3 → 3/3」这种可复现、可量化的结论，才敢放心长期迭代。

  QA 评估判「语义对不对」，对细节出入较宽容，区别于传统单测的逐字断言。
  测试集要持续积累：每个线上 badcase 都连同「应该怎么答」加进去，防止老问题复发。
  —— 这是整个课程的收尾：从「会搭」到「搭得好且能持续验证」。
"""
