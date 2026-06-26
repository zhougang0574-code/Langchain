"""
第14课：评估与测试 —— 怎么知道效果好不好

为什么是这一课：前13课学完了整套 RAG/Agent/结构化输出的搭建方法，但搭出来的链
效果好不好，光靠人眼看几个例子是不靠谱的——今天看着挺好的回答，改了个 Prompt 或
换了个 chunk_size，怎么知道是变好了还是变差了？这一课学的是"用 LLM 当裁判"，
给回答打分、判断对错，把"感觉还不错"变成"通过率从80%提升到92%"这种可量化的结论。

学习要点：
1. Criteria Evaluator —— 用 LLM 按一个标准（简洁性、相关性等）给回答打分
2. 自定义 criteria —— 比如"忠实度"：答案有没有编造参考资料之外的内容（RAG 场景特别需要）
3. QA Evaluator —— 给定标准答案（reference），判断预测答案是否正确，适合有 ground truth 的场景
4. 简单回归测试思路 —— 固定一组 Q&A 测试集，批量跑链、批量打分，统计通过率，
   这样改动 Prompt/参数后，可以一键检测"到底是变好了还是变差了"

Python 小贴士（给新手）：
- 这一课的"测试集"就是一个 Python list，每个元素是一个 dict（{question, reference}），
  用 for 循环遍历，本质和你平时写单元测试的思路一样，只是"断言"换成了"让 LLM 判断对不对"
"""

import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.evaluation import load_evaluator

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# ─────────────────────────────────────────────
# 1. Criteria Evaluator —— 用 LLM 按标准打分
# ─────────────────────────────────────────────
print("=== Criteria Evaluator：内置标准（conciseness 简洁性）===")
conciseness_evaluator = load_evaluator("criteria", criteria="conciseness", llm=llm)
result = conciseness_evaluator.evaluate_strings(
    input="什么是 LangChain？",
    prediction="LangChain是一个开源框架，用于构建基于大语言模型的应用，提供模块化组件简化开发流程。",
)
print(f"判定: {result['value']}（score={result['score']}）")
print(f"理由（截取前80字）: {result['reasoning'][:80]}...")


# ─────────────────────────────────────────────
# 2. 自定义 criteria —— "忠实度"：有没有编造参考资料之外的内容
# ─────────────────────────────────────────────
# 这是 RAG 场景里最值得关注的一类问题：答案看起来很通顺，但其中一部分信息
# 是模型自己编的，并没有出现在检索到的上下文（reference）里。
print("\n=== 自定义 criteria：忠实度（RAG 幻觉检测）===")
faithfulness_criteria = {
    "faithfulness": "答案中的每一条信息是否都能在给定的参考资料(reference)中找到依据，不能包含参考资料之外编造的内容"
}
# 注意：带 reference 的自定义标准要用 "labeled_criteria"（不是 "criteria"），
# 因为要对照参考资料才能判断"是不是编的"
faithfulness_evaluator = load_evaluator("labeled_criteria", criteria=faithfulness_criteria, llm=llm)
hallucination_check = faithfulness_evaluator.evaluate_strings(
    input="LangChain 是谁创建的？",
    prediction="LangChain 由 Harrison Chase 于 2022 年 10 月创建，他之前在 Robust Intelligence 工作。",
    reference="LangChain 是一个用于构建大语言模型应用的开源框架，由 Harrison Chase 于 2022 年 10 月创建。",
)
print(f"判定: {hallucination_check['value']}（应该是 N，因为'之前在xx工作'这条信息参考资料里没有）")
print(f"理由（截取前100字）: {hallucination_check['reasoning'][:100]}...")


# ─────────────────────────────────────────────
# 3. QA Evaluator —— 对照标准答案判断正确性
# ─────────────────────────────────────────────
print("\n=== QA Evaluator：对照标准答案判断对错 ===")
qa_evaluator = load_evaluator("qa", llm=llm)
qa_result = qa_evaluator.evaluate_strings(
    input="LangChain是什么时候创建的？",
    prediction="LangChain创建于2022年。",  # 答案不完整但方向是对的
    reference="LangChain由Harrison Chase于2022年10月创建。",
)
print(f"判定: {qa_result['value']}（QA Evaluator 通常对'方向对、细节略有出入'比较宽容）")


# ─────────────────────────────────────────────
# 4. 综合实战：给第7课的 RAG 链做一次"回归测试"
# ─────────────────────────────────────────────
class AliyunEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


embeddings = AliyunEmbeddings(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"), model="text-embedding-v1"
)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据上下文回答问题，不知道就说不知道。\n上下文：{context}"),
    ("human", "{question}"),
])
rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)

# 固定的测试集：每条记录一个问题 + 一个"标准答案应该包含的事实"
# 真实项目里这份测试集会随着产品迭代越攒越多，每次改了 Prompt/chunk_size 都重新跑一遍
test_set = [
    {"question": "LangChain是什么时候创建的？", "reference": "LangChain由Harrison Chase于2022年10月创建。"},
    {"question": "FAISS是哪个公司开源的？", "reference": "FAISS是Facebook开源的。"},
    {"question": "今天股市行情怎么样？", "reference": "知识库中没有股市相关信息，应该回答不知道，不能瞎编。"},
]

print("\n=== 综合实战：RAG 链回归测试 ===")
passed = 0
for case in test_set:
    answer = rag_chain.invoke(case["question"])
    grade = qa_evaluator.evaluate_strings(
        input=case["question"], prediction=answer, reference=case["reference"]
    )
    is_correct = grade["value"] == "CORRECT"
    passed += int(is_correct)
    print(f"\nQ: {case['question']}")
    print(f"A: {answer[:60]}...")
    print(f"判定: {grade['value']} {'✅' if is_correct else '❌'}")

print(f"\n通过率: {passed}/{len(test_set)}")
print("这套流程就是最简单的'回归测试'：下次改了 Prompt 或 chunk_size 之后，")
print("重新跑一遍这个脚本，对比通过率有没有下降，就知道改动是不是把效果搞差了。")


if __name__ == "__main__":
    pass


"""
执行流程图：

Criteria Evaluator（无需参考答案）：
  (问题, 预测答案) ──► LLM 按给定标准（如"简洁性"）打分 ──► Y/N + 打分理由

Labeled Criteria Evaluator（自定义标准 + 需要参考资料）：
  (问题, 预测答案, 参考资料) ──► LLM 对照参考资料检查预测答案 ──► Y/N + 打分理由
  （常用于检测 RAG 幻觉：预测答案里是否有参考资料之外编造的内容）

QA Evaluator（需要标准答案）：
  (问题, 预测答案, 标准答案) ──► LLM 判断预测答案是否正确 ──► CORRECT/INCORRECT

回归测试流程：
  固定测试集 [{question, reference}, ...]
        │
        │ 对每一条：rag_chain.invoke(question) 拿到实际答案
        ▼
  [{question, 实际答案, reference}, ...]
        │
        │ qa_evaluator 逐条打分
        ▼
  统计通过率 ──► 改动 Prompt/参数后重新跑一遍，对比通过率变化


核心知识点 ★：

★ "用 LLM 当裁判"（LLM-as-judge）是目前评估 LLM 应用效果最实用的方式：
  没有标准答案的维度（简洁性、礼貌程度）用 Criteria Evaluator；
  有标准答案的维度（事实是否正确）用 QA Evaluator。
★ 自定义 criteria 时如果需要对照参考资料，必须用 "labeled_criteria"（不是 "criteria"）——
  忘记这个区别会导致 evaluator 拿不到 reference，没法判断"是不是编的"。
★ "忠实度"（faithfulness）评估是 RAG 场景的核心质检项：检测答案有没有引入
  检索到的上下文之外的编造信息，这是 RAG 比普通问答更容易出问题的地方。
★ QA Evaluator 对"方向对、细节略有出入"通常比较宽容，它判断的是"语义层面对不对"，
  不是要求逐字匹配标准答案——这跟字符串 diff 式的传统单元测试断言完全不同。
★ 回归测试的价值：把"改了 Prompt 之后感觉好像还行"变成"测试集通过率从 2/3 变成 3/3"，
  这种可复现、可量化的反馈，才能支撑长期、放心地迭代 Prompt 和检索参数。
★ 测试集要随着产品迭代持续积累：每次线上发现一个回答得不好的案例，
  都应该把它加进测试集（连同"应该怎么答"的参考），这样以后改动不会让老问题复发。
"""
