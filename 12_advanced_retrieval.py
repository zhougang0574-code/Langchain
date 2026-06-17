"""
第12课：高级检索查漏 —— 重排序 & 混合检索

为什么是这一课：第8课的 MultiQuery/MMR 改进的仍然是"纯向量检索"本身的召回方式，
但向量检索有个天生的弱点——它靠语义相似度匹配，对精确的关键词、数字、专有名词
不一定敏感（比如查"2022年10月"，向量检索可能因为语义上更靠近别的内容而漏掉它）。
解决思路有两个方向：①混合检索——把"关键词检索"(BM25) 和"向量检索"结合，
取长补短；②重排序——先粗召回一批候选，再用更强的能力（这里用 LLM）精确排序。

学习要点：
1. BM25 —— 经典的关键词检索算法，原理是"词频统计"，对精确关键词/数字敏感
2. 中文分词的坑：BM25Retriever 默认按空格切词，中文没有空格会导致整句变成一个"词"
3. EnsembleRetriever —— 把向量检索和 BM25 按权重融合，取长补短
4. LLMListwiseRerank —— 先用普通 retriever 粗召回较多候选，再用 LLM 给文档排序精选
5. ContextualCompressionRetriever —— 把"重排序"包装成一个标准 Retriever，可以直接接入 RAG 链

Python 小贴士（给新手）：
- BM25（Best Matching 25）不需要 LLM 也不需要训练，纯粹靠统计"词出现的频率"打分，
  这也是为什么它对生僻专有名词反而很准——只要词出现过，统计上就能匹配到
"""

import os
import jieba  # 中文分词库，详见下方第2节的踩坑说明
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


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
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model="text-embedding-v1",
)

# 复用第7课持久化的向量库
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
all_docs = list(vectorstore.docstore._dict.values())  # 拿到全部 Document，给 BM25 用
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# ─────────────────────────────────────────────
# 1. BM25Retriever —— 先用默认配置，观察一个真实的坑
# ─────────────────────────────────────────────
print("=== BM25Retriever 默认配置（先观察问题）===")
bm25_default = BM25Retriever.from_documents(all_docs)
bm25_default.k = 2

# default_preprocessing_func 的实现就是 text.lower().split()——按空格切词
from langchain_community.retrievers.bm25 import default_preprocessing_func
print(f"默认分词在中文上的效果: {default_preprocessing_func('LangChain 是一个用于构建大语言模型应用的开源框架')}")
print("⚠️ 整句中文除了空格分隔的英文单词外，剩下的中文部分被当成了【一个词】！")
print("   原因：默认分词函数只会按空格切，中文句子内部没有空格，等于没切词，")
print("   BM25 统计词频时把整句中文当一个 token，关键词匹配能力基本失效。")

query_with_date = "2022年10月"
result_default = bm25_default.invoke(query_with_date)
print(f"\n用默认BM25查 '{query_with_date}'，结果（可能不准）: {[d.page_content[:30] for d in result_default]}")


# ─────────────────────────────────────────────
# 2. 修复：用 jieba 做中文分词，再重建 BM25Retriever
# ─────────────────────────────────────────────
print("\n=== 修复：用 jieba 分词 ===")
print(f"jieba 分词效果: {list(jieba.cut('LangChain 是一个用于构建大语言模型应用的开源框架'))}")


def jieba_tokenize(text: str) -> List[str]:
    # 过滤掉空字符串（标点、空格切出来的空 token）
    return [w for w in jieba.cut(text.lower()) if w.strip()]


bm25_retriever = BM25Retriever.from_documents(all_docs, preprocess_func=jieba_tokenize)
bm25_retriever.k = 2

result_fixed = bm25_retriever.invoke(query_with_date)
print(f"用 jieba 分词后查 '{query_with_date}': {[d.page_content[:30] for d in result_fixed]}")
result_vector = vector_retriever.invoke(query_with_date)
print(f"对比：纯向量检索查 '{query_with_date}': {[d.page_content[:30] for d in result_vector]}")
print("可以看到向量检索因为'语义匹配'容易漏掉这种精确日期，而修好分词后的 BM25 能准确命中。")


# ─────────────────────────────────────────────
# 3. EnsembleRetriever —— 向量检索 + BM25 按权重融合
# ─────────────────────────────────────────────
# weights=[0.5, 0.5] 表示两边各占一半权重；底层用 RRF（Reciprocal Rank Fusion，倒数排名融合）
# 把两份排序结果合并成一份，而不是简单地把两边结果拼起来。
print("\n=== EnsembleRetriever：混合检索 ===")
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)
for q in ["2022年10月", "Weaviate 混合搜索"]:
    docs = ensemble_retriever.invoke(q)
    print(f"\nQ: {q}")
    print(f"混合检索结果（{len(docs)}个，已去重融合): {[d.page_content[:30] for d in docs]}")


# ─────────────────────────────────────────────
# 4. LLMListwiseRerank —— 用 LLM 给候选文档精确排序
# ─────────────────────────────────────────────
# 思路：先用普通 retriever 多召回几个候选（这里 k=4，故意比平时大），
# 再让 LLM 看着问题，把这些候选文档按真实相关性重新排序，只保留 top_n 个。
# 好处：LLM 能理解语义细节（比如"哪个文档其实只是擦边提到，哪个才是真正回答了问题"），
# 这是向量距离这种纯数值排序做不到的。
print("\n=== LLMListwiseRerank：用 LLM 精确重排序 ===")
broad_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # 先粗召回4个候选
reranker = LLMListwiseRerank.from_llm(llm=llm, top_n=2)  # 重排序后只保留最相关的2个
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=broad_retriever,
)

query = "LangChain 有哪些核心组件"
before = broad_retriever.invoke(query)
after = rerank_retriever.invoke(query)
print(f"重排序前（粗召回 {len(before)} 个，按向量距离排序）:")
for d in before:
    print(f"  - {d.page_content[:40]}...")
print(f"重排序后（LLM 精排，只保留 {len(after)} 个最相关的）:")
for d in after:
    print(f"  - {d.page_content[:40]}...")


if __name__ == "__main__":
    pass


"""
执行流程图：

混合检索（EnsembleRetriever）：
查询 ──► 向量检索（语义相似度排序） ──┐
     └─► BM25检索（关键词词频排序） ──┤
                                       │ RRF（倒数排名融合）按权重合并两份排序
                                       ▼
                              融合后的最终排序结果

重排序（LLMListwiseRerank）：
查询 ──► 普通 retriever 粗召回较多候选（如 k=4）
              │
              │ LLM 阅读"问题 + 全部候选文档"，给出真实相关性排序
              ▼
        只保留 top_n 个最相关的（如 top_n=2）


核心知识点 ★：

★ 向量检索和 BM25 是互补的：向量检索擅长"语义相近但用词不同"的情况，
  BM25 擅长"精确关键词/数字/专有名词"命中，混合使用能覆盖两者各自的弱点。
★ 【实测查漏】BM25Retriever 默认分词在中文上基本失效：default_preprocessing_func
  只按空格切词，中文句子内部没有空格，整句会被当成一个 token，词频统计形同虚设。
  解决：用 jieba 等中文分词库自定义 preprocess_func 参数传给 BM25Retriever.from_documents()。
  这是中文 RAG 项目里最容易被忽略、又最容易让 BM25"看起来在跑、其实没用"的坑。
★ EnsembleRetriever 用 RRF（倒数排名融合）合并多个检索器的结果，不是简单拼接，
  weights 参数控制每个检索器的影响力占比。
★ LLMListwiseRerank 的核心思路：先用便宜的方式（向量检索）粗召回一批候选，
  再用更贵但更准的方式（LLM 阅读理解）精排，是"先广撒网、再精筛选"的经典套路。
★ ContextualCompressionRetriever 是个"包装器"：把任意 compressor（重排序器/过滤器）
  和 base_retriever 组合成一个新的标准 Retriever，可以无缝接入第7课学的 RAG 链，
  不需要改动 RAG 链本身的代码。
★ 选择建议：知识库不大、对成本不敏感 → 直接上重排序；中文关键词/专有名词较多的场景
  → 一定要给 BM25 配中文分词，否则它基本不起作用。
"""
