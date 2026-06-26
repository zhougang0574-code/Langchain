"""
course_v2 共享工具
==================
旧版（archive_v1）的毛病：每个 RAG 文件都把同一个 Embeddings 类重抄一遍。
这里把「会被多课复用」的东西收拢到一处，演示工程上最基本的「复用」。

目前只放一样东西：阿里百炼的 Embeddings 封装。
- 阿里百炼的向量化接口走的是 OpenAI 兼容协议，所以可以直接用 openai 客户端调，
  不必额外安装 dashscope 这个 SDK。
- LangChain 的检索组件（FAISS、retriever 等）只认「实现了 Embeddings 接口的对象」，
  所以我们继承 langchain_core 的 Embeddings 抽象基类，实现两个方法即可接入整个生态。

各课的 LLM（ChatOpenAI）仍然在每个文件里就地创建——它只有 4 行，且是讲解的核心，
就地写更利于阅读；而 Embeddings 类有十几行样板，才值得抽出来复用。
"""

import os
from typing import List

from openai import OpenAI
from langchain_core.embeddings import Embeddings


class AliyunEmbeddings(Embeddings):
    """把阿里百炼的向量化接口包装成 LangChain 的 Embeddings 对象。

    Embeddings 接口要求实现两个方法：
      embed_documents(texts) -> 把「一批文档」变成「一批向量」（建库时用）
      embed_query(text)      -> 把「一条查询」变成「一个向量」（检索时用）
    """

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-v1"):
        # 复用 openai 客户端：百炼兼容 OpenAI 协议，base_url 指向 compatible-mode 即可
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        # 查询向量化复用 embed_documents，保证 query 和文档用同一个模型、落在同一向量空间
        return self.embed_documents([text])[0]


def get_embeddings(model: str = "text-embedding-v1") -> AliyunEmbeddings:
    """工厂函数：从环境变量读 key/base_url，建一个 Embeddings 对象。"""
    return AliyunEmbeddings(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=model,
    )
