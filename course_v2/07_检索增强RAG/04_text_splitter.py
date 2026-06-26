"""
【07 检索增强RAG / 04】文本切分 —— 把长文档切成适合检索的小块
=========================================================
〔07/03〕加载出来的 Document 可能很长（一篇文章、一整个 PDF）。直接整篇向量化有两个
问题：① 太长超出 embedding 限制；② 检索粒度太粗，一篇文章里只有一句相关也整篇命中。
所以要先「切块」。

新概念（只有这一个）：
  RecursiveCharacterTextSplitter —— 最常用的切分器。
    chunk_size      每块的目标大小（字符数）
    chunk_overlap   相邻块的重叠字符数（避免把一句话从中间切断、丢上下文）
    split_documents(docs)  把 list[Document] 切成更多更小的 Document

为什么叫 Recursive：
  它按一组分隔符（段落→句子→词）逐级尝试切分，尽量在「自然边界」断开，
  而不是粗暴地每 N 个字符一刀，能更好地保住语义完整。
"""

import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    # 加载知识库（整文件一个 Document）
    docs = TextLoader(os.path.join(HERE, "assets", "knowledge.txt"), encoding="utf-8").load()
    print(f"切分前：{len(docs)} 个 Document，长度 {len(docs[0].page_content)} 字符")

    # ── 切分：每块约 120 字，相邻块重叠 20 字 ──────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20,        # 重叠：避免关键句被切在两块的边界而丢失
    )
    chunks = splitter.split_documents(docs)

    print(f"切分后：{len(chunks)} 个小块\n")
    for i, c in enumerate(chunks[:4], 1):
        print(f"[块{i}] ({len(c.page_content)}字) {c.page_content[:50].strip()}...")

    # ── 直观看「重叠」：相邻两块结尾/开头会有一段重复 ──────────────────────
    if len(chunks) >= 2:
        print("\n=== 观察重叠（块1结尾 与 块2开头 有重复）===")
        print("块1结尾：...", chunks[0].page_content[-25:].strip())
        print("块2开头：", chunks[1].page_content[:25].strip(), "...")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  一个长 Document
        │ RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
        ▼
  [块1] [块2] [块3] ...   ← 每块约 120 字，相邻块有 20 字重叠

  chunk_overlap 的作用：
    无重叠： "...向量化存入向量库。│提问时检索最相关..."  ← 一句话被切两半，各自都不完整
    有重叠： 块1结尾和块2开头共享一段，关键句在某一块里是完整的

★ 核心规律：
  切分是 RAG 质量的关键旋钮：块太大→检索不精准、噪声多；块太小→上下文不完整。
  chunk_overlap 用「相邻块重叠」来对冲「边界切断语义」的风险。
  没有万能值，需按文档类型和实际检索效果调（RAG 优化里数据/分块的优先级很高）。

  切好的小块，接下来就送进向量库建索引（〔07/02〕的 from_texts/from_documents），
  下一课〔07/05〕把「加载→切分→建库→检索→生成」串成完整 RAG 链。
"""
