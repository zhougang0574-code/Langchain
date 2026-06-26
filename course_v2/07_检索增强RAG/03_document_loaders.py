"""
【07 检索增强RAG / 03】文档加载器 —— 把各种格式读成统一的 Document
==============================================================
〔07/02〕的知识是我们手敲进 .txt 的。真实场景里资料是 PDF、Word、CSV、Markdown……
各种格式。LangChain 为每种格式提供了「加载器」，统一读成 Document 对象。

新概念（只有这一个）：
  各种 DocumentLoader —— 把某种格式的文件读成 list[Document]。
    Document 有两个核心属性：
      .page_content  正文文本
      .metadata      来源信息（文件名、页码、行号等）
    常用加载器：TextLoader / PyPDFLoader / CSVLoader / UnstructuredMarkdownLoader /
              Docx2txtLoader …… 名字不同，但 .load() 都返回 list[Document]。

为什么重要：
  「统一成 Document」是 RAG 流水线的入口——无论原始格式是什么，后续切分、
  向量化、检索都只跟 Document 打交道。
"""

import os

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
)

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")


def show(title: str, docs):
    print(f"\n=== {title}：{len(docs)} 个 Document ===")
    d = docs[0]
    print("page_content[:60]:", d.page_content[:60].replace("\n", " "))
    print("metadata         :", d.metadata)


if __name__ == "__main__":
    # ── txt：最简单，整文件一个 Document ───────────────────────────────────
    show("TextLoader (.txt)", TextLoader(os.path.join(A, "sample.txt"), encoding="utf-8").load())

    # ── pdf：PyPDFLoader 按「页」拆，每页一个 Document，metadata 带页码 ─────
    show("PyPDFLoader (.pdf)", PyPDFLoader(os.path.join(A, "sample.pdf")).load())

    # ── csv：CSVLoader 按「行」拆，每行一个 Document ───────────────────────
    show("CSVLoader (.csv)", CSVLoader(os.path.join(A, "sample.csv"), encoding="utf-8").load())

    # ── docx：Docx2txtLoader（需要 pip install docx2txt）──────────────────
    try:
        show("Docx2txtLoader (.docx)", Docx2txtLoader(os.path.join(A, "sample.docx")).load())
    except Exception as e:
        print(f"\n=== Docx2txtLoader (.docx)：跳过 ===\n  需要 docx2txt 库：pip install docx2txt（{type(e).__name__}）")

    print("\n提示：Markdown/HTML 等可用 langchain_community 里对应的 Unstructured*Loader，")
    print("      都遵循同一个约定：loader.load() → list[Document]。")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  各种格式文件
   .txt  → TextLoader
   .pdf  → PyPDFLoader（按页拆）
   .csv  → CSVLoader（按行拆）
   .docx → Docx2txtLoader
        │  .load()
        ▼
  list[Document]（page_content + metadata）  ← 统一入口，后续只认 Document

★ 核心规律：
  不管原始格式是什么，加载器都把它统一成 list[Document]，下游流程不用关心来源格式。
  不同加载器的「拆分粒度」不同：PDF 按页、CSV 按行、txt 整篇——会影响后续切分策略。

  metadata 很有用：检索到答案时能溯源「来自哪个文件、第几页」，是做引用标注的基础。
  部分加载器要装额外库（docx2txt、unstructured 等），见根目录 requirements.txt。
"""
