"""
第9课：Document Loaders —— 把真实数据接入 RAG

为什么需要这一课：第7、8课的"知识库"其实是一段手写的 Python 字符串（raw_text），
但真实项目里数据躺在 PDF、网页、CSV、一堆本地文件里，不会是现成的字符串。
Document Loaders 就是 LangChain 里"把各种格式的原始数据，统一转换成 Document 对象"的那一层。
只要变成了 Document（page_content + metadata），后面分块、向量化、检索的代码跟第7课完全一样。

学习要点：
1. Document 对象再回顾：page_content（正文）+ metadata（来源、页码等附加信息）
2. PyPDFLoader —— 加载 PDF，按页拆分成多个 Document，metadata 自动带页码
3. CSVLoader —— 每一行变成一个 Document，列名自动拼进 page_content
4. DirectoryLoader —— 批量加载一个文件夹里的所有匹配文件
5. WebBaseLoader —— 加载网页，用 BeautifulSoup 提取正文文字
6. 多来源数据合并后，复用第7课的 splitter + FAISS，构建一条真实可用的端到端 RAG 管道

Python 小贴士（给新手）：
- 本课会用 reportlab 现造一个 PDF、用 csv 模块现造一个 CSV，这样不用你手动准备文件就能跑通全部代码
- 会用标准库 http.server 在本地启动一个"迷你网页服务器"，只是为了让 WebBaseLoader 有网页可抓；
  真实项目里 WebBaseLoader 直接指向你想抓的真实网址即可，不需要自己起服务器
"""

import os
import csv
import time
import threading
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from dotenv import load_dotenv
from openai import OpenAI
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
    TextLoader,
    WebBaseLoader,
)

load_dotenv()
# WebBaseLoader 底层用 requests 发 HTTP 请求，没设置 USER_AGENT 会有警告（不影响功能，但设置一下更规范）
os.environ.setdefault("USER_AGENT", "langchain-lesson09-demo")


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

DATA_DIR = "lesson09_data"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(f"{DATA_DIR}/docs", exist_ok=True)


# ─────────────────────────────────────────────
# 0. 准备示例数据（PDF / CSV / 多个txt文件 / 网页）
# ─────────────────────────────────────────────
# 0-1. 生成一个 2 页的示例 PDF（用 reportlab 现造，真实项目里这一步不需要，PDF 本来就在你手上）
print("=== 准备示例数据 ===")
from reportlab.pdfgen import canvas

pdf_path = f"{DATA_DIR}/langchain_intro.pdf"
c = canvas.Canvas(pdf_path)
c.drawString(80, 750, "LangChain Introduction - Page 1")
c.drawString(80, 720, "LangChain is an open-source framework for building LLM applications.")
c.drawString(80, 700, "It was created by Harrison Chase in October 2022.")
c.showPage()  # 翻到第二页
c.drawString(80, 750, "LangChain Introduction - Page 2")
c.drawString(80, 720, "Core components include LCEL, Prompt Template, Memory, Tools and RAG.")
c.save()
print(f"已生成 PDF: {pdf_path}")

# 0-2. 生成一个 FAQ 形式的 CSV
csv_path = f"{DATA_DIR}/faq.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question", "answer"])
    writer.writerow(["什么是 LCEL", "LangChain Expression Language，用 | 管道符组合各组件的表达式语言"])
    writer.writerow(["什么是向量数据库", "存储 Embedding 向量并支持相似度检索的数据库，如 FAISS、Chroma"])
    writer.writerow(["Agent 和普通链有什么区别", "Agent 由 LLM 自主决定下一步做什么，链则是固定的执行顺序"])
print(f"已生成 CSV: {csv_path}")

# 0-3. 生成几个 txt 文件，模拟"团队共享文件夹"
with open(f"{DATA_DIR}/docs/note_memory.txt", "w", encoding="utf-8") as f:
    f.write("笔记：Memory 模块的关键是 session_id，用它区分不同用户的对话历史。")
with open(f"{DATA_DIR}/docs/note_agent.txt", "w", encoding="utf-8") as f:
    f.write("笔记：Agent 的核心循环是 推理 -> 调用工具 -> 观察结果 -> 再推理，直到给出最终答案。")
print("已生成 docs/ 文件夹下的 2 个 txt 笔记")

# 0-4. 生成一个网页文件，准备用本地服务器把它"伪装"成一个真实网站
html_content = """
<html><body>
<h1>LangChain 官方文档摘要（示例页面）</h1>
<p>RAG（检索增强生成）通过先检索相关文档、再让 LLM 基于检索结果回答问题，
解决了大模型无法访问私有知识和最新信息的问题。</p>
</body></html>
"""
with open(f"{DATA_DIR}/page.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("已生成示例网页 page.html")


# ─────────────────────────────────────────────
# 1. PyPDFLoader —— 加载 PDF，按页拆分
# ─────────────────────────────────────────────
print("\n=== PyPDFLoader ===")
pdf_loader = PyPDFLoader(pdf_path)
pdf_docs = pdf_loader.load()
print(f"PDF 共 {len(pdf_docs)} 页，每页是一个 Document")
for d in pdf_docs:
    # metadata 里自带 page（页码，从0开始）和 source（文件路径），溯源全靠它
    print(f"  page={d.metadata['page']} | {d.page_content[:50]}...")


# ─────────────────────────────────────────────
# 2. CSVLoader —— 每一行变成一个 Document
# ─────────────────────────────────────────────
print("\n=== CSVLoader ===")
csv_loader = CSVLoader(csv_path, encoding="utf-8")
csv_docs = csv_loader.load()
print(f"CSV 共 {len(csv_docs)} 行，每行是一个 Document")
for d in csv_docs:
    # page_content 自动把"列名: 值"拼接起来，metadata 里的 row 记录是第几行（从0开始）
    print(f"  row={d.metadata['row']} | {d.page_content!r}")


# ─────────────────────────────────────────────
# 3. DirectoryLoader —— 批量加载一个文件夹
# ─────────────────────────────────────────────
print("\n=== DirectoryLoader ===")
# glob="*.txt"：只加载这个扩展名的文件；loader_cls 指定文件夹里每个文件具体用哪个 Loader 来读
dir_loader = DirectoryLoader(
    f"{DATA_DIR}/docs",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
dir_docs = dir_loader.load()
print(f"文件夹里共加载 {len(dir_docs)} 个文件")
for d in dir_docs:
    print(f"  source={d.metadata['source']} | {d.page_content}")


# ─────────────────────────────────────────────
# 4. WebBaseLoader —— 加载网页
# ─────────────────────────────────────────────
# WebBaseLoader 本质是：用 requests 发 HTTP GET，再用 BeautifulSoup 把 HTML 标签去掉，只留正文文字。
# 真实项目里直接 WebBaseLoader("https://真实网址") 即可；这里为了让代码不依赖外部网络、
# 任何环境都能跑通，临时在本机起一个"迷你网页服务器"把 page.html 发布出去。
print("\n=== WebBaseLoader ===")

# ThreadingHTTPServer 是 Python 标准库自带的简易 HTTP 服务器；
# 用 threading.Thread(daemon=True) 把它放到后台线程跑，daemon=True 表示
# "主程序结束时这个线程也跟着结束"，不会让脚本卡住退不出去。
handler_cls = partial(SimpleHTTPRequestHandler, directory=DATA_DIR)
local_server = ThreadingHTTPServer(("127.0.0.1", 8943), handler_cls)
server_thread = threading.Thread(target=local_server.serve_forever, daemon=True)
server_thread.start()
time.sleep(0.3)  # 给服务器一点启动时间，避免请求发得太快连不上

web_loader = WebBaseLoader("http://127.0.0.1:8943/page.html")
web_docs = web_loader.load()
print(f"网页加载到 {len(web_docs)} 个 Document")
print(f"  {web_docs[0].page_content.strip()[:80]}...")

local_server.shutdown()  # 用完关掉本地服务器


# ─────────────────────────────────────────────
# 5. 合并多来源数据，构建一条真实的端到端 RAG 管道
# ─────────────────────────────────────────────
print("\n=== 合并多来源数据，构建向量库 ===")
all_docs: list[Document] = pdf_docs + csv_docs + dir_docs + web_docs
print(f"四种来源合计 {len(all_docs)} 个 Document（PDF{len(pdf_docs)} + CSV{len(csv_docs)} + txt{len(dir_docs)} + 网页{len(web_docs)}）")

# 复用第7课的分割器：不同来源的 Document 格式统一后，分块逻辑完全不需要关心数据原来是什么格式
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = splitter.split_documents(all_docs)  # split_documents：对已有 Document 列表分块，会自动保留原 metadata
print(f"分块后共 {len(chunks)} 块（注意每块的 metadata 仍然带着来源信息）")
for ch in chunks[:3]:
    print(f"  metadata={ch.metadata} | {ch.page_content[:40]}...")

multi_source_vectorstore = FAISS.from_documents(chunks, embeddings)
multi_source_vectorstore.save_local("faiss_index_multisource")
print(f"\n向量库构建完成并保存到 ./faiss_index_multisource/，共 {multi_source_vectorstore.index.ntotal} 条向量")

# 验证检索：问一个只在 CSV 里出现的问题，看是否能从正确来源召回
result_docs = multi_source_vectorstore.similarity_search("Agent 和链有什么区别", k=1)
print(f"\n检索验证: {result_docs[0].page_content} (来源 metadata: {result_docs[0].metadata})")


if __name__ == "__main__":
    pass


"""
执行流程图：

PDF文件 ──PyPDFLoader──┐
CSV文件 ──CSVLoader────┤
txt文件夹──DirectoryLoader─┤──► 统一变成 [Document, Document, ...]（page_content + metadata）
网页 ──WebBaseLoader────┘                       │
                                                  │ RecursiveCharacterTextSplitter.split_documents()
                                                  ▼
                                          [chunk_0, chunk_1, ...]（保留原 metadata，可溯源）
                                                  │ embeddings.embed_documents()
                                                  ▼
                                          FAISS.from_documents() → 向量库
                                                  │
                                                  ▼
                                    后续检索/RAG链 —— 跟第7课代码完全一样


核心知识点 ★：

★ Document Loaders 的本质：把任意格式的原始数据，统一转换成 Document(page_content, metadata) 对象。
  一旦变成了 Document，后面分块/向量化/检索的代码不需要关心数据原来是 PDF 还是网页还是 CSV。
★ PyPDFLoader 按页拆分：每一页是一个独立 Document，metadata 里的 page 字段就是页码（从0开始）。
★ CSVLoader 按行拆分：每一行是一个独立 Document，page_content 自动把"列名: 值"拼接成文字。
★ DirectoryLoader 是"批量加载器"，本身不解析内容，靠 loader_cls 参数指定具体用哪个 Loader 读每个文件。
★ WebBaseLoader 底层是 requests + BeautifulSoup：发 HTTP 请求拿到 HTML，再把标签去掉只留正文。
  真实项目直接传真实网址即可，本课用本地服务器只是为了让代码不依赖外部网络、随时能跑通。
★ split_documents() vs create_documents()：
  - create_documents([原始字符串]) 用于"手头只有裸字符串"的情况（第7课）
  - split_documents([Document, ...]) 用于"已经是 Document 列表"的情况（本课），会自动保留 metadata
★ metadata 是溯源的关键：合并多个来源后，每个 chunk 仍然带着"我来自哪个文件/哪一页/哪一行"的信息，
  生产环境里回答问题时可以把来源展示给用户，增加可信度。
"""
