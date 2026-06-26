"""
【03 输出解析 / 02】JsonOutputParser —— 让模型吐结构化 JSON
=========================================================
〔03/01〕拿到的是一坨纯文本。但很多场景你要的是「结构化数据」：从评论里抽出
{情感, 关键词, 评分}，好让程序接着处理。光靠 prompt 说「输出 JSON」模型可能
夹带 ```json 代码块、多余解释，解析容易翻车。

新概念（只有这一个）：
  JsonOutputParser —— 两件事一起做：
    ① get_format_instructions()：生成一段「该怎么输出 JSON」的说明，塞进 prompt
    ② invoke()：把模型返回的文本解析成 Python dict（自动剥掉 ```json 包裹）

为什么需要 format_instructions：
  解析器和模型要「约定好格式」。format_instructions 就是这份约定，
  必须拼进 prompt，模型才知道按什么结构输出。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,                  # 结构化任务用 0，输出更稳定
)


# ── 1. 建解析器，拿到「格式说明」───────────────────────────────────────────
parser = JsonOutputParser()
print("=== format_instructions（会拼进 prompt）===")
print(parser.get_format_instructions())


# ── 2. 把格式说明塞进 prompt（用 partial 预填，见〔02/04〕）─────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个情感分析助手。\n{format_instructions}"),
    ("human", "分析这条评论，给出字段 sentiment（正/负/中）、score（0-1）、keywords（列表）：\n{review}"),
]).partial(format_instructions=parser.get_format_instructions())


if __name__ == "__main__":
    filled = prompt.invoke({"review": "这家店服务态度超好，菜也很好吃，就是有点贵。"})
    raw = llm.invoke(filled)
    print("\n=== 模型原始输出（可能带 ```json 包裹）===")
    print(repr(raw.content))

    result = parser.invoke(raw)       # 解析成 dict，自动剥掉包裹
    print("\n=== parser 解析后 ===")
    print(type(result).__name__, "→", result)
    print("可以直接当字典用：sentiment =", result["sentiment"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  parser.get_format_instructions()  ──拼进──→  prompt
                                                  │  llm
                                                  ▼
                                       "```json\n{...}\n```"  (带包裹的文本)
                                                  │  JsonOutputParser().invoke()
                                                  ▼
                                          {"sentiment": ..., ...}  (Python dict)

★ 核心规律：
  结构化输出 = 「format_instructions 进 prompt」+「parser 解析返回」两件事配套。
  少了前者，模型不知道该输出什么格式，解析就会失败。

  JsonOutputParser 给的是「松散」的 dict——它不校验字段是否齐全、类型对不对。
  要强约束字段和类型，用下一课的 PydanticOutputParser〔03/03〕。
"""
