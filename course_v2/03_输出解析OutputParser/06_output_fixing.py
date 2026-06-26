"""
【03 输出解析 / 06】OutputFixingParser —— 解析失败时自动修复〔进阶〕
==============================================================
〔03/03〕的 PydanticOutputParser 很严格：模型只要输出格式不对（少个引号、多句解释、
字段名写错），parser.invoke() 就直接抛异常。生产环境里模型偶尔就是会"手抖"输出脏数据，
为这点小毛病整条链报错不划算。

新概念（只有这一个）：
  OutputFixingParser.from_llm(parser=原解析器, llm=...) —— 给一个解析器套上"自动修复"。
    第一次按原 parser 解析；若失败，它把"出错的原文 + 期望格式"再发给 LLM，
    让 LLM 把脏输出改成合法格式，然后重新解析。对调用方透明。

和谁配合：
  包在任意结构化 parser（Pydantic/Json）外面。它解决的是"格式偶发错误"，
  类比〔09/01〕的 with_retry 之于网络抖动——都是"自动兜底，别为偶发问题报错"。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"), temperature=0,
)


class Actor(BaseModel):
    name: str = Field(description="演员姓名")
    films: list[str] = Field(description="代表作列表")


# ── 1. 原始严格解析器 ──────────────────────────────────────────────────────
base_parser = PydanticOutputParser(pydantic_object=Actor)

# ── 2. 套一层自动修复 ──────────────────────────────────────────────────────
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)


if __name__ == "__main__":
    # 故意构造一段"格式不合法"的脏输出：用了单引号、还夹带了解释文字
    bad_output = "这是结果：{'name': '周星驰', 'films': ['大话西游', '功夫']}"

    print("=== 直接用严格 parser：会抛异常 ===")
    try:
        base_parser.invoke(bad_output)
    except Exception as e:
        print(f"  ✗ 解析失败：{type(e).__name__}")

    print("\n=== 用 OutputFixingParser：自动调 LLM 修成合法格式 ===")
    actor = fixing_parser.invoke(bad_output)     # 内部：解析失败 → 让 LLM 修 → 重新解析
    print(f"  ✓ 修复成功：{type(actor).__name__} → name={actor.name}, films={actor.films}")


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  脏输出 "{'name': ...}"（单引号/夹带解释，非合法 JSON）
        │ OutputFixingParser.invoke
        ▼
  ① 先按原 parser 解析 ── 失败
        │
  ② 把「出错原文 + 期望格式」发给 LLM，让它改成合法格式
        ▼
  ③ 用修好的文本重新解析 ── 成功 → 返回 Actor 对象

★ 核心规律：
  OutputFixingParser 给结构化解析加「失败自动修复」：第一次失败不报错，而是让 LLM
  对照期望格式把脏输出改对再解析。包在 Pydantic/Json parser 外面即可。

  代价：修复要多一次 LLM 调用（失败时才触发）。它治"格式手抖"，不治"内容编造"——
  内容对不对是〔10 评估〕的事。能用 with_structured_output〔03/04〕从源头少出错时优先用它，
  仍可能出脏数据的链路再加 OutputFixingParser 兜底。
  （另有 RetryOutputParser：会带着「原始问题」重试，适合连问题一起才能修对的场景。）
"""
