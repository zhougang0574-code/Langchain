"""
【03 输出解析 / 05】TypedDict 定义结构〔进阶〕
============================================
〔03/03〕〔03/04〕用 Pydantic 的 BaseModel 定义结构。但有时你不想引入「对象」，
只想要一个普通 dict，又希望把「有哪些字段、什么类型、什么含义」写清楚。
TypedDict 就是「带类型标注的 dict」。

新概念（只有这一个）：
  TypedDict + Annotated —— 另一种给 with_structured_output 定义结构的方式。
    class X(TypedDict):
        字段: Annotated[类型, "给模型看的字段说明"]
  结果拿到的是普通 dict（不是对象），用 d["字段"] 访问。

什么时候用它而不是 Pydantic：
  想要轻量、就是个 dict、不需要 field_validator 这类校验逻辑时。
  需要校验 / 对象方法时，仍用 Pydantic〔03/03〕。
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# ── 1. 用 TypedDict + Annotated 定义结构 ───────────────────────────────────
# Annotated[类型, "说明"]：第一个是类型，第二个字符串是给模型看的字段描述
class MovieReview(TypedDict):
    title: Annotated[str, "电影名"]
    rating: Annotated[float, "评分，0-10"]
    summary: Annotated[str, "一句话短评"]
    recommend: Annotated[bool, "是否推荐"]


# ── 2. 同样用 with_structured_output，传 TypedDict 类 ──────────────────────
structured_llm = llm.with_structured_output(MovieReview)


if __name__ == "__main__":
    result = structured_llm.invoke("点评电影《盗梦空间》")
    print("=== 拿到的是普通 dict（不是对象）===")
    print(type(result).__name__)             # dict
    print(result)
    print("\n用 d['key'] 访问：rating =", result["rating"], "| recommend =", result["recommend"])


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  结构定义的三种方式（都可喂给 with_structured_output）：

    Pydantic BaseModel  → 返回「对象」，有类型校验/校验器/方法     〔03/03〕〔03/04〕
    TypedDict           → 返回「dict」，轻量、无校验逻辑           本课
    （还有 json schema 字典，最底层，少用）

★ 核心规律：
  TypedDict 是「带类型标注的 dict」：用 Annotated 给每个字段配「类型 + 说明」，
  with_structured_output 会把这些说明传给模型，结果以普通 dict 返回。

  选型口诀：要校验/对象方法 → Pydantic；只要个带类型的 dict、图轻量 → TypedDict。
"""
