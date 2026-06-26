"""
【03 输出解析 / 04】with_structured_output —— 现代首选写法（重点对照）
==================================================================
〔03/03〕用 PydanticOutputParser 要做三件事：拼 format_instructions 进 prompt、
调模型、再 parser.invoke()。步骤多、容易漏。现代 LangChain 把这套封装成一个方法。

新概念（只有这一个）：
  llm.with_structured_output(你的Pydantic模型) —— 返回一个「保证吐结构化对象」的新模型。
  直接 .invoke(问题) 就拿到 Pydantic 对象，不用手动拼格式说明、不用单独挂 parser。

它和上一课的关系：
  底层目标一样（拿合法的结构化对象），但 with_structured_output 优先用模型的
  原生能力（function calling / json schema），通常比「prompt 里塞格式说明」更稳、更省事。
  →「要结构化输出，先想 with_structured_output；它满足不了的特殊场景再退回 parser。」
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# ── 1. 还是用 Pydantic 定义结构（和上一课同一个套路）─────────────────────
class Product(BaseModel):
    name: str = Field(description="产品名称")
    category: str = Field(description="产品类别")
    price: float = Field(description="价格，单位元")
    description: str = Field(description="一句话产品简介")


# ── 2. 一行：把普通模型变成「结构化输出模型」──────────────────────────────
# 注意：不需要 format_instructions，也不需要单独的 parser
structured_llm = llm.with_structured_output(Product)


# ── 3. 可以直接 invoke，也可以串进链 ───────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是产品信息助手。"),
    ("human", "为「{topic}」编一条产品信息"),
])
chain = prompt | structured_llm     # 链的输出直接就是 Product 对象


if __name__ == "__main__":
    print("=== 直接 invoke 结构化模型 ===")
    p = structured_llm.invoke("给我编一条「机械键盘」的产品信息")
    print(type(p).__name__, "→", p.model_dump())

    print("\n=== 串进链（prompt | structured_llm）===")
    p2 = chain.invoke({"topic": "降噪耳机"})
    print("name:", p2.name, "| price:", p2.price)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  写法对照（目标相同，都拿到 Product 对象）：

  〔03/03〕经典 parser 写法（3 步，手动）：
     prompt(含 format_instructions)  →  llm  →  PydanticOutputParser.invoke()

  〔03/04〕现代写法（1 步，封装好）：
     llm.with_structured_output(Product)  →  .invoke()  → Product

★ 核心规律：
  要结构化输出，默认首选 with_structured_output：少写格式说明、少挂 parser，
  且优先走模型原生的 function calling，稳定性通常更好。

  什么时候仍用 parser（〔03/02〕〔03/03〕）：模型/网关不支持原生结构化输出，
  或你需要对解析过程做特殊定制时。两者都掌握，按场景选。
"""
