"""
【03 输出解析 / 03】PydanticOutputParser —— 带类型校验的结构化输出
================================================================
〔03/02〕的 JsonOutputParser 给的是松散 dict：字段缺了、类型错了、值不合法，
它都照单全收。生产环境需要「拿到的数据一定合法」——这就要靠 Pydantic 模型。

新概念（只有这一个）：
  PydanticOutputParser(pydantic_object=你的模型类)
    - 用一个 Pydantic BaseModel 定义「期望的结构 + 字段类型 + 校验规则」
    - 解析时自动按模型校验：类型不对会转换，违反 field_validator 会报错
    - 返回的是「模型对象」，可以 .字段 访问，IDE 有补全

比 JsonOutputParser 多了什么：类型安全 + 字段校验 + 对象化访问。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# ── 1. 用 Pydantic 定义期望结构 ────────────────────────────────────────────
class Product(BaseModel):
    name: str = Field(description="产品名称")
    category: str = Field(description="产品类别")
    price: float = Field(description="价格，单位元")
    description: str = Field(description="一句话产品简介")

    # field_validator：自定义校验规则。description 太短就判为不合格，强制重视质量
    @field_validator("description")
    def desc_long_enough(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("description 至少 10 个字")
        return v


# ── 2. 建解析器，格式说明里会自动包含字段定义 ──────────────────────────────
parser = PydanticOutputParser(pydantic_object=Product)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是产品信息助手，只输出符合要求的 JSON。\n{format_instructions}"),
    ("human", "为「{topic}」编一条产品信息"),
]).partial(format_instructions=parser.get_format_instructions())


if __name__ == "__main__":
    filled = prompt.invoke({"topic": "华为 Mate 70 折叠屏手机"})
    raw = llm.invoke(filled)

    product = parser.invoke(raw)       # 解析 + 校验，返回 Product 对象
    print("=== 解析后是 Product 对象，不是 dict ===")
    print(type(product).__name__)
    print("name    :", product.name)         # 可以 .字段 访问，有类型
    print("price   :", product.price, type(product.price).__name__)  # float
    print("desc    :", product.description)
    print("\n转回 dict：", product.model_dump())


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  class Product(BaseModel): name/category/price/description + 校验规则
        │  PydanticOutputParser(pydantic_object=Product)
        │  .get_format_instructions() 自动据字段生成格式说明
        ▼
  prompt → llm → 文本 → parser.invoke() → Product 对象（已校验）

  JsonOutputParser   → dict（松散，不校验）
  PydanticOutputParser → 模型对象（强类型 + 字段校验 + .属性访问）

★ 核心规律：
  需要「数据一定合法、字段一定齐全」就用 Pydantic：模型定义即契约，
  price 写成 float 就保证转成数字，field_validator 不通过就直接抛错而不是默默放行。

  不过——这套「parser + 手动拼 format_instructions」是较经典的写法。
  现代 LangChain 更推荐 with_structured_output（下一课〔03/04〕），更省心，重点对照。
"""
