"""
第4课：输出解析器（Output Parsers）

学习要点：
1. StrOutputParser —— 提取 AIMessage.content 为字符串（复习）
2. JsonOutputParser —— 解析 LLM 输出的 JSON，返回 Python dict/list
3. PydanticOutputParser —— 解析为 Pydantic 模型实例，自动类型校验
4. get_format_instructions() —— 告诉 LLM 应该输出什么格式
5. 三种解析器的适用场景与区别
"""

import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,  # 结构化输出任务用 temperature=0，结果更稳定
)


# ─────────────────────────────────────────────
# 1. StrOutputParser（复习）
# ─────────────────────────────────────────────
str_chain = (
    ChatPromptTemplate.from_template("用一句话介绍 {topic}")
    | llm
    | StrOutputParser()
)
print("=== StrOutputParser ===")
print(str_chain.invoke({"topic": "LangChain"}))


# ─────────────────────────────────────────────
# 2. JsonOutputParser —— 无 schema，灵活解析
# ─────────────────────────────────────────────
# get_format_instructions() 返回一段提示文本，告诉 LLM 输出合法 JSON
json_parser = JsonOutputParser()

json_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个数据提取助手，严格按要求输出 JSON。"),
    ("human", "{query}\n\n{format_instructions}"),
]).partial(format_instructions=json_parser.get_format_instructions())
# partial 预填 format_instructions，invoke 时只需传 query

json_chain = json_prompt | llm | json_parser

print("\n=== JsonOutputParser ===")
result = json_chain.invoke({
    "query": "提取以下信息为 JSON：姓名张三，年龄28，城市北京，技能Python和Go"
})
print(type(result), result)  # <class 'dict'>


# ─────────────────────────────────────────────
# 3. PydanticOutputParser —— 有 schema，类型安全
# ─────────────────────────────────────────────
# 先定义 Pydantic 模型，字段描述越清晰 LLM 输出越准确
class Movie(BaseModel):
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")
    genres: List[str] = Field(description="类型标签列表，如 ['动作', '科幻']")
    rating: float = Field(description="评分，0-10 的浮点数")

# PydanticOutputParser 根据 Movie 的 schema 自动生成格式指令
pydantic_parser = PydanticOutputParser(pydantic_object=Movie)

pydantic_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个电影数据库助手。"),
    ("human", "给我推荐一部科幻电影并按格式输出。\n\n{format_instructions}"),
]).partial(format_instructions=pydantic_parser.get_format_instructions())

pydantic_chain = pydantic_prompt | llm | pydantic_parser

print("\n=== PydanticOutputParser ===")
movie = pydantic_chain.invoke({})
print(type(movie))        # <class '__main__.Movie'>
print(movie)              # Movie(title=..., year=..., ...)
print("标题:", movie.title)
print("类型:", movie.genres)
print("评分:", movie.rating)


# ─────────────────────────────────────────────
# 4. 对比 get_format_instructions() 输出
# ─────────────────────────────────────────────
print("\n=== format_instructions 对比 ===")
print("-- JsonOutputParser --")
print(JsonOutputParser().get_format_instructions())

print("\n-- PydanticOutputParser --")
print(pydantic_parser.get_format_instructions())
# Pydantic 版本会包含 JSON schema，更精确


# ─────────────────────────────────────────────
# 5. JsonOutputParser 支持流式（JsonOutputParser 特有能力）
# ─────────────────────────────────────────────
# 注意：PydanticOutputParser 不支持流式，因为必须拿到完整 JSON 才能实例化
print("\n=== JsonOutputParser stream ===")
for chunk in json_chain.stream({
    "query": "给我一个包含 name/age/city 字段的 JSON"
}):
    print(chunk)  # 每个 chunk 是逐步填充的 dict（部分解析）


if __name__ == "__main__":
    pass


"""
执行流程图：

              用户输入
                 │
                 ▼
        ChatPromptTemplate
     (含 format_instructions)
                 │
                 ▼
            ChatOpenAI
          (temperature=0)
                 │
                 ▼
         LLM 输出的文本

    ┌────────────┼────────────┐
    ▼            ▼            ▼
StrOutputParser  JsonOutputParser  PydanticOutputParser
    │            │                  │
    ▼            ▼                  ▼
  str          dict/list        Pydantic 实例
                                (类型已校验)


核心知识点 ★：

★ temperature=0 在结构化输出任务中更可靠，减少 LLM 乱输格式的概率
★ get_format_instructions() 是解析器和 Prompt 之间的桥梁，必须注入 Prompt
★ JsonOutputParser：无 schema，灵活，支持流式；适合结构不固定的场景
★ PydanticOutputParser：有 schema，类型安全，不支持流式；适合结构固定的场景
★ partial() 预填 format_instructions，使 invoke 时接口更干净
★ Pydantic 字段的 Field(description=...) 直接影响 LLM 输出质量，写清楚描述很重要
"""
