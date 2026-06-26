"""
第10课：结构化输出进阶 —— with_structured_output()

为什么是这一课：第4课学的 PydanticOutputParser 属于"提示词流派"——
靠在 Prompt 里塞一段"请按这个 JSON 格式输出"的说明（get_format_instructions()），
模型自己用文本生成 JSON，再用 Parser 把文本解析成 Pydantic 对象。
这套方式有个隐患：LLM 是在"写文本"，偶尔会漏个引号、多个逗号，导致解析失败。

with_structured_output() 是更稳的"工具调用流派"——本质是把你的 Pydantic 模型
转换成一个工具定义（就是第6课学的 tool），强制 LLM 通过 tool_calls 这个结构化通道
返回参数，根本不会有"格式没写对"的问题，因为这不是模型在"写文本"，
而是模型在"填参数表单"。

学习要点：
1. with_structured_output() 的两种底层机制：function_calling（走 tool_calls）和 json_mode/json_schema
2. method 参数：显式指定走哪种机制，不指定时 LangChain 会按模型能力自动选择
3. 嵌套 Pydantic 模型、Optional 可选字段、Enum 枚举字段、List 列表字段的写法
4. include_raw=True —— 同时拿到"原始 AIMessage"和"解析后的对象"，调试利器
5. 对比第4课：同一个任务，两种方式的代码量和稳定性差异

Python 小贴士（给新手）：
- Optional[X] 等价于 "X 或者 None"，表示这个字段可以不填
- Enum（枚举）用来限定一个字段只能是几个固定值之一，比 str 自由文本更不容易出歧义
- List[str] 表示"字符串组成的列表"，这些都是 Python 的类型注解（type hints），
  本身不会在运行时强制检查，但 Pydantic 会在创建对象时真正校验这些类型
"""

import os
from enum import Enum
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# ─────────────────────────────────────────────
# 1. 回顾第4课的方式：PydanticOutputParser（提示词流派）
# ─────────────────────────────────────────────
class SimpleAnswer(BaseModel):
    answer: str = Field(description="问题的答案")


print("=== 回顾：PydanticOutputParser（第4课的方式）===")
parser = PydanticOutputParser(pydantic_object=SimpleAnswer)
old_prompt = ChatPromptTemplate.from_messages([
    ("system", "回答用户问题。\n{format_instructions}"),
    ("human", "{question}"),
]).partial(format_instructions=parser.get_format_instructions())
old_chain = old_prompt | llm | parser
old_result = old_chain.invoke({"question": "1+1等于几"})
print(f"结果: {old_result}")
print("⚠️ 这种方式依赖 LLM 把 JSON 文本写对，需要在 Prompt 里塞一长段格式说明（见上面 format_instructions）")


# ─────────────────────────────────────────────
# 2. with_structured_output()（工具调用流派）—— 同一个任务，代码更短
# ─────────────────────────────────────────────
print("\n=== with_structured_output（本课的方式）===")
# 不需要写 Prompt，不需要 get_format_instructions()，直接把 Pydantic 模型传给 llm
structured_llm = llm.with_structured_output(SimpleAnswer)
new_result = structured_llm.invoke("1+1等于几")
print(f"结果: {new_result}")
print("✅ 少了 Prompt 里那一大段格式说明，因为 LLM 不是在'写JSON文本'，而是在'填参数表单'")


# ─────────────────────────────────────────────
# 3. method 参数：看清楚底层到底走的是哪条路
# ─────────────────────────────────────────────
# with_structured_output 默认会按模型能力自动选择机制（OpenAI 兼容接口通常用 json_schema/json_mode）。
# 显式传 method="function_calling"，可以强制走第6课学过的 tool_calls 通道，方便对照原理。
print("\n=== 用 include_raw + method 看清楚底层机制 ===")
debug_llm = llm.with_structured_output(SimpleAnswer, method="function_calling", include_raw=True)
debug_result = debug_llm.invoke("3+3等于几")
print(f"raw.content: {debug_result['raw'].content!r}（走 tool_calls 时这里通常是空字符串）")
print(f"raw.tool_calls: {debug_result['raw'].tool_calls}")
print(f"parsed（最终解析结果）: {debug_result['parsed']}")

default_debug_llm = llm.with_structured_output(SimpleAnswer, include_raw=True)  # 不指定 method，用默认机制
default_debug_result = default_debug_llm.invoke("3+3等于几")
print(f"\n默认机制 raw.content: {default_debug_result['raw'].content!r}（走 json 模式时这里是一段JSON文本）")
print(f"默认机制 raw.tool_calls: {default_debug_result['raw'].tool_calls}（这里通常是空列表）")


# ─────────────────────────────────────────────
# 4. 综合实战：嵌套模型 + Optional + Enum + List
# ─────────────────────────────────────────────
class Priority(str, Enum):
    # 枚举：限定 priority 字段只能是这三个值之一，LLM 不会随便造一个 "比较急" 出来
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Address(BaseModel):
    city: str = Field(description="城市")
    district: Optional[str] = Field(default=None, description="区/县，文本中没提到就留空")


class SupportTicket(BaseModel):
    title: str = Field(description="工单标题，简要概括问题")
    priority: Priority = Field(description="根据描述的紧急程度判断优先级")
    tags: List[str] = Field(description="问题相关的标签列表")
    address: Optional[Address] = Field(default=None, description="涉及的地址信息，文本中没有就留空")


print("\n=== 综合实战：客服工单结构化提取 ===")
ticket_llm = llm.with_structured_output(SupportTicket)
ticket = ticket_llm.invoke(
    "用户反馈：上海徐汇区的门店收银系统崩溃了，非常紧急，影响所有顾客结账。"
)
print(f"标题: {ticket.title}")
print(f"优先级: {ticket.priority}（类型: {type(ticket.priority)}）")
print(f"标签: {ticket.tags}")
print(f"地址: {ticket.address}")

# 再测一条没有提到具体地址的描述，验证 Optional 字段确实能正常留空
print("\n--- 测试 Optional 字段为空的情况 ---")
ticket2 = ticket_llm.invoke("用户反馈网站登录按钮点击没反应，不是很急。")
print(f"标题: {ticket2.title} | 优先级: {ticket2.priority} | 地址: {ticket2.address}")
print("⚠️ 实测发现：这里地址字段往往不是 None，而是被编出了一个看似合理的城市/区县！")
print("   说明 Optional + Field(description='没提到就留空') 这种'温和提示'不够强，")
print("   LLM 仍然可能在信息缺失时编造一个值，而不是诚实地返回 null。")

# 修复方法：把约束从"字段描述里的温和提示"升级成"system message 里的明确指令"。
# with_structured_output 不影响你正常传 messages 列表，只是最后一步多了"强制结构化"，
# 所以完全可以像第1课那样用 SystemMessage 给出更强的行为约束。
print("\n--- 修复：用 SystemMessage 明确禁止编造 ---")
from langchain_core.messages import SystemMessage, HumanMessage

fixed_result = ticket_llm.invoke([
    SystemMessage(content=(
        "提取信息时，如果原文没有明确提到某个字段的具体内容，必须将该字段设为 null。"
        "绝对不要编造或猜测任何没有出现在原文中的具体值（如城市名、地区名）。"
    )),
    HumanMessage(content="用户反馈网站登录按钮点击没反应，不是很急。"),
])
print(f"标题: {fixed_result.title} | 地址: {fixed_result.address}（这次应为 None）")


if __name__ == "__main__":
    pass


"""
执行流程图：

PydanticOutputParser（第4课，提示词流派）：
  Pydantic模型 → get_format_instructions() → 拼进 Prompt
       │
       ▼
  LLM 生成一段"看起来像JSON"的文本
       │
       │ parser.parse() 用正则/json.loads 解析文本
       ▼
  Pydantic 对象（解析失败就报错）

with_structured_output（本课，工具调用流派）：
  Pydantic模型 → 自动转换成一个 tool 定义（字段=参数，Field描述=参数说明）
       │
       │ bind_tools() 绑定给 LLM（第6课学过的机制）
       ▼
  LLM 直接在 tool_calls 里返回结构化参数（或走 json_schema 强制约束输出）
       │
       │ LangChain 自动把参数还原成 Pydantic 对象
       ▼
  Pydantic 对象（机制上更难"格式跑偏"）


核心知识点 ★：

★ with_structured_output 的本质：把 Pydantic 模型转换成一个 tool 定义，
  让 LLM 像调用工具一样"填参数"，而不是像写作文一样"写JSON文本"——这是它比
  PydanticOutputParser 更稳定的根本原因。
★ method 参数控制走哪条路："function_calling"（tool_calls 通道，第6课机制的复用）、
  "json_schema"/"json_mode"（让模型在 JSON 约束下直接生成文本）。
  不指定时 LangChain 会按模型支持情况自动选最合适的。
★ include_raw=True 调试技巧：能同时拿到原始 AIMessage（看 content/tool_calls 长什么样）
  和解析后的 Pydantic 对象，定位"结构化失败到底是哪一步出问题"很有用。
★ Enum 枚举字段：把一个字段限定在几个固定值之内，比让 LLM 自由写字符串更可控，
  避免出现"比较急"、"挺紧急的"这种无法被下游代码处理的自由文本。
★ 【实测查漏】Optional[X] + Field(description="没提到就留空") 不是可靠的防编造手段：
  实测发现 LLM 仍可能在信息完全缺失时编出一个看似合理的值（如编一个城市名）。
  字段描述（Field description）的约束力比 system message 弱很多，真正需要"绝对不编造"
  这种强约束时，要把指令写进 SystemMessage，而不是寄望于温和的字段描述。
  这也提醒我们：结构化输出只保证"形状对"（类型、字段都符合 Schema），
  不保证"内容真实"，这是 LLM 生成的本质限制，不是 LangChain 的 bug。
★ 嵌套 Pydantic 模型（Address 嵌进 SupportTicket）：LLM 能正确处理多层结构，
  这意味着信息抽取任务可以设计成贴近业务实体的层级结构，而不必拍平成一堆字段。
★ 什么时候还用 PydanticOutputParser：调用一个不支持 tool calling 的模型/接口时，
  prompt-instruction 流派是唯一选择；只要模型支持 tool calling，
  with_structured_output 几乎总是更好的默认选择。
"""
