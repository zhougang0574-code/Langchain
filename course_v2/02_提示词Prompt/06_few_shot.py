"""
【02 提示词Prompt / 06】Few-shot 提示 —— 给模型几个例子，让它照葫芦画瓢〔进阶〕
========================================================================
前几课的 prompt 都只用「文字描述」告诉模型要干嘛（zero-shot）。但有些任务光靠
描述说不清楚——输出格式很特殊、风格要统一、判断标准很微妙。这时给几个「输入→输出」
的范例，比写一大段说明更有效，这就是 few-shot（少样本）。

新概念（只有这一个）：
  FewShotChatMessagePromptTemplate —— 把一组「示例」自动展开成多轮 Human/AI 消息，
  插进最终 prompt 里。
    examples         一组示例 dict，如 {"input": ..., "output": ...}
    example_prompt   单个示例怎么排成消息（一条 human + 一条 ai）
  最终 prompt = System(规则) + 展开的示例对话 + 当前 Human 输入。

为什么有效：
  模型从示例里「归纳」出你要的格式和风格，往往比纯文字指令更准、更稳。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"), temperature=0,
)


# ── 1. 准备示例：把「公司名 → 拟人化吉祥物点子」这种特殊任务用例子说清楚 ────
examples = [
    {"input": "卖咖啡的连锁店", "output": "一只睡不醒但很努力的浣熊，抱着比自己还大的咖啡杯。"},
    {"input": "做云计算的公司", "output": "一朵戴着工程帽、随时长出更多手臂干活的卡通云。"},
]

# ── 2. 单个示例如何排成消息（一条 human + 一条 ai）──────────────────────────
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# ── 3. 把示例展开成「示范对话」────────────────────────────────────────────
few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# ── 4. 组装最终 prompt：规则 + 示例对话 + 当前输入 ─────────────────────────
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是品牌创意助手。根据公司业务，给一个吉祥物创意，风格要和示例保持一致：拟人化、有反差萌、一句话。"),
    few_shot,                       # ← 这里展开成多轮示范对话
    ("human", "{input}"),
])
chain = final_prompt | llm | StrOutputParser()


if __name__ == "__main__":
    # 先看看示例被展开成了什么消息（理解 few-shot 的本质）
    print("=== 展开后的消息列表（含示范对话）===")
    for m in final_prompt.invoke({"input": "做在线教育的公司"}).to_messages():
        print(f"  [{m.__class__.__name__}] {m.content}")

    print("\n=== 模型输出（会模仿示例的风格）===")
    print(chain.invoke({"input": "做在线教育的公司"}))


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  examples=[{input, output}, ...]
        │ FewShotChatMessagePromptTemplate 展开
        ▼
  [Human(例1输入), AI(例1输出), Human(例2输入), AI(例2输出)]   ← 示范对话
        │ 拼进最终 prompt
        ▼
  System(规则) + 示范对话 + Human(当前输入) → 模型照样子输出

★ 核心规律：
  Few-shot = 用「输入→输出」范例替代/补充文字说明，让模型从例子里归纳格式与风格。
  范例被展开成多轮 Human/AI 消息插进 prompt——本质还是〔02/03〕「往模板里塞消息」。

  什么时候用：格式特殊、风格要统一、判断标准微妙、zero-shot 不稳时。
  例子选「有代表性」的 2~5 个即可；例子太多会占 token，也可能让模型过拟合到某个例子。
  （示例很多、要按输入动态挑最相关的几个，可再上「示例选择器」example selector。）
"""
