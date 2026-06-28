"""
【06 工具与Agent / 05】反思 Agent —— 让模型批评并改进自己的输出〔进阶〕
=====================================================================
〔06/04〕Agent 调几个工具、一遍把答案跑出来就结束了。但「一遍过」的输出未必好。
本课引入「反思」：先生成草稿 → 让模型挑自己的毛病 → 据此修订，循环几轮，质量自我爬升。

新概念（只有这一个）：
  Reflection（反思）模式 —— 用两个角色循环：
    生成者(generate) 产出草稿 → 评审者(reflect) 指出不足 → 生成者据反馈改写。
  它不是新 API，是一种「用消息/链编排出来的」Agent 设计模式（手搓即可），
  和〔06/02〕手搓工具循环一样，重点在看清模式本身。

为什么需要：
  复杂任务（写作、改代码、推理）里，模型第一版常有疏漏。给它一个「自我批评再重写」
  的循环，往往比单次生成显著更好——这是 Reflexion / Self-Refine 这类方法的核心思路。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# ── 1. 生成者：给定任务（可带上一轮的批评）产出/改写草稿 ─────────────────────
writer = ChatPromptTemplate.from_messages([
    ("system", "你是一名严谨的技术作者。根据任务写作；若给了「修改意见」，就按意见重写。"),
    ("human", "任务：{task}\n\n上一版草稿：{draft}\n\n修改意见：{critique}\n\n请给出改进后的版本。"),
]) | llm | StrOutputParser()

# ── 2. 评审者：挑草稿的毛病；满意就只回「PASS」 ─────────────────────────────
critic = ChatPromptTemplate.from_messages([
    ("system", "你是一名挑剔的评审。指出草稿的具体不足（准确性/完整性/表达），最多 3 条。"
               "若已经足够好，只回复一个词：PASS。"),
    ("human", "任务：{task}\n\n草稿：{draft}\n\n你的意见："),
]) | llm | StrOutputParser()


if __name__ == "__main__":
    task = "用 3 句话向初学者解释「什么是向量数据库」。"

    draft = "（暂无草稿，请先写第一版）"
    critique = "（暂无意见，请先写第一版）"

    for round_no in range(1, 4):                 # 最多反思 3 轮
        draft = writer.invoke({"task": task, "draft": draft, "critique": critique})
        print(f"\n=== 第 {round_no} 版草稿 ===\n{draft}")

        critique = critic.invoke({"task": task, "draft": draft})
        print(f"\n--- 评审意见 ---\n{critique}")

        if critique.strip().upper().startswith("PASS"):
            print("\n[评审通过，停止反思]")
            break

    print("\n=== 最终输出 ===")
    print(draft)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  task
        │  writer 写第一版
        ▼
  draft ──► critic 挑毛病 ──► critique
        │                        │
        │  critique == PASS ?    │
        │   ├─ 是 → 结束          │
        │   └─ 否 → writer 据 critique 重写 ──┘ （回到上面，循环）
        ▼
  最终 draft

★ 核心规律：
  反思 = 生成者 ⇄ 评审者 的循环：产草稿 → 批评 → 据批评重写，直到评审 PASS 或到轮次上限。
  停止条件（PASS 或 max_round）一定要有，否则会无限改下去——和〔06/02〕的循环上限同理。

  本课用两条独立的链手搓了这个模式；当反思流程更复杂（多评审者、要回溯、要持久化）时，
  就交给 LangGraph（你的另一个课程）用「图」来编排——create_agent〔06/03〕也是它的预制件。
"""
