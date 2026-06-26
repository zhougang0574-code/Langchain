"""
【02 提示词Prompt / 01】PromptTemplate —— 把提示词做成「带空填的模板」
==================================================================
〔01 基础〕里我们把问题写死在代码里（"用一句话介绍 Python"）。真实应用中，
问题来自用户输入、循环变量，需要「模板 + 变量」动态拼装，而不是字符串拼接。

新概念（只有这一个）：
  PromptTemplate —— 文本模板，用 {变量名} 占位，调用时再填值。
    PromptTemplate.from_template("...{topic}...")  建模板
    template.invoke({"topic": "..."})              填值，得到 PromptValue

为什么不用 f-string 直接拼？
  模板是「一等公民」：能校验变量、能 partial 预填、能被 LCEL 串进链里、能从文件加载。
  这些是裸字符串拼接给不了的（后面几课会逐个用到）。
"""

from langchain_core.prompts import PromptTemplate

# ── 1. from_template：最常用的建模板方式 ───────────────────────────────────
# {topic} 是占位符，input_variables 会被自动推断出来
prompt = PromptTemplate.from_template("请用一句话解释「{topic}」，面向{audience}。")

print("=== 模板声明的变量 ===")
print(prompt.input_variables)        # ['audience', 'topic']


# ── 2. invoke：填值，得到 PromptValue ──────────────────────────────────────
# 注意：填完得到的不是 str，而是 PromptValue（链里的标准中间产物）
value = prompt.invoke({"topic": "递归", "audience": "小学生"})
print("\n=== invoke 后的类型 ===")
print(type(value).__name__)          # StringPromptValue
print(value.to_string())             # 转成最终发给模型的纯文本


# ── 3. 真实用法：模板 + 模型（手动两步，下一域会用 | 串起来）────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()
    llm = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=os.getenv("MODEL"),
    )

    # 同一个模板，换不同变量，复用提示词逻辑
    for topic in ["递归", "闭包", "装饰器"]:
        filled = prompt.invoke({"topic": topic, "audience": "Python 初学者"})
        print(f"\n[{topic}]", llm.invoke(filled).content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  PromptTemplate("...{topic}...{audience}...")
            │  .invoke({"topic": ..., "audience": ...})
            ▼
        PromptValue  ──.to_string()──→  最终文本  ──→ 喂给模型

★ 核心规律：
  PromptTemplate 面向「纯文本补全」场景，产出 StringPromptValue。
  但聊天模型更适合「带角色的消息列表」——所以下一课的 ChatPromptTemplate
  〔02/02〕才是日常主力；本课先建立「模板=带空填的对象」这个基本认知。

  填值用 invoke（链式统一风格）；也有等价的老写法 prompt.format(topic=...) 直接得 str。
"""
