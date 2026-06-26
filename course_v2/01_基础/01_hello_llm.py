"""
【01 基础 / 01】Hello LLM —— 第一个调用，看清返回值的结构
=====================================================
这是整个课程的起点，只做一件最小的事：把一句话发给大模型，拿到回复。

新概念（只有这一个）：
  ChatOpenAI —— LangChain 对「聊天模型」的封装。
  阿里百炼兼容 OpenAI 协议，所以用 ChatOpenAI + base_url 就能接入，不需要专门的 SDK。

本课的重点不是「怎么调」，而是「调完拿到的到底是什么」：
  llm.invoke("...") 返回的不是字符串，而是一个 AIMessage 对象。
  .content          才是文本
  .response_metadata 是模型/网关返回的元信息
  .usage_metadata    是 token 消耗（调试成本时很有用）
看清这一点，后面所有课才不会困惑「为什么打印出来是一坨对象」。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# load_dotenv() 会从当前目录向上逐级查找 .env，所以 .env 放在项目根目录一份就够
load_dotenv()


# ── 1. 创建模型对象 ────────────────────────────────────────────────────────
# 三个参数都从 .env 读：把密钥写死在代码里迟早会误提交到 git
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),   # 指向百炼的 compatible-mode 接口
    model=os.getenv("MODEL"),         # qwen-plus
    temperature=0.7,                  # 随机性：0=最确定，1=最发散
)


# ── 2. 最小调用 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 传一个字符串，LangChain 会自动把它当成一条 HumanMessage
    response = llm.invoke("用一句话介绍 LangChain")

    # 关键认知：response 不是 str，是 AIMessage 对象
    print("返回类型：", type(response).__name__)        # AIMessage
    print("content ：", response.content)               # ← 这才是文本

    # 元信息：模型名、结束原因等（不同网关字段略有差异）
    print("\nresponse_metadata：", response.response_metadata)

    # token 消耗：input / output / total，排查「为什么这么费钱」时第一眼看它
    print("usage_metadata   ：", response.usage_metadata)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
执行流程：

  "用一句话介绍 LangChain"  (str)
            │  llm.invoke()
            ▼
        AIMessage
        ├─ content          → "LangChain 是一个用于构建 LLM 应用的开源框架。"
        ├─ response_metadata→ {model, finish_reason, ...}
        └─ usage_metadata   → {input_tokens, output_tokens, total_tokens}

★ 核心规律：
  llm.invoke(x) 永远返回一个 AIMessage 对象，.content 才是文本。
  这是后面一切的基础——StrOutputParser（〔03/01〕）的作用就是自动帮你取出 .content。

  temperature 控制随机性：结构化/确定性任务用 0，创意/闲聊用 0.7~1。
"""
