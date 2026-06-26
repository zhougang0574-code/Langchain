"""
【02 提示词Prompt / 05】从外部文件加载 prompt〔进阶〕
==================================================
前几课 prompt 都写在 .py 代码里。但在团队协作 / 频繁调优 prompt 的项目里，
更好的做法是把 prompt 抽到独立的 json / yaml 文件：改 prompt 不用动代码，
也方便非工程同学维护、做版本管理。

新概念（只有这一个）：
  「prompt 与代码解耦」—— 把模板正文存到外部文件，运行时读进来再建 PromptTemplate。

实现方式：自己用标准库 json / yaml 把文件读成 dict，再喂给 PromptTemplate。
  （LangChain 旧版有个 load_prompt() 一步到位，但已在 1.2 标记弃用，不推荐再用；
    自己读文件这套写法更透明、不依赖会被移除的 API，且 json/yaml 想放什么字段都行。）
"""

import os
import json

import yaml
from langchain_core.prompts import PromptTemplate

HERE = os.path.dirname(os.path.abspath(__file__))


# ── 1. 通用加载函数：读文件 → dict → PromptTemplate ────────────────────────
def load_prompt_file(path: str) -> PromptTemplate:
    with open(path, "r", encoding="utf-8") as f:
        # 按扩展名选解析器：json 用 json.load，yaml 用 yaml.safe_load
        data = json.load(f) if path.endswith(".json") else yaml.safe_load(f)
    # data 里有 template 和 input_variables 两个字段，直接构造模板
    return PromptTemplate(
        template=data["template"],
        input_variables=data["input_variables"],
    )


# ── 2. 从 json / yaml 各加载一份（内容等价）────────────────────────────────
prompt_from_json = load_prompt_file(os.path.join(HERE, "assets", "prompt.json"))
prompt_from_yaml = load_prompt_file(os.path.join(HERE, "assets", "prompt.yaml"))

print("=== 从 json 加载 ===")
print("变量：", prompt_from_json.input_variables)
print(prompt_from_json.invoke({"product": "扫地机器人"}).to_string())

print("\n=== 从 yaml 加载（格式更易读）===")
print(prompt_from_yaml.invoke({"product": "蓝牙音箱"}).to_string())


if __name__ == "__main__":
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()
    llm = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=os.getenv("MODEL"),
    )
    print("\n=== 实跑 ===")
    print(llm.invoke(prompt_from_json.invoke({"product": "智能手表"})).content)


# ── 说明 ────────────────────────────────────────────────────────────────────
"""
  prompt.json / prompt.yaml  (磁盘上的文件)
        │  open + json/yaml 解析 → dict
        ▼
  PromptTemplate(template=..., input_variables=...)
        │
        ▼  和代码里 from_template 建出来的完全一样

★ 核心规律：
  「prompt 外置」的本质就是把模板正文当配置：运行时读进来构造 PromptTemplate。
  自己读文件比依赖 load_prompt() 这类便捷函数更稳——后者已被弃用，且字段格式被框架写死。

  小坑：yaml 需要 pyyaml（装 langchain 时通常已带）；路径用「本文件目录 + assets」拼，
  避免换个工作目录运行就找不到文件。
"""
