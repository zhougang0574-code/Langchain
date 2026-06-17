"""
第11课：流式输出 & 可观测性

为什么是这一课：前10课所有例子都用 .invoke()——发一个请求，等模型把全部内容
生成完，一次性拿到完整结果。这在真实产品里体验很差：用户要等好几秒屏幕才动一下。
"流式输出"就是边生成边把内容吐给你，也就是 ChatGPT 网页那种"打字机效果"的原理。
"可观测性"则是另一件事：链变复杂后（第7-9课的RAG链已经有好几步），
出了问题怎么知道是哪一步、花了多久——这就要靠 Callback 和调试工具。

学习要点：
1. .stream() —— 逐块（chunk）返回结果，而不是等全部生成完再返回
2. LCEL 链天生支持流式：因为 prompt/llm/parser 都实现了同一套 Runnable 接口
3. .astream() —— 异步版的 stream，配合 asyncio，适合 Web 服务端场景
4. .astream_events() —— 拿到链内部"每一步"的细粒度事件（prompt开始/llm开始/逐token/parser结束...）
5. 自定义 CallbackHandler —— 在 LLM 调用的关键节点（开始/每个token/结束）插入自己的逻辑，
   比如统计耗时、记录日志，且不需要修改链本身的代码
6. set_debug(True) —— 全局调试开关，一行代码看清链每一步的输入输出和耗时

Python 小贴士（给新手）：
- for chunk in llm.stream(...): 这里 .stream() 返回的是一个"生成器"（generator），
  不是一次性把所有结果都准备好的列表，而是"你问一次它给一块"，这样才能做到"边生成边返回"
- async/await 是 Python 处理"异步任务"的语法：让程序在等待一个慢操作（比如网络请求）时，
  可以先去做别的事，而不是干等着；asyncio.run(main()) 是运行异步函数的入口
"""

import os
import time
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.globals import set_debug

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
    temperature=0,
)


# ─────────────────────────────────────────────
# 1. .stream() —— 逐块返回，而不是等全部生成完
# ─────────────────────────────────────────────
print("=== .stream() 基础用法 ===")
print("逐块输出: ", end="", flush=True)
chunk_count = 0
for chunk in llm.stream("用两句话介绍 Python 语言"):
    # 每个 chunk 是一个 AIMessageChunk，content 通常是几个字/一个词，不是完整句子
    print(chunk.content, end="", flush=True)
    chunk_count += 1
print(f"\n（一共收到 {chunk_count} 个 chunk，invoke() 的话只会收到1个完整结果）")


# ─────────────────────────────────────────────
# 2. LCEL 链同样支持流式：因为整条链都是 Runnable
# ─────────────────────────────────────────────
print("\n=== 整条链的流式输出 ===")
chain = (
    ChatPromptTemplate.from_template("用一句话讲个关于{topic}的冷笑话")
    | llm
    | StrOutputParser()  # StrOutputParser 也支持流式：把每个 chunk 的 content 转成字符串再吐出来
)
print("链式流式输出: ", end="", flush=True)
for chunk in chain.stream({"topic": "程序员"}):
    print(chunk, end="", flush=True)  # 经过 StrOutputParser 后，chunk 直接是字符串
print()


# ─────────────────────────────────────────────
# 3. .astream() —— 异步流式，适合 Web 服务端场景
# ─────────────────────────────────────────────
# 真实场景：FastAPI 接口要把生成结果用 Server-Sent Events 推给前端，
# 整个 Web 框架是异步的，这时候要用 .astream() 而不是 .stream()。
async def demo_astream():
    print("\n=== .astream() 异步流式 ===")
    print("异步流式输出: ", end="", flush=True)
    async for chunk in llm.astream("用一句话介绍异步编程"):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(demo_astream())


# ─────────────────────────────────────────────
# 4. .astream_events() —— 拿到链内部每一步的细粒度事件
# ─────────────────────────────────────────────
# 普通 .stream() 只能看到"最终输出"在流式吐字；但链内部有 prompt、llm、parser 好几步，
# astream_events() 能让你看到"现在具体是哪一步在跑、跑到哪了"，调试复杂链（比如RAG链）时很有用。
async def demo_astream_events():
    print("\n=== .astream_events() 细粒度事件 ===")
    event_counts: dict[str, int] = {}
    async for event in chain.astream_events({"topic": "猫"}, version="v2"):
        event_type = event["event"]
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        # 只把"LLM 正在吐字"这一类事件的具体内容打出来，模拟"既要看到生成过程，又要知道是哪步产生的"
        if event_type == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            print(token, end="", flush=True)
    print(f"\n\n本次调用经过的事件类型及次数: {event_counts}")

asyncio.run(demo_astream_events())


# ─────────────────────────────────────────────
# 5. 自定义 CallbackHandler —— 在关键节点插入自己的逻辑
# ─────────────────────────────────────────────
# Callback 和 .stream() 是两个不同维度的机制：
#   .stream() 是"我主动要求一段一段拿结果"的调用方式；
#   Callback 是"无论你用 invoke 还是 stream，关键节点都会触发"的钩子，不需要改变调用方式。
class TimingAndTokenHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.start_time = time.time()
        self.token_count = 0

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 每收到一个 token 就会触发一次，invoke() 时如果底层走了流式协议也同样会触发
        self.token_count += 1

    def on_llm_end(self, response, **kwargs) -> None:
        elapsed = time.time() - self.start_time
        print(f"[callback] 本次调用耗时 {elapsed:.2f}s，共收到 {self.token_count} 个 token")

    def on_llm_error(self, error, **kwargs) -> None:
        print(f"[callback] 调用出错: {error}")


print("\n=== 自定义 CallbackHandler ===")
handler = TimingAndTokenHandler()
# 通过 config={"callbacks": [...]} 把 handler 挂到这一次调用上，不需要修改 llm 或链的定义
result = chain.invoke({"topic": "数据库"}, config={"callbacks": [handler]})
print(f"最终结果: {result}")
print("⚠️ 实测发现：on_llm_new_token 的触发次数是 0！")
print("   原因：普通 .invoke() 默认不会要求底层用流式协议返回，所以根本没有逐 token 推送这一步。")
print("   on_llm_new_token 只在真正发生了流式传输时才会触发——要么用 .stream()/.astream()，")
print("   要么在构造 ChatOpenAI 时传 streaming=True，让它即使被 invoke() 调用也走流式协议。")

streaming_llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"), model=os.getenv("MODEL"),
    temperature=0, streaming=True,  # 关键参数：即使用 invoke() 调用，底层也走流式协议
)
streaming_chain = ChatPromptTemplate.from_template("用一句话讲个关于{topic}的冷笑话") | streaming_llm | StrOutputParser()
handler2 = TimingAndTokenHandler()
streaming_chain.invoke({"topic": "网络"}, config={"callbacks": [handler2]})
print("（加了 streaming=True 后，上面这次调用的 token 数应该 > 0 了）")


# ─────────────────────────────────────────────
# 6. set_debug(True) —— 一行代码看清链每一步
# ─────────────────────────────────────────────
print("\n=== set_debug(True) 全局调试开关 ===")
print("（debug 模式下会打印每一步 Runnable 的输入输出和耗时，下面只截取关键部分观察）")
set_debug(True)
chain.invoke({"topic": "向量数据库"})
set_debug(False)  # 用完记得关掉，否则后面所有调用都会打印一长串调试信息


if __name__ == "__main__":
    pass


"""
执行流程图：

.invoke()：
  请求 ──► LLM 内部生成完整内容 ──► 一次性返回完整结果
  （用户体感：等待 N 秒后，结果突然全部出现）

.stream() / .astream()：
  请求 ──► LLM 开始生成
              │ 每生成一小段就立刻 yield 出来（生成器特性）
              ▼
        chunk1 → chunk2 → chunk3 → ...（你边收边显示）
  （用户体感：和 ChatGPT 网页一样，文字一个字一个字蹦出来）

.astream_events()：
  请求 ──► 链开始（on_chain_start）
              │
              ├─► prompt 节点开始/结束（on_prompt_start/end）
              ├─► llm 节点开始 → 逐token（on_chat_model_stream） → 结束
              └─► parser 节点开始/结束（on_parser_start/end）
  （比 .stream() 多了"现在具体是哪一步在跑"的信息）

CallbackHandler 的触发时机（不管用 invoke 还是 stream 都会触发）：
  on_llm_start → on_llm_new_token（可能触发多次） → on_llm_end（或 on_llm_error）


核心知识点 ★：

★ .stream() 返回的是生成器：一边生成一边吐出来，而不是等全部完成后再返回，
  这是实现"打字机效果"的根本机制。
★ LCEL 链天生支持流式：因为 prompt | llm | parser 里每一环都实现了同一套 Runnable 接口
  （invoke/stream/batch 全都支持），所以整条链可以直接 .stream()，不需要额外改造。
★ .astream() 是异步版本，Web 后端（如 FastAPI）要把流式结果推给前端时必须用这个，
  因为整个异步框架要求所有 I/O 操作都用 async/await。
★ .astream_events() 比 .stream() 更细粒度：能看到链内部每一步（prompt/llm/parser）
  各自的开始结束和中间过程，调试多步骤的复杂链（比如RAG链）时比单纯看最终输出更有效率。
★ CallbackHandler 和 .stream() 是两个独立维度：Callback 是"挂在关键节点上的钩子"，
  用 config={"callbacks":[...]} 挂载，不需要改链本身的代码，适合做统计、日志这类"旁路"逻辑。
★ 【实测查漏】on_llm_new_token 不是"只要挂了 Callback 就会触发"：普通 .invoke() 默认走
  非流式协议，根本没有逐token推送这一步，所以 on_llm_new_token 触发次数是 0。
  必须真正发生流式传输才会触发——用 .stream()/.astream()，或者在构造 ChatOpenAI 时
  传 streaming=True，让它即使被 invoke() 调用也走流式协议。
★ set_debug(True) 是排查"链到底执行到哪一步、参数是什么"最快的方式，
  打印的信息很详细，调试完一定记得 set_debug(False) 关掉，否则会持续刷屏。
"""
