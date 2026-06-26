"""
生成 langchain_notes.html —— 把课程内容汇总成一份带折叠导航的网页笔记。
内容以结构化数据维护在本文件里；改完内容跑一次 `python course_v2/build_notes.py` 重新生成。
（这样不必手写上千行 HTML，导航/手风琴/滚动高亮等样板由模板统一产出。）
"""
import os, html

# 每个域：(锚, 标题, 主题色, [课])；每课：(锚, 标题, 概念, 核心规律, 代码)
DOMAINS = [
 ("d01","01 基础","#3b82f6",[
   ("l0101","01 Hello LLM","ChatOpenAI 接百炼；llm.invoke() 返回的是 AIMessage 对象，.content 才是文本，usage_metadata 看 token。",
    "llm.invoke(x) 永远返回 AIMessage，.content 是文本；temperature 控随机性（0 确定 / 0.7 创意）。",
    'llm = ChatOpenAI(api_key=..., base_url=..., model="qwen-plus")\nresp = llm.invoke("用一句话介绍 LangChain")\nresp.content          # 文本\nresp.usage_metadata   # token 消耗'),
   ("l0102","02 消息类型","用 System/Human/AI 三种消息代替裸字符串：System 设角色、Human 用户、AI 历史回复。",
    "模型本身无状态，所谓记忆就是每次把过往消息再发一遍；要设角色/带历史必须用消息列表。",
    'from langchain_core.messages import SystemMessage, HumanMessage, AIMessage\nllm.invoke([\n  SystemMessage(content="你是简洁助手"),\n  HumanMessage(content="LangChain 是什么？"),\n])'),
   ("l0103","03 init_chat_model","模型工厂：用同一个函数 + 字符串参数建模型，换 provider 只改字符串。",
    "接百炼这类 OpenAI 兼容服务 provider 填 'openai'；非官方模型名必须显式传 model_provider。",
    'from langchain.chat_models import init_chat_model\nllm = init_chat_model(model="qwen-plus", model_provider="openai",\n                      api_key=..., base_url=...)'),
   ("l0104","04 四种调用","invoke（等完整）/ stream（逐片段）/ batch（并发批量）/ ainvoke（异步）。",
    "这四个方法是 Runnable 接口的统一方法——prompt、parser、整条链都有，用法一致。",
    'llm.invoke(msgs)\nfor c in llm.stream(msgs): ...\nllm.batch([msgs1, msgs2])\nawait llm.ainvoke(msgs)'),
 ]),
 ("d02","02 提示词 Prompt","#8b5cf6",[
   ("l0201","01 PromptTemplate","带 {变量} 占位的文本模板，调用时填值；模板是一等公民，能校验/partial/进链/外部加载。",
    "PromptTemplate 面向纯文本场景，产出 StringPromptValue；聊天场景用 ChatPromptTemplate。",
    'p = PromptTemplate.from_template("用一句话解释 {topic}")\np.invoke({"topic": "递归"})'),
   ("l0202","02 ChatPromptTemplate","能填空的「消息列表」模板，日常主力；tuple/dict/消息类三种写法等价。",
    "聊天模型一律用 ChatPromptTemplate；元组 ('role','模板') 写法最常用。",
    'ChatPromptTemplate.from_messages([\n  ("system", "你是{style}翻译助手"),\n  ("human", "翻译：{text}"),\n])'),
   ("l0203","03 MessagesPlaceholder","在模板里留一个「插历史」的槽，填值时传一串消息原样展开。",
    "带记忆对话的标准结构：System → 历史槽 → 当前输入；〔05/02〕会自动往槽里灌历史。",
    'ChatPromptTemplate.from_messages([\n  ("system", "你是助手"),\n  MessagesPlaceholder("history"),\n  ("human", "{input}"),\n])'),
   ("l0204","04 partial","预先固定一部分变量，返回只差剩余变量的新模板；值可为常量或无参函数（动态求值）。",
    "把「建模板时就能定的变量」提前固定，调用点只关心真正变化的变量；常用于预填 format_instructions。",
    'base.partial(language="文言文")\nPromptTemplate.from_template("{time}:{q}").partial(time=now_func)'),
   ("l0205","05 外部加载〔进阶〕","把 prompt 抽到 json/yaml，改 prompt 不动代码；自己读文件构造 PromptTemplate（不用已弃用的 load_prompt）。",
    "prompt 与代码解耦，本质是把模板正文当配置运行时读入；加载出的对象和代码里建的没区别。",
    'data = json.load(open(path)) # 或 yaml.safe_load\nPromptTemplate(template=data["template"],\n               input_variables=data["input_variables"])'),
 ]),
 ("d03","03 输出解析 OutputParser","#06b6d4",[
   ("l0301","01 StrOutputParser","输入 AIMessage，输出 .content 字符串；LCEL 链最常见的收尾组件。",
    "OutputParser 是流水线最后一段，把模型文本整理成目标形态；StrOutputParser 只取 .content。",
    'chain = llm | StrOutputParser()   # 输出直接是 str'),
   ("l0302","02 JsonOutputParser","让模型吐 JSON：get_format_instructions() 塞进 prompt + invoke() 解析成 dict。",
    "结构化输出 = format_instructions 进 prompt + parser 解析返回，两件事配套；JsonOutputParser 返回松散 dict 不校验。",
    'parser = JsonOutputParser()\nprompt = ...partial(format_instructions=parser.get_format_instructions())\nparser.invoke(llm.invoke(prompt))'),
   ("l0303","03 PydanticOutputParser","用 Pydantic 模型定义结构 + 字段校验，返回强类型对象，可 .字段 访问。",
    "要「数据一定合法/字段齐全」用 Pydantic：模型即契约，field_validator 不通过直接抛错。",
    'class Product(BaseModel):\n    name: str = Field(description="名称")\n    price: float\nPydanticOutputParser(pydantic_object=Product)'),
   ("l0304","04 with_structured_output ★","一行把模型变成「保证吐结构化对象」的新模型，免手拼格式说明、免挂 parser。",
    "要结构化输出默认首选 with_structured_output（走模型原生 function calling，更稳）；parser 留给不支持的场景。",
    'structured = llm.with_structured_output(Product)\nstructured.invoke("编一条机械键盘信息")  # → Product'),
   ("l0305","05 TypedDict〔进阶〕","用 Annotated[类型,'说明'] 定义结构，返回普通 dict，轻量无校验。",
    "选型：要校验/对象方法 → Pydantic；只要个带类型的 dict、图轻量 → TypedDict。",
    'class Review(TypedDict):\n    rating: Annotated[float, "评分0-10"]\nllm.with_structured_output(Review)'),
 ]),
 ("d04","04 LCEL 与 Runnable","#10b981",[
   ("l0401","01 管道 |","a | b 把组件串成 RunnableSequence；prompt|model|parser 是最经典的链。",
    "前一步输出类型须能当后一步输入；整条链又是 Runnable，自动有 invoke/stream/batch。",
    'chain = prompt | llm | StrOutputParser()\nchain.invoke({"topic": "梯度下降"})'),
   ("l0402","02 RunnableLambda","把任意 Python 函数接进链；itemgetter 从 dict 抽字段更简洁。",
    "RunnableLambda 是「函数↔链」转接头：入参=上一步输出，返回=下一步输入。",
    'chain = prompt | llm | parser | RunnableLambda(add_sig)\n... | itemgetter("upper") | ...'),
   ("l0403","03 RunnableParallel","同一输入扇出到多条子链并发执行，结果按 key 合并成 dict。",
    "{'k': 子链} 字典在 LCEL 里被自动当 RunnableParallel；它会重构 dict，只留声明的 key。",
    'RunnableParallel({"优点": pros_chain, "缺点": cons_chain})'),
   ("l0404","04 Passthrough / assign","Passthrough 原样透传；.assign 在原 dict 上追加字段、原 key 保留。",
    "加工保留原字段→assign；彻底重构→Parallel；原样透传→Passthrough。RAG 链靠这套备 context+question。",
    'RunnablePassthrough.assign(upper=lambda d: d["text"].upper())\n# {"text":..., "upper":...}  原 key 保留'),
   ("l0405","05 RunnableBranch","按条件选走不同子链：(条件,链)...+默认链，顺序匹配第一个 True。",
    "纯 if-else 选链用它就够，不必上 LangGraph（需循环/状态才用）。",
    'RunnableBranch(\n  (is_translate, translate_chain),\n  (is_math, math_chain),\n  chat_chain)'),
   ("l0406","06 bind / config〔进阶〕",".bind(参数=值) 预绑调用参数（如 stop）；with_config 加 run_name/tags 便于追踪。",
    ".bind 不执行只返回预设参数的新 Runnable；bind_tools〔06/01〕同理。with_config 只加可观测信息。",
    'llm.bind(stop=["2."])\n(prompt|llm|parser).with_config(run_name="要点链")'),
 ]),
 ("d05","05 记忆 Memory","#f59e0b",[
   ("l0501","01 ChatMessageHistory","手动维护历史容器：add_user_message/add_ai_message + .messages 回填模板。",
    "记忆=存消息容器+每轮「读历史→调用→写历史」三步；手写一遍看清没有魔法。",
    'history = InMemoryChatMessageHistory()\nchain.invoke({"history": history.messages, "input": q})\nhistory.add_user_message(q); history.add_ai_message(a)'),
   ("l0502","02 RunnableWithMessageHistory","把链包成自动带记忆，只管传 input；session_id 隔离多会话。",
    "自动化〔05/01〕的读写历史；session_id 是会话隔离的钥匙，天然支持多用户。现代标准做法。",
    'RunnableWithMessageHistory(chain, get_history,\n  input_messages_key="input", history_messages_key="history")\n.invoke({"input": q}, config={"configurable":{"session_id":"A"}})'),
   ("l0503","03 trim_messages〔进阶〕","按预算裁剪历史，控制 token；strategy='last' 留最近，include_system 护角色。",
    "长对话里先裁剪再调用，让记忆可持续；token_counter 决定预算单位（len 按条数/真实计数按 token）。",
    'trim_messages(history, max_tokens=4, token_counter=len,\n  strategy="last", include_system=True)'),
 ]),
 ("d06","06 工具与 Agent","#ef4444",[
   ("l0601","01 @tool / bind_tools","@tool 把函数变工具，bind_tools 绑给模型；模型回复里多出 .tool_calls（调用意图）。",
    "模型只「决定」不「执行」：输出 tool_calls（选哪个+传什么参数），靠 docstring 理解工具。",
    '@tool\ndef add(a:int,b:int)->int:\n    """两数相加"""\n    return a+b\nllm.bind_tools([add]).invoke("23加19").tool_calls'),
   ("l0602","02 手动工具循环","执行 tool_calls 后用 ToolMessage(tool_call_id=...) 回灌，循环到模型不再要求调工具。",
    "工具循环=反复「invoke→有tool_calls就执行并回灌→再invoke」；这就是 ReAct 骨架。",
    'for call in ai.tool_calls:\n    r = tools_by_name[call["name"]].invoke(call["args"])\n    messages.append(ToolMessage(content=str(r), tool_call_id=call["id"]))'),
   ("l0603","03 create_agent","一行建 ReAct Agent，内部自动跑工具循环（底层 LangGraph）。",
    "传 model+tools+system_prompt 即得 Agent；输入 {'messages':[...]}，result['messages'] 是完整轨迹。",
    'agent = create_agent(llm, tools=[add, multiply],\n                     system_prompt="计算调工具")\nagent.invoke({"messages":[{"role":"user","content":"..."}]})'),
   ("l0604","04 多工具协作","Agent 用「上一个工具的输出」决定下一步，自主编排调用顺序。",
    "Agent 相对固定链的优势是动态编排；工具 docstring 要清楚、任何情况都要有合法返回（别返回 None）。",
    'create_agent(llm, tools=[search_products, check_inventory], ...)\n# 先搜到 id → 再用 id 查库存，顺序由 Agent 自定'),
 ]),
 ("d07","07 检索增强 RAG","#ec4899",[
   ("l0701","01 Embedding 与相似度","把文本变向量，语义近则向量近；余弦相似度衡量接近程度。",
    "检索本质=问题向量化找最近的文档向量；这是 RAG「按语义检索」的根基。",
    'vec = embeddings.embed_query("什么是 LangChain？")\ncosine(q_vec, doc_vec)  # 越接近1越相似'),
   ("l0702","02 FAISS 向量库","from_texts 建库 / similarity_search 检索 / save_local·load_local 持久化。",
    "向量库=批量存向量+快速最近邻+可落盘复用；FAISS 返回 L2 距离（越小越相似）。",
    'vs = FAISS.from_texts(chunks, embeddings)\nvs.save_local(dir)\nvs.similarity_search("RAG 是什么", k=2)'),
   ("l0703","03 文档加载器","各种 Loader 把 txt/pdf/csv/docx 统一读成 list[Document]（page_content+metadata）。",
    "不管原格式如何统一成 Document；拆分粒度不同（PDF按页/CSV按行）；metadata 可溯源。",
    'TextLoader(path).load()\nPyPDFLoader(path).load()   # 按页\nCSVLoader(path).load()    # 按行'),
   ("l0704","04 文本切分","RecursiveCharacterTextSplitter 按 chunk_size/chunk_overlap 切块。",
    "切分是 RAG 质量关键旋钮：太大噪声多、太小上下文断；overlap 对冲边界切断语义。",
    'RecursiveCharacterTextSplitter(chunk_size=120,\n  chunk_overlap=20).split_documents(docs)'),
   ("l0705","05 完整 RAG 链","{context: 检索+拼接, question: 透传} | prompt | llm | parser。",
    "retriever 是 Runnable 可直接进链；prompt 里「只依据上下文、没有就说不知道」是少幻觉的来源。",
    'rag = ({"context": retriever|RunnableLambda(format_docs),\n        "question": RunnablePassthrough()}\n       | prompt | llm | StrOutputParser())'),
   ("l0706","06 检索策略〔进阶〕","MMR（相关+不重复去冗余）；分数过滤（不够相关就不返回）。",
    "MMR 用 fetch_k 多召回再精选；分数过滤阈值必须按自己数据实测（L2 量级随模型变，别照搬）。",
    'vs.as_retriever(search_type="mmr",\n  search_kwargs={"k":3,"fetch_k":8,"lambda_mult":0.5})'),
   ("l0707","07 MultiQueryRetriever〔进阶〕","LLM 把问题扩写成多个子查询，分别检索再合并去重，提升召回。",
    "对抗「用户措辞和库里措辞不一致」的漏检；可 .llm_chain.invoke() 看生成的子查询。",
    'MultiQueryRetriever.from_llm(\n  retriever=vs.as_retriever(search_kwargs={"k":2}), llm=llm)'),
   ("l0708","08 对话式 RAG〔进阶〕","检索前先「问题重写」把指代词还原成独立问题，再走 RAG + 记忆。",
    "实测坑：改写指令在长历史里会失效（把改写做成作答），须强约束+示例+打印中间问题核对。",
    'standalone = contextualize_chain.invoke({"history":h, "input":q})\nretriever.invoke(standalone)  # 用还原后的独立问题检索'),
 ]),
 ("d08","08 流式与回调","#14b8a6",[
   ("l0801","01 stream","chain.stream() 逐 chunk 产出打字机效果；astream 是异步版。",
    "流式是整条链的能力；StrOutputParser 把模型增量 chunk 直接吐成字符串片段。",
    'for chunk in chain.stream({"topic": x}):\n    print(chunk, end="", flush=True)'),
   ("l0802","02 astream_events","按时间吐出链内部所有事件（模型逐token/检索开始结束…）。",
    "stream 只给最终文本，astream_events 给链内部全事件；按 event['event'] 类型分流做细粒度 UI。",
    'async for ev in chain.astream_events(x):\n    if ev["event"]=="on_chat_model_stream":\n        ev["data"]["chunk"].content'),
   ("l0803","03 Callbacks〔进阶〕","继承 BaseCallbackHandler 重写 on_xxx 钩子，自动监控耗时/token。",
    "回调是「推」式可观测：注册 Handler 自动触发，不改链代码就加监控；astream_events 是「拉」式。",
    'class Stats(BaseCallbackHandler):\n    def on_llm_end(self, response, **kw): ...\nllm.invoke(q, config={"callbacks":[Stats()]})'),
 ]),
 ("d09","09 工程化与可靠性","#6366f1",[
   ("l0901","01 with_retry","失败自动重试，对调用方透明，免手写 try/except。",
    "扛「偶发抖动」：再试一次可能就好；stop_after_attempt 控次数；返回新 Runnable。",
    'RunnableLambda(flaky).with_retry(stop_after_attempt=5)'),
   ("l0902","02 with_fallbacks","主方案彻底失败时按序切到备用方案。",
    "扛「整体故障」：主模型挂了换备用；和 retry 互补，可叠加使用。",
    'broken_llm.with_fallbacks([good_llm])'),
   ("l0903","03 set_llm_cache","相同输入命中缓存，不重复调 LLM，省钱省时。",
    "InMemoryCache 开发用（重启即丢），生产换 SQLiteCache；命中条件是输入完全相同（会抹掉随机性）。",
    'set_llm_cache(InMemoryCache())\nllm.invoke(q)  # 第二次相同输入命中缓存'),
   ("l0904","04 何时不上 LangGraph","纯 if-else 选链用 RunnableBranch；需循环/状态持久化/人工干预才上 LangGraph。",
    "工程好品味=用刚好够用的工具；create_agent 这类循环调工具的场景底层才用 LangGraph。",
    'RunnableBranch((cond1, chain1), (cond2, chain2), default)'),
 ]),
 ("d10","10 评估 Evaluation","#a855f7",[
   ("l1001","01 Criteria 评估","用 LLM 当裁判按标准打分；labeled_criteria 对照参考资料查 RAG 忠实度（幻觉）。",
    "主观维度用 criteria；要对照资料判断是否编造用 labeled_criteria（忘带 labeled 拿不到 reference）。",
    'load_evaluator("criteria", criteria="conciseness", llm=llm)\nload_evaluator("labeled_criteria", criteria={"faithfulness":"..."}, llm=llm)'),
   ("l1002","02 QA 评估 + 回归测试","对照标准答案判 CORRECT/INCORRECT；套在固定测试集上跑出通过率。",
    "把「感觉还行」变成「通过率2/3→3/3」可量化结论；测试集持续积累每个 badcase。",
    'qa = load_evaluator("qa", llm=llm)\nqa.evaluate_strings(input=q, prediction=ans, reference=ref)'),
 ]),
]

def esc(s): return html.escape(s)

def build():
    # 每个域一个渐变 banner（暗色主题，参考 LangGraph 笔记配色）
    GRADIENTS = [
        "linear-gradient(135deg,#1d4ed8 0%,#4f46e5 100%)",  # 01 蓝
        "linear-gradient(135deg,#6d28d9 0%,#0ea5e9 100%)",  # 02 紫
        "linear-gradient(135deg,#0e7490 0%,#0284c7 100%)",  # 03 青
        "linear-gradient(135deg,#065f46 0%,#0891b2 100%)",  # 04 绿
        "linear-gradient(135deg,#92400e 0%,#b45309 100%)",  # 05 橙
        "linear-gradient(135deg,#9f1239 0%,#e11d48 100%)",  # 06 红
        "linear-gradient(135deg,#a21caf 0%,#db2777 100%)",  # 07 品红
        "linear-gradient(135deg,#0f766e 0%,#14b8a6 100%)",  # 08 蓝绿
        "linear-gradient(135deg,#4338ca 0%,#6366f1 100%)",  # 09 靛
        "linear-gradient(135deg,#7e22ce 0%,#a855f7 100%)",  # 10 紫
    ]
    nav, content = [], []
    for di, (did, dtitle, color, lessons) in enumerate(DOMAINS):
        grad = GRADIENTS[di % len(GRADIENTS)]
        subs = "".join(f'<a href="#{lid}" class="nav-sub-item">{esc(lt)}</a>' for lid,lt,*_ in lessons)
        nav.append(f'<div class="nav-domain" data-domain="{did}">'
                   f'<div class="nav-domain-head"><span class="dot" style="background:{color}"></span>'
                   f'{esc(dtitle)}<span class="arrow">▶</span></div>'
                   f'<div class="nav-sub">{subs}</div></div>')
        cards = []
        for lid, lt, concept, rule, code in lessons:
            cards.append(
                f'<div class="lesson" id="{lid}" data-domain="{did}" style="border-left-color:{color}">'
                f'<h3>{esc(lt)}</h3>'
                f'<p class="concept">{esc(concept)}</p>'
                f'<pre><code>{esc(code)}</code></pre>'
                f'<div class="rule"><span>★ 核心规律</span>{esc(rule)}</div></div>')
        content.append(
            f'<section class="domain" id="{did}">'
            f'<div class="banner" style="background:{grad}">{esc(dtitle)}</div>'
            f'<div class="cards">{"".join(cards)}</div></section>')
    return TEMPLATE.replace("{{NAV}}","".join(nav)).replace("{{CONTENT}}","".join(content))

TEMPLATE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LangChain 课程 v2 · 图文笔记</title>
<style>
:root {
  --sidebar-w: 280px;
  --bg: #0f1722;          /* 页面背景：柔和深蓝灰 */
  --surface: #1a2433;     /* 卡片：比背景略亮 */
  --code-bg: #0b1220;     /* 代码 / 侧栏：最深 */
  --text: #e6edf5;        /* 正文：柔和浅色，非纯白 */
  --text-sub: #9fb0c3;
  --text-muted: #6b7a8d;
  --border: #2a3850;
  --primary: #3b82f6;
  --orange: #f59e0b;
  --radius: 10px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
       "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
       font-size: 17px; line-height: 1.8; color: var(--text); background: var(--bg);
       display: flex; }

/* ── 侧边栏 ── */
#sidebar { width: var(--sidebar-w); height: 100vh; position: fixed; top: 0; left: 0;
           overflow-y: auto; overflow-x: hidden; background: var(--code-bg);
           border-right: 1px solid var(--border); padding-bottom: 40px; z-index: 100;
           scrollbar-width: thin; scrollbar-color: #243045 transparent; }
#sidebar::-webkit-scrollbar { width: 4px; }
#sidebar::-webkit-scrollbar-thumb { background: #243045; border-radius: 2px; }
.sidebar-header { padding: 26px 20px 18px; border-bottom: 1px solid #1e293b; }
.sidebar-header .logo { font-size: 11px; color: #5a6b80; text-transform: uppercase;
                        letter-spacing: 1.5px; margin-bottom: 6px; }
.sidebar-header h1 { font-size: 18px; font-weight: 700; color: #f1f5f9; line-height: 1.3; }
.sidebar-header p { font-size: 12.5px; color: #64748b; margin-top: 6px; }
.nav-domain { margin-bottom: 1px; }
.nav-domain-head { display: flex; align-items: center; gap: 10px; padding: 11px 20px;
                   font-size: 15px; font-weight: 600; color: #a8b6c8; cursor: pointer;
                   user-select: none; transition: all .15s; }
.nav-domain-head:hover { color: #e2e8f0; background: #1b2740; }
.dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.arrow { margin-left: auto; font-size: 10px; color: #475569; transition: transform .2s ease; }
.nav-domain.open .arrow { transform: rotate(90deg); color: #60a5fa; }
.nav-sub { max-height: 0; overflow: hidden; transition: max-height .25s ease; }
.nav-domain.open .nav-sub { max-height: 600px; }
.nav-sub-item { display: block; padding: 6px 20px 6px 39px; font-size: 13.5px;
                color: #7d8da0; text-decoration: none; line-height: 1.5; transition: color .15s; }
.nav-sub-item:hover { color: #b7c4d6; }
.nav-sub-item.active { color: #60a5fa; font-weight: 600; background: #16294a; }

/* ── 主内容 ── */
#main { margin-left: var(--sidebar-w); flex: 1; padding: 0 0 80px; }
.domain { max-width: 980px; }
.banner { color: #fff; font-size: 23px; font-weight: 800; padding: 30px 40px;
          margin: 30px 40px 6px; border-radius: 14px;
          box-shadow: 0 6px 20px rgba(0,0,0,.35); }
.cards { padding: 0 40px; }
.lesson { background: var(--surface); margin: 18px 0; padding: 22px 26px;
          border-radius: var(--radius); border: 1px solid var(--border);
          border-left: 4px solid var(--primary); }
.lesson h3 { font-size: 18px; color: #f1f5f9; margin-bottom: 10px; }
.concept { color: var(--text-sub); line-height: 1.8; margin-bottom: 14px; font-size: 16.5px; }
pre { background: var(--code-bg); color: #e2e8f0; padding: 16px 20px; border-radius: 8px;
      overflow-x: auto; font-size: 14px; line-height: 1.75; border: 1px solid #1e2a3f;
      font-family: "JetBrains Mono", "Fira Code", "SF Mono", Menlo, Consolas, monospace; }
pre code { font-family: inherit; }
.rule { margin-top: 14px; background: #2c2410; border-left: 4px solid var(--orange);
        padding: 12px 16px; border-radius: 8px; line-height: 1.8; color: #f0e6d2; font-size: 16px; }
.rule span { display: block; font-weight: 700; color: #fbbf24; margin-bottom: 4px; font-size: 14px; }

/* ── 回到顶部 ── */
#back-top { position: fixed; bottom: 28px; right: 28px; width: 44px; height: 44px;
            background: var(--primary); color: #fff; border-radius: 50%; display: flex;
            align-items: center; justify-content: center; font-size: 18px; cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,.5); text-decoration: none; opacity: .9;
            transition: opacity .15s, transform .15s; z-index: 200; }
#back-top:hover { opacity: 1; transform: translateY(-2px); }

@media (max-width: 768px) {
  #sidebar { transform: translateX(-100%); }
  #main { margin-left: 0; }
  .banner, .cards { margin-left: 16px; margin-right: 16px; padding-left: 0; padding-right: 0; }
}
</style></head>
<body>
<nav id="sidebar">
  <div class="sidebar-header">
    <div class="logo">LANGCHAIN COURSE</div>
    <h1>LangChain 课程 v2</h1>
    <p>10 概念域 · 44 课 · 暗色护眼</p>
  </div>
  {{NAV}}
</nav>
<main id="main">{{CONTENT}}</main>
<a href="#" id="back-top">↑</a>
<script>
// 手风琴：点击域标题展开 / 收起
document.querySelectorAll('.nav-domain-head').forEach(h => {
  h.addEventListener('click', () => h.parentElement.classList.toggle('open'));
});
// 滚动高亮 + 自动展开当前域（IntersectionObserver 监听正文）
const io = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    const id = e.target.id;
    document.querySelectorAll('.nav-sub-item').forEach(a =>
      a.classList.toggle('active', a.getAttribute('href') === '#' + id));
    const did = e.target.dataset.domain;
    document.querySelectorAll('.nav-domain').forEach(d =>
      d.classList.toggle('open', d.dataset.domain === did));
  });
}, { rootMargin: '-45% 0px -50% 0px' });
document.querySelectorAll('.lesson').forEach(l => io.observe(l));
document.querySelector('.nav-domain')?.classList.add('open');  // 默认展开第一个域
</script></body></html>"""

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "langchain_notes.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build())
    n = sum(len(d[3]) for d in DOMAINS)
    print(f"已生成 {out}（{len(DOMAINS)} 域 / {n} 课）")
