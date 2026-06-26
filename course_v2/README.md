# LangChain 课程 v2 —— 分类轴 + 域内细梯度

本版把课程从「纯线性 14 课」重排为「**按概念域分组**（分类轴），每个域内部保留**一文件一个新概念、小步递进**（域内梯度）」，并对照速通课资料补全了一批知识点（多 provider、文档加载器全家桶、向量相似度、`with_structured_output` 等）。

- 旧版（原始线性 14 课）完整留底在 `../archive_v1/`，未做任何改动。
- 本目录是当前学习版本，**目录结构即学习路径**。

---

## 一、设计哲学（为什么这样分）

LangChain 的心智模型是「**一条由组件拼成的流水线**」：`Prompt | Model | Parser`，再用 **LCEL** 把它们粘起来。所以课程的前半部分按 **组件（原语）** 切分，后半部分按 **能力（用例）** 切分：

```
前半「原语轴」：先把零件一个个学会
  01 基础(Model)  →  02 Prompt  →  03 Parser  →  04 LCEL(粘合层)

后半「能力轴」：再学这些零件能拼出什么
  05 记忆  →  06 工具与Agent  →  07 RAG  →  08 流式  →  09 工程化  →  10 评估
```

这是刻意的「混合轴」：和教编程语言「先语法、再应用」是同一个道理。零件（01-04）必须先学，因为后面每一个能力都是它们的组合。

---

## 二、目录结构（即学习顺序）

域与域之间已按前置依赖排序，从上往下读即可；每个文件夹内部从 `01` 到末尾是小步递进。
标〔进阶〕的可第一遍跳过，不影响主线。

```
01_基础/
  01_hello_llm.py          ChatOpenAI / invoke / AIMessage 的结构
  02_messages.py           System / Human / AI 三种消息 + 多轮对话
  03_init_chat_model.py    init_chat_model 统一入口，一行换 provider     ★融合速通
  04_invoke_stream_batch.py  invoke / stream / batch / ainvoke 四种调用

02_提示词Prompt/
  01_prompt_template.py    PromptTemplate.from_template / format
  02_chat_prompt_template.py  ChatPromptTemplate：tuple / dict / 消息对象三种写法
  03_placeholder.py        MessagesPlaceholder 把历史消息插进模板
  04_partial.py            partial_variables / .partial() 预填部分变量
  05_load_external.py      从 json / yaml 外部文件加载 prompt          〔进阶〕
  06_few_shot.py           Few-shot 少样本：给几个示例让模型照样输出   〔进阶〕

03_输出解析OutputParser/
  01_str_parser.py         StrOutputParser：从 AIMessage 取纯文本
  02_json_parser.py        JsonOutputParser + get_format_instructions
  03_pydantic_parser.py    PydanticOutputParser + 字段校验
  04_structured_output.py  with_structured_output（现代首选写法）★重点对照
  05_typed_dict.py         TypedDict + Annotated 描述结构                〔进阶〕
  06_output_fixing.py      OutputFixingParser 解析失败自动修复          〔进阶〕

04_LCEL与Runnable/
  01_pipe_chain.py         prompt | model | parser：第一条链
  02_runnable_lambda.py    RunnableLambda 接任意函数 + itemgetter
  03_runnable_parallel.py  RunnableParallel 并发多路
  04_passthrough_assign.py RunnablePassthrough vs .assign（关键区别）
  05_runnable_branch.py    RunnableBranch 条件分支
  06_bind_config.py        .bind() 预绑定参数 + with_config            〔进阶〕

05_记忆Memory/
  01_chat_history.py       InMemoryChatMessageHistory 手动维护历史
  02_runnable_with_history.py  RunnableWithMessageHistory 自动管理 + session_id
  03_trim_messages.py      trim_messages 裁剪历史控制 token            〔进阶〕

06_工具与Agent/
  01_tools.py              @tool / bind_tools / 读 tool_calls
  02_tool_loop.py          手动执行 tool_calls，用 ToolMessage 回灌结果
  03_create_agent.py       create_agent 一行建 ReAct Agent
  04_multi_tool_agent.py   多工具协作（搜索 + 查库的电商例子）

07_检索增强RAG/
  01_embeddings.py         文本向量化 + 余弦相似度直观演示
  02_vector_store.py       FAISS 建库 / similarity_search / 本地持久化
  03_document_loaders.py   txt / pdf / csv / md / docx 多格式加载       ★融合速通
  04_text_splitter.py      RecursiveCharacterTextSplitter / chunk_overlap
  05_basic_rag.py          完整 RAG 链：加载→分块→向量化→检索→生成
  06_retriever_strategy.py MMR / 分数过滤                              〔进阶〕
  07_multi_query.py        MultiQueryRetriever 多角度召回              〔进阶〕
  08_conversational_rag.py 对话式 RAG + 问题重写（含实测踩坑）         〔进阶〕
  09_compression_rerank.py 上下文压缩 + 重排序：检索后处理精炼          〔进阶〕

08_流式与回调/
  01_stream.py             stream / astream 打字机效果
  02_astream_events.py     astream_events：token 级 + 全事件类型
  03_callbacks.py          CallbackHandler 监控耗时 / token 成本        〔进阶〕

09_工程化与可靠性/
  01_with_retry.py         .with_retry() 自动重试
  02_with_fallbacks.py     .with_fallbacks() 降级到备用模型
  03_llm_cache.py          set_llm_cache() 缓存重复调用省钱
  04_runnable_branch_eng.py  RunnableBranch 轻量分支（何时不必上 LangGraph）

10_评估Evaluation/
  01_criteria_eval.py      Criteria / labeled_criteria（含 RAG 忠实度幻觉检测）
  02_qa_eval.py            QA Evaluator + 回归测试集（可量化迭代）
```

### 域间排序的依据（前置依赖）
- **01-04 是四件套原语**：Model → Prompt → Parser → LCEL，先有零件才能拼链；
  Parser 放在 LCEL 前面，是为了让 LCEL 第一课就能演示完整的 `prompt | model | parser`。
- **05 记忆**依赖 02 的 `MessagesPlaceholder` 和消息列表；
- **06 工具与 Agent**依赖 01 的消息和 03 的结构化输出；
- **07 RAG**是 01-04 的综合应用，自己内部也有梯度（先零件 embedding/向量库/加载器/分块，再拼链，再进阶检索）；
- **08 流式 / 09 工程化 / 10 评估**是横切能力，复用前面几乎所有积木，放最后。

---

## 三、相对 archive_v1（旧 14 课）的优化

| 优化 | 说明 |
|---|---|
| 拆分原子化 | 旧版一课塞多个概念（如旧 03 LCEL 一课讲 8 个 Runnable）→ 新版一文件一个新概念 |
| 补 `init_chat_model` | 融合速通课：统一入口 + 一行换 provider（Qwen/DeepSeek/Ollama 同一套代码） |
| 补 `with_structured_output` | 旧版只有 PydanticOutputParser；新版补上现代首选写法并与旧写法对照 |
| 补文档加载器全家桶 | 融合速通：txt/pdf/csv/md/docx 各演示一遍，不只 txt |
| 补向量相似度直觉 | 融合速通：先用余弦相似度看懂「向量为什么能检索」，再上向量库 |
| 抽 `_common.py` | 修掉旧版「每个 RAG 文件重抄一遍 Embeddings 类」的毛病，演示工程上的「复用」 |
| 补进阶提质技巧 | few-shot 提示〔02/06〕、解析失败兜底 OutputFixingParser〔03/06〕、检索后处理压缩/重排〔07/09〕 |
| 固定教学模板 | 每个 `.py` docstring 统一为：【域/序号】标题 → 与上一课的差异 → 新概念（只一个）→ 为什么 → 分段代码 → ★核心规律 + 交叉引用 |

---

## 四、运行方式

`.env` 放在仓库根目录（`../.env`），`load_dotenv()` 会自动向上查找，无需在每个子文件夹复制。

```
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
API_KEY=sk-xxxx
MODEL=qwen-plus
```

在仓库根目录用项目 `.venv` 直接运行任意一课，例如：

```bash
python "course_v2/04_LCEL与Runnable/01_pipe_chain.py"
```

> 标〔进阶〕的文件第一遍可跳过。RAG 域的 `02` 会先建好本地 `faiss_index/`，后面几课复用它，建议按序号跑。
