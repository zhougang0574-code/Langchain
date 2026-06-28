# CLAUDE.md

给后续 Claude Code 会话的项目说明，打开即读。**先读完这份再动手。**

## 这是什么

LangChain 的**中文教学项目**：一套循序渐进的代码课 + 一份网页图文笔记。
学习者以「回顾、查漏补缺」为主 —— 代码注释 / 文档 / 讲解一律用中文。
LLM 用阿里百炼 **qwen-plus**（兼容 OpenAI 接口）。

> 姊妹项目：`../LangGraph`（同一学习者的 LangGraph 课程）。复杂 Agent、多 Agent、
> 状态持久化、人工干预等放在那边讲，本项目不重复——两套课互补。

## 权威版本（最重要）

- **`course_v2/` 是唯一在维护的学习版本，一切以它为准。**
- `archive_v1/`（旧版线性 14 课）**已废弃**，只作回溯参考，**不要再改动它**。
- 课程的权威目录树和学习顺序见 **`course_v2/README.md`**（改了结构要同步更新它）。

## 课程结构

10 个「概念域」、50 个 `.py`，**目录结构即学习路径**，从上往下学：

```
01 基础 → 02 提示词Prompt → 03 输出解析 → 04 LCEL与Runnable → 05 记忆Memory
→ 06 工具与Agent → 07 检索增强RAG → 08 流式与回调 → 09 工程化与可靠性 → 10 评估
```

设计轴：**分类轴（按概念域分组）+ 域内梯度（每域内 01→N 小步递进）**。
前半「原语轴」（Model/Prompt/Parser/LCEL 四件套）→ 后半「能力轴」（记忆/Agent/RAG/流式/工程化/评估）。
标〔进阶〕的课放各域末尾，第一遍可跳过。

## 教学法（硬性原则）

1. **一个文件只引入一个新概念**；其余代码尽量和上一课保持一致，靠"对比上一课哪里不一样"来教。
2. **宁可多拆几课，也不要一课塞太多**（旧版一课塞多个概念，是这次重构要修的核心问题）。
3. **概念累积、不回退**：后面的课在前面的基础上加一层。

## 文件内约定

每个 `.py` 顶部 docstring 固定结构：
1. 标题行 `【域名 / 序号】`
2. 与上一课的区别
3. `新概念（只有这一个）`
4. 为什么需要

文件末尾用多行字符串写「执行流程图 + ★核心规律」。跨课引用统一写 **`〔域号/课号〕`**，例如 `〔04/04〕`、`〔07/05〕`、`〔05/02〕`。

> ⚠️ 改了课程编号 / 顺序，必须同步更新所有 `【…】` 标题和 `〔…〕` 引用。

## 共享代码

`course_v2/_common.py` 放「会被多课复用」的东西，目前是阿里百炼的 `AliyunEmbeddings`
（OpenAI 兼容接口，不需要装 dashscope）。RAG / 评估域的文件靠下面这段把它导进来：

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import get_embeddings
```

LLM（ChatOpenAI）仍在每个文件就地创建（只有 4 行，是讲解核心，就地写更利于阅读）。

## HTML 笔记是「生成」的，不是手写的（与 LangGraph 项目不同）

`course_v2/langchain_notes.html` 由 **`course_v2/build_notes.py` 生成**。
**不要手改 HTML**——内容以结构化数据维护在 `build_notes.py` 顶部的 `DOMAINS` 列表里。

- 每课一条：`(锚id, 标题, 概念, 核心规律, 代码片段)`；每个域：`(锚, 标题, 主题色, [课...])`。
- 改完跑一次重新生成：`python course_v2/build_notes.py`
- 课程数、侧栏 meta 等都从 `DOMAINS` 自动算，无需手填（别再硬编码课数）。
- 暗色护眼主题，CSS 变量在 `TEMPLATE` 的 `:root`；侧栏手风琴 + IntersectionObserver 滚动高亮。

## 新增 / 调整一课的步骤

1. 在对应域文件夹放 `NN_xxx.py`（序号紧接该域已有文件；插在中间则后续文件和所有 `〔…〕` 引用都要重编号）。
2. 写 docstring（标题 / 区别 / 新概念 / 为什么）+ 代码 + 末尾流程图，遵循「单概念」原则。
3. 在 `build_notes.py` 的 `DOMAINS` 里对应域加一条课数据，然后 `python course_v2/build_notes.py` 重新生成 HTML。
4. 更新 `course_v2/README.md` 的目录树（根 `README.md` 不写死课数，一般不用动）。
5. 校验：`find course_v2 -name "*.py" -print0 | xargs -0 python -m py_compile` 全过；
   能跑的实跑一遍（RAG 域要先跑 `07/02` 建好 `faiss_index/`）；HTML 课卡片数 == 导航项数 == `.py` 文件数。

## 运行与环境

- `.env` 只放在**仓库根目录**（`load_dotenv()` 向上查找），格式：
  ```
  BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  API_KEY=sk-xxxx
  MODEL=qwen-plus
  ```
- 用项目根的 `.venv` 运行，例：`python "course_v2/06_工具与Agent/03_create_agent.py"`
- 依赖见 `requirements.txt`（langchain 1.3.x）。几处可选库不装会 ImportError，已 try/except 或在文档标注：
  `docx2txt`（07/03 加载 .docx）、`unstructured`（加载 .md/.html）。
- **RAG 域先建索引**：先跑 `07/02_vector_store.py` 生成 `course_v2/07_检索增强RAG/faiss_index/`，
  `07/06 07/07 07/08 07/09` 和 `10/02` 都加载它。该目录已被 `.gitignore`，可随时重建。
- 向量化用 `text-embedding-v1`；其 FAISS L2 距离量级在数千~两万（相关 <12000），分数过滤阈值按此校准。
