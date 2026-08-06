# CFDFlow-Agent

该项目是基于 LangGraph + Gradio + Tool System 构建 CFD 后处理智能分析工作台，支持上传 CSV/Excel/PDF 数据后自动完成数据清洗、任务分类、工具调用、指标计算、图表生成和报告输出，并可视化展示节点执行流、结构化上下文和 trace 产物。

该项目配备了jupyter notebook作为主要展示，您也可以在.env中配置对应API,实现Gradio工作台可视化


## 项目亮点

| 特性 | 说明 |
|------|------|
| 专业工具路由 | Agent 根据列名自动选择残差收敛、升阻力、压力分布、速度剖面、网格无关性分析工具 |
| 确定性计算 | 残差下降阶数、Cl/Cd 统计、Richardson 外推、GCI 由 Python 工具完成 |
| LangGraph 编排 | 将 prepare、clean_data、select_tool、execute_tool、optional_search、synthesize_report、persist 固化为可观测节点 |
| 上下文工程 | 使用 structured_facts 保存稳定事实，使用 recent_messages 保留最近 N 轮原始证据 |
| 结构化报告 | 固定 7 章节：算例概况→收敛判断→关键指标→图表→异常点→工程解释→局限性 |
| 可追踪性 | 每次运行保存 `agent_trace_langgraph.json`、`final_report.md`、`figures/` |

## 快速开始

```bash
# 1. 激活环境（建议激活虚拟环境 python >=3.10)
conda activate 

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 ZHIPUAI_API_KEY

# 4. 生成示例数据（如未生成）
python data/generate_sample_data.py

# 5. 运行 Jupyter Notebook
jupyter notebook main.ipynb

# 6. 离线 smoke test（不调用 LLM / Tavily）
python smoke_test.py

# 7. 启动 Gradio 可视化工作台
python app_gradio.py

# 8. Gradio UI smoke test（不启动 Web 服务）
python smoke_test_ui.py
```

启动后在浏览器打开 `http://127.0.0.1:7860`。默认 `share=False`，不会主动暴露公网链接；如启用 LLM 报告或 Tavily 检索，需要在本地 `.env` 中配置密钥，但工作台不会展示密钥内容。

## 5 个 CFD 分析工具

### ResidualAnalysisTool — 残差收敛分析
- 输入：残差历史 CSV（iteration + 各残差列）
- 计算：每个残差的下降阶数 = log₁₀(初值/终值)，收敛判定（≥3阶 + 末10%稳定）
- 输出：半对数残差图 + `[CONVERGED]` 或 `[WARNING: NOT CONVERGED]` 状态

### ForceAnalysisTool — 升阻力系数分析
- 输入：力系数 CSV（iteration + Cl/Cd/Cm）
- 计算：均值、标准差、振荡检测（std/mean > 阈值）、稳态判定
- 输出：力系数时间历程图 + 均值线 + ±1σ 带 + `[STEADY]` 或 `[WARNING: FORCE OSCILLATION]`

### PressureAnalysisTool — 压力分布分析
- 输入：Cp CSV（x/c + Cp_upper/Cp_lower）
- 计算：吸力峰值位置和大小、驻点识别
- 输出：Cp vs x/c 图（y 轴反转，符合气动惯例）

### VelocityAnalysisTool — 速度剖面分析
- 输入：速度剖面 CSV（y_plus/u_plus 或 y/u）
- 计算：与壁面律（u⁺=y⁺）和对数律（u⁺=1/κ·ln(y⁺)+B）对比，边界层厚度估计
- 输出：速度剖面图 + 理论曲线叠加

### GridStudyTool — 网格无关性分析
- 输入：网格研究 CSV（mesh_level, cell_count, Cl, Cd 等）
- 计算：Richardson 外推（表观收敛阶 p）、GCI（Fs=1.25）
- 输出：解 vs 网格尺寸图 + 不确定度带

## 项目结构

```
CFDFlow-Agent/
├── main.ipynb                          # 入口 notebook，5 个 Demo
├── requirements.txt
├── .env.example
├── data/                               # 合成 CFD 数据
│   ├── generate_sample_data.py
│   ├── residual_history.csv
│   ├── force_coefficients.csv
│   ├── pressure_distribution.csv
│   ├── velocity_profile.csv
│   └── mesh_study.csv
├── src/cfd_analysis_agent/
│   ├── agent_runner.py                 # 核心编排器 + Scientific ReAct Runner
│   ├── cfd_classifier.py              # 列名模式匹配 → CFD 任务分类
│   ├── data_context.py                # 数据上下文构建 + CFD 分类集成
│   ├── prompts.py                     # CFD 领域提示词
│   ├── presentation.py                # Notebook 展示辅助
│   ├── reporting.py                   # 报告提取与持久化
│   ├── config.py                      # 运行时配置
│   ├── llm.py                         # LLM 构建
│   ├── plotting.py                    # 绘图辅助
│   ├── tool_protocol.py               # 工具协议
│   ├── document_ingestion.py          # PDF 文档解析
│   ├── vision_review.py               # 视觉审稿
│   └── tools/
│       ├── python_interpreter.py      # Python 代码执行
│       ├── tavily_search.py           # 联网检索
│       ├── residual_analysis.py       # 残差收敛分析
│       ├── force_analysis.py          # 升阻力分析
│       ├── pressure_analysis.py       # 压力分布分析
│       ├── velocity_analysis.py       # 速度剖面分析
│       └── grid_study.py             # 网格无关性分析
└── outputs/                           # 运行输出（自动创建）
```

## 技术架构

```
用户输入 (CSV/PDF)
    │
    ▼
Document Ingestion → 表格提取
    │
    ▼
Data Context Builder → CFD 列名分类 (cfd_classifier.py)
    │                   → 任务类型 + 推荐工具
    ▼
Scientific ReAct Runner (JSON 驱动的 ReAct 控制器)
    │
    ├─ Stage 1: 数据清洗 → cleaned_data.csv (PythonInterpreterTool)
    │
    ├─ Stage 2: CFD 分析 → 专业工具路由
    │   ├─ ResidualAnalysisTool    (残差数据)
    │   ├─ ForceAnalysisTool       (力系数数据)
    │   ├─ PressureAnalysisTool    (Cp 分布数据)
    │   ├─ VelocityAnalysisTool    (速度数据)
    │   └─ GridStudyTool           (网格研究数据)
    │
    ├─ 收敛性保护 (三层)
    │   ├─ 工具层: [CONVERGED] / [WARNING: NOT CONVERGED]
    │   ├─ Observation 层: 强制要求 Agent 回应警告
    │   └─ Reviewer 层: 审查遗漏的收敛警告
    │
    ▼
结构化报告 (7 章节) + trace.json + figures/
```

## 收敛性保护机制

1. **工具层**：每个 CFD 工具的返回值以标签开头（`[CONVERGED]` / `[WARNING: NOT CONVERGED]`）
2. **Observation Prompt 层**：明确要求 Agent 在报告中正面回应任何 WARNING
3. **Reviewer 层**（standard/publication 模式）：审查报告是否遗漏收敛警告，遗漏则 REJECT
