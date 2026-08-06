"""Gradio workstation for the LangGraph CFDFlow-Agent copy."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cfd_analysis_agent import run_analysis


ALLOWED_SUFFIXES = {".csv", ".xlsx", ".xls", ".pdf"}
SAMPLE_DATASETS = {
    path.name: path
    for path in sorted((PROJECT_ROOT / "data").glob("*"))
    if path.suffix.lower() in ALLOWED_SUFFIXES
}


def _uploaded_file_path(uploaded_file: Any) -> Path | None:
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, (str, Path)):
        return Path(uploaded_file)
    name = getattr(uploaded_file, "name", None)
    return Path(name) if name else None


def _resolve_data_path(uploaded_file: Any, sample_dataset: str | None) -> Path:
    uploaded_path = _uploaded_file_path(uploaded_file)
    if uploaded_path is not None:
        suffix = uploaded_path.suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
            raise ValueError(f"Unsupported file type: {suffix}. Allowed: {allowed}")
        return uploaded_path

    if sample_dataset and sample_dataset in SAMPLE_DATASETS:
        return SAMPLE_DATASETS[sample_dataset]

    if SAMPLE_DATASETS:
        return next(iter(SAMPLE_DATASETS.values()))

    raise FileNotFoundError("No uploaded file or bundled sample dataset is available.")


def _step_trace_dataframe(result: Any) -> pd.DataFrame:
    rows = [
        {
            "step": trace.step_index,
            "node": trace.node,
            "action": trace.action,
            "status": trace.status,
            "tool": trace.tool_name,
            "duration_ms": trace.duration_ms,
            "summary": trace.summary,
        }
        for trace in result.step_traces
    ]
    return pd.DataFrame(rows)


def _gallery_items(result: Any) -> list[tuple[str, str]]:
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    items: list[tuple[str, str]] = []
    for image_path in sorted(result.figures_dir.rglob("*")):
        if image_path.suffix.lower() in image_suffixes and image_path.exists():
            items.append((str(image_path), image_path.name))
    return items


def _artifact_files(result: Any) -> list[str]:
    paths = [
        result.report_path,
        result.trace_path,
        result.cleaned_data_path,
    ]
    paths.extend(path for path, _caption in _gallery_items(result))
    return [str(Path(path)) for path in paths if Path(path).exists()]


def _load_trace_payload(result: Any) -> dict[str, Any]:
    if result.trace_path.exists():
        return json.loads(result.trace_path.read_text(encoding="utf-8"))
    return {}


def _summary_markdown(result: Any) -> str:
    status = "✅ 完成" if result.workflow_complete else "⚠️ 部分完成"
    return (
        f"### {status}\n"
        f"- Agent 类型：`{result.agent_type}`\n"
        f"- CFD 任务类型：`{result.cfd_task_type}`\n"
        f"- 选中工具：`{result.selected_tool}`\n"
        f"- Context 策略：`{result.context_strategy}`\n"
        f"- 最近消息窗口：`{result.context_window_turns}` 轮，当前保留 `{len(result.recent_messages)}` 条消息\n"
        f"- 报告路径：`{result.report_path}`\n"
        f"- Trace 路径：`{result.trace_path}`"
    )


def run_workflow(
    uploaded_file: Any,
    sample_dataset: str | None,
    query: str,
    context_window_turns: int,
    use_llm_report: bool,
    enable_search: bool,
    quality_mode: str,
    latency_mode: str,
) -> tuple[str, str, pd.DataFrame, list[tuple[str, str]], dict[str, Any], list[dict[str, str]], dict[str, Any], list[str]]:
    """Run the LangGraph workflow from Gradio controls."""

    try:
        data_path = _resolve_data_path(uploaded_file, sample_dataset)
        result = run_analysis(
            data_path,
            query=query.strip() or "请对这个 CFD 后处理数据进行分析。",
            output_dir=PROJECT_ROOT / "outputs" / "gradio_runs",
            env_file=PROJECT_ROOT / ".env",
            context_window_turns=max(1, int(context_window_turns)),
            use_llm_report=bool(use_llm_report),
            enable_search=bool(enable_search),
            quality_mode=quality_mode,
            latency_mode=latency_mode,
            verbose=False,
        )
        return (
            _summary_markdown(result),
            result.report_markdown,
            _step_trace_dataframe(result),
            _gallery_items(result),
            result.structured_facts,
            list(result.recent_messages),
            _load_trace_payload(result),
            _artifact_files(result),
        )
    except Exception as exc:  # Gradio should surface a readable error panel.
        empty_trace = pd.DataFrame(columns=["step", "node", "action", "status", "tool", "duration_ms", "summary"])
        return (
            f"### ❌ 运行失败\n- 错误类型：`{type(exc).__name__}`\n- 错误信息：{exc}",
            "",
            empty_trace,
            [],
            {},
            [],
            {"error_type": type(exc).__name__, "error": str(exc)},
            [],
        )


def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks app without launching it."""

    default_sample = next(iter(SAMPLE_DATASETS.keys()), None)
    with gr.Blocks(title="CFDFlow-Agent LangGraph Workbench") as demo:
        gr.Markdown(
            """
            # CFDFlow-Agent 可视化工作台

            上传 CFD 后处理数据或选择内置样例，工作台会展示 LangGraph 节点执行流、
            结构化事实摘要、最近消息滑动窗口、最终报告和生成图表。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                uploaded_file = gr.File(
                    label="上传 CSV / Excel / PDF",
                    file_types=sorted(ALLOWED_SUFFIXES),
                    type="filepath",
                )
                sample_dataset = gr.Dropdown(
                    label="或选择内置样例",
                    choices=list(SAMPLE_DATASETS.keys()),
                    value=default_sample,
                )
                query = gr.Textbox(
                    label="分析问题",
                    value="请分析该 CFD 后处理数据，判断收敛状态并生成工程解释。",
                    lines=4,
                )
                context_window_turns = gr.Slider(
                    label="最近原始消息窗口 N",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=3,
                )
                use_llm_report = gr.Checkbox(
                    label="启用 LLM 报告润色（需要 .env配置）",
                    value=False,
                )
                enable_search = gr.Checkbox(
                    label="启用 Tavily 外部检索（需要 .env配置）",
                    value=False,
                )
                quality_mode = gr.Dropdown(
                    label="质量模式",
                    choices=["draft", "standard", "publication"],
                    value="draft",
                )
                latency_mode = gr.Dropdown(
                    label="延迟模式",
                    choices=["fast", "auto", "thorough"],
                    value="fast",
                )
                run_button = gr.Button("运行 LangGraph 工作流", variant="primary")

            with gr.Column(scale=2):
                status_markdown = gr.Markdown(label="运行状态")
                with gr.Tabs():
                    with gr.Tab("报告"):
                        report_markdown = gr.Markdown()
                    with gr.Tab("LangGraph 节点"):
                        step_trace = gr.Dataframe(
                            headers=["step", "node", "action", "status", "tool", "duration_ms", "summary"],
                            wrap=True,
                            interactive=False,
                        )
                    with gr.Tab("图表"):
                        gallery = gr.Gallery(label="生成图表", columns=2, height=420)
                    with gr.Tab("结构化事实摘要"):
                        structured_facts = gr.JSON(label="structured_facts")
                    with gr.Tab("最近消息窗口"):
                        recent_messages = gr.JSON(label="recent_messages")
                    with gr.Tab("原始 Trace"):
                        raw_trace = gr.JSON(label="agent_trace_langgraph.json")
                    with gr.Tab("产物下载"):
                        artifacts = gr.File(label="报告 / Trace / 清洗数据 / 图表", file_count="multiple")

        run_button.click(
            fn=run_workflow,
            inputs=[
                uploaded_file,
                sample_dataset,
                query,
                context_window_turns,
                use_llm_report,
                enable_search,
                quality_mode,
                latency_mode,
            ],
            outputs=[
                status_markdown,
                report_markdown,
                step_trace,
                gallery,
                structured_facts,
                recent_messages,
                raw_trace,
                artifacts,
            ],
        )

    return demo


if __name__ == "__main__":
    build_demo().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
