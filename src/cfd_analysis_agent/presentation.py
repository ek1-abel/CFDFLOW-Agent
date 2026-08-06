"""Notebook-oriented presentation helpers for CFD analysis results."""

from __future__ import annotations

import html
from typing import Any, Iterable


TraceLike = Any
RunResultLike = Any


def _tool_label(tool_name: str | None) -> str:
    labels = {
        "PythonInterpreterTool": "Python 代码执行",
        "TavilySearchTool": "联网背景检索",
        "ResidualAnalysisTool": "残差收敛分析",
        "ForceAnalysisTool": "升阻力分析",
        "PressureAnalysisTool": "压力分布分析",
        "VelocityAnalysisTool": "速度剖面分析",
        "GridStudyTool": "网格无关性分析",
    }
    if tool_name in labels:
        return labels[tool_name]
    if tool_name:
        return tool_name
    return "报告生成"


def _status_label(status: str) -> str:
    mapping = {
        "success": "成功",
        "partial": "部分完成",
        "error": "失败",
        "unknown": "未知",
        "skipped": "跳过",
        "attempted": "已尝试",
    }
    return mapping.get(status, status)


def _escape(value: object) -> str:
    return html.escape(str(value))


def _trace_attr(trace: TraceLike, name: str, default: object = "") -> object:
    value = getattr(trace, name, default)
    return default if value is None else value


def _trace_status(trace: TraceLike) -> str:
    status = _trace_attr(trace, "tool_status", None)
    if status is None:
        status = _trace_attr(trace, "status", "unknown")
    return str(status or "unknown")


def _trace_decision(trace: TraceLike) -> str:
    return str(
        _trace_attr(trace, "decision", "")
        or _trace_attr(trace, "action", "")
        or _trace_attr(trace, "node", "")
        or _trace_attr(trace, "summary", "")
    )


def _stage_label(trace: TraceLike) -> str:
    action = str(_trace_attr(trace, "action", ""))
    node = str(_trace_attr(trace, "node", ""))
    tool_name = str(_trace_attr(trace, "tool_name", ""))
    node_labels = {
        "prepare": "准备数据上下文",
        "clean_data": "数据清洗",
        "select_tool": "工具路由",
        "execute_tool": "工具执行",
        "optional_search": "联网背景检索",
        "synthesize_report": "最终报告",
        "persist": "保存产物",
    }

    if action == "call_tool" or tool_name:
        label = _tool_label(tool_name or None)
        return f"{label} ({tool_name})" if tool_name else label
    if node:
        return f"{node_labels.get(node, node)} ({node})"
    return "最终报告"


def _trace_short_observation(trace: TraceLike) -> str:
    observation_preview = str(_trace_attr(trace, "observation_preview", ""))
    if observation_preview:
        return observation_preview
    observation = str(_trace_attr(trace, "observation", ""))
    if observation:
        return " ".join(observation.split())[:220]
    return ""


def _trace_diagnostic_text(trace: TraceLike) -> str:
    return str(
        _trace_attr(trace, "observation", "")
        or _trace_attr(trace, "parse_error", "")
        or _trace_attr(trace, "observation_preview", "")
        or _trace_attr(trace, "summary", "")
        or "No diagnostic text available."
    )


def _iter_failed_traces(step_traces: Iterable[TraceLike]) -> list[TraceLike]:
    failed = []
    for trace in step_traces:
        diagnostic_text = _trace_diagnostic_text(trace)
        if _trace_status(trace) == "error" or "Traceback" in diagnostic_text:
            failed.append(trace)
    return failed


def render_trace_table(result: RunResultLike):
    from IPython.display import HTML

    rows = []
    for trace in result.step_traces:
        rows.append(
            """
            <tr>
              <td style="border:1px solid #d1d5db; padding:8px; vertical-align:top;">{step}</td>
              <td style="border:1px solid #d1d5db; padding:8px; vertical-align:top;">{stage}</td>
              <td style="border:1px solid #d1d5db; padding:8px; vertical-align:top;">{decision}</td>
              <td style="border:1px solid #d1d5db; padding:8px; vertical-align:top;">{status}</td>
              <td style="border:1px solid #d1d5db; padding:8px; vertical-align:top;">{observation}</td>
              <td style="border:1px solid #d1d5db; padding:8px; vertical-align:top;">{notes}</td>
            </tr>
            """.format(
                step=_escape(_trace_attr(trace, "step_index", "")),
                stage=_escape(_stage_label(trace)),
                decision=_escape(_trace_decision(trace)),
                status=_escape(_status_label(_trace_status(trace))),
                observation=_escape(_trace_short_observation(trace) or "无"),
                notes=_escape(_trace_attr(trace, "summary", "") or _trace_attr(trace, "parse_error", "") or "无"),
            )
        )

    html_content = """
    <h2>CFD Agent 推理轨迹表</h2>
    <table style="width:100%; border-collapse:collapse; font-size:14px;">
      <thead>
        <tr style="background:#f3f4f6;">
          <th style="border:1px solid #d1d5db; padding:8px;">Step</th>
          <th style="border:1px solid #d1d5db; padding:8px;">Stage / Tool</th>
          <th style="border:1px solid #d1d5db; padding:8px;">Decision</th>
          <th style="border:1px solid #d1d5db; padding:8px;">Status</th>
          <th style="border:1px solid #d1d5db; padding:8px;">Short Observation</th>
          <th style="border:1px solid #d1d5db; padding:8px;">Notes</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """.format(rows="".join(rows))
    return HTML(html_content)


def render_full_report(result: RunResultLike):
    from IPython.display import Markdown
    return Markdown("## 完整报告正文\n\n" + result.report_markdown)


def render_diagnostics(result: RunResultLike):
    from IPython.display import HTML

    failed_traces = _iter_failed_traces(result.step_traces)
    if not failed_traces:
        return HTML("<h2>错误与诊断详情</h2><p>本次运行无工具级异常。</p>")

    details_blocks = []
    for trace in failed_traces:
        trace_label = _trace_attr(trace, "node", "") or _trace_attr(trace, "tool_name", "") or _trace_attr(trace, "action", "")
        title = f"Step {_trace_attr(trace, 'step_index', '')} {trace_label} Diagnostics".strip()
        body = _escape(_trace_diagnostic_text(trace))
        details_blocks.append(
            f"""
            <details style="margin-bottom:12px;">
              <summary style="cursor:pointer; font-weight:600;">{_escape(title)}</summary>
              <pre style="white-space:pre-wrap; background:#111827; color:#f9fafb; padding:12px; border-radius:8px; margin-top:8px;">{body}</pre>
            </details>
            """
        )

    return HTML("<h2>错误与诊断详情</h2>" + "".join(details_blocks))
