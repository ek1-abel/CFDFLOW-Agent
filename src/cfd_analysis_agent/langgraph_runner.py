"""LangGraph-first CFDFlow-Agent runner with context-engineered memory.

This runner changes the orchestration layer from the original hand-written
HelloAgents ReAct controller to an explicit LangGraph workflow. The CFD tools
remain deterministic Python tools, while the agent context is managed as:

1. structured facts summary: stable facts extracted from data/tool outputs;
2. recent raw message sliding window: only the latest N turns are passed to LLM.
"""

from __future__ import annotations

import copy
import json
import operator
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from openai import OpenAI
from typing_extensions import Annotated, TypedDict

from .config import RuntimeConfig, load_runtime_config
from .data_context import DataContextSummary, build_data_context
from .document_ingestion import IngestionResult, ingest_input_document
from .prompts import DEFAULT_QUERY
from .reporting import ReportTelemetry, extract_report_and_telemetry, save_markdown_report
from .tools.force_analysis import ForceAnalysisTool
from .tools.grid_study import GridStudyTool
from .tools.pressure_analysis import PressureAnalysisTool
from .tools.python_interpreter import PythonInterpreterTool
from .tools.residual_analysis import ResidualAnalysisTool
from .tools.tavily_search import TavilySearchTool
from .tools.velocity_analysis import VelocityAnalysisTool

try:  # Optional dependency until users install requirements.txt in this copy.
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError as exc:  # pragma: no cover - only hit without langgraph.
    END = None
    StateGraph = None
    _LANGGRAPH_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None


EventHandler = Callable[[str, dict[str, Any]], None]

SPECIALIZED_TOOL_FACTORIES = {
    "ResidualAnalysisTool": ResidualAnalysisTool,
    "ForceAnalysisTool": ForceAnalysisTool,
    "PressureAnalysisTool": PressureAnalysisTool,
    "VelocityAnalysisTool": VelocityAnalysisTool,
    "GridStudyTool": GridStudyTool,
    "PythonInterpreterTool": PythonInterpreterTool,
}

SEARCH_SIGNAL_KEYWORDS = (
    "residual",
    "convergence",
    "turbulence",
    "reynolds",
    "mach",
    "boundary layer",
    "grid independence",
    "richardson extrapolation",
    "gci",
    "naca",
    "airfoil",
    "drag coefficient",
    "lift coefficient",
    "pressure coefficient",
    "残差",
    "收敛",
    "湍流",
    "雷诺数",
    "马赫数",
    "边界层",
    "网格无关性",
    "升力系数",
    "阻力系数",
    "压力系数",
)


@dataclass(frozen=True)
class LangGraphStepTrace:
    """One observable LangGraph node execution record."""

    step_index: int
    node: str
    action: str
    status: str
    summary: str
    tool_name: str = ""
    observation_preview: str = ""
    duration_ms: int = 0


@dataclass(frozen=True)
class LangGraphCFDRunResult:
    """Return object for the LangGraph version."""

    data_context: DataContextSummary
    report_markdown: str
    report_path: Path
    output_dir: Path
    run_dir: Path
    data_dir: Path
    figures_dir: Path
    logs_dir: Path
    trace_path: Path
    cleaned_data_path: Path
    agent_type: str
    step_traces: tuple[LangGraphStepTrace, ...]
    telemetry: ReportTelemetry
    methods_used: tuple[str, ...]
    detected_domain: str
    tools_used: tuple[str, ...]
    search_status: str
    search_notes: str
    workflow_complete: bool
    workflow_warnings: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    quality_mode: str = "standard"
    review_enabled: bool = False
    review_status: str = "skipped"
    review_rounds_used: int = 0
    review_critique: str = "LangGraph copy does not run the legacy reviewer by default."
    review_log_paths: tuple[Path, ...] = ()
    input_kind: str = "tabular"
    document_ingestion_status: str = "not_needed"
    document_ingestion_summary: str = ""
    document_ingestion_duration_ms: int = 0
    document_ingestion_log_path: Path | None = None
    candidate_table_count: int = 0
    selected_table_id: str = ""
    selected_table_shape: tuple[int, int] | None = None
    pdf_multi_table_mode: bool = False
    latency_mode: str = "auto"
    vision_review_mode: str = "off"
    vision_review_enabled: bool = False
    vision_review_status: str = "skipped"
    vision_review_summary: str = "LangGraph copy does not run visual review by default."
    vision_review_duration_ms: int = 0
    vision_review_log_paths: tuple[Path, ...] = ()
    total_duration_ms: int = 0
    llm_duration_ms: int = 0
    tool_duration_ms: int = 0
    review_duration_ms: int = 0
    timing_breakdown: dict[str, int] = field(default_factory=dict)
    cfd_task_type: str = "general_cfd"
    selected_tool: str = ""
    tool_status: str = "unknown"
    context_strategy: str = "structured_facts_summary + recent_raw_message_sliding_window"
    context_window_turns: int = 3
    structured_facts: dict[str, Any] = field(default_factory=dict)
    recent_messages: tuple[dict[str, str], ...] = ()


class CFDGraphState(TypedDict, total=False):
    """LangGraph state passed between nodes."""

    data_path: str
    query: str
    output_dir: str
    report_path: str | None
    env_file: str | None
    started_at: float
    use_llm_report: bool
    enable_search: bool
    context_window_turns: int
    document_ingestion_mode: str
    max_pdf_pages: int
    max_candidate_tables: int
    selected_table_id: str | None
    runtime_config: RuntimeConfig | None
    data_context: DataContextSummary
    document_ingestion: IngestionResult
    selected_tool: str
    tool_input: dict[str, Any]
    tool_observation: str
    search_observation: str
    report_markdown: str
    telemetry: ReportTelemetry
    run_dir: str
    data_dir: str
    figures_dir: str
    logs_dir: str
    final_report_path: str
    trace_path: str
    cleaned_data_path: str
    facts: dict[str, Any]
    recent_messages: list[dict[str, str]]
    step_traces: Annotated[list[dict[str, Any]], operator.add]
    timing_breakdown: dict[str, int]
    trace: dict[str, Any]


def _ensure_langgraph_available() -> None:
    if StateGraph is None or END is None:
        raise ModuleNotFoundError(
            "LangGraph is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from _LANGGRAPH_IMPORT_ERROR


def _emit_event(event_handler: Optional[EventHandler], event_type: str, **payload: Any) -> None:
    if event_handler is not None:
        event_handler(event_type, payload)


def build_plaintext_event_handler() -> EventHandler:
    """Build a lightweight stdout event handler for notebooks and scripts."""

    def handle_event(event_type: str, payload: dict[str, Any]) -> None:
        message = payload.get("message")
        if message:
            print(f"[{event_type}] {message}")
        elif event_type == "node_started":
            print(f"LangGraph node started: {payload.get('node')}")
        elif event_type == "node_finished":
            print(f"LangGraph node finished: {payload.get('node')} | status={payload.get('status', 'ok')}")
        elif event_type == "report_saved":
            print(f"Report: {payload.get('report_path')}")
            print(f"Trace: {payload.get('trace_path')}")

    return handle_event


def _elapsed_ms(start_time: float) -> int:
    return int(round((time.perf_counter() - start_time) * 1000))


def _create_run_directory(output_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    run_dir = Path(output_dir) / f"run_langgraph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir = run_dir / "data"
    figures_dir = run_dir / "figures" / "review_round_1"
    logs_dir = run_dir / "logs"
    for directory in (data_dir, figures_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return run_dir, data_dir, figures_dir, logs_dir


def _truncate_text(text: Any, limit: int = 1200) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + " ... [truncated]"


def _read_json_observation(raw_observation: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_observation)
    except json.JSONDecodeError:
        return {"status": "unknown", "text": raw_observation, "data": {}}
    return payload if isinstance(payload, dict) else {"status": "unknown", "text": str(payload), "data": {}}


def _status_from_observation(raw_observation: str) -> str:
    return str(_read_json_observation(raw_observation).get("status", "unknown"))


def _safe_jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, tuple):
            return [_safe_jsonable(item) for item in value]
        if hasattr(value, "__dict__"):
            return {key: _safe_jsonable(item) for key, item in vars(value).items()}
        return str(value)


def _initial_facts(query: str) -> dict[str, Any]:
    return {
        "schema_version": "cfdflow-context-facts-v1",
        "user_query": query,
        "data": {},
        "workflow": [],
        "tool_calls": [],
        "warnings": [],
        "figures": [],
        "search": {
            "status": "not_used",
            "notes": "No online knowledge retrieval was triggered.",
            "results": [],
        },
    }


def _append_recent_message(
    messages: list[dict[str, str]],
    *,
    role: str,
    content: str,
    context_window_turns: int,
) -> list[dict[str, str]]:
    updated = [*messages, {"role": role, "content": content}]
    max_messages = max(1, int(context_window_turns)) * 2
    return updated[-max_messages:]


def _append_workflow_fact(facts: dict[str, Any], *, node: str, status: str, summary: str) -> dict[str, Any]:
    updated = copy.deepcopy(facts)
    updated.setdefault("workflow", []).append(
        {
            "node": node,
            "status": status,
            "summary": summary,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return updated


def _extract_warning_lines(text: str) -> list[str]:
    warning_lines = []
    for line in str(text or "").splitlines():
        if "WARNING" in line.upper() or "未收敛" in line or "不宜作为稳态结论" in line:
            warning_lines.append(line.strip())
    return warning_lines


def _extract_tool_facts(tool_name: str, raw_observation: str) -> dict[str, Any]:
    payload = _read_json_observation(raw_observation)
    data = payload.get("data", {})
    text = str(payload.get("text", ""))
    tool_fact = {
        "tool_name": tool_name,
        "status": str(payload.get("status", "unknown")),
        "text_preview": _truncate_text(text, 900),
        "data": {},
    }

    if isinstance(data, dict):
        for key in (
            "convergence_status",
            "residual_details",
            "force_statistics",
            "pressure_metrics",
            "velocity_metrics",
            "grid_convergence",
            "figure_path",
            "figures_saved",
            "stdout",
            "stderr",
            "warnings",
        ):
            if key in data:
                tool_fact["data"][key] = _safe_jsonable(data[key])

    figure_paths = []
    for value in (tool_fact["data"].get("figure_path"), tool_fact["data"].get("figures_saved")):
        if isinstance(value, str) and value:
            figure_paths.append(value)
        elif isinstance(value, list):
            figure_paths.extend(str(item) for item in value if str(item))

    warnings = _extract_warning_lines(text)
    if isinstance(tool_fact["data"].get("warnings"), list):
        warnings.extend(str(item) for item in tool_fact["data"]["warnings"])
    if tool_fact["status"] not in {"success", "partial"}:
        warnings.append(f"{tool_name} returned status={tool_fact['status']}")

    return {
        "tool_fact": tool_fact,
        "figure_paths": figure_paths,
        "warnings": warnings,
    }


def _add_tool_facts(facts: dict[str, Any], *, tool_name: str, raw_observation: str) -> dict[str, Any]:
    extracted = _extract_tool_facts(tool_name, raw_observation)
    updated = copy.deepcopy(facts)
    updated.setdefault("tool_calls", []).append(extracted["tool_fact"])
    for figure_path in extracted["figure_paths"]:
        if figure_path not in updated.setdefault("figures", []):
            updated["figures"].append(figure_path)
    for warning in extracted["warnings"]:
        if warning and warning not in updated.setdefault("warnings", []):
            updated["warnings"].append(warning)
    return updated


def _build_cleaning_code(source_path: Path, cleaned_data_path: Path) -> str:
    return f"""
from pathlib import Path
import pandas as pd

source_path = Path({json.dumps(source_path.as_posix())})
cleaned_data_path = Path({json.dumps(cleaned_data_path.as_posix())})
cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)

if source_path.suffix.lower() == ".csv":
    df = pd.read_csv(source_path)
else:
    df = pd.read_excel(source_path)

df = df.dropna(how="all")
df.columns = [str(col).strip() for col in df.columns]
for col in df.columns:
    original_non_null = int(df[col].notna().sum())
    converted = pd.to_numeric(df[col], errors="coerce")
    if original_non_null == 0 or int(converted.notna().sum()) == original_non_null:
        df[col] = converted

df.to_csv(cleaned_data_path, index=False)
print(f"Cleaned data saved to: {{cleaned_data_path.as_posix()}}")
print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
print("Columns:", list(df.columns))
print("Missing cells:", int(df.isna().sum().sum()))
""".strip()


def _build_general_analysis_code(cleaned_data_path: Path, figures_dir: Path) -> str:
    return f"""
from pathlib import Path
import pandas as pd

data_path = Path({json.dumps(cleaned_data_path.as_posix())})
figures_dir = Path({json.dumps(figures_dir.as_posix())})
figures_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)
print("General CFD data overview")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.describe(include="all").to_string())
""".strip()


def _build_tool_input(selected_tool: str, cleaned_data_path: Path, figures_dir: Path) -> dict[str, Any]:
    if selected_tool == "PythonInterpreterTool":
        return {"code": _build_general_analysis_code(cleaned_data_path, figures_dir)}
    payload = {
        "data_path": cleaned_data_path.as_posix(),
        "figures_dir": figures_dir.as_posix(),
    }
    return {"input": json.dumps(payload, ensure_ascii=False)}


def _should_enable_search(data_context: DataContextSummary, query: str, *, enable_search: bool) -> bool:
    if not enable_search or not os.getenv("TAVILY_API_KEY"):
        return False
    searchable_text = " ".join([query, *data_context.columns]).lower()
    return any(keyword in searchable_text for keyword in SEARCH_SIGNAL_KEYWORDS)


def _build_search_query(data_context: DataContextSummary, user_query: str) -> str:
    task_queries = {
        "residual_convergence": "CFD residual convergence criteria residual drop orders steady simulation",
        "force_coefficients": "CFD lift drag coefficient convergence oscillation criterion aerodynamic coefficients",
        "pressure_distribution": "airfoil pressure coefficient Cp distribution suction peak interpretation",
        "velocity_profile": "CFD boundary layer velocity profile y plus u plus log law interpretation",
        "grid_independence": "CFD grid independence study Richardson extrapolation GCI method",
    }
    return task_queries.get(data_context.cfd_task_type, f"CFD post processing {user_query}")[:240]


def _build_report_messages(state: CFDGraphState) -> list[dict[str, str]]:
    facts = state.get("facts", {})
    recent_messages = state.get("recent_messages", [])
    data_context = state["data_context"]
    facts_json = json.dumps(_safe_jsonable(facts), ensure_ascii=False, indent=2)
    recent_json = json.dumps(recent_messages, ensure_ascii=False, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "你是 CFD 后处理分析智能体。你必须采用上下文工程策略："
                "优先读取 StructuredFactsSummary 中的确定性事实，只把 RecentRawMessageWindow "
                "当作最近步骤的原始证据补充。不要编造任何数值，所有数值必须来自工具事实。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{state.get('query') or DEFAULT_QUERY}\n\n"
                f"数据上下文快照：\n{data_context.context_text}\n\n"
                "<StructuredFactsSummary>\n"
                f"{facts_json}\n"
                "</StructuredFactsSummary>\n\n"
                f"<RecentRawMessageWindow turns=\"{state.get('context_window_turns', 3)}\">\n"
                f"{recent_json}\n"
                "</RecentRawMessageWindow>\n\n"
                "请生成完整 Markdown CFD 后处理报告，必须包含：算例概况、收敛判断、关键气动/流动指标、"
                "图表、异常点、工程解释、局限性。最后必须追加一个 <telemetry>{...}</telemetry> JSON 块，"
                "字段包含 methods、domain、tools_used、search_used、search_notes、cleaned_data_saved、"
                "cleaned_data_path、figures_generated、cfd_task_type、convergence_status。"
            ),
        },
    ]


def _telemetry_payload_from_facts(state: CFDGraphState) -> dict[str, Any]:
    facts = state.get("facts", {})
    tool_calls = facts.get("tool_calls", [])
    tools_used = [
        item.get("tool_name")
        for item in tool_calls
        if isinstance(item, dict) and item.get("tool_name")
    ]
    convergence_status = ""
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        data = item.get("data", {})
        if isinstance(data, dict) and data.get("convergence_status"):
            convergence_status = str(data["convergence_status"])
    search = facts.get("search", {}) if isinstance(facts.get("search"), dict) else {}
    return {
        "methods": tools_used,
        "domain": "CFD_post_processing",
        "tools_used": tools_used,
        "search_used": search.get("status") == "used",
        "search_notes": search.get("notes", "No online knowledge retrieval was triggered."),
        "cleaned_data_saved": Path(state["cleaned_data_path"]).exists(),
        "cleaned_data_path": state["cleaned_data_path"],
        "figures_generated": facts.get("figures", []),
        "cfd_task_type": state["data_context"].cfd_task_type,
        "convergence_status": convergence_status,
    }


def _ensure_report_telemetry(report: str, state: CFDGraphState) -> str:
    if re.search(r"<telemetry>\s*\{[\s\S]*?\}\s*</telemetry>\s*$", report.strip(), re.IGNORECASE):
        return report.strip()
    telemetry_json = json.dumps(_telemetry_payload_from_facts(state), ensure_ascii=False)
    return f"{report.strip()}\n\n<telemetry>{telemetry_json}</telemetry>"


def _fallback_report(state: CFDGraphState) -> str:
    facts = state.get("facts", {})
    telemetry_json = json.dumps(_telemetry_payload_from_facts(state), ensure_ascii=False)
    warnings = facts.get("warnings") or ["暂无显著 warning。"]
    figures = facts.get("figures") or []
    tool_calls = facts.get("tool_calls") or []
    latest_tool = tool_calls[-1] if tool_calls else {}
    latest_preview = latest_tool.get("text_preview", "暂无工具观测。") if isinstance(latest_tool, dict) else "暂无工具观测。"
    figure_lines = "\n".join(f"- ![图表]({path})" for path in figures) if figures else "- 暂无图表。"
    warning_lines = "\n".join(f"- {warning}" for warning in warnings)
    return (
        "# CFDFlow-Agent LangGraph 分析报告\n\n"
        "## 1. 算例概况\n\n"
        f"- CFD 任务类型：{state['data_context'].cfd_task_type}\n"
        f"- 选择工具：{state.get('selected_tool', 'unknown')}\n"
        f"- 清洗数据：{state['cleaned_data_path']}\n\n"
        "## 2. 收敛判断\n\n"
        f"{latest_preview}\n\n"
        "## 3. 关键气动/流动指标\n\n"
        "- 关键指标来自工具层结构化输出，未由 LLM 自行生成。\n\n"
        "## 4. 图表\n\n"
        f"{figure_lines}\n\n"
        "## 5. 异常点\n\n"
        f"{warning_lines}\n\n"
        "## 6. 工程解释\n\n"
        "- 请结合原始算例设置、边界条件和网格策略复核以上工具输出。\n\n"
        "## 7. 局限性\n\n"
        "- 本报告使用结构化事实摘要与最近消息滑动窗口生成，完整证据请查看 trace 文件。\n\n"
        f"<telemetry>{telemetry_json}</telemetry>"
    )


def _call_openai_compatible_llm(runtime_config: RuntimeConfig, messages: list[dict[str, str]]) -> str:
    client = OpenAI(
        api_key=runtime_config.api_key,
        base_url=runtime_config.base_url,
        timeout=runtime_config.timeout,
    )
    response = client.chat.completions.create(
        model=runtime_config.model_id,
        messages=messages,
        temperature=0,
    )
    content = response.choices[0].message.content
    return str(content or "").strip()


def _trace_record(
    *,
    step_index: int,
    node: str,
    action: str,
    status: str,
    summary: str,
    tool_name: str = "",
    observation_preview: str = "",
    duration_ms: int = 0,
) -> dict[str, Any]:
    return asdict(
        LangGraphStepTrace(
            step_index=step_index,
            node=node,
            action=action,
            status=status,
            summary=summary,
            tool_name=tool_name,
            observation_preview=observation_preview,
            duration_ms=duration_ms,
        )
    )


def build_cfd_langgraph(event_handler: Optional[EventHandler] = None):
    """Build the LangGraph workflow."""

    _ensure_langgraph_available()
    builder = StateGraph(CFDGraphState)

    def prepare_node(state: CFDGraphState) -> CFDGraphState:
        node_started_at = time.perf_counter()
        _emit_event(event_handler, "node_started", node="prepare")
        run_dir, data_dir, figures_dir, logs_dir = _create_run_directory(state.get("output_dir", "outputs"))
        cleaned_data_path = data_dir / "cleaned_data.csv"
        final_report_path = run_dir / "final_report.md"
        trace_path = logs_dir / "agent_trace_langgraph.json"

        runtime_config = None
        use_llm_report = bool(state.get("use_llm_report", True))
        if use_llm_report:
            try:
                runtime_config = load_runtime_config(env_file=state.get("env_file"))
            except Exception:
                use_llm_report = False

        source_path = Path(state["data_path"]).resolve()
        input_kind = "pdf" if source_path.suffix.lower() == ".pdf" else "tabular"
        document_ingestion = ingest_input_document(
            source_path,
            run_dir=run_dir,
            data_dir=data_dir,
            logs_dir=logs_dir,
            mode=state.get("document_ingestion_mode", "auto"),
            max_pdf_pages=int(state.get("max_pdf_pages", 20)),
            max_candidate_tables=int(state.get("max_candidate_tables", 5)),
            selected_table_id=state.get("selected_table_id"),
        )
        if document_ingestion.status == "failed":
            raise ValueError(document_ingestion.summary)

        data_context = build_data_context(
            document_ingestion.normalized_data_path,
            input_kind=document_ingestion.input_kind or input_kind,
            parsed_document_path=document_ingestion.parsed_document_path,
        )
        facts = _initial_facts(state.get("query") or DEFAULT_QUERY)
        facts["data"] = {
            "source_path": source_path.as_posix(),
            "normalized_data_path": document_ingestion.normalized_data_path.as_posix(),
            "shape": list(data_context.shape),
            "columns": list(data_context.columns),
            "cfd_task_type": data_context.cfd_task_type,
            "recommended_tool": data_context.cfd_recommended_tool,
            "matched_columns": data_context.cfd_matched_columns or {},
            "input_kind": document_ingestion.input_kind,
        }
        facts = _append_workflow_fact(
            facts,
            node="prepare",
            status="success",
            summary=f"Prepared run directory and data context for {data_context.cfd_task_type}.",
        )
        recent_messages = _append_recent_message(
            [],
            role="user",
            content=(
                f"{state.get('query') or DEFAULT_QUERY}\n"
                f"Data context snapshot:\n{data_context.context_text}"
            ),
            context_window_turns=int(state.get("context_window_turns", 3)),
        )
        summary = f"data_context ready | task={data_context.cfd_task_type}"
        _emit_event(event_handler, "node_finished", node="prepare", status="success")
        return {
            "runtime_config": runtime_config,
            "use_llm_report": use_llm_report,
            "document_ingestion": document_ingestion,
            "data_context": data_context,
            "run_dir": run_dir.as_posix(),
            "data_dir": data_dir.as_posix(),
            "figures_dir": figures_dir.as_posix(),
            "logs_dir": logs_dir.as_posix(),
            "cleaned_data_path": cleaned_data_path.as_posix(),
            "final_report_path": final_report_path.as_posix(),
            "trace_path": trace_path.as_posix(),
            "facts": facts,
            "recent_messages": recent_messages,
            "timing_breakdown": {"prepare_duration_ms": _elapsed_ms(node_started_at)},
            "step_traces": [
                _trace_record(
                    step_index=1,
                    node="prepare",
                    action="build_data_context",
                    status="success",
                    summary=summary,
                    duration_ms=_elapsed_ms(node_started_at),
                )
            ],
        }

    def clean_data_node(state: CFDGraphState) -> CFDGraphState:
        node_started_at = time.perf_counter()
        _emit_event(event_handler, "node_started", node="clean_data")
        data_context = state["data_context"]
        cleaned_data_path = Path(state["cleaned_data_path"])
        tool = PythonInterpreterTool()
        code = _build_cleaning_code(data_context.absolute_path, cleaned_data_path)
        observation = tool.run({"code": code})
        status = _status_from_observation(observation)
        facts = _add_tool_facts(state["facts"], tool_name="PythonInterpreterTool", raw_observation=observation)
        facts = _append_workflow_fact(
            facts,
            node="clean_data",
            status=status,
            summary=f"Cleaned data written to {cleaned_data_path.as_posix()}.",
        )
        recent_messages = _append_recent_message(
            state.get("recent_messages", []),
            role="assistant",
            content=f"PythonInterpreterTool cleaning observation:\n{observation}",
            context_window_turns=int(state.get("context_window_turns", 3)),
        )
        timing = dict(state.get("timing_breakdown", {}))
        duration_ms = _elapsed_ms(node_started_at)
        timing["clean_data_duration_ms"] = duration_ms
        _emit_event(event_handler, "node_finished", node="clean_data", status=status)
        return {
            "facts": facts,
            "recent_messages": recent_messages,
            "timing_breakdown": timing,
            "step_traces": [
                _trace_record(
                    step_index=2,
                    node="clean_data",
                    action="call_tool",
                    status=status,
                    summary=f"Python cleaning completed with status={status}.",
                    tool_name="PythonInterpreterTool",
                    observation_preview=_truncate_text(_read_json_observation(observation).get("text", ""), 240),
                    duration_ms=duration_ms,
                )
            ],
        }

    def select_tool_node(state: CFDGraphState) -> CFDGraphState:
        node_started_at = time.perf_counter()
        _emit_event(event_handler, "node_started", node="select_tool")
        data_context = state["data_context"]
        selected_tool = data_context.cfd_recommended_tool
        if selected_tool not in SPECIALIZED_TOOL_FACTORIES:
            selected_tool = "PythonInterpreterTool"

        tool_input = _build_tool_input(
            selected_tool,
            Path(state["cleaned_data_path"]),
            Path(state["figures_dir"]),
        )
        facts = _append_workflow_fact(
            state["facts"],
            node="select_tool",
            status="success",
            summary=f"Selected {selected_tool} for {data_context.cfd_task_type}.",
        )
        timing = dict(state.get("timing_breakdown", {}))
        duration_ms = _elapsed_ms(node_started_at)
        timing["select_tool_duration_ms"] = duration_ms
        _emit_event(event_handler, "node_finished", node="select_tool", status="success")
        return {
            "selected_tool": selected_tool,
            "tool_input": tool_input,
            "facts": facts,
            "timing_breakdown": timing,
            "step_traces": [
                _trace_record(
                    step_index=3,
                    node="select_tool",
                    action="route_tool",
                    status="success",
                    summary=f"Selected {selected_tool}.",
                    tool_name=selected_tool,
                    duration_ms=duration_ms,
                )
            ],
        }

    def execute_tool_node(state: CFDGraphState) -> CFDGraphState:
        node_started_at = time.perf_counter()
        _emit_event(event_handler, "node_started", node="execute_tool")
        selected_tool = state["selected_tool"]
        tool = SPECIALIZED_TOOL_FACTORIES[selected_tool]()
        observation = tool.run(state["tool_input"])
        status = _status_from_observation(observation)
        facts = _add_tool_facts(state["facts"], tool_name=selected_tool, raw_observation=observation)
        facts = _append_workflow_fact(
            facts,
            node="execute_tool",
            status=status,
            summary=f"{selected_tool} executed against cleaned_data.csv.",
        )
        recent_messages = _append_recent_message(
            state.get("recent_messages", []),
            role="assistant",
            content=f"{selected_tool} observation:\n{observation}",
            context_window_turns=int(state.get("context_window_turns", 3)),
        )
        timing = dict(state.get("timing_breakdown", {}))
        duration_ms = _elapsed_ms(node_started_at)
        timing["execute_tool_duration_ms"] = duration_ms
        timing["tool_duration_ms"] = timing.get("tool_duration_ms", 0) + duration_ms
        _emit_event(event_handler, "node_finished", node="execute_tool", status=status)
        return {
            "tool_observation": observation,
            "facts": facts,
            "recent_messages": recent_messages,
            "timing_breakdown": timing,
            "step_traces": [
                _trace_record(
                    step_index=4,
                    node="execute_tool",
                    action="call_tool",
                    status=status,
                    summary=f"{selected_tool} completed with status={status}.",
                    tool_name=selected_tool,
                    observation_preview=_truncate_text(_read_json_observation(observation).get("text", ""), 240),
                    duration_ms=duration_ms,
                )
            ],
        }

    def optional_search_node(state: CFDGraphState) -> CFDGraphState:
        node_started_at = time.perf_counter()
        _emit_event(event_handler, "node_started", node="optional_search")
        facts = copy.deepcopy(state["facts"])
        recent_messages = list(state.get("recent_messages", []))
        status = "skipped"
        observation = ""
        summary = "Online search skipped by policy or missing credential."

        if _should_enable_search(
            state["data_context"],
            state.get("query") or DEFAULT_QUERY,
            enable_search=bool(state.get("enable_search", True)),
        ):
            search_query = _build_search_query(state["data_context"], state.get("query") or DEFAULT_QUERY)
            observation = TavilySearchTool().run({"query": search_query})
            payload = _read_json_observation(observation)
            status = str(payload.get("status", "unknown"))
            results = payload.get("data", {}).get("results", []) if isinstance(payload.get("data"), dict) else []
            facts["search"] = {
                "status": "used" if status == "success" and results else "attempted",
                "notes": _truncate_text(payload.get("text", ""), 600),
                "results": [
                    {
                        "title": str(item.get("title", ""))[:120],
                        "url": str(item.get("url", ""))[:240],
                    }
                    for item in results[:3]
                    if isinstance(item, dict)
                ],
            }
            recent_messages = _append_recent_message(
                recent_messages,
                role="assistant",
                content=f"TavilySearchTool observation:\n{observation}",
                context_window_turns=int(state.get("context_window_turns", 3)),
            )
            summary = f"Tavily search completed with status={status}."
        else:
            facts["search"] = {
                "status": "skipped",
                "notes": summary,
                "results": [],
            }

        facts = _append_workflow_fact(facts, node="optional_search", status=status, summary=summary)
        timing = dict(state.get("timing_breakdown", {}))
        duration_ms = _elapsed_ms(node_started_at)
        timing["optional_search_duration_ms"] = duration_ms
        _emit_event(event_handler, "node_finished", node="optional_search", status=status)
        return {
            "search_observation": observation,
            "facts": facts,
            "recent_messages": recent_messages,
            "timing_breakdown": timing,
            "step_traces": [
                _trace_record(
                    step_index=5,
                    node="optional_search",
                    action="maybe_call_tool",
                    status=status,
                    summary=summary,
                    tool_name="TavilySearchTool" if observation else "",
                    observation_preview=_truncate_text(_read_json_observation(observation).get("text", ""), 240) if observation else "",
                    duration_ms=duration_ms,
                )
            ],
        }

    def synthesize_report_node(state: CFDGraphState) -> CFDGraphState:
        node_started_at = time.perf_counter()
        _emit_event(event_handler, "node_started", node="synthesize_report")
        report = ""
        llm_duration_ms = 0

        if state.get("use_llm_report") and state.get("runtime_config"):
            messages = _build_report_messages(state)
            llm_started_at = time.perf_counter()
            try:
                report = _call_openai_compatible_llm(state["runtime_config"], messages)  # type: ignore[arg-type]
            except Exception:
                report = ""
            llm_duration_ms = _elapsed_ms(llm_started_at)

        if not report:
            report = _fallback_report(state)
        else:
            report = _ensure_report_telemetry(report, state)

        extraction = extract_report_and_telemetry(report)
        facts = _append_workflow_fact(
            state["facts"],
            node="synthesize_report",
            status="success",
            summary="Generated final Markdown report from structured facts and sliding window.",
        )
        recent_messages = _append_recent_message(
            state.get("recent_messages", []),
            role="assistant",
            content=f"Final report draft:\n{report}",
            context_window_turns=int(state.get("context_window_turns", 3)),
        )
        timing = dict(state.get("timing_breakdown", {}))
        duration_ms = _elapsed_ms(node_started_at)
        timing["synthesize_report_duration_ms"] = duration_ms
        timing["llm_duration_ms"] = timing.get("llm_duration_ms", 0) + llm_duration_ms
        _emit_event(event_handler, "node_finished", node="synthesize_report", status="success")
        return {
            "report_markdown": extraction.report_markdown,
            "telemetry": extraction.telemetry,
            "facts": facts,
            "recent_messages": recent_messages,
            "timing_breakdown": timing,
            "step_traces": [
                _trace_record(
                    step_index=6,
                    node="synthesize_report",
                    action="generate_report",
                    status="success",
                    summary="Report generated using structured facts summary + sliding window.",
                    duration_ms=duration_ms,
                )
            ],
        }

    def persist_node(state: CFDGraphState) -> CFDGraphState:
        node_started_at = time.perf_counter()
        _emit_event(event_handler, "node_started", node="persist")
        final_report_path = Path(state["final_report_path"])
        trace_path = Path(state["trace_path"])
        saved_report_path = save_markdown_report(state["report_markdown"], final_report_path)
        if state.get("report_path"):
            save_markdown_report(state["report_markdown"], Path(str(state["report_path"])))

        timing = dict(state.get("timing_breakdown", {}))
        timing["persist_duration_ms"] = _elapsed_ms(node_started_at)
        timing["total_duration_ms"] = int(round((time.perf_counter() - state["started_at"]) * 1000))
        persist_trace_record = _trace_record(
            step_index=7,
            node="persist",
            action="save_artifacts",
            status="success",
            summary="Saved report and LangGraph trace artifacts.",
            duration_ms=_elapsed_ms(node_started_at),
        )

        trace = {
            "agent_type": "LangGraphContextEngineeredWorkflow",
            "framework": "langgraph",
            "context_strategy": "structured_facts_summary + recent_raw_message_sliding_window",
            "context_window_turns": state.get("context_window_turns", 3),
            "graph_nodes": [
                "prepare",
                "clean_data",
                "select_tool",
                "execute_tool",
                "optional_search",
                "synthesize_report",
                "persist",
            ],
            "data_path": state["data_path"],
            "query": state.get("query") or DEFAULT_QUERY,
            "run_dir": state["run_dir"],
            "selected_tool": state["selected_tool"],
            "cfd_task_type": state["data_context"].cfd_task_type,
            "structured_facts": _safe_jsonable(state.get("facts", {})),
            "recent_messages": state.get("recent_messages", []),
            "step_traces": [*state.get("step_traces", []), persist_trace_record],
            "telemetry": _safe_jsonable(state.get("telemetry", ReportTelemetry())),
            "cleaned_data_path": state["cleaned_data_path"],
            "report_path": saved_report_path.as_posix(),
            "timing_breakdown": timing,
        }
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        _emit_event(
            event_handler,
            "report_saved",
            report_path=saved_report_path.as_posix(),
            trace_path=trace_path.as_posix(),
        )
        _emit_event(event_handler, "node_finished", node="persist", status="success")
        return {
            "trace": trace,
            "timing_breakdown": timing,
            "step_traces": [persist_trace_record],
        }

    builder.add_node("prepare", prepare_node)
    builder.add_node("clean_data", clean_data_node)
    builder.add_node("select_tool", select_tool_node)
    builder.add_node("execute_tool", execute_tool_node)
    builder.add_node("optional_search", optional_search_node)
    builder.add_node("synthesize_report", synthesize_report_node)
    builder.add_node("persist", persist_node)

    builder.set_entry_point("prepare")
    builder.add_edge("prepare", "clean_data")
    builder.add_edge("clean_data", "select_tool")
    builder.add_edge("select_tool", "execute_tool")
    builder.add_edge("execute_tool", "optional_search")
    builder.add_edge("optional_search", "synthesize_report")
    builder.add_edge("synthesize_report", "persist")
    builder.add_edge("persist", END)
    return builder.compile()


def run_analysis(
    data_path: str | Path,
    *,
    query: str = DEFAULT_QUERY,
    output_dir: str | Path = "outputs",
    report_path: Optional[str | Path] = None,
    env_file: Optional[str | Path] = None,
    context_window_turns: int = 3,
    use_llm_report: bool = True,
    enable_search: bool = True,
    document_ingestion_mode: str = "auto",
    max_pdf_pages: int = 20,
    max_candidate_tables: int = 5,
    selected_table_id: str | None = None,
    quality_mode: str = "standard",
    latency_mode: str = "auto",
    max_steps: int = 6,
    max_reviews: Optional[int] = None,
    vision_review_mode: str = "off",
    vision_max_images: int = 3,
    vision_max_image_side: int = 1024,
    event_handler: Optional[EventHandler] = None,
    verbose: bool = False,
) -> LangGraphCFDRunResult:
    """Run the LangGraph CFD analysis workflow.

    Some legacy keyword arguments are accepted for notebook compatibility. The
    LangGraph copy expresses the workflow as graph nodes rather than a free-form
    ReAct loop, so `max_steps`, `max_reviews`, and vision-review knobs are not
    used by default.
    """

    del max_steps, max_reviews, vision_max_images, vision_max_image_side
    if event_handler is None and verbose:
        event_handler = build_plaintext_event_handler()

    graph = build_cfd_langgraph(event_handler=event_handler)
    final_state: CFDGraphState = graph.invoke(
        {
            "data_path": str(data_path),
            "query": query,
            "output_dir": str(output_dir),
            "report_path": str(report_path) if report_path is not None else None,
            "env_file": str(env_file) if env_file is not None else None,
            "started_at": time.perf_counter(),
            "context_window_turns": max(1, int(context_window_turns)),
            "use_llm_report": bool(use_llm_report),
            "enable_search": bool(enable_search),
            "document_ingestion_mode": document_ingestion_mode,
            "max_pdf_pages": max_pdf_pages,
            "max_candidate_tables": max_candidate_tables,
            "selected_table_id": selected_table_id,
            "step_traces": [],
        }
    )

    telemetry = final_state.get("telemetry", ReportTelemetry())
    facts = final_state.get("facts", {})
    search = facts.get("search", {}) if isinstance(facts.get("search"), dict) else {}
    warnings = tuple(str(item) for item in facts.get("warnings", []) if str(item))
    missing_artifacts = []
    for artifact_path in (
        final_state.get("cleaned_data_path", ""),
        final_state.get("final_report_path", ""),
        final_state.get("trace_path", ""),
    ):
        if artifact_path and not Path(artifact_path).exists():
            missing_artifacts.append(str(artifact_path))

    step_traces = tuple(
        LangGraphStepTrace(**item)
        for item in final_state.get("step_traces", [])
        if isinstance(item, dict)
    )
    tool_status = _status_from_observation(final_state.get("tool_observation", ""))
    timing = final_state.get("timing_breakdown", {})
    return LangGraphCFDRunResult(
        data_context=final_state["data_context"],
        report_markdown=final_state["report_markdown"],
        report_path=Path(final_state["final_report_path"]),
        output_dir=Path(final_state["run_dir"]),
        run_dir=Path(final_state["run_dir"]),
        data_dir=Path(final_state["data_dir"]),
        figures_dir=Path(final_state["figures_dir"]),
        logs_dir=Path(final_state["logs_dir"]),
        trace_path=Path(final_state["trace_path"]),
        cleaned_data_path=Path(final_state["cleaned_data_path"]),
        agent_type="LangGraphContextEngineeredWorkflow",
        step_traces=step_traces,
        telemetry=telemetry,
        methods_used=telemetry.methods,
        detected_domain=telemetry.domain,
        tools_used=telemetry.tools_used,
        search_status=str(search.get("status", "not_used")),
        search_notes=str(search.get("notes", "No online knowledge retrieval was triggered.")),
        workflow_complete=not missing_artifacts,
        workflow_warnings=warnings,
        missing_artifacts=tuple(missing_artifacts),
        quality_mode=quality_mode,
        input_kind=final_state["document_ingestion"].input_kind,
        document_ingestion_status=final_state["document_ingestion"].status,
        document_ingestion_summary=final_state["document_ingestion"].summary,
        document_ingestion_duration_ms=final_state["document_ingestion"].duration_ms,
        document_ingestion_log_path=final_state["document_ingestion"].log_path,
        candidate_table_count=final_state["document_ingestion"].candidate_table_count,
        selected_table_id=final_state["document_ingestion"].selected_table_id,
        selected_table_shape=final_state["document_ingestion"].selected_table_shape,
        pdf_multi_table_mode=final_state["document_ingestion"].pdf_multi_table_mode,
        latency_mode=latency_mode,
        vision_review_mode=vision_review_mode,
        total_duration_ms=int(timing.get("total_duration_ms", 0)),
        llm_duration_ms=int(timing.get("llm_duration_ms", 0)),
        tool_duration_ms=int(timing.get("tool_duration_ms", 0)),
        timing_breakdown=dict(timing),
        cfd_task_type=final_state["data_context"].cfd_task_type,
        selected_tool=final_state["selected_tool"],
        tool_status=tool_status,
        context_window_turns=max(1, int(context_window_turns)),
        structured_facts=copy.deepcopy(facts),
        recent_messages=tuple(final_state.get("recent_messages", [])),
    )


def run_cfd_langgraph_analysis(*args: Any, **kwargs: Any) -> LangGraphCFDRunResult:
    """Backward-compatible alias for earlier sidecar examples."""

    return run_analysis(*args, **kwargs)
