"""CFDFlow Analysis Agent package."""

from .config import RuntimeConfig, apply_token_counter_patch, load_runtime_config
from .data_context import DataContextSummary, build_data_context
from .langgraph_runner import (
    LangGraphCFDRunResult,
    build_cfd_langgraph,
    build_plaintext_event_handler,
    run_analysis,
    run_cfd_langgraph_analysis,
)
from .presentation import render_diagnostics, render_full_report, render_trace_table
from .tools.python_interpreter import PythonInterpreterTool
from .tools.tavily_search import TavilySearchTool
from .tools.residual_analysis import ResidualAnalysisTool
from .tools.force_analysis import ForceAnalysisTool
from .tools.pressure_analysis import PressureAnalysisTool
from .tools.velocity_analysis import VelocityAnalysisTool
from .tools.grid_study import GridStudyTool

AnalysisRunResult = LangGraphCFDRunResult

__all__ = [
    "AnalysisRunResult",
    "DataContextSummary",
    "PythonInterpreterTool",
    "RuntimeConfig",
    "LangGraphCFDRunResult",
    "TavilySearchTool",
    "ResidualAnalysisTool",
    "ForceAnalysisTool",
    "PressureAnalysisTool",
    "VelocityAnalysisTool",
    "GridStudyTool",
    "apply_token_counter_patch",
    "build_cfd_langgraph",
    "build_data_context",
    "build_plaintext_event_handler",
    "load_runtime_config",
    "render_diagnostics",
    "render_full_report",
    "render_trace_table",
    "run_analysis",
    "run_cfd_langgraph_analysis",
]
