"""Offline smoke test for the LangGraph CFDFlow-Agent copy.

The test exercises the full graph without calling external LLM or Tavily APIs:
prepare -> clean_data -> select_tool -> execute_tool -> optional_search
-> synthesize_report -> persist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cfd_analysis_agent import run_analysis


DATA_PATH = PROJECT_ROOT / "data" / "residual_history.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "smoke_test_langgraph"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing sample dataset: {DATA_PATH}")

    result = run_analysis(
        DATA_PATH,
        query="offline smoke test for residual convergence",
        output_dir=OUTPUT_DIR,
        context_window_turns=2,
        use_llm_report=False,
        enable_search=False,
        quality_mode="draft",
        latency_mode="fast",
        verbose=False,
    )

    trace_payload = json.loads(result.trace_path.read_text(encoding="utf-8"))
    figure_candidates = list(result.figures_dir.rglob("residual_convergence.png"))
    expected_nodes = [
        "prepare",
        "clean_data",
        "select_tool",
        "execute_tool",
        "optional_search",
        "synthesize_report",
        "persist",
    ]
    trace_nodes = [item["node"] for item in trace_payload.get("step_traces", [])]

    checks = {
        "workflow_complete": result.workflow_complete,
        "cleaned_data_exists": result.cleaned_data_path.exists(),
        "report_exists": result.report_path.exists(),
        "trace_exists": result.trace_path.exists(),
        "figure_exists": bool(figure_candidates),
        "framework_is_langgraph": trace_payload.get("framework") == "langgraph",
        "context_strategy": result.context_strategy
        == "structured_facts_summary + recent_raw_message_sliding_window",
        "sliding_window_applied": len(result.recent_messages) <= result.context_window_turns * 2,
        "structured_facts_present": bool(result.structured_facts.get("tool_calls")),
        "graph_nodes_recorded": trace_nodes == expected_nodes,
        "selected_residual_tool": result.selected_tool == "ResidualAnalysisTool",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Smoke test failed: {', '.join(failed)}")

    print("SMOKE_TEST_OK")
    print(f"agent_type={result.agent_type}")
    print(f"context_strategy={result.context_strategy}")
    print(f"context_window_turns={result.context_window_turns}")
    print(f"recent_message_count={len(result.recent_messages)}")
    print(f"selected_tool={result.selected_tool}")
    print(f"run_dir={result.run_dir}")
    print(f"report={result.report_path}")
    print(f"trace={result.trace_path}")
    print(f"figure={figure_candidates[0]}")


if __name__ == "__main__":
    main()
