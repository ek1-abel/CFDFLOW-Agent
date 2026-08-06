"""Simple smoke test for the Gradio workstation.

The test imports the UI, builds the Blocks object, and runs the callback once
without starting a web server or calling external LLM/search APIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import app_gradio


def main() -> None:
    demo = app_gradio.build_demo()
    if not hasattr(demo, "launch"):
        raise AssertionError("Gradio Blocks app was not built correctly.")

    outputs = app_gradio.run_workflow(
        uploaded_file=None,
        sample_dataset="residual_history.csv",
        query="UI smoke test: analyze residual convergence.",
        context_window_turns=2,
        use_llm_report=False,
        enable_search=False,
        quality_mode="draft",
        latency_mode="fast",
    )
    (
        status_markdown,
        report_markdown,
        step_trace,
        gallery,
        structured_facts,
        recent_messages,
        raw_trace,
        artifacts,
    ) = outputs

    checks = {
        "status_ok": "完成" in status_markdown,
        "report_generated": "# CFDFlow-Agent" in report_markdown,
        "step_trace_dataframe": isinstance(step_trace, pd.DataFrame) and len(step_trace) == 7,
        "gallery_has_figure": bool(gallery),
        "facts_have_tool_calls": bool(structured_facts.get("tool_calls")),
        "sliding_window_applied": len(recent_messages) <= 4,
        "trace_framework": raw_trace.get("framework") == "langgraph",
        "artifacts_exist": all(Path(path).exists() for path in artifacts),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"UI smoke test failed: {', '.join(failed)}")

    print("UI_SMOKE_TEST_OK")
    print(f"trace_nodes={json.dumps(raw_trace.get('graph_nodes', []), ensure_ascii=False)}")
    print(f"artifact_count={len(artifacts)}")


if __name__ == "__main__":
    main()
