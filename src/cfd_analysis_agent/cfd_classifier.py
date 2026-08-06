"""CFD task type classifier based on column name pattern matching."""

from __future__ import annotations

import re
from typing import Any


_ITERATION_PATTERNS = re.compile(
    r"^(iteration|iter|step|time_step|timestep|time|n)$", re.IGNORECASE
)

_RESIDUAL_PATTERNS = re.compile(
    r"(continuity|x[-_]?velocity|y[-_]?velocity|z[-_]?velocity|"
    r"energy|k|omega|epsilon|nut|tke|sdr|"
    r"residual|res[-_])", re.IGNORECASE
)

_FORCE_PATTERNS = re.compile(
    r"^(cl|cd|cm|c_l|c_d|c_m|lift|drag|moment|"
    r"lift[-_]?coefficient|drag[-_]?coefficient)$", re.IGNORECASE
)

_PRESSURE_PATTERNS = re.compile(
    r"^(cp|c_p|pressure[-_]?coefficient|cp[-_]?upper|cp[-_]?lower)$", re.IGNORECASE
)

_POSITION_PATTERNS = re.compile(
    r"^(x/c|x_over_c|x[-_]?chord|x)$", re.IGNORECASE
)

_VELOCITY_PROFILE_PATTERNS = re.compile(
    r"^(u_plus|y_plus|u\+|y\+|u_inf|u/u_inf|y/delta|y[-_]?norm)$", re.IGNORECASE
)

_GRID_PATTERNS = re.compile(
    r"^(cell[-_]?count|elements|nodes|cells|mesh[-_]?level|"
    r"grid[-_]?level|grid|mesh|refinement[-_]?level|h|grid[-_]?size)$", re.IGNORECASE
)

CFD_TASK_TYPES = (
    "residual_convergence",
    "force_coefficients",
    "pressure_distribution",
    "velocity_profile",
    "grid_independence",
    "general_cfd",
)

_TOOL_RECOMMENDATION = {
    "residual_convergence": "ResidualAnalysisTool",
    "force_coefficients": "ForceAnalysisTool",
    "pressure_distribution": "PressureAnalysisTool",
    "velocity_profile": "VelocityAnalysisTool",
    "grid_independence": "GridStudyTool",
    "general_cfd": "PythonInterpreterTool",
}


def classify_cfd_task(
    columns: list[str],
) -> tuple[str, dict[str, list[str]], str]:
    """Classify the CFD task type from column names.

    Returns:
        task_type: one of CFD_TASK_TYPES
        matched_columns: dict mapping semantic roles to actual column names
        recommended_tool: the tool name best suited for this data
    """
    cols_lower = [c.strip() for c in columns]
    matched: dict[str, list[str]] = {}

    iteration_cols = [c for c in cols_lower if _ITERATION_PATTERNS.match(c)]
    residual_cols = [c for c in cols_lower if _RESIDUAL_PATTERNS.search(c) and c not in iteration_cols]
    force_cols = [c for c in cols_lower if _FORCE_PATTERNS.match(c)]
    pressure_cols = [c for c in cols_lower if _PRESSURE_PATTERNS.match(c)]
    position_cols = [c for c in cols_lower if _POSITION_PATTERNS.match(c)]
    velocity_cols = [c for c in cols_lower if _VELOCITY_PROFILE_PATTERNS.match(c)]
    grid_cols = [c for c in cols_lower if _GRID_PATTERNS.match(c)]

    # Also check for generic u and y columns (velocity profile without plus notation)
    generic_u = [c for c in cols_lower if re.match(r"^u$", c, re.IGNORECASE)]
    generic_y = [c for c in cols_lower if re.match(r"^y$", c, re.IGNORECASE)]

    # Priority-ordered classification
    if residual_cols and (iteration_cols or len(residual_cols) >= 2):
        matched["iteration"] = iteration_cols
        matched["residuals"] = residual_cols
        return "residual_convergence", matched, _TOOL_RECOMMENDATION["residual_convergence"]

    if force_cols and iteration_cols:
        matched["iteration"] = iteration_cols
        matched["forces"] = force_cols
        return "force_coefficients", matched, _TOOL_RECOMMENDATION["force_coefficients"]

    if pressure_cols and position_cols:
        matched["position"] = position_cols
        matched["pressure"] = pressure_cols
        return "pressure_distribution", matched, _TOOL_RECOMMENDATION["pressure_distribution"]

    if velocity_cols or (generic_u and generic_y and not force_cols):
        matched["velocity"] = velocity_cols or generic_u
        matched["wall_distance"] = [c for c in cols_lower if re.match(r"^(y_plus|y\+|y|y/delta|y[-_]?norm)$", c, re.IGNORECASE)]
        return "velocity_profile", matched, _TOOL_RECOMMENDATION["velocity_profile"]

    if grid_cols:
        solution_cols = [c for c in cols_lower if c not in grid_cols and c not in iteration_cols]
        matched["grid"] = grid_cols
        matched["solution_quantities"] = solution_cols
        return "grid_independence", matched, _TOOL_RECOMMENDATION["grid_independence"]

    return "general_cfd", {}, _TOOL_RECOMMENDATION["general_cfd"]


def get_tool_recommendation(task_type: str) -> str:
    return _TOOL_RECOMMENDATION.get(task_type, "PythonInterpreterTool")
