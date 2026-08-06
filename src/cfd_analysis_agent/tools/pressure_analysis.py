"""Pressure distribution analysis tool for CFD simulations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hello_agents.tools import Tool, ToolParameter

from ..plotting import apply_publication_style, beautify_axes, save_figure
from ..tool_protocol import ToolErrorCode, ToolResponse


class PressureAnalysisTool(Tool):
    """Analyze surface pressure coefficient (Cp) distribution from CFD results."""

    def __init__(self):
        super().__init__(
            name="PressureAnalysisTool",
            description=(
                "Analyze surface pressure coefficient (Cp) distribution from CFD results. "
                "Input: JSON string with keys 'data_path' (path to CSV with x/c and Cp columns) "
                "and 'figures_dir' (directory for output figures). "
                "Returns suction peak location/magnitude, stagnation point, and generates Cp vs x/c plot."
            ),
        )
        apply_publication_style()

    def _parse_input(self, parameters: Dict[str, Any]) -> tuple[Path, Path]:
        raw = parameters.get("input", parameters.get("code", ""))
        if not raw:
            raise ValueError("Empty input")
        try:
            parsed = json.loads(raw)
            return Path(parsed["data_path"]), Path(parsed["figures_dir"])
        except (json.JSONDecodeError, KeyError):
            raise ValueError(
                "Input must be a JSON string with 'data_path' and 'figures_dir' keys."
            )

    def execute(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            data_path, figures_dir = self._parse_input(parameters)
        except ValueError as exc:
            return ToolResponse.error(code=ToolErrorCode.INVALID_PARAM, message=str(exc))

        try:
            df = pd.read_csv(data_path)
        except Exception as exc:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"Failed to read data file: {exc}",
            )

        # Find position column
        pos_col = None
        for col in df.columns:
            if re.match(r"^(x/c|x_over_c|x[-_]?chord|x)$", col, re.IGNORECASE):
                pos_col = col
                break
        if pos_col is None:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="No position column (x/c) found in data.",
            )

        # Find Cp columns
        cp_cols = {}
        for col in df.columns:
            cl = col.strip().lower()
            if cl in ("cp_upper", "cp upper"):
                cp_cols["upper"] = col
            elif cl in ("cp_lower", "cp lower"):
                cp_cols["lower"] = col
            elif cl in ("cp", "c_p", "pressure_coefficient") and "upper" not in cp_cols:
                cp_cols["combined"] = col

        if not cp_cols:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="No pressure coefficient (Cp) columns found in data.",
            )

        results = {}
        x_vals = df[pos_col].values.astype(float)

        for surface, col in cp_cols.items():
            cp_vals = df[col].values.astype(float)
            valid = ~np.isnan(cp_vals)
            cp_valid = cp_vals[valid]
            x_valid = x_vals[valid]

            suction_idx = np.argmin(cp_valid)
            suction_peak_cp = float(cp_valid[suction_idx])
            suction_peak_x = float(x_valid[suction_idx])

            stag_idx = np.argmin(np.abs(cp_valid - 1.0))
            stagnation_cp = float(cp_valid[stag_idx])
            stagnation_x = float(x_valid[stag_idx])

            trailing_edge_cp = float(cp_valid[-1]) if len(cp_valid) > 0 else None

            results[surface] = {
                "column_name": col,
                "suction_peak_cp": round(suction_peak_cp, 4),
                "suction_peak_x_over_c": round(suction_peak_x, 4),
                "stagnation_cp": round(stagnation_cp, 4),
                "stagnation_x_over_c": round(stagnation_x, 4),
                "trailing_edge_cp": round(trailing_edge_cp, 4) if trailing_edge_cp is not None else None,
                "cp_range": [round(float(np.min(cp_valid)), 4), round(float(np.max(cp_valid)), 4)],
            }

        # Generate plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt, _ = apply_publication_style()

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = {"upper": "#2196F3", "lower": "#F44336", "combined": "#333333"}

        for surface, col in cp_cols.items():
            cp_vals = df[col].values.astype(float)
            label = f"Cp ({surface})" if surface != "combined" else "Cp"
            ax.plot(x_vals, cp_vals, color=colors.get(surface, "#333333"),
                   linewidth=2.0, marker="o", markersize=3, label=label)

        ax.invert_yaxis()  # CFD convention: negative Cp up
        ax.set_xlabel("x/c")
        ax.set_ylabel("$C_p$")
        ax.set_title("Pressure Coefficient Distribution")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

        # Mark suction peak
        for surface, info in results.items():
            ax.annotate(
                f"Cp_min = {info['suction_peak_cp']:.3f}\nx/c = {info['suction_peak_x_over_c']:.3f}",
                xy=(info["suction_peak_x_over_c"], info["suction_peak_cp"]),
                xytext=(info["suction_peak_x_over_c"] + 0.15, info["suction_peak_cp"] - 0.3),
                arrowprops=dict(arrowstyle="->", color="#333"),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )
            break  # Only annotate the first surface

        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = figures_dir / "pressure_distribution.png"
        save_figure(str(fig_path))
        plt.close(fig)

        summary_lines = ["Pressure Distribution Analysis Results:", ""]
        for surface, info in results.items():
            summary_lines.append(f"  Surface: {surface}")
            summary_lines.append(f"    Suction peak: Cp = {info['suction_peak_cp']:.4f} at x/c = {info['suction_peak_x_over_c']:.4f}")
            summary_lines.append(f"    Stagnation point: Cp = {info['stagnation_cp']:.4f} at x/c = {info['stagnation_x_over_c']:.4f}")
            if info["trailing_edge_cp"] is not None:
                summary_lines.append(f"    Trailing edge: Cp = {info['trailing_edge_cp']:.4f}")
        summary_lines.append(f"\nFigure saved to: {fig_path.as_posix()}")

        return ToolResponse.success(
            text="\n".join(summary_lines),
            data={"pressure_details": results, "figure_path": fig_path.as_posix()},
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        return self.execute(parameters).to_json()

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description=(
                    'JSON string with keys "data_path" and "figures_dir". '
                    'Example: {"data_path": "/path/to/cp.csv", "figures_dir": "/path/to/figures/"}'
                ),
                required=True,
            )
        ]
