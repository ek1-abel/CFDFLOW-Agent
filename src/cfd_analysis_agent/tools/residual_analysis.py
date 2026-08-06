"""Residual convergence analysis tool for CFD simulations."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hello_agents.tools import Tool, ToolParameter

from ..plotting import apply_publication_style, beautify_axes, save_figure
from ..tool_protocol import ToolErrorCode, ToolResponse


class ResidualAnalysisTool(Tool):
    """Analyze CFD solver residual history for convergence assessment."""

    def __init__(self):
        super().__init__(
            name="ResidualAnalysisTool",
            description=(
                "Analyze CFD solver residual convergence history. "
                "Input: JSON string with keys 'data_path' (path to CSV with iteration and residual columns) "
                "and 'figures_dir' (directory for output figures). "
                "Returns convergence status, drop orders for each residual, and generates a semi-log residual plot."
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
                "Input must be a JSON string with 'data_path' and 'figures_dir' keys. "
                f'Example: {{"data_path": "/path/to/data.csv", "figures_dir": "/path/to/figures/"}}'
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

        import re
        iter_col = None
        for col in df.columns:
            if re.match(r"^(iteration|iter|step|time_step|timestep|n)$", col, re.IGNORECASE):
                iter_col = col
                break
        if iter_col is None:
            iter_col = df.columns[0]

        residual_cols = [
            col for col in df.columns
            if col != iter_col and df[col].dtype in [np.float64, np.float32, float, np.int64, np.int32]
        ]
        if not residual_cols:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="No numeric residual columns found in the data.",
            )

        results = {}
        all_converged = True
        for col in residual_cols:
            values = df[col].dropna().values.astype(float)
            values = values[values > 0]
            if len(values) < 10:
                results[col] = {"status": "insufficient_data", "drop_order": 0.0}
                all_converged = False
                continue

            initial = np.mean(values[:5])
            final = np.mean(values[-5:])
            if initial <= 0 or final <= 0:
                results[col] = {"status": "invalid_values", "drop_order": 0.0}
                all_converged = False
                continue

            drop_order = np.log10(initial / final)
            last_10pct = values[int(len(values) * 0.9):]
            variation = np.log10(np.max(last_10pct) / (np.min(last_10pct) + 1e-30))

            if drop_order >= 3.0 and variation < 1.0:
                status = "converged"
            elif drop_order >= 2.0:
                status = "marginally_converged"
                all_converged = False
            else:
                status = "not_converged"
                all_converged = False

            results[col] = {
                "status": status,
                "drop_order": round(float(drop_order), 2),
                "initial_value": f"{initial:.2e}",
                "final_value": f"{final:.2e}",
                "tail_variation_order": round(float(variation), 2),
            }

        # Generate plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt, _ = apply_publication_style()

        fig, ax = plt.subplots(figsize=(12, 7))
        x = df[iter_col].values
        for col in residual_cols:
            vals = df[col].values.astype(float)
            mask = vals > 0
            ax.semilogy(x[mask], vals[mask], label=col, linewidth=1.8)

        ax.axhline(y=1e-6, color="red", linestyle="--", alpha=0.5, label="Reference 1e-6")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual (log scale)")
        ax.set_title("Residual Convergence History")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = figures_dir / "residual_convergence.png"
        save_figure(str(fig_path))
        plt.close(fig)

        convergence_tag = "[CONVERGED]" if all_converged else "[WARNING: NOT CONVERGED]"
        summary_lines = [convergence_tag, "", "Residual Convergence Analysis Results:", ""]
        for col, info in results.items():
            summary_lines.append(
                f"  {col}: status={info['status']}, drop_order={info.get('drop_order', 'N/A')}"
            )
        summary_lines.append(f"\nFigure saved to: {fig_path.as_posix()}")

        return ToolResponse.success(
            text="\n".join(summary_lines),
            data={
                "convergence_status": "converged" if all_converged else "not_converged",
                "residual_details": results,
                "figure_path": fig_path.as_posix(),
            },
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
                    'Example: {"data_path": "/path/to/residuals.csv", "figures_dir": "/path/to/figures/"}'
                ),
                required=True,
            )
        ]
