"""Force coefficient analysis tool for CFD simulations."""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from hello_agents.tools import Tool, ToolParameter

from ..plotting import apply_publication_style, beautify_axes, save_figure
from ..tool_protocol import ToolErrorCode, ToolResponse


class ForceAnalysisTool(Tool):
    """Analyze aerodynamic force coefficients (Cl, Cd, Cm) from CFD simulations."""

    def __init__(self):
        super().__init__(
            name="ForceAnalysisTool",
            description=(
                "Analyze aerodynamic force coefficients (Cl, Cd, Cm) from CFD simulation results. "
                "Input: JSON string with keys 'data_path' (path to CSV with iteration and Cl/Cd/Cm columns) "
                "and 'figures_dir' (directory for output figures). "
                "Returns mean/std/oscillation status for each coefficient and generates time-history plots."
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

        iter_col = None
        for col in df.columns:
            if re.match(r"^(iteration|iter|step|time_step|timestep|time|n)$", col, re.IGNORECASE):
                iter_col = col
                break
        if iter_col is None:
            iter_col = df.columns[0]

        force_mapping = {}
        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in ("cl", "c_l", "lift", "lift_coefficient"):
                force_mapping["Cl"] = col
            elif col_lower in ("cd", "c_d", "drag", "drag_coefficient"):
                force_mapping["Cd"] = col
            elif col_lower in ("cm", "c_m", "moment", "moment_coefficient"):
                force_mapping["Cm"] = col

        if not force_mapping:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="No force coefficient columns (Cl/Cd/Cm) found in data.",
            )

        results = {}
        any_oscillation = False
        thresholds = {"Cl": 0.05, "Cd": 0.02, "Cm": 0.05}

        for label, col in force_mapping.items():
            values = df[col].dropna().values.astype(float)
            n = len(values)
            if n < 10:
                results[label] = {"status": "insufficient_data"}
                continue

            last_half = values[n // 2:]
            mean_val = float(np.mean(last_half))
            std_val = float(np.std(last_half))
            min_val = float(np.min(last_half))
            max_val = float(np.max(last_half))

            rel_oscillation = abs(std_val / mean_val) if abs(mean_val) > 1e-10 else float("inf")
            threshold = thresholds.get(label, 0.05)
            oscillating = rel_oscillation > threshold

            if oscillating:
                any_oscillation = True

            # Check stationarity: compare last 20% mean vs last 40% mean
            last_20 = values[int(n * 0.8):]
            last_40 = values[int(n * 0.6):]
            drift = abs(np.mean(last_20) - np.mean(last_40))
            stationary = drift < 2 * std_val if std_val > 0 else True

            results[label] = {
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "min": round(min_val, 6),
                "max": round(max_val, 6),
                "relative_oscillation": round(rel_oscillation, 4),
                "oscillating": bool(oscillating),
                "stationary": bool(stationary),
                "column_name": col,
            }

        # Generate plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt, _ = apply_publication_style()

        n_plots = len(force_mapping)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        x = df[iter_col].values
        colors = {"Cl": "#2196F3", "Cd": "#F44336", "Cm": "#4CAF50"}

        for idx, (label, col) in enumerate(force_mapping.items()):
            ax = axes[idx]
            vals = df[col].values.astype(float)
            info = results.get(label, {})
            mean_val = info.get("mean")
            std_val = info.get("std")
            color = colors.get(label, "#333333")

            ax.plot(x, vals, color=color, linewidth=1.2, alpha=0.8, label=f"{label} history")

            if mean_val is not None:
                n_half = len(vals) // 2
                ax.axhline(y=mean_val, color=color, linestyle="--", linewidth=1.5,
                          alpha=0.7, label=f"Mean = {mean_val:.4f}")
                if std_val is not None and std_val > 0:
                    ax.axhspan(mean_val - std_val, mean_val + std_val,
                              alpha=0.1, color=color, label=f"$\\pm 1\\sigma$ = {std_val:.4f}")

            ax.set_ylabel(label)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if info.get("oscillating"):
                ax.set_title(f"{label} [OSCILLATING]", color="red", fontweight="bold")
            else:
                ax.set_title(f"{label} [STEADY]")

        axes[-1].set_xlabel("Iteration")
        fig.suptitle("Force Coefficient History", fontsize=16, fontweight="bold", y=1.02)

        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = figures_dir / "force_coefficients.png"
        save_figure(str(fig_path))
        plt.close(fig)

        convergence_tag = (
            "[WARNING: FORCE OSCILLATION DETECTED]" if any_oscillation
            else "[STEADY]"
        )
        summary_lines = [convergence_tag, "", "Force Coefficient Analysis Results:", ""]
        for label, info in results.items():
            if "mean" in info:
                summary_lines.append(
                    f"  {label}: mean={info['mean']:.6f}, std={info['std']:.6f}, "
                    f"oscillating={info['oscillating']}, stationary={info['stationary']}"
                )
            else:
                summary_lines.append(f"  {label}: {info.get('status', 'unknown')}")
        summary_lines.append(f"\nFigure saved to: {fig_path.as_posix()}")

        return ToolResponse.success(
            text="\n".join(summary_lines),
            data={
                "overall_status": "oscillating" if any_oscillation else "steady",
                "force_details": results,
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
                    'Example: {"data_path": "/path/to/forces.csv", "figures_dir": "/path/to/figures/"}'
                ),
                required=True,
            )
        ]
