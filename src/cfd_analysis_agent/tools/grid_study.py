"""Grid independence study tool for CFD simulations."""

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


class GridStudyTool(Tool):
    """Perform grid independence analysis with Richardson extrapolation and GCI."""

    def __init__(self):
        super().__init__(
            name="GridStudyTool",
            description=(
                "Perform grid independence study with Richardson extrapolation and GCI computation. "
                "Input: JSON string with keys 'data_path' (path to CSV with mesh levels and solution quantities) "
                "and 'figures_dir' (directory for output figures). "
                "Requires at least 3 grid levels. Returns extrapolated values, convergence order, and GCI."
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

    def _richardson_extrapolation(self, h_values: np.ndarray, f_values: np.ndarray):
        """Perform Richardson extrapolation on the finest 3 grids."""
        if len(h_values) < 3:
            return None

        # Use 3 finest grids (smallest h)
        idx = np.argsort(h_values)[:3]
        h1, h2, h3 = h_values[idx[0]], h_values[idx[1]], h_values[idx[2]]
        f1, f2, f3 = f_values[idx[0]], f_values[idx[1]], f_values[idx[2]]

        r21 = h2 / h1
        r32 = h3 / h2

        epsilon_21 = f2 - f1
        epsilon_32 = f3 - f2

        if abs(epsilon_21) < 1e-15 or abs(epsilon_32) < 1e-15:
            return {"extrapolated": float(f1), "order": float("nan"), "gci_fine": 0.0}

        ratio = epsilon_32 / epsilon_21
        if ratio <= 0:
            return {"extrapolated": float(f1), "order": float("nan"), "gci_fine": 0.0,
                    "note": "oscillatory_convergence"}

        # Apparent order of convergence
        p = abs(np.log(abs(ratio)) / np.log(r21))

        # Extrapolated value
        f_exact = f1 + (f1 - f2) / (r21 ** p - 1)

        # GCI (Grid Convergence Index) with safety factor Fs = 1.25
        Fs = 1.25
        gci_fine = Fs * abs((f2 - f1) / f1) / (r21 ** p - 1) * 100  # percentage

        return {
            "extrapolated": round(float(f_exact), 6),
            "order": round(float(p), 4),
            "gci_fine_percent": round(float(gci_fine), 4),
            "f1": round(float(f1), 6),
            "f2": round(float(f2), 6),
            "f3": round(float(f3), 6),
            "r21": round(float(r21), 4),
        }

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

        # Find grid size column
        grid_col = None
        for col in df.columns:
            if re.match(r"^(cell[-_]?count|elements|nodes|cells|n[-_]?cells)$", col, re.IGNORECASE):
                grid_col = col
                break
        if grid_col is None:
            for col in df.columns:
                if re.match(r"^(mesh[-_]?level|grid[-_]?level|grid|mesh|refinement[-_]?level|h|grid[-_]?size)$", col, re.IGNORECASE):
                    grid_col = col
                    break
        if grid_col is None:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="No grid size column (cell_count/elements/nodes) found in data.",
            )

        # Identify solution quantity columns
        solution_cols = [
            col for col in df.columns
            if col != grid_col
            and not re.match(r"^(mesh[-_]?level|grid[-_]?level|grid|mesh|refinement[-_]?level)$", col, re.IGNORECASE)
            and df[col].dtype in [np.float64, np.float32, float, np.int64, np.int32]
        ]
        if not solution_cols:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="No solution quantity columns found in data.",
            )

        # Sort by grid size ascending
        df = df.sort_values(grid_col, ascending=True).reset_index(drop=True)
        grid_values = df[grid_col].values.astype(float)

        # Compute effective grid spacing (assuming 2D if max cells < 1M, else 3D)
        dim = 2 if np.max(grid_values) < 1e6 else 3
        h_values = 1.0 / (grid_values ** (1.0 / dim))

        results = {}
        for col in solution_cols:
            f_values = df[col].values.astype(float)
            valid = ~np.isnan(f_values)
            if np.sum(valid) < 3:
                results[col] = {"status": "insufficient_data"}
                continue

            re_result = self._richardson_extrapolation(h_values[valid], f_values[valid])
            if re_result is None:
                results[col] = {"status": "insufficient_data"}
                continue

            monotonic = all(
                (f_values[i+1] - f_values[i]) * (f_values[1] - f_values[0]) >= 0
                for i in range(len(f_values) - 1)
            ) if len(f_values) > 1 else False

            results[col] = {
                "values": [round(float(v), 6) for v in f_values],
                "grid_sizes": [int(g) for g in grid_values],
                "richardson_extrapolation": re_result,
                "monotonic_convergence": monotonic,
                "relative_change_finest": round(
                    abs((f_values[-1] - f_values[-2]) / f_values[-1]) * 100 if len(f_values) >= 2 and abs(f_values[-1]) > 1e-15 else 0, 4
                ),
            }

        # Generate plots
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt, _ = apply_publication_style()

        n_quantities = len(solution_cols)
        fig, axes = plt.subplots(1, n_quantities, figsize=(6 * n_quantities, 6))
        if n_quantities == 1:
            axes = [axes]

        for idx, col in enumerate(solution_cols):
            ax = axes[idx]
            f_values = df[col].values.astype(float)
            ax.plot(grid_values, f_values, "bo-", markersize=8, linewidth=2.0, label="CFD results")

            info = results.get(col, {})
            re_info = info.get("richardson_extrapolation", {})
            if "extrapolated" in re_info and not np.isnan(re_info.get("order", float("nan"))):
                ax.axhline(y=re_info["extrapolated"], color="red", linestyle="--",
                          linewidth=1.5, alpha=0.7,
                          label=f"Richardson extrap. = {re_info['extrapolated']:.4f}")
                if re_info.get("gci_fine_percent", 0) > 0:
                    gci = re_info["gci_fine_percent"]
                    extrap = re_info["extrapolated"]
                    ax.fill_between(
                        [grid_values[0] * 0.8, grid_values[-1] * 1.2],
                        extrap * (1 - gci / 100), extrap * (1 + gci / 100),
                        alpha=0.15, color="red", label=f"GCI = {gci:.2f}%"
                    )

            ax.set_xlabel("Cell Count")
            ax.set_ylabel(col)
            ax.set_title(f"Grid Convergence: {col}")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("Grid Independence Study", fontsize=16, fontweight="bold")

        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = figures_dir / "grid_independence.png"
        save_figure(str(fig_path))
        plt.close(fig)

        summary_lines = ["Grid Independence Study Results:", ""]
        summary_lines.append(f"  Grid levels: {len(df)}")
        summary_lines.append(f"  Assumed dimensionality: {dim}D")
        for col, info in results.items():
            if "richardson_extrapolation" not in info:
                summary_lines.append(f"  {col}: {info.get('status', 'unknown')}")
                continue
            re_info = info["richardson_extrapolation"]
            summary_lines.append(f"  {col}:")
            summary_lines.append(f"    Finest grid value: {info['values'][-1]}")
            summary_lines.append(f"    Richardson extrapolated: {re_info.get('extrapolated', 'N/A')}")
            summary_lines.append(f"    Apparent order: {re_info.get('order', 'N/A')}")
            summary_lines.append(f"    GCI (fine): {re_info.get('gci_fine_percent', 'N/A')}%")
            summary_lines.append(f"    Monotonic convergence: {info.get('monotonic_convergence', 'N/A')}")
            summary_lines.append(f"    Relative change (finest pair): {info.get('relative_change_finest', 'N/A')}%")
        summary_lines.append(f"\nFigure saved to: {fig_path.as_posix()}")

        return ToolResponse.success(
            text="\n".join(summary_lines),
            data={"grid_study_details": results, "figure_path": fig_path.as_posix()},
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
                    'Example: {"data_path": "/path/to/mesh.csv", "figures_dir": "/path/to/figures/"}'
                ),
                required=True,
            )
        ]
