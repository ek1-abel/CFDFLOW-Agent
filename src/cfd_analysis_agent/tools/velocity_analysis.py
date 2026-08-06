"""Velocity profile analysis tool for CFD simulations."""

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


class VelocityAnalysisTool(Tool):
    """Analyze boundary layer velocity profiles from CFD simulations."""

    def __init__(self):
        super().__init__(
            name="VelocityAnalysisTool",
            description=(
                "Analyze boundary layer velocity profiles from CFD results. "
                "Input: JSON string with keys 'data_path' (path to CSV with y_plus/u_plus or y/u columns) "
                "and 'figures_dir' (directory for output figures). "
                "Returns boundary layer analysis with law of wall comparison and generates velocity profile plot."
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

        # Detect coordinate system
        y_col = None
        u_col = None
        is_wall_coords = False

        for col in df.columns:
            cl = col.strip().lower()
            if cl in ("y_plus", "y+", "yplus"):
                y_col = col
                is_wall_coords = True
            elif cl in ("y", "y/delta", "y_norm"):
                y_col = col

        for col in df.columns:
            cl = col.strip().lower()
            if cl in ("u_plus", "u+", "uplus"):
                u_col = col
                is_wall_coords = True
            elif cl in ("u", "u/u_inf", "u_norm"):
                u_col = col

        if y_col is None or u_col is None:
            return ToolResponse.error(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="No velocity profile columns (y_plus/u_plus or y/u) found in data.",
            )

        y_vals = df[y_col].values.astype(float)
        u_vals = df[u_col].values.astype(float)
        valid = (~np.isnan(y_vals)) & (~np.isnan(u_vals)) & (y_vals > 0)
        y_vals = y_vals[valid]
        u_vals = u_vals[valid]

        results = {
            "coordinate_system": "wall_units" if is_wall_coords else "physical",
            "y_column": y_col,
            "u_column": u_col,
            "n_points": int(len(y_vals)),
            "y_range": [round(float(np.min(y_vals)), 4), round(float(np.max(y_vals)), 4)],
            "u_range": [round(float(np.min(u_vals)), 4), round(float(np.max(u_vals)), 4)],
        }

        if is_wall_coords:
            # Compare with law of wall
            kappa = 0.41
            B = 5.2

            # Viscous sublayer (y+ < 5): u+ = y+
            viscous_mask = y_vals < 5
            if np.sum(viscous_mask) > 0:
                u_theory_viscous = y_vals[viscous_mask]
                u_actual_viscous = u_vals[viscous_mask]
                viscous_rmse = float(np.sqrt(np.mean((u_actual_viscous - u_theory_viscous) ** 2)))
                results["viscous_sublayer_rmse"] = round(viscous_rmse, 4)

            # Log layer (y+ > 30): u+ = (1/kappa) * ln(y+) + B
            log_mask = y_vals > 30
            if np.sum(log_mask) > 0:
                u_theory_log = (1.0 / kappa) * np.log(y_vals[log_mask]) + B
                u_actual_log = u_vals[log_mask]
                log_rmse = float(np.sqrt(np.mean((u_actual_log - u_theory_log) ** 2)))
                results["log_layer_rmse"] = round(log_rmse, 4)

            # Estimate friction velocity from log law fit
            log_region = (y_vals > 30) & (y_vals < 300)
            if np.sum(log_region) > 3:
                from numpy.polynomial import polynomial as P
                log_y = np.log(y_vals[log_region])
                coeffs = np.polyfit(log_y, u_vals[log_region], 1)
                fitted_kappa = 1.0 / coeffs[0] if abs(coeffs[0]) > 0.01 else None
                if fitted_kappa:
                    results["fitted_von_karman_constant"] = round(float(fitted_kappa), 4)
        else:
            # Physical coordinates: estimate boundary layer thickness
            if len(u_vals) > 0:
                u_max = np.max(u_vals)
                delta_99_mask = u_vals >= 0.99 * u_max
                if np.any(delta_99_mask):
                    delta_99_idx = np.argmax(delta_99_mask)
                    results["delta_99"] = round(float(y_vals[delta_99_idx]), 6)

                # Displacement thickness
                if u_max > 0:
                    integrand = 1.0 - u_vals / u_max
                    delta_star = float(np.trapz(integrand, y_vals))
                    results["displacement_thickness"] = round(delta_star, 6)

                    # Momentum thickness
                    integrand_theta = (u_vals / u_max) * (1.0 - u_vals / u_max)
                    theta = float(np.trapz(integrand_theta, y_vals))
                    results["momentum_thickness"] = round(theta, 6)

                    if theta > 0:
                        results["shape_factor"] = round(delta_star / theta, 4)

        # Generate plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt, _ = apply_publication_style()

        fig, ax = plt.subplots(figsize=(10, 7))

        if is_wall_coords:
            ax.semilogx(y_vals, u_vals, "bo", markersize=4, alpha=0.7, label="CFD Data")

            # Viscous sublayer
            y_vis = np.linspace(0.1, 5, 50)
            ax.semilogx(y_vis, y_vis, "r-", linewidth=2.0, label="$u^+ = y^+$ (viscous)")

            # Log law
            y_log = np.linspace(30, np.max(y_vals), 100)
            u_log = (1.0 / 0.41) * np.log(y_log) + 5.2
            ax.semilogx(y_log, u_log, "g--", linewidth=2.0, label="Log law ($\\kappa=0.41$, B=5.2)")

            ax.set_xlabel("$y^+$")
            ax.set_ylabel("$u^+$")
            ax.set_title("Velocity Profile in Wall Units")
        else:
            ax.plot(u_vals, y_vals, "b-o", markersize=4, linewidth=1.5, label="CFD Data")
            ax.set_xlabel("$u / U_\\infty$" if "u_inf" in u_col.lower() else "u")
            ax.set_ylabel("y")
            ax.set_title("Boundary Layer Velocity Profile")

            if "delta_99" in results:
                ax.axhline(y=results["delta_99"], color="red", linestyle="--",
                          alpha=0.7, label=f"$\\delta_{{99}}$ = {results['delta_99']:.4f}")

        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = figures_dir / "velocity_profile.png"
        save_figure(str(fig_path))
        plt.close(fig)

        summary_lines = ["Velocity Profile Analysis Results:", ""]
        summary_lines.append(f"  Coordinate system: {results['coordinate_system']}")
        summary_lines.append(f"  Data points: {results['n_points']}")
        if "delta_99" in results:
            summary_lines.append(f"  Boundary layer thickness (delta_99): {results['delta_99']}")
        if "displacement_thickness" in results:
            summary_lines.append(f"  Displacement thickness: {results['displacement_thickness']}")
        if "momentum_thickness" in results:
            summary_lines.append(f"  Momentum thickness: {results['momentum_thickness']}")
        if "shape_factor" in results:
            summary_lines.append(f"  Shape factor (H): {results['shape_factor']}")
        if "viscous_sublayer_rmse" in results:
            summary_lines.append(f"  Viscous sublayer RMSE: {results['viscous_sublayer_rmse']}")
        if "log_layer_rmse" in results:
            summary_lines.append(f"  Log layer RMSE: {results['log_layer_rmse']}")
        if "fitted_von_karman_constant" in results:
            summary_lines.append(f"  Fitted von Karman constant: {results['fitted_von_karman_constant']}")
        summary_lines.append(f"\nFigure saved to: {fig_path.as_posix()}")

        return ToolResponse.success(
            text="\n".join(summary_lines),
            data={"velocity_details": results, "figure_path": fig_path.as_posix()},
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
                    'Example: {"data_path": "/path/to/velocity.csv", "figures_dir": "/path/to/figures/"}'
                ),
                required=True,
            )
        ]
