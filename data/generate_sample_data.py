"""Generate synthetic CFD sample data for CFDFlow-Agent demos."""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
DATA_DIR = Path(__file__).parent


def generate_residual_history():
    n = 2000
    iterations = np.arange(1, n + 1)

    def make_residual(drop_orders, noise_level=0.1):
        base = np.logspace(0, -drop_orders, n)
        # Add plateau around iteration 400-600
        plateau = np.ones(n)
        plateau[400:600] = np.exp(-0.3 * np.abs(np.arange(200) - 100) / 100)
        residual = base * (1 + noise_level * np.random.randn(n))
        residual[400:600] *= 5  # temporary plateau
        residual = np.maximum(residual, 1e-12)
        return residual

    df = pd.DataFrame({
        "iteration": iterations,
        "continuity": make_residual(5.0, 0.08),
        "x-velocity": make_residual(6.0, 0.06),
        "y-velocity": make_residual(5.5, 0.07),
        "energy": make_residual(7.0, 0.05),
        "k": make_residual(4.0, 0.12),
        "omega": make_residual(3.5, 0.15),
    })
    df.to_csv(DATA_DIR / "residual_history.csv", index=False)
    print(f"Generated residual_history.csv: {df.shape}")


def generate_force_coefficients():
    n = 2000
    iterations = np.arange(1, n + 1)

    # Cl: starts with large transient, settles to ~0.45
    cl_steady = 0.45
    cl_transient = 1.5 * np.exp(-iterations / 150) * np.sin(iterations * 0.05)
    cl_noise = 0.002 * np.random.randn(n)
    cl_oscillation = 0.003 * np.sin(iterations * 0.02)
    cl = cl_steady + cl_transient + cl_noise + cl_oscillation

    # Cd: settles to ~0.025
    cd_steady = 0.025
    cd_transient = 0.15 * np.exp(-iterations / 120)
    cd_noise = 0.0005 * np.random.randn(n)
    cd = cd_steady + cd_transient + cd_noise

    # Cm: settles to ~-0.08
    cm_steady = -0.08
    cm_transient = 0.3 * np.exp(-iterations / 180) * np.cos(iterations * 0.03)
    cm_noise = 0.001 * np.random.randn(n)
    cm = cm_steady + cm_transient + cm_noise

    df = pd.DataFrame({
        "iteration": iterations,
        "Cl": cl,
        "Cd": cd,
        "Cm": cm,
    })
    df.to_csv(DATA_DIR / "force_coefficients.csv", index=False)
    print(f"Generated force_coefficients.csv: {df.shape}")


def generate_pressure_distribution():
    n = 100
    x_c = np.linspace(0, 1, n)

    # NACA 0012 at alpha=4 degrees approximation
    # Upper surface
    cp_upper = np.zeros(n)
    cp_upper[0] = 1.0  # stagnation
    for i in range(1, n):
        x = x_c[i]
        # Suction peak around x/c = 0.08
        cp_upper[i] = -1.5 * np.exp(-((x - 0.08) / 0.06) ** 2)
        # Pressure recovery
        cp_upper[i] += 0.6 * x ** 0.5
        # Trailing edge
        cp_upper[i] += 0.15 * x ** 2
    cp_upper[0] = 1.0
    cp_upper += 0.01 * np.random.randn(n)

    # Lower surface
    cp_lower = np.zeros(n)
    cp_lower[0] = 1.0  # stagnation
    for i in range(1, n):
        x = x_c[i]
        cp_lower[i] = -0.3 * np.exp(-((x - 0.15) / 0.1) ** 2)
        cp_lower[i] += 0.4 * x ** 0.5
        cp_lower[i] += 0.1 * x
    cp_lower[0] = 1.0
    cp_lower += 0.008 * np.random.randn(n)

    df = pd.DataFrame({
        "x/c": x_c,
        "Cp_upper": cp_upper,
        "Cp_lower": cp_lower,
    })
    df.to_csv(DATA_DIR / "pressure_distribution.csv", index=False)
    print(f"Generated pressure_distribution.csv: {df.shape}")


def generate_velocity_profile():
    n = 100
    y_plus = np.logspace(-1, 4, n)

    kappa = 0.41
    B = 5.2

    u_plus = np.zeros(n)
    for i, yp in enumerate(y_plus):
        if yp < 5:
            # Viscous sublayer
            u_plus[i] = yp
        elif yp < 30:
            # Buffer layer (Spalding interpolation approximation)
            u_vis = yp
            u_log = (1.0 / kappa) * np.log(yp) + B
            blend = (yp - 5) / 25
            u_plus[i] = (1 - blend) * u_vis + blend * u_log
        else:
            # Log layer with wake
            u_plus[i] = (1.0 / kappa) * np.log(yp) + B
            # Wake function (Coles)
            if yp > 100:
                Pi = 0.55
                eta = np.log(yp) / np.log(y_plus[-1])
                u_plus[i] += (2 * Pi / kappa) * np.sin(np.pi / 2 * eta) ** 2

    # Add noise
    u_plus += 0.03 * u_plus * np.random.randn(n)
    u_plus = np.maximum(u_plus, 0)

    df = pd.DataFrame({
        "y_plus": np.round(y_plus, 4),
        "u_plus": np.round(u_plus, 4),
    })
    df.to_csv(DATA_DIR / "velocity_profile.csv", index=False)
    print(f"Generated velocity_profile.csv: {df.shape}")


def generate_mesh_study():
    cell_counts = [50000, 100000, 200000, 400000, 800000]
    mesh_levels = ["Coarse", "Medium", "Fine", "Very Fine", "Extra Fine"]

    # Simulate second-order convergence
    h = [1.0 / (c ** 0.5) for c in cell_counts]  # 2D
    h_norm = [hi / h[-1] for hi in h]

    # Cl converges from 0.42 to ~0.4512
    cl_exact = 0.4515
    cl_values = [cl_exact + 0.035 * (hn ** 2) + 0.003 * np.random.randn() for hn in h_norm]
    cl_values.sort(reverse=False)  # ensure monotonic trend
    cl_values = [round(v, 5) for v in cl_values]

    # Cd converges from 0.028 to ~0.0245
    cd_exact = 0.0243
    cd_values = [cd_exact + 0.006 * (hn ** 2) + 0.0003 * np.random.randn() for hn in h_norm]
    cd_values.sort(reverse=True)
    cd_values = [round(v, 6) for v in cd_values]

    # Cp_min converges from -1.35 to -1.52
    cpmin_exact = -1.525
    cpmin_values = [cpmin_exact + 0.2 * (hn ** 2) + 0.01 * np.random.randn() for hn in h_norm]
    cpmin_values.sort(reverse=True)
    cpmin_values = [round(v, 4) for v in cpmin_values]

    df = pd.DataFrame({
        "mesh_level": mesh_levels,
        "cell_count": cell_counts,
        "Cl": cl_values,
        "Cd": cd_values,
        "Cp_min": cpmin_values,
    })
    df.to_csv(DATA_DIR / "mesh_study.csv", index=False)
    print(f"Generated mesh_study.csv: {df.shape}")


if __name__ == "__main__":
    generate_residual_history()
    generate_force_coefficients()
    generate_pressure_distribution()
    generate_velocity_profile()
    generate_mesh_study()
    print("\nAll sample data files generated successfully!")
