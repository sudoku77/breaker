from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_comparison(output_dir: str | Path, result: dict[str, np.ndarray]) -> None:
    """保存角度和角速度的实际值/理论值对比图。"""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times = result["times"]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(times, np.degrees(result["actual_theta_rad"]), label="Actual position", linewidth=2)
    axis.plot(times, np.degrees(result["expected_theta_rad"]), "--", label="Expected position", linewidth=2)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Joint position (deg)")
    axis.set_title("Joint Position: Actual vs Expected")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "position_comparison.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(times, np.degrees(result["actual_omega_rad_s"]), label="Actual angular velocity", linewidth=2)
    axis.plot(times, np.degrees(result["expected_omega_rad_s"]), "--", label="Expected angular velocity", linewidth=2)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Angular velocity (deg/s)")
    axis.set_title("Angular Velocity: Actual vs Expected")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "velocity_comparison.png", dpi=150)
    plt.close(fig)

