from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_comparison_csv(path: str | Path, result: dict[str, np.ndarray]) -> None:
    """导出实际与理论角度、角速度对比数据。角度单位为 degree。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    actual_theta_deg = np.degrees(result["actual_theta_rad"])
    actual_omega_deg_s = np.degrees(result["actual_omega_rad_s"])
    expected_theta_deg = np.degrees(result["expected_theta_rad"])
    expected_omega_deg_s = np.degrees(result["expected_omega_rad_s"])

    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "time_s",
                "actual_position_deg",
                "actual_velocity_deg_s",
                "expected_position_deg",
                "expected_velocity_deg_s",
                "position_error_deg",
                "velocity_error_deg_s",
            ]
        )
        for row in zip(
            result["times"],
            actual_theta_deg,
            actual_omega_deg_s,
            expected_theta_deg,
            expected_omega_deg_s,
        ):
            time_s, actual_theta, actual_omega, expected_theta, expected_omega = row
            writer.writerow(
                [
                    time_s,
                    actual_theta,
                    actual_omega,
                    expected_theta,
                    expected_omega,
                    actual_theta - expected_theta,
                    actual_omega - expected_omega,
                ]
            )

