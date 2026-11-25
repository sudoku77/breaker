from __future__ import annotations

import numpy as np


def comparison_metrics(result: dict[str, np.ndarray]) -> dict[str, float]:
    """计算角度和角速度的 MAE、RMSE，角度以 degree 表示。"""
    position_error = np.degrees(
        result["actual_theta_rad"] - result["expected_theta_rad"]
    )
    velocity_error = np.degrees(
        result["actual_omega_rad_s"] - result["expected_omega_rad_s"]
    )
    return {
        "position_mae_deg": float(np.mean(np.abs(position_error))),
        "position_rmse_deg": float(np.sqrt(np.mean(position_error**2))),
        "velocity_mae_deg_s": float(np.mean(np.abs(velocity_error))),
        "velocity_rmse_deg_s": float(np.sqrt(np.mean(velocity_error**2))),
    }

