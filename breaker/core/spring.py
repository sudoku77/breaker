from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DoubleWellSpring:
    """双稳态内部弹簧模型，角度单位 rad，力矩单位 N·m。"""

    theta0: float
    tau_peak: float
    center: float = 0.0

    @property
    def k(self) -> float:
        """非线性系数，单位 N·m/rad³。"""
        return 3.0 * math.sqrt(3.0) * self.tau_peak / (2.0 * self.theta0**3)

    def torque(self, theta: float) -> float:
        x = theta - self.center
        return self.k * x * (self.theta0**2 - x * x)

    def potential(self, theta: float) -> float:
        """势能，单位 J；越中心位置势能为 0。"""
        x = theta - self.center
        return self.k * (x**4 / 4.0 - self.theta0**2 * x * x / 2.0)

    @property
    def k_eff_well(self) -> float:
        """稳定位置附近的等效扭转刚度，单位 N·m/rad。"""
        return 2.0 * self.k * self.theta0**2

    @property
    def barrier(self) -> float:
        """越中心势垒高度，单位 J。"""
        return self.k * self.theta0**4 / 4.0

    def other_turning_point(self, theta_start: float) -> float:
        """无阻尼、由静止出发时同一势阱中的另一侧转折点。"""
        x = theta_start - self.center
        u = 2.0 * self.theta0**2 - x * x
        if x == 0.0 or u <= 0.0:
            raise ValueError("初始位置不满足同一势阱内转折点计算条件")
        return self.center + math.copysign(math.sqrt(u), x)

