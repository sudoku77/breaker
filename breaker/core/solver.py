from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .model import Case, Friction, TorqueTerm
from .spring import DoubleWellSpring


def spring_term(spring: DoubleWellSpring) -> TorqueTerm:
    """将双稳态弹簧包装为通用力矩项。"""

    def torque(theta: float, omega: float) -> float:
        return spring.torque(theta)

    torque.potential = spring.potential
    return torque


def gravity_term(
    mass: float,
    r0: Sequence[float],
    axis: Sequence[float],
    theta_ref: float = 0.0,
    g: Sequence[float] = (0.0, 0.0, -9.81),
) -> TorqueTerm:
    """根据质心位置和转轴方向生成重力力矩项。"""
    if not math.isfinite(mass) or mass < 0:
        raise ValueError("mass 必须是有限非负数")

    r0_array = np.asarray(r0, dtype=float)
    axis_array = np.asarray(axis, dtype=float)
    gravity = np.asarray(g, dtype=float)
    if any(vector.shape != (3,) for vector in (r0_array, axis_array, gravity)):
        raise ValueError("r0、axis、g 都必须是长度为 3 的向量")
    if not all(np.all(np.isfinite(vector)) for vector in (r0_array, axis_array, gravity)):
        raise ValueError("r0、axis、g 必须全部是有限数")

    axis_norm = np.linalg.norm(axis_array)
    if axis_norm < 1e-12:
        raise ValueError("axis 不能是零向量")
    axis_array = axis_array / axis_norm
    r_perp = r0_array - np.dot(r0_array, axis_array) * axis_array

    a = np.cross(axis_array, r_perp)
    c_cos = mass * float(np.dot(gravity, a))
    c_sin = -mass * float(np.dot(gravity, r_perp))

    def torque(theta: float, omega: float) -> float:
        phi = theta - theta_ref
        return c_cos * math.cos(phi) + c_sin * math.sin(phi)

    def potential(theta: float) -> float:
        phi = theta - theta_ref
        return -c_cos * math.sin(phi) + c_sin * math.cos(phi)

    torque.potential = potential
    return torque


def friction_torque(
    omega: float,
    external_torque: float,
    friction: Friction,
    omega_epsilon: float = 1e-8,
) -> float:
    """库仑摩擦近似，静止时由静摩擦抵消阈值内的外力矩。"""
    if abs(omega) <= omega_epsilon:
        if abs(external_torque) <= friction.static:
            return -external_torque
        return -math.copysign(friction.dynamic, external_torque)
    return -math.copysign(friction.dynamic, omega)


class GenericDynamicsSolver:
    """单自由度转动动力学理论求解器。"""

    def __init__(self, I: float, case: Case, dt: float):
        self.I = float(I)
        self.b = float(case.viscous_damping)
        self.dt = float(dt)
        self.friction = case.friction or Friction()
        self.terms = [case.gravity_term, case.spring_term]

    def conservative_torque(self, theta: float, omega: float) -> float:
        return sum(term(theta, omega) for term in self.terms if term is not None)

    def acceleration(self, theta: float, omega: float) -> float:
        torque = self.conservative_torque(theta, omega) - self.b * omega
        torque += friction_torque(omega, torque, self.friction)
        return torque / self.I

    def rk4_step(self, theta: float, omega: float, dt: float) -> tuple[float, float]:
        def derivative(y_theta: float, y_omega: float) -> tuple[float, float]:
            return y_omega, self.acceleration(y_theta, y_omega)

        k1_theta, k1_omega = derivative(theta, omega)
        k2_theta, k2_omega = derivative(
            theta + 0.5 * dt * k1_theta,
            omega + 0.5 * dt * k1_omega,
        )
        k3_theta, k3_omega = derivative(
            theta + 0.5 * dt * k2_theta,
            omega + 0.5 * dt * k2_omega,
        )
        k4_theta, k4_omega = derivative(
            theta + dt * k3_theta,
            omega + dt * k3_omega,
        )

        theta_next = theta + dt / 6.0 * (
            k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta
        )
        omega_next = omega + dt / 6.0 * (
            k1_omega + 2.0 * k2_omega + 2.0 * k3_omega + k4_omega
        )
        return theta_next, omega_next

    def semi_implicit_step(
        self, theta: float, omega: float, dt: float
    ) -> tuple[float, float]:
        torque = self.conservative_torque(theta, omega)
        if abs(omega) < 1e-8 and abs(torque) <= self.friction.static:
            return theta, 0.0
        direction = omega if abs(omega) >= 1e-8 else torque
        torque -= math.copysign(self.friction.dynamic, direction)
        omega_next = (omega + dt * torque / self.I) / (
            1.0 + dt * self.b / self.I
        )
        return theta + dt * omega_next, omega_next

    def simulate_expected(
        self,
        times: np.ndarray,
        theta0: float,
        omega0: float,
        limits: tuple[Optional[float], Optional[float]] = (None, None),
        method: str = "rk4",
        h: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """在给定采样时刻返回角度和角速度；h 是积分最大步长覆盖参数。"""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
            raise ValueError("times 必须是一维有限数数组且不能为空")
        if len(times) > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("times 必须严格递增")
        if times[0] < 0:
            raise ValueError("times 不能包含负数")
        if not math.isfinite(self.I) or self.I <= 0:
            raise ValueError("I 必须是有限正数")
        if not math.isfinite(self.b) or self.b < 0:
            raise ValueError("粘性阻尼必须是有限非负数")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt 必须是有限正数")
        if method not in ("rk4", "semi_implicit"):
            raise ValueError("method 必须是 rk4 或 semi_implicit")

        lower, upper = limits
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("lower_limit 必须小于 upper_limit")
        if lower is not None and theta0 < lower:
            raise ValueError("初始角度低于下限")
        if upper is not None and theta0 > upper:
            raise ValueError("初始角度高于上限")

        max_step = self.dt if h is None else min(self.dt, float(h))
        if not math.isfinite(max_step) or max_step <= 0:
            raise ValueError("h 必须是有限正数")

        theta = float(theta0)
        omega = float(omega0)
        theta_out = np.empty(len(times), dtype=float)
        omega_out = np.empty(len(times), dtype=float)
        theta_out[0], omega_out[0] = theta, omega
        current_time = float(times[0])

        for sample_index in range(1, len(times)):
            target_time = float(times[sample_index])
            while current_time < target_time - 1e-12:
                step_dt = min(max_step, target_time - current_time)
                if method == "rk4":
                    theta_next, omega_next = self.rk4_step(theta, omega, step_dt)
                else:
                    theta_next, omega_next = self.semi_implicit_step(
                        theta, omega, step_dt
                    )

                if lower is not None and theta_next < lower:
                    theta_next = lower
                    if omega_next < 0:
                        omega_next = 0.0
                if upper is not None and theta_next > upper:
                    theta_next = upper
                    if omega_next > 0:
                        omega_next = 0.0

                theta, omega = theta_next, omega_next
                current_time += step_dt

            theta_out[sample_index] = theta
            omega_out[sample_index] = omega

        return theta_out, omega_out

    def total_potential(self, theta: float) -> float:
        """返回所有带势能定义的保守力矩项的总势能。"""
        return sum(
            potential(theta)
            for term in self.terms
            if term is not None
            if (potential := getattr(term, "potential", None)) is not None
        )

