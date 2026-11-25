from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


TorqueTerm = Callable[[float, float], float]


@dataclass
class Friction:
    """关节库仑摩擦参数，单位 N·m。"""

    static: float = 0.0
    dynamic: float = 0.0

    def __post_init__(self) -> None:
        if self.static < 0 or self.dynamic < 0:
            raise ValueError("摩擦力矩不能为负数")


@dataclass
class Case:
    """单自由度拨片工况；角度使用 rad，角速度使用 rad/s。"""

    name: str
    gravity_term: TorqueTerm | None = None
    spring_term: TorqueTerm | None = None
    viscous_damping: float = 0.0
    friction: Friction = field(default_factory=Friction)
    joint_limit: tuple[float, float] = (-math.inf, math.inf)
    theta0_rad: float = 0.0
    omega0_rad_s: float = 0.0

