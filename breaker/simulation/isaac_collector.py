from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from breaker.core import Case, DoubleWellSpring, Friction, GenericDynamicsSolver
from breaker.core.solver import gravity_term, spring_term


def _effective_joint_inertia(articulation: Any, joint_name: str) -> float:
    mass_matrix = articulation.get_mass_matrices().numpy()
    dof_count = len(articulation.dof_names)
    dof_index = articulation.dof_names.index(joint_name)
    joint_column = (mass_matrix.shape[-1] - dof_count) + dof_index
    return float(mass_matrix[0, joint_column, joint_column])


class _PhysicsStepRecorder:
    """在每个物理子步写入内部弹簧力矩并记录拨片状态。"""

    def __init__(self, case, stage, articulation, config, spring):
        from pxr import UsdPhysics

        self.case = case
        self.stage = stage
        self.articulation = articulation
        self.config = config
        self.spring = spring if case.spring_term is not None else None
        self.joint_prim = stage.GetPrimAtPath(config["joint_path"])
        self.dof_index = articulation.get_dof_indices(config["joint_name"])
        self.dof_count = len(articulation.dof_names)
        self.times: list[float] = []
        self.theta: list[float] = []
        self.omega: list[float] = []
        self.spring_torque: list[float] = []
        self.elapsed_time = 0.0
        self._usd_physics = UsdPhysics

    def configure_physics(self) -> None:
        for prim in self.stage.Traverse():
            if prim.IsA(self._usd_physics.Scene):
                scene = self._usd_physics.Scene(prim)
                if self.case.gravity_term is None:
                    scene.GetGravityMagnitudeAttr().Set(0.0)
                else:
                    scene.GetGravityDirectionAttr().Set((0.0, 0.0, -1.0))
                    scene.GetGravityMagnitudeAttr().Set(9.81)

        positions = self.articulation.get_dof_positions().numpy().copy()
        velocities = np.zeros((1, self.dof_count), dtype=np.float32)
        velocity_targets = np.zeros((1, self.dof_count), dtype=np.float32)
        positions[0, self.dof_index] = self.case.theta0_rad
        velocities[0, self.dof_index] = self.case.omega0_rad_s
        self.articulation.set_dof_positions(positions)
        self.articulation.set_dof_velocities(velocities)
        self.articulation.set_dof_velocity_targets(velocity_targets)

        lower, upper = self.case.joint_limit
        revolute_joint = self._usd_physics.RevoluteJoint(self.joint_prim)
        revolute_joint.GetLowerLimitAttr().Set(float(math.degrees(lower)))
        revolute_joint.GetUpperLimitAttr().Set(float(math.degrees(upper)))

        self.articulation.switch_dof_control_mode("effort")
        self.articulation.set_dof_gains(stiffnesses=0.0, dampings=0.0)
        drive_api = self._usd_physics.DriveAPI.Get(self.joint_prim, "angular")
        drive_api.GetDampingAttr().Set(0.0)
        self.articulation.set_dof_friction_properties(
            static_frictions=self.case.friction.static,
            dynamic_frictions=self.case.friction.dynamic,
            viscous_frictions=self.case.viscous_damping,
        )
        self.elapsed_time = 0.0

    def pre_step(self, dt, context=None) -> None:
        positions = self.articulation.get_dof_positions().numpy()
        velocities = self.articulation.get_dof_velocities().numpy()
        theta = float(positions[0, self.dof_index][0])
        omega = float(velocities[0, self.dof_index][0])
        torque = self.spring.torque(theta) if self.spring is not None else 0.0

        efforts = np.zeros((1, self.dof_count), dtype=np.float32)
        efforts[0, self.dof_index] = torque
        self.articulation.set_dof_efforts(efforts)

        self.times.append(self.elapsed_time)
        self.theta.append(theta)
        self.omega.append(omega)
        self.spring_torque.append(torque)
        self.elapsed_time += float(dt)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.times, dtype=float),
            np.asarray(self.theta, dtype=float),
            np.asarray(self.omega, dtype=float),
            np.asarray(self.spring_torque, dtype=float),
        )


def run_isaac_simulation(config: dict[str, Any]) -> dict[str, Any]:
    """统一启动 Isaac Sim、运行单次基准仿真并采集结果。"""
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(config["headless"])})
    # try: 
    return _run_with_app(config)
   
    

def _run_with_app(config: dict[str, Any]) -> dict[str, Any]:
    """SimulationApp 启动后执行 USD、PhysX 和数据采集流程。"""
    from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent
    from isaacsim.core.experimental.prims import Articulation
    import isaacsim.core.experimental.utils.app as app_utils
    import omni.usd
    from pxr import UsdPhysics
    
    usd_path = Path(config["usd_path"]).expanduser().resolve()
    usd_context = omni.usd.get_context()
    usd_context.open_stage(str(usd_path))
    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError(f"USD Stage 加载失败: {usd_path}")

    joint_prim = stage.GetPrimAtPath(config["joint_path"])
    if not joint_prim.IsValid():
        raise RuntimeError(f"找不到 RevoluteJoint: {config['joint_path']}")
    revolute_joint = UsdPhysics.RevoluteJoint(joint_prim)
    local_pos1 = np.asarray(revolute_joint.GetLocalPos1Attr().Get(), dtype=float)
    axis_lookup = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    axis_name = revolute_joint.GetAxisAttr().Get()
    axis = axis_lookup[str(axis_name)]
    mass_prim = stage.GetPrimAtPath(config["handle_path"])
    mass_api = UsdPhysics.MassAPI.Apply(mass_prim)
    mass = float(mass_api.GetMassAttr().Get())
    drive_api = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
    damping_per_degree = float(drive_api.GetDampingAttr().Get())
    damping_rad = damping_per_degree * 180.0 / math.pi

    SimulationManager.set_physics_dt(float(config["physics_dt"]))
    app_utils.play()
    articulation = Articulation(config["base_path"])
    inertia = _effective_joint_inertia(articulation, config["joint_name"])

    gravity = None
    if config["gravity_enabled"]:
        gravity = gravity_term(
            mass=mass,
            r0=-local_pos1,
            axis=axis,
            theta_ref=0.0,
            g=tuple(config["gravity"]),
        )

    spring = DoubleWellSpring(
        theta0=math.radians(float(config["spring"]["stable_angle_deg"])),
        tau_peak=float(config["spring"]["peak_torque_nm"]),
        center=math.radians(float(config["spring"]["center_deg"])),
    )
    damping_ratio = float(config["damping_ratio"])
    viscous_damping = 2.0 * damping_ratio * math.sqrt(inertia * spring.k_eff_well)
    case = Case(
        name=str(config["case_name"]),
        gravity_term=gravity,
        spring_term=spring_term(spring),
        viscous_damping=viscous_damping,
        friction=Friction(
            static=float(config["friction"]["static_nm"]),
            dynamic=float(config["friction"]["dynamic_nm"]),
        ),
        joint_limit=tuple(math.radians(v) for v in config["joint_limit_deg"]),
        theta0_rad=math.radians(float(config["initial_position_deg"])),
        omega0_rad_s=math.radians(float(config["initial_velocity_deg_s"])),
    )

    recorder = _PhysicsStepRecorder(case, stage, articulation, config, spring)
    recorder.configure_physics()
    callback_id = SimulationManager.register_callback(
        recorder.pre_step, event=SimulationEvent.PHYSICS_PRE_STEP
    )
    try:
        for _ in range(int(config["num_steps"])): 
            SimulationManager.step()
    finally:
        SimulationManager.deregister_callback(callback_id)
        app_utils.stop()
    
    times, actual_theta, actual_omega, spring_torque = recorder.arrays()
    solver = GenericDynamicsSolver(I=inertia, case=case, dt=float(config["physics_dt"]))
    expected_theta, expected_omega = solver.simulate_expected(
        times=times,
        theta0=case.theta0_rad,
        omega0=case.omega0_rad_s,
        limits=case.joint_limit,
        method=str(config["solver_method"]),
        h=float(config["solver_max_step"]),
    )
     
    return {
        "times": times,
        "actual_theta_rad": actual_theta,
        "actual_omega_rad_s": actual_omega,
        "spring_torque_nm": spring_torque,
        "expected_theta_rad": expected_theta,
        "expected_omega_rad_s": expected_omega,
        "metadata": {
            "usd_path": str(usd_path),
            "joint_name": config["joint_name"],
            "mass_kg": mass,
            "effective_inertia_kg_m2": inertia,
            "usd_damping_rad": damping_rad,
            "viscous_damping_nm_s_rad": viscous_damping,
            "spring_k_nm_rad3": spring.k,
            "spring_well_stiffness_nm_rad": spring.k_eff_well,
            "spring_barrier_j": spring.barrier,
        },
    }

