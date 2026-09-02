#!/usr/bin/env python3
"""XL330 测试台的 sim2real 验证.

在固定目标角度序列上运行同一个 ONNX 策略, 可以在 MuJoCo (用 BAM M6 执行器模型)
或真实 XL330 (通过 rustypot) 上运行, 记录关节轨迹, 并绘制 sim 与 real 的对比图.

示例工作流
----------------
    # 1) 在 sim 中录制:
    uv run python scripts/testbench_sim2real.py --mode sim  --onnx policy.onnx --out sim.npz
    # 2) 用 USB 连接真实测试台, 然后在硬件上录制:
    uv run python scripts/testbench_sim2real.py --mode real --onnx policy.onnx --out real.npz \
        --port /dev/ttyUSB0 --motor-id 1
    # 3) 对比两条轨迹:
    uv run python scripts/testbench_sim2real.py --compare sim.npz real.npz --out-plot comparison.png

观测布局 (必须与训练 env 一致): [joint_pos, joint_vel, last_action, command]
动作: 1-D 位置偏移 (弧度), 缩放系数 1.0, 加到默认姿态 (0.0) 上.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from mjlab_microduck.robot.testbench_constants import (
    TESTBENCH_ARM_MASS,
    TESTBENCH_XML,
    _set_arm_mass,
)

# --- 与训练 env 对齐 ---
CONTROL_DT = 0.02  # decimation=4 × timestep=0.005  (策略频率 = 50 Hz)
SIM_DT = 0.005  # (记录频率 = 200 Hz — 每个 sim 内步采样一次)
LOG_DT = SIM_DT
DEFAULT_POS = 0.0
MAX_ANGLE = math.radians(80.0)

# XL330 的 present_velocity 由 rustypot 以原始 tick 返回 (i32, 未换算).
# 每个 tick = 0.229 RPM (依 Dynamixel XL330 规范). rad/s = ticks * 0.229 * 2π/60.
DXL_VEL_TICK_TO_RAD_S = 0.229 * 2.0 * math.pi / 60.0  # ≈ 0.02398 rad/s per tick


# ---------------------------------------------------------------------------
# 共享: 目标调度 + 策略封装
# ---------------------------------------------------------------------------


def make_target_schedule(
    total_time: float,
    hold_time: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    """每个控制步返回一个目标角度."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(total_time / CONTROL_DT))
    steps_per_hold = int(round(hold_time / CONTROL_DT))
    targets = np.zeros(n_steps, dtype=np.float32)
    i = 0
    while i < n_steps:
        angle = float(rng.uniform(-MAX_ANGLE, MAX_ANGLE))
        end = min(i + steps_per_hold, n_steps)
        targets[i:end] = angle
        i = end
    return targets


class PolicyRunner:
    """加载 ONNX 策略, 按测试台 obs 布局逐步推进."""

    def __init__(self, onnx_path: str, action_scale: float = 1.0):
        """加载 ONNX session 完成初始化."""
        print(f"加载策略: {onnx_path}  (action_scale={action_scale})")
        self.session = ort.InferenceSession(onnx_path)
        self.in_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        print(f"  input  {self.in_name} shape={in_shape}")
        self.action_scale = action_scale
        self.last_action = np.zeros(1, dtype=np.float32)

    def reset(self):
        """将 last-action 缓冲区清零."""
        self.last_action[:] = 0.0

    def step(self, q: float, qd: float, target: float) -> float:
        """运行策略一步, 返回得到的目标位置."""
        # 与测试台 env 的策略 obs 布局一致:
        #   [joint_pos_rel, joint_vel_rel, last_action, command]  (4-d).
        obs = np.array(
            [q - DEFAULT_POS, qd, self.last_action[0], target],
            dtype=np.float32,
        )[None, :]
        action = self.session.run(None, {self.in_name: obs})[0].reshape(-1)
        self.last_action = action.astype(np.float32)
        return DEFAULT_POS + float(action[0]) * self.action_scale


# ---------------------------------------------------------------------------
# Sim rollout (mujoco, 与训练相同的 XL330 测试台 XML)
# ---------------------------------------------------------------------------


def rollout_sim_bam(onnx_path: str, total_time: float, seed: int, action_scale: float) -> dict:
    """使用 bam 的 MujocoController 在原生 MuJoCo 步进循环上做 sim rollout.

    优点: 200 Hz 内步日志, 不依赖 torch/mjwarp.  缺点: 不是训练时用的那个执行器
    (用 bam 上游, 而非 mjlab 的 M6).
    """
    # 从规范的 bam bundle 加载拟合好的 XL330 m6 参数 (与原来放在
    # mjlab_microduck.actuator.bam_params 中的值完全一致).
    import json as _json

    import mujoco  # 局部导入, 使 --mode real 在没有 mujoco 时也能工作
    from bam.actuators import actuators as bam_actuators
    from bam.model import _resolve_json_path
    from bam.model import models as bam_models
    from bam.mujoco import MujocoController

    with Path(_resolve_json_path(None, "xl330", "m6")).open() as _f:
        DEFAULT_XL330_M6 = _json.load(_f)

    VIN = 7.4
    KP_FW = 200.0
    ACTUATOR_NAME = "1"

    # 构建 BAM 的 M6 模型 + XL330 电压控制执行器. 下面的
    # MujocoController 每一步通过这个模型驱动关节, 把力矩写入
    # data.ctrl 并更新 dof_frictionloss/dof_damping, 让 MuJoCo 求解器
    # 应用 BAM 的 Stribeck + 负载 + 二次摩擦.
    bam_model = bam_models["m6"]()
    bam_model.set_actuator(bam_actuators["xl330"]())
    bam_model.actuator.kp = KP_FW
    bam_model.actuator.vin = VIN
    bam_model.load_parameters_from_dict(DEFAULT_XL330_M6)

    kt = bam_model.kt.value
    R = bam_model.R.value

    spec = mujoco.MjSpec.from_file(str(TESTBENCH_XML))
    _set_arm_mass(spec, TESTBENCH_ARM_MASS)

    # MujocoController 需要一个力矩控制的 motor; XML 中的 XL330 条目是
    # 位置执行器, 因此转换为 motor 并设置电压限幅的力范围. Armature 由
    # MujocoController.__init__ 设置到 dof 上.
    for act in spec.actuators:
        act.set_to_motor()
        act.forcelimited = False
        fl = VIN * kt / R
        act.forcerange = (-fl, fl)
        act.gear = [1.0, 0, 0, 0, 0, 0]
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
            joint.damping = 0.0
            joint.frictionloss = 0.0

    model = spec.compile()
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "1")
    dof_id = int(model.jnt_dofadr[joint_id])
    qpos_id = int(model.jnt_qposadr[joint_id])

    data.qpos[qpos_id] = 0.0
    data.qvel[dof_id] = 0.0
    mujoco.mj_forward(model, data)

    bam_ctrl = MujocoController(bam_model, ACTUATOR_NAME, model, data)
    bam_ctrl.reset(data.qpos)

    runner = PolicyRunner(onnx_path, action_scale=action_scale)
    policy_targets = make_target_schedule(total_time, seed=seed)
    decim = int(round(CONTROL_DT / SIM_DT))

    # 以 SIM_DT (200 Hz) 记录: 每个策略步 decim 个采样.
    N_log = len(policy_targets) * decim
    rec = {k: np.zeros(N_log, dtype=np.float32) for k in ("t", "target", "q", "qd", "action", "ctrl")}

    t = 0.0
    log_i = 0
    for _policy_i, target in enumerate(policy_targets):
        q = float(data.qpos[qpos_id])
        qd = float(data.qvel[dof_id])
        goal = runner.step(q, qd, float(target))
        action_raw = float(runner.last_action[0])

        for _ in range(decim):
            q = float(data.qpos[qpos_id])
            dq = float(data.qvel[dof_id])

            # ---- 以 200 Hz 记录 ----
            rec["t"][log_i] = t
            rec["target"][log_i] = target
            rec["q"][log_i] = q
            rec["qd"][log_i] = dq
            rec["action"][log_i] = action_raw
            rec["ctrl"][log_i] = goal
            log_i += 1

            # BAM 接管控制/力矩/摩擦: 设置目标后 update()
            # 把力矩写入 data.ctrl 并把摩擦/阻尼推到 dof 上,
            # 让 MuJoCo 求解器在下一步应用它们.
            bam_ctrl.set_q_target(ACTUATOR_NAME, goal)
            bam_ctrl.update()
            mujoco.mj_step(model, data)
            t += SIM_DT

    return rec


def rollout_sim_mjlab(onnx_path: str, total_time: float, seed: int, action_scale: float) -> dict:
    """通过实际的 mjlab 测试台 env 做 sim rollout (与策略训练时的 BAM M6 相同).

    启动 make_testbench_env_cfg() 并设 num_envs=1, 每个策略 tick 用我们确定性的调度覆盖
    target_angle 命令, 并用策略动作步进 env. 我们手动复制 ManagerBasedRlEnv.step 的内层
    decimation 循环, 以便在子步之间以 SIM_DT (200 Hz) 记录 q/qd, 与 bam 后端的记录频率一致.
    """
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from mjlab_microduck.tasks.testbench_env_cfg import make_testbench_env_cfg

    env_cfg = make_testbench_env_cfg(play=True)
    env_cfg.scene.num_envs = 1
    # 关闭自动重采样和自动 reset, 让我们的确定性调度和初始姿态贯穿整条 rollout.
    env_cfg.commands["target_angle"].resampling_time_range = (1e6, 1e6)
    env_cfg.episode_length_s = max(total_time + 10.0, env_cfg.episode_length_s)
    # 去掉观测噪声, 使 mjlab 路径成为公平的 sim2real 参考
    # (与不注入噪声的 bam 路径一致).
    env_cfg.observations["policy"].enable_corruption = False

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset(seed=seed)

    cmd_term = env.command_manager.get_term("target_angle")
    robot = env.scene["robot"]

    runner = PolicyRunner(onnx_path, action_scale=action_scale)
    policy_targets = make_target_schedule(total_time, seed=seed)
    decim = env.cfg.decimation
    physics_dt = env.physics_dt
    N_log = len(policy_targets) * decim
    rec = {k: np.zeros(N_log, dtype=np.float32) for k in ("t", "target", "q", "qd", "action", "ctrl")}

    t = 0.0
    log_i = 0
    for target in policy_targets:
        # 注入确定性目标并重算 obs, 让策略在这一 tick 看到它
        # (否则测试台 env 的 TargetAngleCommand 会随机采样).
        cmd_term._target[0, 0] = float(target)
        # update_history=True 至关重要: 测试台 env 的 joint_vel obs
        # 有 1-tick 延迟, 因此历史缓冲区必须每个策略 tick 前进,
        # 否则策略看到的是陈旧速度.
        obs_buf = env.observation_manager.compute(update_history=True)
        policy_obs = obs_buf["policy"][0].detach().cpu().numpy().astype(np.float32)
        ort_out = runner.session.run(None, {runner.in_name: policy_obs[None, :]})[0].reshape(-1)
        runner.last_action = ort_out.astype(np.float32)
        action_raw = float(ort_out[0])
        goal = DEFAULT_POS + action_raw * action_scale

        # 手动运行 ManagerBasedRlEnv.step 使用的 decimation 循环, 以便
        # 以物理频率 (200 Hz) 采样关节状态.
        action = torch.as_tensor(ort_out, device=device).reshape(1, -1)
        env.action_manager.process_action(action)
        for _ in range(decim):
            # 记录步前状态以对齐 bam 后端 (它在每个 mj_step 之前记录 q/qd).
            rec["t"][log_i] = t
            rec["target"][log_i] = float(target)
            rec["q"][log_i] = float(robot.data.joint_pos[0, 0].item())
            rec["qd"][log_i] = float(robot.data.joint_vel[0, 0].item())
            rec["action"][log_i] = action_raw
            rec["ctrl"][log_i] = goal
            log_i += 1

            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(dt=physics_dt)
            t += physics_dt

    env.close()
    return rec


# ---------------------------------------------------------------------------
# Real rollout (rustypot XL330)
# ---------------------------------------------------------------------------


def rollout_real(
    onnx_path: str,
    total_time: float,
    seed: int,
    port: str,
    motor_id: int,
    baudrate: int,
    kp: int,
    action_scale: float,
) -> dict:
    """在真实 XL330 电机上按目标调度 rollout ONNX 策略."""
    from rustypot import Xl330PyController

    ctrl = Xl330PyController(port, baudrate, 0.05)
    assert ctrl.ping(motor_id), f"motor id={motor_id} 在 {port} 上无响应"

    # 与 sim 中使用的固件增益对齐 (BAM kp_fw=200).
    ctrl.write_torque_enable(motor_id, False)
    ctrl.write_operating_mode(motor_id, 3)  # 位置控制
    ctrl.write_position_p_gain(motor_id, kp)
    ctrl.write_position_i_gain(motor_id, 0)
    ctrl.write_position_d_gain(motor_id, 0)
    # 回读以确认增益确实写入 (固件会静默截断超范围值, 验证可尽早发现不匹配).
    readback = ctrl.read_position_p_gain(motor_id)
    if isinstance(readback, (list, tuple)):
        readback = readback[0]
    print(f"  XL330 位置 P 增益: 请求={kp}, 回读={readback}")
    ctrl.write_goal_position(motor_id, 0.0)
    ctrl.write_torque_enable(motor_id, True)
    time.sleep(1.0)  # 让它稳定在零位

    runner = PolicyRunner(onnx_path, action_scale=action_scale)
    policy_targets = make_target_schedule(total_time, seed=seed)

    decim = int(round(CONTROL_DT / LOG_DT))  # 每个策略 tick 的采样数 (200 Hz / 50 Hz 时为 4)
    N_log = len(policy_targets) * decim
    rec = {k: np.zeros(N_log, dtype=np.float32) for k in ("t", "target", "q", "qd", "action", "ctrl")}

    def _scalar(x) -> float:
        if isinstance(x, (list, tuple)):
            return float(x[0])
        return float(x)

    t_start = time.perf_counter()
    prev_q = 0.0
    log_i = 0
    goal = 0.0
    action_raw = 0.0

    for policy_i, target in enumerate(policy_targets):
        tick_start = time.perf_counter()
        target_f = float(target)

        # 在 20 ms 窗口开始时读一次, 跑策略, 写目标.
        q = _scalar(ctrl.read_present_position(motor_id))
        try:
            qd = _scalar(ctrl.read_present_velocity(motor_id)) * DXL_VEL_TICK_TO_RAD_S
        except Exception:
            qd = (q - prev_q) / CONTROL_DT

        goal = runner.step(q, qd, target_f)
        action_raw = float(runner.last_action[0])
        # ctrl.write_goal_position(motor_id, float(np.clip(goal, -MAX_ANGLE, MAX_ANGLE)))
        ctrl.write_goal_position(motor_id, float(goal))

        # 第一个 200 Hz 采样用刚才读到的值 (没有额外 USB 往返).
        rec["t"][log_i] = time.perf_counter() - t_start
        rec["target"][log_i] = target_f
        rec["q"][log_i] = q
        rec["qd"][log_i] = qd
        rec["action"][log_i] = action_raw
        rec["ctrl"][log_i] = goal
        prev_q = q
        log_i += 1

        # 策略窗口内剩余 (decim-1) 个采样: 只读.
        for k in range(1, decim):
            sample_deadline = tick_start + (k + 1) * LOG_DT
            while time.perf_counter() < sample_deadline - 0.001:
                time.sleep(0.0005)
            q = _scalar(ctrl.read_present_position(motor_id))
            try:
                qd = _scalar(ctrl.read_present_velocity(motor_id)) * DXL_VEL_TICK_TO_RAD_S
            except Exception:
                qd = (q - prev_q) / LOG_DT
            prev_q = q

            rec["t"][log_i] = time.perf_counter() - t_start
            rec["target"][log_i] = target_f
            rec["q"][log_i] = q
            rec["qd"][log_i] = qd
            rec["action"][log_i] = action_raw
            rec["ctrl"][log_i] = goal
            log_i += 1

        # 每个新段加心跳, 实时打印状态.
        new_segment = policy_i == 0 or policy_targets[policy_i] != policy_targets[policy_i - 1]
        if new_segment or policy_i % 25 == 0:
            print(
                f"\r  t={rec['t'][log_i - 1]:6.2f}s  target={math.degrees(target_f):+6.1f}°  "
                f"q={math.degrees(q):+6.1f}°  err={math.degrees(q - target_f):+6.1f}°  "
                f"goal={math.degrees(goal):+6.1f}°",
                end="" if not new_segment else "\n",
                flush=True,
            )

        # 如果提前完成, 占满策略窗口剩余时间.
        dt_left = CONTROL_DT - (time.perf_counter() - tick_start)
        if dt_left > 0:
            time.sleep(dt_left)
    print()

    ctrl.write_torque_enable(motor_id, False)
    return rec


# ---------------------------------------------------------------------------
# 绘图 / 分析
# ---------------------------------------------------------------------------


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.mean(np.abs(a[:n] - b[:n])))


def npz_to_bam_log(npz_path: str, json_path: str, *, mass: float, length: float, kp: int, vin: float) -> None:
    """将 rollout .npz (由 rollout_sim/rollout_real 写出) 转换为 BAM log json.

    BAM log 格式 (见 ~/Rhoban/bam/bam/logs.py):
      顶层: mass, length, kp, vin, motor, trajectory, dt
      条目:   position, speed, load, input_volts, temp, goal_position, torque_enable, timestamp
    可用 `python -m bam.plot --logdir <dir> --actuator xl330` 加载.
    """
    import json

    d = dict(np.load(npz_path))
    t = d["t"]
    # 优先用实际记录时间戳算 dt 以应对小幅抖动;
    # 采样数少于 2 时退回到固定控制周期.
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else CONTROL_DT

    entries = []
    for i in range(len(t)):
        entries.append(
            {
                "position": float(d["q"][i]),
                "speed": float(d["qd"][i]),
                "load": 0.0,
                "input_volts": vin,
                "temp": 25.0,
                "goal_position": float(d["ctrl"][i]),
                "torque_enable": True,
                "timestamp": float(t[i]),
            }
        )

    log = {
        "mass": mass,
        "length": length,
        "kp": kp,
        "vin": vin,
        "motor": "xl330",
        "trajectory": "rl_policy",
        "dt": dt,
        "entries": entries,
    }

    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(log, f)
    print(f"已写入 BAM log: {out} ({len(entries)} 条, dt={dt:.4f}s, mass={mass}kg, kp={kp})")
    print(f"  回放命令: (cd ~/Rhoban/bam && python -m bam.plot --logdir {out.parent} --actuator xl330)")


def compare_and_plot(sim_file: str, real_file: str, out_path: str) -> None:
    """对比 sim 与 real 的 rollout 日志并保存诊断图."""
    import matplotlib.pyplot as plt

    sim = dict(np.load(sim_file))
    real = dict(np.load(real_file))
    n = min(len(sim["t"]), len(real["t"]))
    t = sim["t"][:n]

    err_sim = sim["q"][:n] - sim["target"][:n]
    err_real = real["q"][:n] - real["target"][:n]

    print("\n=== 分析 ===")
    print(f"  对比步数           : {n}")
    print(
        f"  MAE q (sim vs real): {_mae(sim['q'], real['q']):.4f} rad ({math.degrees(_mae(sim['q'], real['q'])):.2f}°)"
    )
    print(
        f"  sim   跟踪 MAE     : {float(np.mean(np.abs(err_sim))):.4f} rad "
        f"({math.degrees(float(np.mean(np.abs(err_sim)))):.2f}°)"
    )
    print(
        f"  real  跟踪 MAE     : {float(np.mean(np.abs(err_real))):.4f} rad "
        f"({math.degrees(float(np.mean(np.abs(err_real)))):.2f}°)"
    )
    print(f"  sim   qd RMS       : {float(np.sqrt(np.mean(sim['qd'][:n] ** 2))):.3f} rad/s")
    print(f"  real  qd RMS       : {float(np.sqrt(np.mean(real['qd'][:n] ** 2))):.3f} rad/s")
    print(f"  action MAE         : {_mae(sim['action'], real['action']):.4f} rad")

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(t, sim["target"][:n], "k-", lw=1, label="target", alpha=0.4)
    axes[0].plot(t, sim["q"][:n], "b-", lw=1.2, label="sim q")
    axes[0].plot(t, real["q"][:n], "r-", lw=1.2, label="real q")
    axes[0].plot(t, sim["ctrl"][:n], "b:", lw=0.8, alpha=0.6, label="sim goal (policy)")
    axes[0].plot(t, real["ctrl"][:n], "r:", lw=0.8, alpha=0.6, label="real goal (policy)")
    axes[0].set_ylabel("position [rad]")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"测试台 sim2real — MAE(sim, real) = {_mae(sim['q'], real['q']):.4f} rad")

    axes[1].plot(t, np.degrees(err_sim), "b-", lw=1, label="sim")
    axes[1].plot(t, np.degrees(err_real), "r-", lw=1, label="real")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("tracking error [deg]")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, sim["qd"][:n], "b-", lw=1, label="sim")
    axes[2].plot(t, real["qd"][:n], "r-", lw=1, label="real")
    axes[2].set_ylabel("velocity [rad/s]")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, sim["action"][:n], "b-", lw=1, label="sim action")
    axes[3].plot(t, real["action"][:n], "r-", lw=1, label="real action")
    axes[3].set_ylabel("policy action [rad]")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"\n已保存图表: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI 入口: 运行 sim/real 测试台 rollout 或对比日志."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "real"], help="Rollout 模式")
    ap.add_argument(
        "--sim-backend",
        choices=["bam", "mjlab"],
        default="bam",
        help="Sim 后端: 'bam' 在原生 mujoco 循环上用 bam.MujocoController "
        "(200 Hz 日志, 轻量); 'mjlab' 启动实际的 make_testbench_env_cfg() mjlab env "
        "及其 BamM6Actuator (50 Hz 日志).",
    )
    ap.add_argument("--onnx", type=str, help="训练好的 ONNX 策略路径")
    ap.add_argument("--out", type=str, help="输出 .npz 日志文件")
    ap.add_argument("--duration", type=float, default=30.0, help="总时长 [s]")
    ap.add_argument("--seed", type=int, default=0, help="目标调度种子")
    # 仅 real
    ap.add_argument("--port", type=str, default="/dev/ttyUSB0")
    ap.add_argument("--motor-id", type=int, default=1)
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--kp", type=int, default=200, help="XL330 位置 P 增益")
    ap.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="策略动作在偏移默认姿态前的乘数 (必须与训练 env 的 action scale 一致)",
    )
    # 对比模式
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("SIM_NPZ", "REAL_NPZ"),
        help="并排绘制两条已记录的 run",
    )
    ap.add_argument("--out-plot", type=str, default="testbench_sim2real.png")
    # BAM log 导出
    ap.add_argument(
        "--to-bam",
        nargs=2,
        metavar=("NPZ", "JSON"),
        help="将 rollout NPZ 转为 BAM log 格式 (运行: python -m bam.plot --logdir <dir> --actuator xl330)",
    )
    ap.add_argument("--bam-mass", type=float, default=TESTBENCH_ARM_MASS, help="负载质量 [kg]")
    ap.add_argument("--bam-length", type=float, default=0.1, help="臂长 [m]")
    ap.add_argument("--bam-vin", type=float, default=7.4, help="供电电压 [V]")

    args = ap.parse_args()

    if args.compare:
        compare_and_plot(args.compare[0], args.compare[1], args.out_plot)
        return

    if args.to_bam:
        npz_to_bam_log(
            args.to_bam[0],
            args.to_bam[1],
            mass=args.bam_mass,
            length=args.bam_length,
            kp=args.kp,
            vin=args.bam_vin,
        )
        return

    if not (args.mode and args.onnx and args.out):
        ap.error("rollout 需要 --mode, --onnx 和 --out")

    if args.mode == "sim":
        sim_fn = rollout_sim_mjlab if args.sim_backend == "mjlab" else rollout_sim_bam
        rec = sim_fn(args.onnx, args.duration, args.seed, args.action_scale)
    else:
        rec = rollout_real(
            args.onnx,
            args.duration,
            args.seed,
            args.port,
            args.motor_id,
            args.baudrate,
            args.kp,
            args.action_scale,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **rec)
    err = rec["q"] - rec["target"]
    print(f"\n已保存 {len(rec['t'])} 个采样到 {out}")
    print(f"  跟踪 MAE: {float(np.mean(np.abs(err))):.4f} rad ({math.degrees(float(np.mean(np.abs(err)))):.2f}°)")


if __name__ == "__main__":
    main()
