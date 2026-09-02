#!/usr/bin/env python3
"""在 MuJoCo 中运行 ONNX 策略推理并进行渲染的简单脚本."""

import argparse
import csv
import math
import os
import pickle
import queue
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene.xml"
# MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene_ramps.xml"
# MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene_floor_objects.xml"
# MICRODUCK_XML = "src/mjlab_microduck/robot/microduck/scene_robot_walk.xml"
MICRODUCK_ROLLERS_XML = "src/mjlab_microduck/robot/microduck/scene_rollers.xml"
MICRODUCK_BALL_XML = "src/mjlab_microduck/robot/microduck/scene_ball.xml"

# 躯干姿态命令常量 (必须与训练常量一致)
BODY_CMD_MAX_Z = 0.03  # ±30 mm
BODY_CMD_MAX_XY = 0.02  # ±20 mm
BODY_CMD_MAX_ANGLE = math.radians(30)  # ±30°

# 踢球行为的球初始放置 (必须与 microduck_ball_kick_env_cfg 的
# reset_ball_in_front_of_foot 参数一致: 球心在机器人 yaw 坐标系中).
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035

# 策略使用的默认姿态 (腿部弯曲, 站立位置)
# 这是参考姿态, 满足:
# - 动作是相对它的偏移 (motor_target = DEFAULT_POSE + action * scale)
# - 关节观测是相对它的 (obs_joint_pos = current_pos - DEFAULT_POSE)
# STAND2 姿态 (与 microduck_constants.py 中的 HOME_FRAME 一致): 躯干前移
# ~5mm, 让 CoM 位于踝关节轴上方. 腿部俯仰链相比旧姿态向前倾斜:
# hip_pitch 30°→26.24°, ankle 30°→25.95°, knee 0°→0.28°.
DEFAULT_POSE = np.array(
    [
        0.0,  # left_hip_yaw
        -0.0873,  # left_hip_roll
        -0.4579,  # left_hip_pitch
        -0.0049,  # left_knee
        0.4530,  # left_ankle
        0.3491,  # neck_pitch
        0.3491,  # head_pitch
        0.0,  # head_yaw
        0.0,  # head_roll
        0.0,  # right_hip_yaw
        0.0873,  # right_hip_roll
        0.4579,  # right_hip_pitch
        0.0049,  # right_knee
        -0.4530,  # right_ankle
    ],
    dtype=np.float32,
)


class TerminalInput:
    """从 stdin 读取单按键的读取器 (cbreak 模式, 后台线程).

    替代 MuJoCo viewer 的 key_callback: 在 viewer 窗口中按键也会触发
    viewer 内置的可视化快捷键 (坐标帧, 标签, 渲染开关...), 因此改为从
    终端读取命令. 方向键以 ESC [ A/B/C/D 转义序列到达, 会被翻译成符号
    名称 ("up"/"down"/"left"/"right"); 字母会被转为小写.
    cbreak (非 raw) 模式保留 ISIG, 因此 Ctrl+C 仍然可用.
    """

    _ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}

    def __init__(self):
        """初始化终端读取器并检测 stdin 是否为 TTY."""
        self._queue = queue.Queue()
        self.enabled = sys.stdin.isatty()
        self._fd = sys.stdin.fileno() if self.enabled else -1
        self._old_attrs = None
        self._stop = threading.Event()

    def __enter__(self):
        if not self.enabled:
            print("WARNING: stdin is not a TTY — keyboard control disabled")
            return self
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def _read1(self, timeout):
        """从 stdin 读取一个字节, 超时返回 None.

        os.read (无缓冲): 带缓冲的 sys.stdin.read 会吞掉 select 报告就绪
        之外的转义序列字节.
        """
        r, _, _ = select.select([self._fd], [], [], timeout)
        if not r:
            return None
        data = os.read(self._fd, 1)
        return data.decode(errors="ignore") if data else None

    def _reader(self):
        while not self._stop.is_set():
            ch = self._read1(0.1)
            if not ch:
                continue
            if ch == "\x1b":  # 可能是方向键转义序列
                if self._read1(0.05) == "[":
                    final = self._read1(0.05)
                    name = self._ARROWS.get(final) if final else None
                    if name:
                        self._queue.put(name)
                continue  # 裸 ESC / 未知序列: 忽略
            self._queue.put(ch.lower() if ch.isalpha() else ch)

    def get_keys(self):
        """取出并返回所有待处理按键 (符号名 / 字符)."""
        keys = []
        while True:
            try:
                keys.append(self._queue.get_nowait())
            except queue.Empty:
                return keys


class PolicyInference:
    """在 MuJoCo 模型中运行 ONNX 策略, 支持键盘驱动的命令切换."""

    def __init__(
        self,
        model,
        data,
        walking_onnx_path=None,
        action_scale=1.0,
        delay_min_lag=0,
        delay_max_lag=0,
        standing_onnx_path=None,
        switch_threshold=0.05,
        use_projected_gravity=False,
        ground_pick_onnx_path=None,
        ground_pick_period=4.0,
        sit_onnx_path=None,
        new_cmd_obs=False,
        slope_onnx_path=None,
        sitstand_onnx_path=None,
        kick_left_onnx_path=None,
        kick_right_onnx_path=None,
        roulade_onnx_path=None,
        kick_duration=3.0,
        roulade_duration=2.0,
    ):
        """用 MuJoCo 模型和 ONNX 策略初始化推理运行器."""
        self.model = model
        self.data = data
        self.action_scale = action_scale
        self.use_projected_gravity = use_projected_gravity
        self.delay_min_lag = delay_min_lag
        self.delay_max_lag = delay_max_lag
        self.switch_threshold = switch_threshold
        # 为 True 时: 输出统一的 13D 命令向量, 将 head_offset /
        # body_cmd 当作策略命令处理 (不加入 ctrl, 不修正 joint_pos).
        # 为 False 时: 旧行为 (3D 命令, head_offset 加到 ctrl[5:9]).
        self.new_cmd_obs = new_cmd_obs

        # 加载 walking 策略
        self.walking_session = None
        self.default_gait_period_from_onnx = None
        if walking_onnx_path:
            print(f"Loading walking policy from: {walking_onnx_path}")
            self.walking_session = ort.InferenceSession(walking_onnx_path)
            w_input_shape = self.walking_session.get_inputs()[0].shape
            w_output_shape = self.walking_session.get_outputs()[0].shape
            print(f"Walking policy input: {self.walking_session.get_inputs()[0].name}, shape: {w_input_shape}")
            print(f"Walking policy output: {self.walking_session.get_outputs()[0].name}, shape: {w_output_shape}")

            # 尝试从 ONNX metadata 读取步态周期
            try:
                model_metadata = self.walking_session.get_modelmeta()
                if (
                    hasattr(model_metadata, "custom_metadata_map")
                    and "gait_period" in model_metadata.custom_metadata_map
                ):
                    self.default_gait_period_from_onnx = float(model_metadata.custom_metadata_map["gait_period"])
                    print(f"Found gait period in ONNX metadata: {self.default_gait_period_from_onnx:.4f}s")
            except Exception as e:
                print(f"Could not read gait period from ONNX metadata: {e}")

        # 加载 standing 策略
        self.standing_session = None
        if standing_onnx_path:
            print(f"\nLoading standing policy from: {standing_onnx_path}")
            self.standing_session = ort.InferenceSession(standing_onnx_path)
            s_input_shape = self.standing_session.get_inputs()[0].shape
            s_output_shape = self.standing_session.get_outputs()[0].shape
            print(f"Standing policy input: {self.standing_session.get_inputs()[0].name}, shape: {s_input_shape}")
            print(f"Standing policy output: {self.standing_session.get_outputs()[0].name}, shape: {s_output_shape}")
            if self.walking_session:
                print(f"Policy switching threshold: {switch_threshold} (vel command magnitude)")

        # 加载 ground pick 策略
        self.ground_pick_session = None
        self.ground_pick_mode = False
        self.ground_pick_phase = 0.0
        self.ground_pick_period = ground_pick_period
        if ground_pick_onnx_path:
            print(f"\nLoading ground pick policy from: {ground_pick_onnx_path}")
            self.ground_pick_session = ort.InferenceSession(ground_pick_onnx_path)
            gp_input_shape = self.ground_pick_session.get_inputs()[0].shape
            print(f"Ground pick policy input shape: {gp_input_shape}")

        # 加载 sit 策略. 两种变体共享 Y 键和 self.sit_session:
        #  - --sit (is_sitstand=False): 旧的单向 sit 策略. 在零 twist 命令下
        #    无条件坐下; 起身靠切回 standing/walking session 完成.
        #  - --sitstand (is_sitstand=True): 受控的 sit↔stand 策略.
        #    twist[0] 是姿态标志 (0=stand, 1=sit); 同一个策略完成坐下,
        #    保持, 起身 — Y 只是翻转标志.
        self.sit_session = None
        self.sit_mode = False
        self.is_sitstand = False
        if sit_onnx_path and sitstand_onnx_path:
            raise ValueError("Provide only one of --sit / --sitstand")
        if sit_onnx_path:
            print(f"\nLoading sit policy from: {sit_onnx_path}")
            self.sit_session = ort.InferenceSession(sit_onnx_path)
            sit_input_shape = self.sit_session.get_inputs()[0].shape
            print(f"Sit policy input shape: {sit_input_shape}")
        elif sitstand_onnx_path:
            if not self.new_cmd_obs:
                raise ValueError("--sitstand policies use the unified 13D command obs (61D); run with --new-cmd-obs")
            print(f"\nLoading sitstand policy from: {sitstand_onnx_path}")
            self.sit_session = ort.InferenceSession(sitstand_onnx_path)
            self.is_sitstand = True
            ss_input_shape = self.sit_session.get_inputs()[0].shape
            print(f"Sitstand policy input shape: {ss_input_shape}")

        # 加载 slope 策略 (被动下滑, 在零 twist 命令下运行)
        self.slope_session = None
        self.slope_mode = False
        if slope_onnx_path:
            print(f"\nLoading slope policy from: {slope_onnx_path}")
            self.slope_session = ort.InferenceSession(slope_onnx_path)
            sl_input_shape = self.slope_session.get_inputs()[0].shape
            print(f"Slope policy input shape: {sl_input_shape}")

        # 情节式行为策略 (左/右踢球, roulade). 三者都使用统一的 61D obs
        # 布局, 配以全零 13D 命令 (训练时 twist 强制为 ~0, head/body 槽
        # 零填充), 因此触发其中一个就是简单的 session 切换; `duration` 秒
        # 后控制权交回 walking/standing (行为策略自身会以站立姿态结束).
        self.behavior_sessions = {}
        self.behavior_durations = {}
        self.behavior_mode = None  # 当前运行的行为名称, 或 None
        self.behavior_time_left = 0.0
        for name, path, duration in (
            ("kick_left", kick_left_onnx_path, kick_duration),
            ("kick_right", kick_right_onnx_path, kick_duration),
            ("roulade", roulade_onnx_path, roulade_duration),
        ):
            if not path:
                continue
            if not self.new_cmd_obs:
                raise ValueError(
                    f"--{name.replace('_', '-')} policies use the unified 13D command obs (61D); run with --new-cmd-obs"
                )
            print(f"\nLoading {name} policy from: {path}")
            self.behavior_sessions[name] = ort.InferenceSession(path)
            self.behavior_durations[name] = duration
            print(
                f"{name} policy input shape: {self.behavior_sessions[name].get_inputs()[0].shape}"
                f"  (auto-return after {duration:.1f}s)"
            )

        # 校验至少加载了一个策略. sitstand 策略可以单独运行
        # (在 flag=0 时保持站立), 不像旧的单向 sit 策略.
        if not self.walking_session and not self.standing_session and not self.is_sitstand:
            raise ValueError("At least one of --walking, --standing or --sitstand must be provided")

        # 确定初始活动 session 和策略
        if self.walking_session:
            self.current_policy = "walking"
            self.ort_session = self.walking_session
        elif self.standing_session:
            self.current_policy = "standing"
            self.ort_session = self.standing_session
        else:
            # 仅 sitstand: 以站立姿态开始 (姿态标志 0).
            self.current_policy = "sit"
            self.ort_session = self.sit_session

        # 从活动 session 获取输入/输出名
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        # 获取传感器 ID 和 body ID
        self.imu_ang_vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        self.trunk_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

        # Trunk freejoint 的 qpos 地址 (用于在机器人 yaw 坐标系中放置球)
        # 以及可选的 ball freejoint (存在于 scene_ball.xml 中).
        _trunk_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self._trunk_qpos_adr = int(model.jnt_qposadr[_trunk_jid])
        _ball_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        if _ball_jid >= 0:
            self.ball_qpos_adr = int(model.jnt_qposadr[_ball_jid])
            self.ball_qvel_adr = int(model.jnt_dofadr[_ball_jid])
        else:
            self.ball_qpos_adr = None
            self.ball_qvel_adr = None

        print("Sensors found:")
        print(f"  imu_ang_vel: id={self.imu_ang_vel_id}")
        print("Body IDs:")
        print(f"  trunk_base: id={self.trunk_base_id}")

        # 关节信息
        self.n_joints = model.nu

        # 对于带被动/交错关节的机器人 (例如轮滑鞋), 执行关节在 qpos/qvel 中
        # 不连续. 从执行器 transmission 的 joint ID 计算正确的索引, 以便
        # 任意关节顺序都能正确提取.
        self.joint_qpos_indices = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]
        self.joint_qvel_indices = [int(model.jnt_dofadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]

        # 策略的默认姿态 (腿部弯曲)
        self.default_pose = DEFAULT_POSE[: self.n_joints]
        print(f"Number of actuators: {self.n_joints}")
        print(f"Default pose: {self.default_pose}")
        print(f"Action scale: {self.action_scale}")

        # 上一个动作 (用于观测历史)
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)

        # 速度命令 [lin_vel_x, lin_vel_y, ang_vel_z] — 控制 walking / 策略切换
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        # 按键步长和限制 (在 main() 中按模式覆盖)
        self.vel_step_x = 0.05
        self.vel_step_y = 0.05
        self.vel_step_ang = 0.3
        self.vel_max_x = 0.3
        self.vel_min_x = -0.3
        self.vel_max_y = 0.3
        self.vel_min_y = -0.3
        self.vel_max_ang = 1.5
        # 躯干姿态命令. 在 new_cmd_obs 模式下是 6D
        #   [x, y, z, roll, pitch, yaw] (m, m, m, rad, rad, rad)
        # 在 legacy 模式下只有 [z, pitch, roll] (前 3 个索引复用为
        # [z, pitch, roll], 以保持 legacy 归一化路径可用).
        self.body_cmd = np.zeros(6 if self.new_cmd_obs else 3, dtype=np.float32)
        # obs 命令向量 (legacy 模式下 3D, new_cmd_obs=True 时 13D).
        self.command = np.zeros(13 if self.new_cmd_obs else 3, dtype=np.float32)

        # 躯干姿态模式 (类似 head 模式, 但用于站立时的躯干姿态控制)
        self.body_pose_mode = False
        self.body_cmd_step_xy = 0.005  # 每次按键 5 mm (4 次到最大)
        self.body_cmd_step_z = 0.01  # 每次按键 10 mm (3 次到最大)
        self.body_cmd_step_angle = math.radians(10)  # 每次按键 10° (3 次到最大)

        # 头部控制模式. legacy 模式下 head_offset 加到 ctrl[5:9] 上;
        # new_cmd_obs 模式下它是输入给策略的 *命令*.
        # 训练时每个关节的最终限位: neck/head_pitch ±1.1, head_yaw ±1.4,
        # head_roll ±0.31. 滑块最大值 = 最宽的关节限位; head_roll 自然
        # 会被策略裁剪, 因为它训练时从未超过 0.31.
        self.head_mode = False
        self.head_offset = np.zeros(4, dtype=np.float32)
        if self.new_cmd_obs:
            self.head_max = 1.4
            self.head_step = 0.1
        else:
            self.head_max = 2.5
            self.head_step = 0.83

        # 动作延迟缓冲
        self.use_delay = self.delay_max_lag > 0
        if self.use_delay:
            buffer_size = self.delay_max_lag + 1
            self.action_buffer = [np.zeros(self.n_joints, dtype=np.float32) for _ in range(buffer_size)]
            self.buffer_index = 0
            self.current_lag = np.random.randint(self.delay_min_lag, self.delay_max_lag + 1)
            print("\nActuator delay enabled:")
            print(f"  Min lag: {self.delay_min_lag} timesteps")
            print(f"  Max lag: {self.delay_max_lag} timesteps")
            print(f"  Sampled lag: {self.current_lag} timesteps")
            print(f"  Buffer size: {buffer_size}")
        else:
            self.action_buffer = None
            self.current_lag = 0

    def _update_command(self):
        """根据当前策略和命令更新 self.command (送入 obs).

        Legacy 模式 (new_cmd_obs=False): self.command 是 3D.
        新模式 (new_cmd_obs=True): self.command 是 13D:
            [vx, vy, vtheta,                                  ← twist
             neck_pitch, head_pitch, head_yaw, head_roll,     ← head_pose deltas
             body_x, body_y, body_z, body_roll, body_pitch, body_yaw]  ← body_pose
        我们保留现有的键盘映射: head_offset (4D) 驱动 head 槽;
        body_cmd[0..2] 当前表示 (Δz, Δpitch, Δroll), 路由到 body_pose 槽
        [z, pitch, roll]; x/y/yaw 保持为 0 (键盘暂未暴露).
        ground_pick 仍然占用 [0..2] 槽用于相位编码.
        """
        if self.new_cmd_obs:
            if self.behavior_mode is not None:
                # Kick/roulade 训练时使用全零 13D 命令
                # (twist ~0, head/body 槽零填充) — 喂入陈旧的
                # head/body 命令会超出分布.
                self.command = np.zeros(13, dtype=np.float32)
                return
            cmd = np.zeros(13, dtype=np.float32)
            # twist 槽 (或 ground_pick 的相位编码 — 在那里被覆盖)
            if self.current_policy == "walking":
                cmd[0:3] = self.vel_cmd
            elif self.current_policy == "sit" and self.is_sitstand:
                # Sitstand 姿态标志: 1 = sit, 0 = stand. 不是零 —
                # 全零 twist 是该策略的 STAND 命令, 这就是为什么喂入
                # 旧 sit 策略的零命令毫无效果的原因.
                cmd[0] = 1.0 if self.sit_mode else 0.0
            # 其余 standing/old-sit/ground_pick: twist 保持 0 (ground_pick
            # 稍后写入相位编码)
            cmd[3:7] = self.head_offset
            cmd[7:13] = self.body_cmd  # [x, y, z, roll, pitch, yaw]
            self.command = cmd
            return

        # Legacy 3D 命令
        if self.current_policy == "walking":
            self.command = self.vel_cmd.copy()
        elif self.current_policy == "sit":
            # Sit 训练时使用近零 twist 命令.
            self.command = np.zeros(3, dtype=np.float32)
        elif self.current_policy == "standing":
            # 归一化躯干姿态命令以匹配训练的 body_pose_cmd_obs
            self.command = np.array(
                [
                    self.body_cmd[0] / BODY_CMD_MAX_Z,
                    self.body_cmd[1] / BODY_CMD_MAX_ANGLE,
                    self.body_cmd[2] / BODY_CMD_MAX_ANGLE,
                ],
                dtype=np.float32,
            )
        elif self.current_policy == "slope":
            # 被动下滑: 零命令 (类似 standing coast)
            self.command = np.zeros(3, dtype=np.float32)
        # ground_pick: 命令由 update_ground_pick_phase 直接设置

    def _update_policy_session(self):
        """根据 vel_cmd 幅度在 walking 和 standing session 之间切换."""
        if not (self.walking_session and self.standing_session):
            return  # 只加载了一个策略, 不切换
        if self.ground_pick_mode:
            return  # ground pick 期间不切换
        if self.sit_mode:
            return  # 坐下时不切换
        if self.slope_mode:
            return  # slope 模式期间不切换
        if self.behavior_mode is not None:
            return  # kick/roulade 期间不切换

        magnitude = float(np.linalg.norm(self.vel_cmd))
        new_policy = "standing" if magnitude <= self.switch_threshold else "walking"
        if new_policy != self.current_policy:
            self.current_policy = new_policy
            self.ort_session = self.standing_session if new_policy == "standing" else self.walking_session
            print(f"Switched to {self.current_policy} policy (vel magnitude: {magnitude:.3f})")
            self._update_command()

    def set_vel_cmd(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
        """设置速度命令 (用于 walking / 策略切换)."""
        self.vel_cmd = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)
        self._update_policy_session()
        self._update_command()
        print(f"Vel cmd: [{lin_vel_x:.2f}, {lin_vel_y:.2f}, {ang_vel_z:.2f}] [{self.current_policy}]")

    def toggle_body_pose_mode(self):
        """开关躯干姿态控制模式."""
        self.body_pose_mode = not self.body_pose_mode
        if self.body_pose_mode:
            print("Body pose mode: ON")
            print(f"  UP/DOWN: Δz ±{self.body_cmd_step_z * 1000:.0f}mm  (max ±{BODY_CMD_MAX_Z * 1000:.0f}mm)")
            print(
                f"  LEFT/RIGHT: Δpitch ±{math.degrees(self.body_cmd_step_angle):.0f}°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)"
            )
            print(
                f"  A/E: Δroll ±{math.degrees(self.body_cmd_step_angle):.0f}°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)"
            )
            if self.new_cmd_obs:
                print(
                    f"  Z/S: Δyaw ±{math.degrees(self.body_cmd_step_angle):.0f}°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)"
                )
            print("  SPACE: reset body pose to zero")
            self._print_body_cmd()
        else:
            print("Body pose mode: OFF")

    def toggle_slope_mode(self):
        """开关 slope 策略模式 (被动下滑, 零 twist 命令)."""
        if self.slope_session is None:
            print("Slope unavailable: no --slope policy loaded")
            return
        if self.behavior_mode is not None:
            print(f"Cannot toggle slope mode during {self.behavior_mode}")
            return
        self.slope_mode = not self.slope_mode
        if self.slope_mode:
            self.ort_session = self.slope_session
            self.current_policy = "slope"
            self.set_vel_cmd(0.0, 0.0, 0.0)  # 被动下滑: 零命令
            print("Slope mode: ON (passive descent)")
        else:
            self.vel_cmd = np.zeros(3, dtype=np.float32)
            if self.walking_session:
                self.current_policy = "walking"
                self.ort_session = self.walking_session
            else:
                self.current_policy = "standing"
                self.ort_session = self.standing_session
            self._update_command()
            print("Slope mode: OFF")

    def _print_body_cmd(self):
        if self.new_cmd_obs:
            x, y, z, roll, pitch, yaw = self.body_cmd
            print(
                f"Body cmd: x={x * 1000:5.1f}mm  y={y * 1000:5.1f}mm  z={z * 1000:5.1f}mm  "
                f"roll={math.degrees(roll):5.1f}°  pitch={math.degrees(pitch):5.1f}°  "
                f"yaw={math.degrees(yaw):5.1f}°"
            )
        else:
            print(
                f"Body cmd: z={self.body_cmd[0] * 1000:.1f}mm  "
                f"pitch={math.degrees(self.body_cmd[1]):.1f}°  "
                f"roll={math.degrees(self.body_cmd[2]):.1f}°"
            )

    # --- 躯干命令增量器 (legacy 3D 和新 6D 的索引不同) ---
    def _body_idx(self, axis: str) -> int:
        """根据当前模式将轴名映射到 body_cmd 索引."""
        if self.new_cmd_obs:
            return {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}[axis]
        return {"z": 0, "pitch": 1, "roll": 2}[axis]

    def bump_body(self, axis: str, delta: float):
        """按 delta 调整一个躯干命令轴, 限制在允许范围内."""
        idx = self._body_idx(axis)
        cap = BODY_CMD_MAX_Z if axis == "z" else BODY_CMD_MAX_XY if axis in ("x", "y") else BODY_CMD_MAX_ANGLE
        self.body_cmd[idx] = float(np.clip(self.body_cmd[idx] + delta, -cap, cap))
        self._update_command()
        self._print_body_cmd()

    def quat_rotate_inverse(self, quat, vec):
        """用四元数 [w, x, y, z] 的逆旋转向量."""
        w = quat[0]
        xyz = quat[1:4]
        t = np.cross(xyz, vec) * 2
        return vec - w * t + np.cross(xyz, t)

    def get_raw_accelerometer(self):
        """从 MuJoCo 传感器读取原始加速度计读数."""
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        if sensor_id < 0:
            raise ValueError("Sensor 'imu_accel' not found in model")

        sensor_adr = self.model.sensor_adr[sensor_id]
        accel_raw = self.data.sensordata[sensor_adr : sensor_adr + 3].copy().astype(np.float32)
        accel_negated = -accel_raw
        mag = np.linalg.norm(accel_negated)
        if mag > 0.1:
            return accel_negated / mag
        else:
            quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
            world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            return self.quat_rotate_inverse(quat, world_gravity)

    def get_projected_gravity(self):
        """获取 body 坐标系下的投影重力."""
        quat = self.data.xquat[self.trunk_base_id].copy().astype(np.float32)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return self.quat_rotate_inverse(quat, world_gravity)

    def get_base_ang_vel(self):
        """从 IMU 陀螺仪传感器获取基座角速度."""
        sensor_adr = self.model.sensor_adr[self.imu_ang_vel_id]
        return self.data.sensordata[sensor_adr : sensor_adr + 3].copy().astype(np.float32)

    def get_joint_pos_relative(self):
        """获取相对于默认姿态的关节位置."""
        current_pos = self.data.qpos[self.joint_qpos_indices].copy().astype(np.float32)
        return current_pos - self.default_pose

    def get_joint_vel(self):
        """获取关节速度."""
        return self.data.qvel[self.joint_qvel_indices].copy().astype(np.float32)

    def get_observations(self):
        """收集与策略输入匹配的观测.

        velocity/standing 任务的顺序:
        1. base_ang_vel (3D)
        2. raw_accelerometer 或 projected_gravity (3D)
        3. joint_pos (14D) - 相对默认姿态
        4. joint_vel (14D)
        5. actions (14D) - 上一个动作
        6. command (3D) - vel cmd (walking) 或归一化的 body pose cmd (standing)
        总计: 51D
        """
        obs = []

        obs.append(self.get_base_ang_vel())

        if self.use_projected_gravity:
            obs.append(self.get_projected_gravity())
        else:
            obs.append(self.get_raw_accelerometer())

        obs.append(self.get_joint_pos_relative())
        obs.append(self.get_joint_vel())
        obs.append(self.last_action)
        obs.append(self.command)

        return np.concatenate(obs).astype(np.float32)

    def trigger_ground_pick(self):
        """启动一个 ground pick 周期.

        完成后自动返回 walking.
        """
        if self.ground_pick_session is None:
            print("Ground pick unavailable: no --ground-pick policy loaded")
            return
        if self.ground_pick_mode:
            print("Ground pick already in progress")
            return
        if self.sit_mode:
            print("Cannot ground pick while sitting (press Y to stand up first)")
            return
        if self.behavior_mode is not None:
            print(f"Cannot ground pick during {self.behavior_mode}")
            return
        self.ground_pick_mode = True
        self.ground_pick_phase = 0.0
        self.ort_session = self.ground_pick_session
        self.current_policy = "ground_pick"
        print(f"Ground pick: started (period={self.ground_pick_period:.1f}s)")

    def _end_ground_pick(self):
        """在一个 ground pick 周期完成后切回."""
        self.ground_pick_mode = False
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        if self.walking_session:
            self.current_policy = "walking"
            self.ort_session = self.walking_session
        else:
            self.current_policy = "standing"
            self.ort_session = self.standing_session
        self._update_command()
        print(f"Ground pick: done → back to {self.current_policy}")

    def update_ground_pick_phase(self, dt: float):
        """推进 ground pick 相位; 完成一个完整周期后自动退出."""
        if not self.ground_pick_mode:
            return
        new_phase = self.ground_pick_phase + dt / self.ground_pick_period
        if new_phase >= 0.7:
            self._end_ground_pick()
            return
        self.ground_pick_phase = new_phase
        # ground_pick 策略使用前 3 个槽 (twist) 作为相位编码.
        # 更高槽位 (head/body) 保持在 _update_command 设置的值.
        self.command[0] = np.cos(2 * np.pi * self.ground_pick_phase)
        self.command[1] = np.sin(2 * np.pi * self.ground_pick_phase)
        self.command[2] = 0.0

    def trigger_behavior(self, name):
        """启动一个情节式行为 (kick_left / kick_right / roulade).

        行为策略训练时从站立开始, 使用全零命令, 并以站立结束, 因此
        触发就是一次 session 切换; 之后用计时器把控制权交回 walking/standing.
        """
        session = self.behavior_sessions.get(name)
        if session is None:
            print(f"{name} unavailable: no --{name.replace('_', '-')} policy loaded")
            return
        if self.behavior_mode is not None:
            print(f"Cannot start {name}: {self.behavior_mode} already in progress")
            return
        if self.ground_pick_mode:
            print(f"Cannot start {name} during ground pick")
            return
        if self.sit_mode:
            print(f"Cannot start {name} while sitting (press Y to stand up first)")
            return
        if self.slope_mode:
            print(f"Cannot start {name} during slope mode")
            return
        if name in ("kick_left", "kick_right"):
            self._place_ball(name)
        self.behavior_mode = name
        self.behavior_time_left = self.behavior_durations[name]
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.current_policy = name
        self.ort_session = session
        self._update_command()
        print(f"{name}: started (auto-return in {self.behavior_time_left:.1f}s)")

    def _place_ball(self, behavior):
        """将球瞬移到踢球脚前方, 匹配训练的 reset_ball_in_front_of_foot (偏移在.

        机器人 yaw 坐标系中).
        """
        if self.ball_qpos_adr is None or self.ball_qvel_adr is None:
            print("No ball in scene (kick will swing at air)")
            return
        adr = self._trunk_qpos_adr
        x, y = float(self.data.qpos[adr]), float(self.data.qpos[adr + 1])
        qw, qx, qy, qz = self.data.qpos[adr + 3 : adr + 7]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        off_y = -BALL_OFFSET_ABS_Y if behavior == "kick_right" else BALL_OFFSET_ABS_Y
        bx = x + math.cos(yaw) * BALL_OFFSET_X - math.sin(yaw) * off_y
        by = y + math.sin(yaw) * BALL_OFFSET_X + math.cos(yaw) * off_y
        self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 7] = [
            bx,
            by,
            BALL_RADIUS,
            1,
            0,
            0,
            0,
        ]
        self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 6] = 0.0
        foot = behavior.split("_")[1]
        print(f"Ball placed at ({bx:.3f}, {by:.3f}) in front of the {foot} foot")

    def update_behavior(self, dt: float):
        """推进行为计时器; 完成后交回 walking/standing."""
        if self.behavior_mode is None:
            return
        self.behavior_time_left -= dt
        if self.behavior_time_left <= 0.0:
            self._end_behavior()

    def _end_behavior(self):
        name = self.behavior_mode
        self.behavior_mode = None
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        if self.walking_session:
            self.current_policy = "walking"
            self.ort_session = self.walking_session
        elif self.standing_session:
            self.current_policy = "standing"
            self.ort_session = self.standing_session
        else:
            # 仅 sitstand 配置: sitstand 策略保持站立 (flag 0).
            self.current_policy = "sit"
            self.ort_session = self.sit_session
        self._update_command()
        print(f"{name}: done → back to {self.current_policy}")

    def toggle_sit(self):
        """开关坐下 (Y 键).

        旧单向 sit 策略 (--sit): Y 关闭时切回 standing/walking session,
        由它完成起身. Sitstand 策略 (--sitstand): Y 只是翻转姿态标志 —
        同一个策略完成坐下, 保持坐姿, 并平稳起身 (对标志翻转的训练响应
        是 ~2 s 的平滑过渡). 起身后 session 保持活动 (它保持站立);
        速度命令会像往常一样切回 walking/standing.
        """
        if self.sit_session is None:
            print("Sit unavailable: no --sit/--sitstand policy loaded")
            return
        if self.ground_pick_mode:
            print("Cannot sit during ground pick")
            return
        if self.behavior_mode is not None:
            print(f"Cannot sit during {self.behavior_mode}")
            return
        self.sit_mode = not self.sit_mode
        if self.sit_mode:
            self.vel_cmd = np.zeros(3, dtype=np.float32)
            self.current_policy = "sit"
            self.ort_session = self.sit_session
            print("Sit: ON" + (" (sitstand flag=1; Y again to stand up)" if self.is_sitstand else ""))
        elif self.is_sitstand:
            # 留在 sitstand session — 它自己起身 (flag → 0).
            # 不要在这里切到 standing 策略: 它会接管
            # 它未训练过的坐姿到起身中间状态.
            print("Sit: OFF → sitstand policy standing up (flag=0)")
        else:
            if self.standing_session:
                self.current_policy = "standing"
            else:
                self.current_policy = "walking"
            self.ort_session = self.standing_session if self.current_policy == "standing" else self.walking_session
            print(f"Sit: OFF → back to {self.current_policy}")
        self._update_command()

    def toggle_head_mode(self):
        """开关头部控制模式."""
        self.head_mode = not self.head_mode
        if self.head_mode:
            print("Head mode: ON")
            print(
                f"  Z/S: neck_pitch  |  UP/DOWN: head_pitch  |  LEFT/RIGHT: head_yaw  |  A/E: head_roll  |  SPACE: reset  (max ±{self.head_max:.2f} rad)"
            )
        else:
            print("Head mode: OFF")

    def infer(self):
        """运行策略推理并返回动作."""
        obs = self.get_observations()
        obs_batch = obs.reshape(1, -1)
        action = self.ort_session.run([self.output_name], {self.input_name: obs_batch})[0]
        action = action.squeeze(0).astype(np.float32)
        self.last_action = action.copy()
        return action

    def apply_action(self, action):
        """将动作应用到 MuJoCo 控制上, 可选延迟."""
        if self.use_delay:
            self.action_buffer[self.buffer_index] = action.copy()
            delayed_index = (self.buffer_index - self.current_lag) % len(self.action_buffer)
            delayed_action = self.action_buffer[delayed_index]
            self.buffer_index = (self.buffer_index + 1) % len(self.action_buffer)
            target_positions = self.default_pose + delayed_action * self.action_scale
        else:
            target_positions = self.default_pose + action * self.action_scale

        self.data.ctrl[:] = target_positions
        # Legacy 模式: head_offset 是叠加在策略输出之上的外部扰动.
        # 新模式: head_offset 是输入到策略 obs 的命令, 因此策略自身
        # 产生带偏移的头部姿态.
        if not self.new_cmd_obs:
            self.data.ctrl[5:9] += self.head_offset


def main():
    """CLI 入口: 在 MuJoCo 仿真中运行 ONNX 策略."""
    parser = argparse.ArgumentParser(description="Run ONNX policy in MuJoCo")
    parser.add_argument(
        "--roller",
        action="store_true",
        help="使用 roller skate 机器人 XML (robot_walk_rollers.xml)",
    )
    parser.add_argument("--walking", type=str, default=None, help="walking 策略 ONNX 文件路径")
    parser.add_argument(
        "--standing",
        "-s",
        type=str,
        default=None,
        help="standing 策略 ONNX 文件路径",
    )
    parser.add_argument(
        "--ground-pick",
        type=str,
        default=None,
        help="ground pick 策略 ONNX 文件路径 (按 G 激活)",
    )
    parser.add_argument(
        "--sit",
        type=str,
        default=None,
        help="旧的单向 sitting 策略 ONNX 文件路径 (按 Y 坐下, 再按 Y 切回 standing/walking 策略)",
    )
    parser.add_argument(
        "--sitstand",
        type=str,
        default=None,
        help="sitstand 策略 ONNX 路径 (受控 sit<->stand; 按 Y 坐下, 再按 Y 同一策略起身). 需要 --new-cmd-obs. 可单独运行.",
    )
    parser.add_argument(
        "--slope",
        type=str,
        default=None,
        help="slope 策略 ONNX 文件路径 (按 Y 切换)",
    )
    parser.add_argument(
        "--kick-left",
        type=str,
        default=None,
        help="左脚踢球策略 ONNX 路径 (按 K 触发). 需要 --new-cmd-obs. 加载带球的场景.",
    )
    parser.add_argument(
        "--kick-right",
        type=str,
        default=None,
        help="右脚踢球策略 ONNX 路径 (按 L 触发). 需要 --new-cmd-obs. 加载带球的场景.",
    )
    parser.add_argument(
        "--roulade",
        type=str,
        default=None,
        help="roulade (前滚翻) 策略 ONNX 路径 (按 R 触发). 需要 --new-cmd-obs.",
    )
    parser.add_argument(
        "--kick-duration",
        type=float,
        default=3.0,
        help="踢球策略保持活动的秒数, 之后交回 standing/walking (默认: 3.0)",
    )
    parser.add_argument(
        "--roulade-duration",
        type=float,
        default=2.0,
        help="roulade 策略保持活动的秒数, 之后交回 standing/walking (默认: 2.0, 约为翻滚本身; 站立/行走策略接管稳定阶段)",
    )
    parser.add_argument(
        "--lin-vel-x",
        type=float,
        default=0.0,
        help="初始线速度 X 命令 (m/s)",
    )
    parser.add_argument(
        "--lin-vel-y",
        type=float,
        default=0.0,
        help="初始线速度 Y 命令 (m/s)",
    )
    parser.add_argument(
        "--ang-vel-z",
        type=float,
        default=0.0,
        help="初始角速度 Z 命令 (rad/s)",
    )
    parser.add_argument("--action-scale", type=float, default=1.0, help="动作缩放 (默认: 1.0)")
    parser.add_argument(
        "--raw-accelerometer",
        action="store_true",
        help="使用原始加速度计而非投影重力",
    )
    parser.add_argument(
        "--delay",
        type=int,
        nargs="*",
        default=None,
        help="启用执行器延迟: --delay MIN MAX 或 --delay LAG",
    )
    parser.add_argument("--debug", action="store_true", help="打印观测和动作")
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="将观测和动作保存到 CSV 文件",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="启用录制模式: 在 Ctrl+C 时将观测保存到 pickle 文件",
    )
    parser.add_argument(
        "--switch-threshold",
        type=float,
        default=0.05,
        help="walking/standing 切换的速度命令幅度阈值 (默认: 0.05)",
    )
    parser.add_argument(
        "--ground-pick-period",
        type=float,
        default=4.0,
        help="ground pick 相位周期, 单位秒 (默认: 4.0)",
    )
    parser.add_argument(
        "--new-cmd-obs",
        action="store_true",
        help="使用统一的 13D 命令 obs 布局 (twist+head_pose+body_pose). "
        "使用新 pose-command-tracking 设置训练的策略需要此选项. "
        "旧策略 (51D obs, head_offset 加到 ctrl) 需要关闭此标志.",
    )
    parser.add_argument(
        "--current-limit",
        type=float,
        default=1.75,
        help="XL330 固件电流限制 [A]. 执行器力矩被裁剪到 "
        "+/- current_limit * kt (kt 来自 bam 包), 匹配训练中建模的"
        "电流饱和. <=0 禁用.",
    )
    parser.add_argument(
        "--foot-friction",
        type=float,
        default=None,
        help="覆盖脚部滑动摩擦 (mu), 以模拟真实的高抓地力 "
        "PU 鞋底. 训练使用 mu~1.0 (范围 0.7-1.3); 真实 PU 大约 "
        "~1.5-2.5. 例如 --foot-friction 2.0",
    )
    parser.add_argument(
        "--foot-solref",
        type=float,
        default=None,
        help="软化脚部接触: 脚部 geom 的 solref 时间常数 (s) "
        "(默认仿真 ~0.02 = 硬/刚性). 越大越软, 用于模拟 "
        "柔顺的 PU 鞋底. 例如 --foot-solref 0.04",
    )
    args = parser.parse_args()

    if not args.walking and not args.standing and not args.sitstand:
        parser.error("At least one of --walking, --standing or --sitstand must be provided")
    if args.sitstand and not args.new_cmd_obs:
        parser.error("--sitstand policies use the unified 13D command obs (61D); add --new-cmd-obs")
    if (args.kick_left or args.kick_right or args.roulade) and not args.new_cmd_obs:
        parser.error(
            "--kick-left/--kick-right/--roulade policies use the unified 13D command obs (61D); add --new-cmd-obs"
        )
    if (args.kick_left or args.kick_right or args.roulade) and args.roller:
        parser.error("kick/roulade policies are trained on the walking robot, not the roller model")

    # 解析延迟参数
    delay_min_lag = 0
    delay_max_lag = 0
    if args.delay is not None:
        if len(args.delay) == 0:
            delay_min_lag = 1
            delay_max_lag = 2
        elif len(args.delay) == 1:
            delay_min_lag = args.delay[0]
            delay_max_lag = args.delay[0]
        elif len(args.delay) == 2:
            delay_min_lag = args.delay[0]
            delay_max_lag = args.delay[1]
        else:
            print("Error: --delay accepts 0, 1, or 2 arguments")
            return

    # 加载 MuJoCo 模型. 踢球策略使用带球的场景.
    if args.roller:
        xml_path = MICRODUCK_ROLLERS_XML
    elif args.kick_left or args.kick_right:
        xml_path = MICRODUCK_BALL_XML
    else:
        xml_path = MICRODUCK_XML
    print(f"Loading MuJoCo model from: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    # XL330 固件电流限制. 电机在 ~1.75 A 处饱和电流; 由于
    # torque = kt * current, 这将执行器力限制在 +/- kt * I_max.
    # 这里的 MuJoCo 位置执行器不是 BAM 电压模型, 但裁剪其输出力
    # 可以重现策略训练时所用的电流饱和 (见 BamActuator.max_current).
    # kt 来自 bam 包.
    if args.current_limit and args.current_limit > 0:
        from bam.model import load_model

        kt = load_model(motor_name="xl330", model="m6").kt.value
        torque_limit = kt * args.current_limit
        model.actuator_forcerange[:, 0] = -torque_limit
        model.actuator_forcerange[:, 1] = torque_limit
        model.actuator_forcelimited[:] = 1
        print(f"Current limit: {args.current_limit:.2f} A -> torque limit +/-{torque_limit:.4f} Nm (kt={kt:.4f})")

    # 脚部接触覆盖 — 模拟真实的高抓地力 + 柔软 PU 鞋底, 检查
    # 是否能重现机器人在高速时前扑的现象. 训练使用 mu~1.0 的
    # 刚性脚; 真实鞋底更抓地 (mu 更高) 且更柔顺 (solref 更软).
    # 仅应用于脚部碰撞 geom.
    if args.foot_friction is not None or args.foot_solref is not None:
        import re as _re

        n_feet = 0
        for g in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            if gname and _re.match(r"^(left|right)_foot_collision$", gname):
                if args.foot_friction is not None:
                    model.geom_friction[g, 0] = args.foot_friction  # tangential mu
                if args.foot_solref is not None:
                    model.geom_solref[g, 0] = args.foot_solref  # softer contact
                    model.geom_solref[g, 1] = 1.0
                n_feet += 1
        print(
            f"Foot override on {n_feet} geoms: "
            f"mu={args.foot_friction if args.foot_friction is not None else 'default'}, "
            f"solref={args.foot_solref if args.foot_solref is not None else 'default'}"
        )

    # 初始化策略
    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=args.walking,
        action_scale=args.action_scale,
        delay_min_lag=delay_min_lag,
        delay_max_lag=delay_max_lag,
        standing_onnx_path=args.standing,
        switch_threshold=args.switch_threshold,
        use_projected_gravity=not args.raw_accelerometer,
        ground_pick_onnx_path=args.ground_pick,
        ground_pick_period=args.ground_pick_period,
        sit_onnx_path=args.sit,
        new_cmd_obs=args.new_cmd_obs,
        slope_onnx_path=args.slope,
        sitstand_onnx_path=args.sitstand,
        kick_left_onnx_path=args.kick_left,
        kick_right_onnx_path=args.kick_right,
        roulade_onnx_path=args.roulade,
        kick_duration=args.kick_duration,
        roulade_duration=args.roulade_duration,
    )
    policy.set_vel_cmd(args.lin_vel_x, args.lin_vel_y, args.ang_vel_z)

    # 为 roller 推理设置真实的轮轴承摩擦 (必须以编程方式完成 —
    # XML 中非零 frictionloss 会破坏训练)
    if args.roller:
        import re

        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and re.match(r"^passive_.*", name):
                dof_adr = model.jnt_dofadr[j]
                model.dof_frictionloss[dof_adr] = 0.003

    # 按模式的速度命令限制, 匹配训练范围
    if args.roller:
        policy.vel_step_x = 0.05  # lin_vel_x 步长 (范围 -0.5..0.6)
        policy.vel_step_y = 0.0  # roller 无侧向命令
        policy.vel_step_ang = 0.1  # 航向误差步长 (范围 ±1.0 rad)
        policy.vel_max_x = 0.6
        policy.vel_min_x = -0.5  # 负值 = 制动
        policy.vel_max_y = 0.0
        policy.vel_min_y = 0.0
        policy.vel_max_ang = 1.0  # ±1.0 rad 航向误差
    else:
        policy.vel_max_x = 0.3
        policy.vel_min_x = -0.3
        policy.vel_max_y = 0.2
        policy.vel_min_y = -0.2
        policy.vel_max_ang = 1.5

    # 设置初始位置为默认姿态
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    data.qpos[qpos_adr + 0] = 0.0
    data.qpos[qpos_adr + 1] = 0.0
    data.qpos[qpos_adr + 2] = 0.1385 if args.roller else 0.125  # rollers 增加 13.5mm 高度
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    data.ctrl[:] = policy.default_pose
    mujoco.mj_forward(model, data)

    # 校验观测尺寸
    test_obs = policy.get_observations()
    cmd_dim = 13 if policy.new_cmd_obs else 3
    expected_obs_size = 3 + 3 + policy.n_joints + policy.n_joints + policy.n_joints + cmd_dim
    breakdown = (
        f"3(ang_vel) + 3(proj_grav) + {policy.n_joints}(joint_pos) + "
        f"{policy.n_joints}(joint_vel) + {policy.n_joints}(last_action) + {cmd_dim}(command)"
    )

    if test_obs.size != expected_obs_size:
        print("\nWARNING: Observation size mismatch!")
        print(f"  Expected: {expected_obs_size}")
        print(f"  Got: {test_obs.size}")
        print(f"  Breakdown: {breakdown}")
        print()

    print("\n" + "=" * 80)
    print("MicroDuck Policy Inference")
    print("=" * 80)
    print("Control frequency: 50 Hz (decimation: 4)")
    print(f"Simulation timestep: {model.opt.timestep}s")
    print(f"Observation size: {test_obs.size} (expected: {expected_obs_size})")
    if policy.walking_session:
        print("Walking policy: loaded")
    if policy.standing_session:
        print(
            f"Standing policy: loaded  (body pose: z=±{BODY_CMD_MAX_Z * 1000:.0f}mm, pitch/roll=±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)"
        )
    if policy.walking_session and policy.standing_session:
        print(f"  Switch threshold: {policy.switch_threshold} (vel cmd magnitude)")
    if policy.ground_pick_session:
        print("Ground pick policy: loaded  (press G)")
    if policy.sit_session:
        kind = "Sitstand" if policy.is_sitstand else "Sit"
        print(f"{kind} policy: loaded  (press Y to toggle)")
    if policy.slope_session:
        print("Slope policy: loaded  (press Y to toggle, passive descent)")
    _behavior_keys = {"kick_left": "K", "kick_right": "L", "roulade": "R"}
    for _name in policy.behavior_sessions:
        print(
            f"{_name} policy: loaded  (press {_behavior_keys[_name]}, "
            f"auto-return after {policy.behavior_durations[_name]:.1f}s)"
        )
    print(f"Active policy: {policy.current_policy}")
    print("Close viewer window to exit")
    print()

    decimation = 4
    control_step_count = 0
    control_dt = decimation * model.opt.timestep

    # 滑动缓冲区, 保存过去 1 s 内 trunk 世界坐标系下的 xy 速度, 用于
    # 打印移动平均值, 以便比较命令速度和实际速度.
    from collections import deque

    _vel_window_steps = max(1, int(round(1.0 / control_dt)))  # ≈ 50 @ 50 Hz
    vel_history = deque(maxlen=_vel_window_steps)

    csv_data = [] if args.save_csv else None
    recorded_observations = [] if args.record else None
    policy_enabled = not args.record
    policy_enable_time = None
    original_kp = None
    if args.record:
        original_kp = model.actuator_gainprm[:, 0].copy()

    # 缓存 trunk freejoint 的 qvel 地址, 以便 push 处理器可以直接写入
    # trunk 的世界坐标系线速度 (qvel[0..3]).
    _freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    _trunk_qvel_adr = int(model.jnt_dofadr[_freejoint_id])
    PUSH_MAX = 1.0  # 匹配 velstand 最终 push_magnitude curriculum 上限

    def random_push():
        """将 trunk 的世界坐标系 xy 速度设为 PUSH_MAX 大小的随机向量, 模拟.

        push_by_setting_velocity 训练事件.

        不累积 — 直接覆盖当前线速度.
        """
        import random

        angle = random.uniform(0, 2 * np.pi)
        vx = PUSH_MAX * np.cos(angle)
        vy = PUSH_MAX * np.sin(angle)
        data.qvel[_trunk_qvel_adr + 0] = vx
        data.qvel[_trunk_qvel_adr + 1] = vy
        print(f"PUSH applied: v=[{vx:.2f}, {vy:.2f}, 0] m/s (angle={np.degrees(angle):.0f}°)")

    # 按键来自 TERMINAL (原始 stdin, 见 TerminalInput) — 不来自
    # MuJoCo viewer 窗口, 因为 viewer 窗口中的按键也会触发内置可视化
    # 快捷键. `key` 是符号名: "up"/"down"/"left"/"right", " ", 或
    # 小写字母.
    quit_requested = False

    def handle_key(key):
        nonlocal policy_enabled, quit_requested
        try:
            if key == "up":
                if policy.head_mode:
                    policy.head_offset[1] = np.clip(
                        policy.head_offset[1] + policy.head_step,
                        -policy.head_max,
                        policy.head_max,
                    )
                    policy._update_command()
                    print(
                        f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}"
                    )
                elif policy.body_pose_mode:
                    policy.bump_body("z", policy.body_cmd_step_z)
                else:
                    policy.set_vel_cmd(policy.vel_max_x, policy.vel_cmd[1], policy.vel_cmd[2])
            elif key == "down":
                if policy.head_mode:
                    policy.head_offset[1] = np.clip(
                        policy.head_offset[1] - policy.head_step,
                        -policy.head_max,
                        policy.head_max,
                    )
                    policy._update_command()
                    print(
                        f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}"
                    )
                elif policy.body_pose_mode:
                    policy.bump_body("z", -policy.body_cmd_step_z)
                else:
                    policy.set_vel_cmd(policy.vel_min_x, policy.vel_cmd[1], policy.vel_cmd[2])
            elif key == "right":
                if policy.head_mode:
                    policy.head_offset[2] = np.clip(
                        policy.head_offset[2] - policy.head_step,
                        -policy.head_max,
                        policy.head_max,
                    )
                    policy._update_command()
                    print(
                        f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}"
                    )
                elif policy.body_pose_mode:
                    policy.bump_body("pitch", -policy.body_cmd_step_angle)
                elif args.roller:
                    new_ang = np.clip(
                        policy.vel_cmd[2] - policy.vel_step_ang,
                        -policy.vel_max_ang,
                        policy.vel_max_ang,
                    )
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], new_ang)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_min_y, policy.vel_cmd[2])
            elif key == "left":
                if policy.head_mode:
                    policy.head_offset[2] = np.clip(
                        policy.head_offset[2] + policy.head_step,
                        -policy.head_max,
                        policy.head_max,
                    )
                    policy._update_command()
                    print(
                        f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}"
                    )
                elif policy.body_pose_mode:
                    policy.bump_body("pitch", policy.body_cmd_step_angle)
                elif args.roller:
                    new_ang = np.clip(
                        policy.vel_cmd[2] + policy.vel_step_ang,
                        -policy.vel_max_ang,
                        policy.vel_max_ang,
                    )
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], new_ang)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_max_y, policy.vel_cmd[2])
            elif key == " ":
                if policy.head_mode:
                    policy.head_offset[:] = 0.0
                    policy._update_command()
                    print("Head offset reset to zero")
                elif policy.body_pose_mode:
                    policy.body_cmd[:] = 0.0
                    policy._update_command()
                    print("Body pose cmd reset to zero")
                else:
                    policy.set_vel_cmd(0.0, 0.0, 0.0)
            elif key == "t":
                # 开关策略推理. 关闭时控制器停止
                # 查询 ONNX 策略, 电机保持最后应用的
                # 目标 (没有新的 ctrl 写入).
                policy_enabled = not policy_enabled
                print(f"Policy inference: {'ON' if policy_enabled else 'OFF (paused)'}")
            elif key == "g":
                policy.trigger_ground_pick()
            elif key == "k":
                policy.trigger_behavior("kick_left")
            elif key == "l":
                policy.trigger_behavior("kick_right")
            elif key == "r":
                policy.trigger_behavior("roulade")
            elif key == "q":
                quit_requested = True
                print("Quit requested")
            elif key == "y":
                # Y 切换已加载的辅助策略 (--sit 或 --slope).
                if policy.sit_session is not None:
                    policy.toggle_sit()
                else:
                    policy.toggle_slope_mode()
            elif key == "h":
                policy.toggle_head_mode()
            elif key == "b":
                policy.toggle_body_pose_mode()
            elif key == "p":
                random_push()
            elif key == "a":
                if policy.head_mode:
                    policy.head_offset[3] = np.clip(
                        policy.head_offset[3] + policy.head_step,
                        -policy.head_max,
                        policy.head_max,
                    )
                    policy._update_command()
                    print(
                        f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}"
                    )
                elif policy.body_pose_mode:
                    policy.bump_body("roll", policy.body_cmd_step_angle)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], policy.vel_max_ang)
            elif key == "e":
                if policy.head_mode:
                    policy.head_offset[3] = np.clip(
                        policy.head_offset[3] - policy.head_step,
                        -policy.head_max,
                        policy.head_max,
                    )
                    policy._update_command()
                    print(
                        f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}"
                    )
                elif policy.body_pose_mode:
                    policy.bump_body("roll", -policy.body_cmd_step_angle)
                else:
                    policy.set_vel_cmd(policy.vel_cmd[0], policy.vel_cmd[1], -policy.vel_max_ang)
            elif key == "z":
                if policy.head_mode:
                    policy.head_offset[0] = np.clip(
                        policy.head_offset[0] + policy.head_step,
                        -policy.head_max,
                        policy.head_max,
                    )
                    policy._update_command()
                    print(
                        f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}"
                    )
                elif policy.body_pose_mode and policy.new_cmd_obs:
                    policy.bump_body("yaw", policy.body_cmd_step_angle)
            elif key == "s":
                if policy.head_mode:
                    policy.head_offset[0] = np.clip(
                        policy.head_offset[0] - policy.head_step,
                        -policy.head_max,
                        policy.head_max,
                    )
                    policy._update_command()
                    print(
                        f"Head offset: neck={policy.head_offset[0]:.2f} pitch={policy.head_offset[1]:.2f} yaw={policy.head_offset[2]:.2f} roll={policy.head_offset[3]:.2f}"
                    )
                elif policy.body_pose_mode and policy.new_cmd_obs:
                    policy.bump_body("yaw", -policy.body_cmd_step_angle)
        except Exception as e:
            print(f"Key press error: {e}")

    print("\nKeyboard controls (type in THIS terminal — the viewer window no longer captures keys):")
    print("  [ Velocity mode (default) ]")
    print("  UP arrow:         increase lin_vel_x (push/accelerate)")
    print("  DOWN arrow:       decrease lin_vel_x (0=coast, negative=brake)")
    if args.roller:
        print("  LEFT/RIGHT arrow: turn left/right (ang_vel_z heading error)")
        print("  A / E:            turn left/right (ang_vel_z, incremental)")
    else:
        print("  LEFT/RIGHT arrow: strafe left/right (lin_vel_y)")
        print("  A / E:            turn left/right (ang_vel_z)")
    print("  SPACE:            coast (zero all commands)")
    print("  T:                toggle policy inference on/off (paused = motors hold last target)")
    print("  G:                trigger ground pick (requires --ground-pick)")
    print("  Y:                toggle sit (with --sit/--sitstand) or slope mode (with --slope)")
    print("  K:                kick with LEFT foot (requires --kick-left)")
    print("  L:                kick with RIGHT foot (requires --kick-right)")
    print("  R:                roulade / forward roll (requires --roulade)")
    print(f"  P:                random push (trunk vel = {PUSH_MAX:.1f} m/s in random direction)")
    print("  Q:                quit")
    print("  [ Body pose mode — press B to toggle ]")
    print(f"  UP/DOWN arrow:    Δz ±10mm  (max ±{BODY_CMD_MAX_Z * 1000:.0f}mm)")
    print(f"  LEFT/RIGHT arrow: Δpitch ±10°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    print(f"  A / E:            Δroll ±10°  (max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    if args.new_cmd_obs:
        print(f"  Z / S:            Δyaw ±10°  (new_cmd_obs only, max ±{math.degrees(BODY_CMD_MAX_ANGLE):.0f}°)")
    print("  SPACE:            reset body pose to zero")
    print("  [ Head mode — press H to toggle ]")
    print("  Z / S:            neck_pitch ±step")
    print("  UP/DOWN arrow:    head_pitch ±step")
    print("  LEFT/RIGHT arrow: head_yaw ±step")
    print("  A / E:            head_roll ±step")
    print("  SPACE:            reset head offset to zero")

    with (
        TerminalInput() as term,
        mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer,
    ):
        viewer.sync()
        start_time = time.time()

        if args.record:
            policy_enable_time = start_time + 1.0
            print("Recording mode: policy will be enabled after 1 second standby")
            for i in range(model.nu):
                model.actuator_gainprm[i, 0] = 2.0
                model.actuator_biasprm[i, 1] = -2.0
            print("  Standby mode: kp set to 2.0")

        try:
            prev_step_time = time.time()

            while viewer.is_running() and not quit_requested:
                step_start = time.time()

                for key in term.get_keys():
                    handle_key(key)

                if not policy_enabled and policy_enable_time is not None and step_start >= policy_enable_time:
                    policy_enabled = True
                    if original_kp is not None:
                        for i in range(model.nu):
                            kp = original_kp[i]
                            model.actuator_gainprm[i, 0] = kp
                            model.actuator_biasprm[i, 1] = -kp
                        print("Policy inference enabled (after 1s standby)")
                        print(
                            f"  Restored original kp gains (range: [{original_kp.min():.2f}, {original_kp.max():.2f}])"
                        )

                actual_dt = step_start - prev_step_time
                prev_step_time = step_start

                policy.update_ground_pick_phase(actual_dt)
                policy.update_behavior(actual_dt)

                if policy_enabled:
                    action = policy.infer()
                    policy.apply_action(action)
                else:
                    # 暂停: 保持上一个 ctrl, 不查询策略. 电机
                    # 保持位置. 使用零动作只是为了让下游
                    # 日志 (csv/debug) 看到一致的内容.
                    action = np.zeros(policy.n_joints, dtype=np.float32)

                control_step_count += 1

                # 跟踪 BODY 坐标系的前向/侧向速度 + yaw 速率, 每秒
                # 打印一次 1 秒移动平均值与命令值对比.
                # 用 body 坐标系以便 "前进"/"转向" 与命令 (在机器人
                # 坐标系中) 直接可比: 让我们看到策略是否真正实现了
                # 命令的前进速度和转向速率.
                quat = data.qpos[qpos_adr + 3 : qpos_adr + 7].astype(np.float32)
                v_world = np.array(
                    [
                        data.qvel[_trunk_qvel_adr + 0],
                        data.qvel[_trunk_qvel_adr + 1],
                        data.qvel[_trunk_qvel_adr + 2],
                    ],
                    dtype=np.float32,
                )
                v_body = policy.quat_rotate_inverse(quat, v_world)
                yaw_rate = float(data.qvel[_trunk_qvel_adr + 5])  # body-frame wz
                vel_history.append((float(v_body[0]), float(v_body[1]), yaw_rate))
                if control_step_count % _vel_window_steps == 0 and len(vel_history) > 0:
                    n = len(vel_history)
                    avg_fwd = sum(v[0] for v in vel_history) / n
                    avg_lat = sum(v[1] for v in vel_history) / n
                    avg_yaw = sum(v[2] for v in vel_history) / n
                    cmd_x, cmd_y, cmd_yaw = (
                        policy.vel_cmd[0],
                        policy.vel_cmd[1],
                        policy.vel_cmd[2],
                    )
                    trunk_z = float(data.qpos[qpos_adr + 2])
                    print(
                        f"[vel 1s avg] achieved/cmd  fwd={avg_fwd:+.2f}/{cmd_x:+.2f}  "
                        f"lat={avg_lat:+.2f}/{cmd_y:+.2f} m/s  "
                        f"yaw={avg_yaw:+.2f}/{cmd_yaw:+.2f} rad/s   "
                        f"trunk_z={trunk_z * 1000:.1f} mm"
                    )

                if csv_data is not None:
                    obs = policy.get_observations()
                    row = {
                        "step": control_step_count,
                        "time": control_step_count * control_dt,
                    }
                    for i in range(obs.size):
                        row[f"obs_{i}"] = obs[i]
                    for i in range(action.size):
                        row[f"action_{i}"] = action[i]
                    csv_data.append(row)

                if recorded_observations is not None:
                    obs = policy.get_observations()
                    timestamp = time.time() - start_time
                    recorded_observations.append({"timestamp": timestamp, "observation": obs.tolist()})

                if args.debug:
                    should_print = control_step_count <= 10 or control_step_count % 50 == 0
                    if should_print:
                        obs = policy.get_observations()
                        pos = data.qpos[qpos_adr : qpos_adr + 3]
                        quat = data.qpos[qpos_adr + 3 : qpos_adr + 7]
                        com_height = pos[2]

                        print(f"\n{'=' * 70}")
                        print(f"Step {control_step_count} DEBUG:")
                        print(f"{'=' * 70}")
                        print(f"Active policy: {policy.current_policy}")
                        print("Base state:")
                        print(f"  Position: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
                        print(f"  CoM height: {com_height:7.4f}")
                        print(f"  Quaternion: [{quat[0]:7.4f}, {quat[1]:7.4f}, {quat[2]:7.4f}, {quat[3]:7.4f}]")
                        print(f"\nObservation (shape {obs.shape}, total {obs.size}):")
                        print(f"  Ang vel [0:3]:        {obs[0:3]}")
                        print(f"  Proj grav [3:6]:      {obs[3:6]}")
                        print(f"  Joint pos [6:{6 + policy.n_joints}]:     {obs[6 : 6 + policy.n_joints]}")
                        print(
                            f"  Joint vel [{6 + policy.n_joints}:{6 + 2 * policy.n_joints}]:    {obs[6 + policy.n_joints : 6 + 2 * policy.n_joints]}"
                        )
                        print(
                            f"  Last action [{6 + 2 * policy.n_joints}:{6 + 3 * policy.n_joints}]:  {obs[6 + 2 * policy.n_joints : 6 + 3 * policy.n_joints]}"
                        )
                        cmd_end = 6 + 3 * policy.n_joints + 3
                        print(
                            f"  Command [{6 + 3 * policy.n_joints}:{cmd_end}]:      {obs[6 + 3 * policy.n_joints : cmd_end]}"
                        )
                        if policy.current_policy == "standing":
                            print(
                                f"  Body cmd (raw): z={policy.body_cmd[0] * 1000:.1f}mm  pitch={math.degrees(policy.body_cmd[1]):.1f}°  roll={math.degrees(policy.body_cmd[2]):.1f}°"
                            )
                        print("\nAction output:")
                        print(f"  Raw action: {action}")
                        print(f"  Action min/max: [{action.min():.4f}, {action.max():.4f}]")
                        if policy.use_delay:
                            print(f"  Delay: {policy.current_lag} timesteps (buffered)")
                        print(f"  Applied ctrl (first 5): {data.ctrl[:5]}")
                        print(f"  Applied ctrl (last 5):  {data.ctrl[-5:]}")

                for _ in range(decimation):
                    mujoco.mj_step(model, data)

                viewer.sync()

                elapsed = time.time() - step_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt received (Ctrl+C). Saving data...")

    print("\nInference stopped.")

    if csv_data is not None and len(csv_data) > 0:
        print(f"\nSaving {len(csv_data)} steps to: {args.save_csv}")
        with Path(args.save_csv).open("w", newline="") as csvfile:
            fieldnames = csv_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print("CSV file saved successfully!")
        print(f"  Columns: {len(fieldnames)}")
        print(f"  Rows: {len(csv_data)}")

    if recorded_observations is not None and len(recorded_observations) > 0:
        print(f"\nSaving {len(recorded_observations)} recorded observations to: {args.record}")
        with Path(args.record).open("wb") as f:
            pickle.dump(recorded_observations, f)
        print(f"Recorded observations saved to {args.record}")
        print(f"  Observations: {len(recorded_observations)}")
        print(f"  Duration: {recorded_observations[-1]['timestamp']:.2f}s")


if __name__ == "__main__":
    main()
