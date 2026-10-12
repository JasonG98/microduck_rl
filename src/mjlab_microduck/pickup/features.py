"""每个 tick 的特征向量 + 暂停时的控制器律 — runtime 契约.

这里的一切都是 robotd 每个 50 Hz tick 能算出的量, 单位一致:

  [0:3]    gyro (rad/s, trunk IMU 系)                   — ImuData.gyro
  [3:6]    projected gravity (单位向量, IMU 系)          — 由 SFLP 四元数得到
  [6:20]   joint pos − HOME (rad)                       — 与 obs[6:20] 相同
  [20:34]  joint vel (rad/s)                            — 与 obs[20:34] 相同
  [34:48]  commanded target − HOME (rad)                — robotd 写给舵机的量
  [48:62]  |present current| (A)                        — Sensors.currents_ma / 1000
  [62]     paused 标志 (检测器保持姿态期间为 1)

关节顺序为 14 舵机的 ctrl 顺序 (0-4 左腿, 5-8 颈/头, 9-13 右腿).

PAUSED 期间 runtime 会在 ``HOLD_RAMP_S`` 内从最后的 policy 目标斜坡过渡到
``PAUSE_POSE_REL`` 并保持 — ``paused_target_rel`` 就是该律的精确形式. PAUSE_POSE_REL 是
velstand 自身的平均站立姿态 (零命令, 对称化): 在地面上以 kp 200 保持 HOME 在 sim 中会翻倒
(1 s 后 50 % 超过 40°, 3 s 后 93 %); 该姿态争取到时间 (11 % / 45 %), 并让机器人落在 policy
期望的站姿上. 因此放下检测必须快 — policy 得在被放下的机器人倾倒前接管.

(暂停时用 hip-pitch "probe" 正弦探测的方案试过并放弃了: 在受控 sim bench 中手的柔顺会让 trunk
随腿一起摇动, 因此它无法区分被握住与在地面, 还会让机器人抖动.)
"""

from __future__ import annotations

import torch

NUM_JOINTS = 14
FEAT_DIM = 63
PAUSED_IDX = 62
CTRL_HZ = 50.0

LEG_IDS = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)  # 舵机 ctrl 索引
CURRENT_SLICE = slice(48, 62)

HOLD_RAMP_S = 0.3
# target − HOME (rad), ctrl 顺序: velstand fhathosb@3750 零命令站立,
# 256 envs × 1 s 的均值, L/R 镜像, hip_yaw/head yaw/roll 置零
PAUSE_POSE_REL = (
    0.0, -0.100, 0.038, -0.145, 0.044,      # 左 hip_yaw, hip_roll, hip_pitch, knee, ankle
    -0.096, 0.008, 0.0, 0.0,                # neck_pitch, head_pitch, head_yaw, head_roll
    0.0, 0.100, -0.038, 0.145, -0.044,      # 右 hip_yaw, hip_roll, hip_pitch, knee, ankle
)

FEATURE_NAMES = (
    [f"gyro_{a}" for a in "xyz"] + [f"grav_{a}" for a in "xyz"]
    + [f"q_{i}" for i in range(14)] + [f"dq_{i}" for i in range(14)]
    + [f"tgt_{i}" for i in range(14)] + [f"cur_{i}" for i in range(14)] + ["paused"]
)
assert len(FEATURE_NAMES) == FEAT_DIM


def paused_target_rel(t_paused: torch.Tensor, target_at_pause: torch.Tensor) -> torch.Tensor:
    """暂停期间的 target − HOME. ``t_paused`` (N,) 为自暂停起的秒数,
    ``target_at_pause`` (N, 14) 为最后的 policy target − HOME."""
    a = (t_paused / HOLD_RAMP_S).clamp(0.0, 1.0).unsqueeze(-1)
    pose = torch.tensor(PAUSE_POSE_REL, device=target_at_pause.device, dtype=target_at_pause.dtype)
    return target_at_pause * (1.0 - a) + pose * a


class CurrentSensorDR:
    """sim 电流 → XL330 上报的值: 逐 env/关节的 gain + offset, 噪声,
    1 mA 量化, 取绝对值 (robotd 存 |current|)."""

    def __init__(self, num_envs, device, gain=(0.7, 1.3), offset_ma=(0.0, 25.0), noise_ma=10.0):
        self.gain = torch.empty(num_envs, NUM_JOINTS, device=device).uniform_(*gain)
        self.offset = torch.empty(num_envs, NUM_JOINTS, device=device).uniform_(*offset_ma) / 1000.0
        self.noise = noise_ma / 1000.0

    def resample(self, env_ids, gain=(0.7, 1.3), offset_ma=(0.0, 25.0)):
        n = len(env_ids)
        dev = self.gain.device
        self.gain[env_ids] = torch.empty(n, NUM_JOINTS, device=dev).uniform_(*gain)
        self.offset[env_ids] = torch.empty(n, NUM_JOINTS, device=dev).uniform_(*offset_ma) / 1000.0

    def __call__(self, current_a: torch.Tensor) -> torch.Tensor:
        c = current_a * self.gain + torch.randn_like(current_a) * self.noise
        c = c.abs() + self.offset
        return torch.round(c * 1000.0) / 1000.0
