"""microduck 61-D env 的左右对称性数据增强.

于 2026-08-13 从旧的 51-D 布局迁移到当前的 61-D 任务族
(velocity/velstand/standup/roulade — twist + head_command + body_command
obs 槽位), 并修复了 augmented-obs 输出键由 "policy" → "actor" 的问题
(mjlab 1.3.0 group 命名; 旧键在 rsl_rl 5.0.1 的 mirror-loss 路径中会
KeyError — 此前因没有 env 启用 symmetry 而一直是死代码).

Actor 观测布局 (61 维平坦张量, 按 term 插入顺序拼接):
    [0:3]   base_ang_vel      (roll, pitch, yaw — 本体坐标系 IMU)
    [3:6]   projected_gravity (gx, gy, gz       — 本体坐标系)
    [6:20]  joint_pos_rel     (14 个关节, 相对于默认姿态)
    [20:34] joint_vel_rel     (14 个关节)
    [34:48] last_action       (14 个关节)
    [48:51] twist command     (lin_vel_x, lin_vel_y, ang_vel_z)
    [51:55] head command      (neck_pitch, head_pitch, head_yaw, head_roll 增量)
    [55:61] body command      (x, y, z, roll, pitch, yaw 增量)

每个 14 维块内的关节顺序 (来自 robot_walk.xml body 树):
    0: left_hip_yaw    5: neck_pitch    9:  right_hip_yaw
    1: left_hip_roll   6: head_pitch    10: right_hip_roll
    2: left_hip_pitch  7: head_yaw      11: right_hip_pitch
    3: left_knee       8: head_roll     12: right_knee
    4: left_ankle                       13: right_ankle

镜像规则 (绕矢状面的左右反射):
- 左腿 (0-4) 与右腿 (9-13) 互换; 中线关节 (5-8) 不动.
- 互换后取反:
    - hip_yaw, hip_roll: yaw/roll 轴在左右反射下方向反转
    - hip_pitch, knee, ankle: home 帧左右采用相反符号约定 (例如
      left_hip_pitch = +0.6, right_hip_pitch = -0.6),
      所以相对偏差也取反
    - head_yaw, head_roll: 同样的 yaw/roll 推理
    - neck_pitch, head_pitch: 矢状面关节, 符号不变
- base_ang_vel: roll ([0]) 与 yaw ([2]) 取反; pitch 不变
- projected_gravity: gy ([4]) 取反; gx 与 gz 不变
- twist command: lin_vel_y ([49]) 与 ang_vel_z ([50]) 取反; lin_vel_x 不变
- head command: head_yaw ([53]) 与 head_roll ([54]) 取反; pitch 不变
- body command: y ([56]), roll ([58]), yaw ([60]) 取反; x, z, pitch 不变
"""

from dataclasses import dataclass

import torch
from mjlab.rl import RslRlPpoAlgorithmCfg
from tensordict import TensorDict


@dataclass
class PpoWithSymmetryCfg(RslRlPpoAlgorithmCfg):
    """PPO 算法 cfg, 扩展了一个可选的 symmetry_cfg 字段."""

    symmetry_cfg: dict | None = None


SYMMETRY_CFG = {
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.5,
    "data_augmentation_func": "mjlab_microduck.tasks.symmetry.microduck_vel_symmetry",
}

# ---------------------------------------------------------------------------
# 置换与符号表
# ---------------------------------------------------------------------------

# 14 关节块内: 左 (0-4) <-> 右 (9-13), 中线 (5-8) 固定
_JOINT_PERM: list[int] = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]

# 置换后施加到每个关节位置的符号
_JOINT_SIGN: list[float] = [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1]

# 完整 61 维 actor obs 置换 (所有 command 槽原地镜像)
_OBS_PERM: list[int] = (
    [0, 1, 2]  # base_ang_vel (索引不变)
    + [3, 4, 5]  # projected_gravity
    + [6 + j for j in _JOINT_PERM]  # joint_pos
    + [20 + j for j in _JOINT_PERM]  # joint_vel
    + [34 + j for j in _JOINT_PERM]  # last_action
    + [48, 49, 50]  # twist command
    + [51, 52, 53, 54]  # head command
    + [55, 56, 57, 58, 59, 60]  # body command
)

# 完整 61 维符号向量
_OBS_SIGN: list[float] = (
    [-1.0, 1.0, -1.0]  # base_ang_vel: 取反 roll, yaw
    + [1.0, -1.0, 1.0]  # projected_gravity: 取反 gy
    + _JOINT_SIGN  # joint_pos
    + _JOINT_SIGN  # joint_vel
    + _JOINT_SIGN  # last_action
    + [1.0, -1.0, -1.0]  # twist: 取反 lin_vel_y, ang_vel_z
    + [1.0, 1.0, -1.0, -1.0]  # head: 取反 head_yaw, head_roll
    + [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # body: 取反 y, roll, yaw
)

# 按设备缓存张量, 避免每次调用都重新分配
_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_tensors(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if device not in _cache:
        obs_perm = torch.tensor(_OBS_PERM, dtype=torch.long, device=device)
        obs_sign = torch.tensor(_OBS_SIGN, dtype=torch.float32, device=device)
        act_perm = torch.tensor(_JOINT_PERM, dtype=torch.long, device=device)
        act_sign = torch.tensor(_JOINT_SIGN, dtype=torch.float32, device=device)
        _cache[device] = (obs_perm, obs_sign, act_perm, act_sign)
    return _cache[device]


# ---------------------------------------------------------------------------
# 公共数据增强函数
# ---------------------------------------------------------------------------


def microduck_vel_symmetry(
    env,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Microduck vel env 的左右对称数据增强 / 镜像函数.

    沿 batch 维返回 [原始, 镜像] 拼接结果.与 rsl_rl PPO 的 ``symmetry_cfg``
    接口兼容 (use_data_augmentation 和/或 use_mirror_loss).

    Args:
    env: 向量化环境 (未使用, 仅为接口兼容而保留).
    obs: TensorDict, 含键 ``"policy"`` 与 ``"critic"``, shape ``[B, obs_dim]``.
         只需镜像 actions 时传 ``None``.
    actions: float tensor, shape ``[B, 14]``.
             只需镜像 obs 时传 ``None``.

    Returns:
    元组 ``(aug_obs, aug_actions)``, 每个非 None 输入沿 batch 轴翻倍为
    ``[原始; 镜像]``.
    """
    aug_obs: TensorDict | None = None
    aug_actions: torch.Tensor | None = None

    if obs is not None:
        actor_orig: torch.Tensor = obs["actor"]  # [B, 51]
        obs_perm, obs_sign, _, _ = _get_tensors(actor_orig.device)
        actor_sym = actor_orig[:, obs_perm] * obs_sign

        critic_orig: torch.Tensor = obs["critic"]
        # Critic obs 镜像未实现 (use_mirror_loss 不需要).
        # 对 use_data_augmentation, critic 看到的是重复未镜像的 obs,
        # 这是个无害的近似, 因为 critic 使用 actor obs 中不存在的
        # 特权信息.
        critic_repeated = torch.cat([critic_orig, critic_orig], dim=0)

        aug_obs = TensorDict(
            {
                "actor": torch.cat([actor_orig, actor_sym], dim=0),
                "critic": critic_repeated,
            },
            batch_size=[actor_orig.shape[0] * 2],
            device=actor_orig.device,
        )

    if actions is not None:
        _, _, act_perm, act_sign = _get_tensors(actions.device)
        actions_sym = actions[:, act_perm] * act_sign
        aug_actions = torch.cat([actions, actions_sym], dim=0)

    return aug_obs, aug_actions
