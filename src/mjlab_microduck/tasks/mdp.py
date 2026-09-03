"""microduck 任务的 MDP 函数."""

import math
from dataclasses import dataclass as _dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.reward_manager import RewardManager as _RewardManager
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import observations as _velocity_obs
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_from_angle_axis,
    wrap_to_pi,
)
from rsl_rl.algorithms.ppo import PPO as _PPO

# ---------------------------------------------------------------------------
# Patch 1: RewardManager.compute — 在 NaN reward 进入 PPO buffer 之前将其清理.
# mjlab 在 reset 环境之前计算 reward, 因此任何作用于 NaN 物理状态的 reward
# 项都会返回 NaN.该 NaN 会传播: NaN reward → NaN advantage → NaN loss →
# NaN gradient → NaN/负 std → 下一个 mini-batch 的 torch.normal 崩溃.
# ---------------------------------------------------------------------------
_orig_reward_compute = _RewardManager.compute


def _nan_safe_reward_compute(self, dt: float) -> torch.Tensor:
    result = _orig_reward_compute(self, dt)
    # _episode_sums 在 compute() 内部更新, 早于 nan_to_num 生效.
    # 就地清理, 防止 per-term 指标显示 NaN.
    for key in self._episode_sums:
        torch.nan_to_num_(self._episode_sums[key], nan=0.0)
    return torch.nan_to_num(result, nan=0.0)


_RewardManager.compute = _nan_safe_reward_compute

# ---------------------------------------------------------------------------
# Patch 2: PPO.compute_returns — 在归一化之前清理 advantage.
# 在 curriculum 突变步 (例如 reward 权重 ×2.5), value function 严重失准:
# 所有 TD 误差整体偏移相同量, std(advantages) → 极小, 且
# (A − mean) / (std + 1e-8) → 极大.这会撑爆 std 的梯度, 使优化器将其推到
# 负值.在归一化之前将 NaN/Inf advantage 置零, 可保持其在安全范围内.
# ---------------------------------------------------------------------------
_orig_compute_returns = _PPO.compute_returns


def _safe_compute_returns(self, obs) -> None:
    _orig_compute_returns(self, obs)
    st = self.storage
    torch.nan_to_num_(st.advantages, nan=0.0, posinf=0.0, neginf=0.0)
    torch.nan_to_num_(st.returns, nan=0.0, posinf=0.0, neginf=0.0)


_PPO.compute_returns = _safe_compute_returns

# Patch 3 (ActorCritic._update_distribution std-clamp) 在 mjlab 1.3.0 迁移中
# 已移除: rsl_rl 5.0.1 重构了 policy (不再有 ActorCritic 类; distribution 现
# 位于 rsl_rl.modules.distribution).它原本是针对 std 变负/NaN 的防御性
# 补丁 (microban 不加也能跑).若 1.3.0 下 std 爆炸问题复发, 需针对新的
# GaussianDistribution 重新加回.

print("[mdp] Patch 1-2 已激活: reward/advantage 的 NaN 防护")

# ---------------------------------------------------------------------------
# Patch 4: exporter_utils.get_base_metadata — 新版 microduck 模型带有 passive
# 关节 (通过等式约束闭合的颚部连杆), 它们属于 articulation 但没有 XML
# actuator.上游 exporter 遍历 robot.joint_names (16) 并索引
# joint_name_to_ctrl_id (14), 在 passive_* 上会因 KeyError 崩溃.这里将
# passive 关节从导出元数据中过滤掉, 使 policy 与 14 维动作空间保持一致.
# ---------------------------------------------------------------------------
from mjlab.envs.mdp.actions import JointPositionAction as _JointAction  # noqa: E402
from mjlab.rl import exporter_utils as _exporter_utils  # noqa: E402


def _get_base_metadata_no_passive(env, run_path):
    robot = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, _JointAction)
    full_names = list(robot.joint_names)
    keep_idx = [i for i, n in enumerate(full_names) if not n.startswith("passive_")]
    joint_names = [full_names[i] for i in keep_idx]
    joint_name_to_ctrl_id = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    ctrl_ids = [joint_name_to_ctrl_id[n] for n in joint_names]
    stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
    damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]
    default_jp = robot.data.default_joint_pos[0].cpu().tolist()
    return {
        "run_path": run_path,
        "joint_names": joint_names,
        "joint_stiffness": stiffness.tolist(),
        "joint_damping": damping.tolist(),
        "default_joint_pos": [default_jp[i] for i in keep_idx],
        "command_names": list(env.command_manager.active_terms),
        "observation_names": env.observation_manager.active_terms["actor"],
        "action_scale": joint_action._scale[0].cpu().tolist()
        if isinstance(joint_action._scale, torch.Tensor)
        else joint_action._scale,
    }


_exporter_utils.get_base_metadata = _get_base_metadata_no_passive
# 同时 patch velocity task exporter 中已导入的引用.
try:
    from mjlab.tasks.velocity.rl import exporter as _vel_exporter  # noqa: E402

    if hasattr(_vel_exporter, "get_base_metadata"):
        _vel_exporter.get_base_metadata = _get_base_metadata_no_passive
except Exception:
    pass

print("[mdp] Patch 4 已激活: ONNX 导出过滤 passive_* 关节")

if TYPE_CHECKING:
    from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
# 模块级单例, 用作 SceneEntityCfg 默认值 (B008: 默认参数中不能有函数调用 —
# 这些是初始化后冻结的共享 dataclass 配置).
_TRUNK_BASE_ASSET_CFG = SceneEntityCfg("robot", body_names=("trunk_base",))
_LEG_JOINTS_ASSET_CFG = SceneEntityCfg("robot", joint_names=(r".*(hip|knee|ankle).*",))
_MOUTH_TIP_ASSET_CFG = SceneEntityCfg("robot", site_names=["mouth_tip"])
_JAW_SOFT_ASSET_CFG = SceneEntityCfg("robot", body_names=["jaw_soft"], site_names=["mouth_tip"])
_FEET_ASSET_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))

# 匹配 4 个 neck/head 驱动关节的名称模式.用于 head_pose 跟踪 reward 和
# UniformPoseCommand 的 asset 绑定.
_NECK_JOINT_PATTERNS = [
    r".*neck_pitch.*",
    r".*head_pitch.*",
    r".*head_yaw.*",
    r".*head_roll.*",
]


def _servo_joint_ids(env: "ManagerBasedRlEnv", asset: Entity) -> list:
    """servo (非 ``passive_``) 关节的 entity 局部索引, 带缓存.

    本模块中所有基于关节索引的 reward/event 参数 (``joint_indices``, ``target_overrides``, qpos 列运算)
    都是针对规范的 14-servo 布局编写的.在带额外非驱动关节的模型上 — backlash 铰链、滚轮、
    颚部连杆, 均命名为 ``passive_*`` — entity 关节数组更宽且交错排列, 因此原始索引会选错关节.
    通过本列表索引以恢复 servo-only 视图; 在普通模型上即为恒等映射.
    """
    cache = env.__dict__.setdefault("_servo_joint_ids_cache", {})
    key = id(asset)
    ids = cache.get(key)
    if ids is None:
        ids, _ = asset.find_joints(r"^(?!passive_).*")
        cache[key] = ids
    return ids


def _servo_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_pos[:, _servo_joint_ids(env, asset)]


def _servo_joint_vel(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.joint_vel[:, _servo_joint_ids(env, asset)]


def _servo_default_joint_pos(env: "ManagerBasedRlEnv", asset: Entity) -> torch.Tensor:
    return asset.data.default_joint_pos[:, _servo_joint_ids(env, asset)]


def reset_with_forward_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float] = (0.3, 0.8),
    fraction_stages: list[dict] | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """以随机前向速度热启动一部分 reset 环境.

    机器人生成时已沿自身体前方向运动, 因此它首先发现高速滑行是什么感觉.
    该比例在训练过程中逐步降低, 迫使其从静止开始逐步挣得该速度.

    Args:
        env: RL 环境.
        env_ids: 待 reset 的环境索引张量.
        velocity_range: (min, max) 前向速度, 单位 m/s.
        fraction_stages: {"step": int, "fraction": float} 字典列表, 按 step 排序.
            使用当前训练步对应的 fraction.
            示例: [{"step":0,"fraction":0.8}, {"step":2000*24,"fraction":0.0}]
        asset_cfg: robot entity 配置.
    """
    if fraction_stages is None:
        fraction_stages = [{"step": 0, "fraction": 0.8}]

    # 从训练步确定当前 fraction
    step = env.common_step_counter
    fraction = fraction_stages[0]["fraction"]
    for stage in fraction_stages:
        if step >= stage["step"]:
            fraction = stage["fraction"]

    if len(env_ids) == 0 or fraction <= 0.0:
        return

    n_warmstart = max(1, int(len(env_ids) * fraction))
    perm = torch.randperm(len(env_ids), device=env.device)[:n_warmstart]
    warmstart_ids = env_ids[perm]

    lo, hi = velocity_range
    vx = lo + torch.rand(n_warmstart, device=env.device) * (hi - lo)

    # 仅由 yaw 构造水平前向方向 — 忽略 pitch/roll.
    # 重要: 从 qpos 读取四元数, 而非从 root_link_quat_w 读取.
    # root_link_quat_w 读取 xquat, 后者依赖 sim.forward() 才会更新.
    # 在 reset_base 将新 yaw 写入 qpos 后, xquat 仍是旧的 (上一 episode).
    # qpos 由 write_root_pose 立即更新, 因此始终是最新的.
    asset: Entity = env.scene[asset_cfg.name]
    qpos_q_adr = asset.data.indexing.free_joint_q_adr[3:7]  # quat indices in qpos
    q = asset.data.data.qpos[warmstart_ids][:, qpos_q_adr]  # (n, 4) [w, x, y, z]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward_world = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)

    velocities = torch.zeros(n_warmstart, 6, device=env.device)
    velocities[:, :3] = vx.unsqueeze(-1) * forward_world

    asset.write_root_link_velocity_to_sim(velocities, env_ids=warmstart_ids)

    # 旋转轮子以匹配前向速度 — 防止瞬间的无滑移制动.
    # 轮半径 = 0.0175 m (实测).
    # 4 个轮子均以 +ω 正转表示前进 (由 test_wheel_direction.py 验证).
    _WHEEL_RADIUS = 0.0175
    all_wheel_ids, _ = asset.find_joints(r"^passive_.*")

    if all_wheel_ids:
        joint_pos = asset.data.joint_pos[warmstart_ids].clone()
        joint_vel = asset.data.joint_vel[warmstart_ids].clone()
        omega = vx / _WHEEL_RADIUS  # (n,) rad/s, positive = forward
        joint_vel[:, all_wheel_ids] = omega.unsqueeze(-1).expand(-1, len(all_wheel_ids))
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=warmstart_ids)


def reset_action_history(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """为正在 reset 的环境重置缓存的 action history.

    这对 action rate 和加速度惩罚项至关重要.

    此函数应在 post_reset 回调中或 episode 终止时调用.

    Args:
        env: 环境
        env_ids: 待 reset 的环境索引
        asset_cfg: Asset 配置
    """
    if len(env_ids) == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]

    # 重置腿部 action rate 缓存
    if hasattr(env, "_prev_leg_actions"):
        # 设为当前 action (若尚无 action 则置零)
        if hasattr(env, "action_manager") and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            env._prev_leg_actions[env_ids] = env.action_manager.action[env_ids][:, leg_joint_indices]
        else:
            env._prev_leg_actions[env_ids] = 0.0

    # 重置颈部 action rate 缓存
    if hasattr(env, "_prev_neck_actions"):
        if hasattr(env, "action_manager") and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            env._prev_neck_actions[env_ids] = env.action_manager.action[env_ids][:, neck_joint_indices]
        else:
            env._prev_neck_actions[env_ids] = 0.0

    # 重置腿部 action 加速度缓存
    if hasattr(env, "_prev_leg_actions_for_acc"):
        if hasattr(env, "action_manager") and env.action_manager.action is not None:
            leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
            current_action = env.action_manager.action[env_ids][:, leg_joint_indices]
            env._prev_leg_actions_for_acc[env_ids] = current_action
            env._prev_prev_leg_actions_for_acc[env_ids] = current_action
        else:
            env._prev_leg_actions_for_acc[env_ids] = 0.0
            env._prev_prev_leg_actions_for_acc[env_ids] = 0.0

    # 重置颈部 action 加速度缓存
    if hasattr(env, "_prev_neck_actions_for_acc"):
        if hasattr(env, "action_manager") and env.action_manager.action is not None:
            neck_joint_indices = list(range(5, 9))
            current_action = env.action_manager.action[env_ids][:, neck_joint_indices]
            env._prev_neck_actions_for_acc[env_ids] = current_action
            env._prev_prev_neck_actions_for_acc[env_ids] = current_action
        else:
            env._prev_neck_actions_for_acc[env_ids] = 0.0
            env._prev_prev_neck_actions_for_acc[env_ids] = 0.0

    # 重置关节速度缓存 (用于关节加速度)
    if hasattr(asset.data, "_prev_joint_vel"):
        # 取 reset 环境的当前关节速度
        joint_vel = asset.data.joint_vel[env_ids, :][:, asset_cfg.joint_ids]
        asset.data._prev_joint_vel[env_ids] = joint_vel

    # 重置接触频率跟踪
    if hasattr(env, "_contact_change_count"):
        env._contact_change_count[env_ids] = 0.0
    if hasattr(env, "_contact_change_timer"):
        env._contact_change_timer[env_ids] = 0.0
    if hasattr(env, "_prev_contacts_for_freq") and "feet_ground_contact" in env.scene.sensors:
        contacts = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2]
        env._prev_contacts_for_freq[env_ids] = contacts

    # 重置脚力平滑度跟踪
    if hasattr(env, "_prev_foot_forces") and "feet_ground_contact" in env.scene.sensors:
        forces = env.scene.sensors["feet_ground_contact"].data.found[env_ids, :2].squeeze(-1)
        env._prev_foot_forces[env_ids] = forces

    # 重置 actuator 力矩变化率跟踪
    if hasattr(env, "_prev_actuator_forces"):
        env._prev_actuator_forces[env_ids] = asset.data.actuator_force[env_ids].clone()


def joint_accelerations_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """用 L2 平方范数惩罚关节加速度.

    关节加速度通过关节速度的有限差分计算.

    Args:
        env: 环境
        asset_cfg: Asset 配置

    Returns:
        形状为 (num_envs,) 的惩罚张量 — 关节加速度平方和
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 取当前关节速度
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # 取上一帧关节速度 (存储在 asset data 中)
    # 注意: 假设环境存储了上一帧关节速度
    if not hasattr(asset.data, "_prev_joint_vel"):
        # 首次调用时初始化
        asset.data._prev_joint_vel = joint_vel.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # 用有限差分计算关节加速度
    dt = env.step_dt
    joint_acc = (joint_vel - asset.data._prev_joint_vel) / dt

    # 保存当前速度供下一帧使用
    asset.data._prev_joint_vel = joint_vel.clone()

    # 返回 L2 平方范数
    return torch.sum(torch.square(joint_acc), dim=1)


def leg_action_rate_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """惩罚腿部 action 的变化率 (action_t - action_{t-1}).

    腿部关节索引为 0-4 和 9-13 (共 10 个关节).

    Args:
        env: 环境
        asset_cfg: Asset 配置

    Returns:
        形状为 (num_envs,) 的惩罚张量
    """
    # 取腿部关节索引
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    # 仅取腿部关节的当前和上一帧 action
    # action 存储在 env 中 (假设 action 可用)
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)

    # 取关节位置 action
    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, "_prev_leg_actions"):
        env._prev_leg_actions = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = leg_actions - env._prev_leg_actions
    env._prev_leg_actions = leg_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def neck_action_rate_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """惩罚颈部 action 的变化率 (action_t - action_{t-1}).

    颈部关节索引为 5-8 (共 4 个关节).

    Args:
        env: 环境
        asset_cfg: Asset 配置

    Returns:
        形状为 (num_envs,) 的惩罚张量
    """
    # 取颈部关节索引
    neck_joint_indices = list(range(5, 9))

    # 仅取颈部关节的当前和上一帧 action
    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, "_prev_neck_actions"):
        env._prev_neck_actions = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_rate = neck_actions - env._prev_neck_actions
    env._prev_neck_actions = neck_actions.clone()

    return torch.sum(torch.square(action_rate), dim=1)


def leg_action_acceleration_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """惩罚腿部 action 的加速度 (action_t - 2*action_{t-1} + action_{t-2}).

    腿部关节索引为 0-4 和 9-13 (共 10 个关节).

    Args:
        env: 环境
        asset_cfg: Asset 配置

    Returns:
        形状为 (num_envs,) 的惩罚张量
    """
    # 取腿部关节索引
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))

    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    leg_actions = actions[:, leg_joint_indices]

    if not hasattr(env, "_prev_leg_actions_for_acc"):
        env._prev_leg_actions_for_acc = leg_actions.clone()
        env._prev_prev_leg_actions_for_acc = leg_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = leg_actions - 2 * env._prev_leg_actions_for_acc + env._prev_prev_leg_actions_for_acc

    env._prev_prev_leg_actions_for_acc = env._prev_leg_actions_for_acc.clone()
    env._prev_leg_actions_for_acc = leg_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def neck_action_acceleration_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """惩罚颈部 action 的加速度 (action_t - 2*action_{t-1} + action_{t-2}).

    颈部关节索引为 5-8 (共 4 个关节).

    Args:
        env: 环境
        asset_cfg: Asset 配置

    Returns:
        形状为 (num_envs,) 的惩罚张量
    """
    # 取颈部关节索引
    neck_joint_indices = list(range(5, 9))

    if not hasattr(env, "action_manager"):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_manager.action
    if actions.shape[1] < 14:
        return torch.zeros(env.num_envs, device=env.device)

    neck_actions = actions[:, neck_joint_indices]

    if not hasattr(env, "_prev_neck_actions_for_acc"):
        env._prev_neck_actions_for_acc = neck_actions.clone()
        env._prev_prev_neck_actions_for_acc = neck_actions.clone()
        return torch.zeros(env.num_envs, device=env.device)

    action_acc = neck_actions - 2 * env._prev_neck_actions_for_acc + env._prev_prev_neck_actions_for_acc

    env._prev_prev_neck_actions_for_acc = env._prev_neck_actions_for_acc.clone()
    env._prev_neck_actions_for_acc = neck_actions.clone()

    return torch.sum(torch.square(action_acc), dim=1)


def _fallen_mask(
    env: ManagerBasedRlEnv,
    asset,
    gate_z_below: float,
    gate_tilt_above_deg: float,
) -> torch.Tensor:
    """逐环境 float 掩码: 1.0 表示机器人算作摔倒 — trunk 高度低于 ``gate_z_below`` 或倾斜超过
    ``gate_tilt_above_deg``.

    用于对 recovery reward 进行门控, 使其仅在真正摔倒时引导, 在正常行走时贡献恰好为零
    (无行走税 / bounce farming).
    """
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    quat = asset.data.root_link_quat_w
    # cos(tilt) = R22 = 1 - 2(qx² + qy²)
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = (z < gate_z_below) | (cos_tilt < math.cos(math.radians(gate_tilt_above_deg)))
    return fallen.float()


def feet_air_time_upright(
    env: ManagerBasedRlEnv,
    gate_tilt_above_deg: float = 40.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    **air_time_kwargs,
) -> torch.Tensor:
    """velocity 模板的 feet_air_time, 在摔倒 (倾斜 > gate) 时归零.

    velstand: 一个躯干趴地的机器人仍能通过 air-time 窗口有节奏地蹬腿 — 这就是观察到的
    "趴地抖腿" 漏洞.Air time 仅在直立时才有意义.
    """
    from mjlab.tasks.velocity.mdp import feet_air_time as _template_air_time

    reward = _template_air_time(env, **air_time_kwargs)
    asset: Entity = env.scene[asset_cfg.name]
    upright = 1.0 - _fallen_mask(env, asset, 0.0, gate_tilt_above_deg)
    return reward * upright


def upright_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """基于势能的直立塑造: 每步 Δcos(tilt).

    为向直立的进展付费, 为向摔倒的进展收费, 保持任何姿态都恰好付费零 —
    因此任何状态都无法 farm 它 (它所替代的门控状态 reward 曾被从坐、趴地、
    以及一个 head-tripod 倾斜中 farm, 跨越三次 velstand run).基于势能的
    塑造是 policy-invariant 的 (Ng et al.): 它加速 recovery 的学习, 而不
    创建新的最优.一次完整的趴地→站立 recovery 共收集 Δ≈+1 (× 权重);
    一次摔倒在下行过程中花费相同.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    cos_tilt = torch.nan_to_num(1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), nan=1.0)
    if not hasattr(env, "_upright_potential_prev"):
        env._upright_potential_prev = cos_tilt.clone()
    # 刚 reset 的环境: 不产生来自上一 episode 姿态的虚假 delta.
    fresh = env.episode_length_buf <= 1
    env._upright_potential_prev[fresh] = cos_tilt[fresh]
    delta = cos_tilt - env._upright_potential_prev
    env._upright_potential_prev = cos_tilt.clone()
    return delta


def height_progress(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ceiling: float = 0.115,
) -> torch.Tensor:
    """基于势能的高度塑造: 每步 Δ min(trunk z, ceiling).

    ``upright_progress`` 的 z 轴伴侣 (velstand 蹲伏终点教训): recovery 的最后一段 —
    从深蹲中伸直膝盖 — 主要是一个 HEIGHT 变化伴随轻微倾斜, 恰好是 Gaussian 直立/
    姿态 reward 平坦且 Δcos(tilt) 极小的区域.上升付费, 下降收费, 保持付费零,
    因此步态起伏净零, 任何状态都无法 farm 它.上限为 ``ceiling`` (略低于全站 trunk
    z ≈ 0.117), 因此在站姿高度以上跳跃不额外付费.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    pot = torch.clamp(z, max=ceiling)
    if not hasattr(env, "_height_potential_prev"):
        env._height_potential_prev = pot.clone()
    fresh = env.episode_length_buf <= 1
    env._height_potential_prev[fresh] = pot[fresh]
    delta = pot - env._height_potential_prev
    env._height_potential_prev = pot.clone()
    return delta


def fallen_state_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_tilt_above_deg: float = 40.0,
    release_tilt_below_deg: float | None = None,
    release_z_above: float | None = None,
) -> torch.Tensor:
    """摔倒期间为 1.0 (权重取负): 对停留的扁平 per-step 税.

    没有它的话, 趴着不动约 0/step, 而尝试 recovery 要付 action-rate/torque
    惩罚 — 等待 fallen_too_long 回收曾是理性策略.(对坏状态的惩罚是安全的;
    被门控在坏状态上的 POSITIVE reward 才会被 farm.)

    设置 ``release_*`` 后, 该税带 HYSTERESIS (velstand 蹲伏终点教训):
    一次摔倒会激活它, 它持续付费直到机器人真正站起 (倾斜 < release_tilt
    且 z > release_z), 而非仅在 arming gate 之下.没有它的话, 刚好在 40°
    gate 以下的蹲姿是一个零成本静止状态 — recovery 学会停在那里而不是
    完成站立.仅在真正摔倒时激活, 因此步态周期的倾斜摆动永不被征税.
    """
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, 0.0, gate_tilt_above_deg).bool()
    if release_tilt_below_deg is None:
        return fallen.float()
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    up = cos_tilt > math.cos(math.radians(release_tilt_below_deg))
    if release_z_above is not None:
        up &= z > release_z_above
    if not hasattr(env, "_fallen_tax_armed"):
        env._fallen_tax_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fresh = env.episode_length_buf <= 1
    env._fallen_tax_armed[fresh] = False
    env._fallen_tax_armed |= fallen
    env._fallen_tax_armed &= ~up
    return env._fallen_tax_armed.float()


def recovery_success(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    fallen_tilt_deg: float = 40.0,
    min_fallen_s: float = 0.5,
    up_tilt_deg: float = 25.0,
    up_z: float = 0.105,
) -> torch.Tensor:
    """对已完成 recovery 的一次性奖励: 在一个环境已摔倒 (倾斜 > fallen_tilt 持续 ≥ min_fallen_s)
    变为真正直立 (倾斜 < up_tilt 且 trunk z > up_z) 的那一帧触发.

    Hysteresis: 仅通过再次摔倒重新激活, 因此在 gate 附近振荡不付费.
    提供密集门控项所缺乏的稀疏但强烈的终点梯度.
    """
    asset: Entity = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    fallen = cos_tilt < math.cos(math.radians(fallen_tilt_deg))
    up = (cos_tilt > math.cos(math.radians(up_tilt_deg))) & (z > up_z)
    if not hasattr(env, "_recovery_fallen_s"):
        env._recovery_fallen_s = torch.zeros(env.num_envs, device=env.device)
        env._recovery_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fresh = env.episode_length_buf <= 1
    env._recovery_fallen_s[fresh] = 0.0
    env._recovery_armed[fresh] = False
    env._recovery_fallen_s = torch.where(
        fallen,
        env._recovery_fallen_s + env.step_dt,
        torch.zeros_like(env._recovery_fallen_s),
    )
    env._recovery_armed |= env._recovery_fallen_s >= min_fallen_s
    fired = env._recovery_armed & up
    env._recovery_armed &= ~fired
    return fired.float()


def body_upright_linear(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
) -> torch.Tensor:
    """对身体直立程度的线性 reward — 在每个倾斜角提供梯度.

    完全直立时返回 +1, 水平时 (趴/仰) 返回 0, 倒立时返回 -1.
    与 flat_orientation (Gaussian) 不同, 它在所有位置都有非零梯度, 因此
    机器人即使从趴地起步也有信号转向直立.

    计算方式为 body 局部 Z 轴在世界坐标系下的 z 分量, 对四元数
    [w, x, y, z] 等于 R[2,2] = 1 - 2*(qx² + qy²).
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4): [w, x, y, z]
    qx = quat[:, 1]
    qy = quat[:, 2]
    reward = 1.0 - 2.0 * (qx * qx + qy * qy)
    if gate_z_below is not None:
        # recovery 门控变体 (velstand): 仅在摔倒时激活, 在正常行走时恰好
        # 为零, 因此不会稀释跟踪 reward.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def body_upright_gaussian(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.1,
) -> torch.Tensor:
    """对倾斜幅度的 Gaussian reward — 向完全垂直的强烈拉力.

    补充 ``body_upright_linear`` (即 ``cos(tilt)``, 其梯度 ``sin(tilt)``
    在目标处 *消失*).本 Gaussian 的梯度在接近垂直时非零, 随着远离
    而递减, 因此在线性版本最弱的区间产生强烈的差异化拉力.

    使用 ``2*(qx² + qy²) = 1 - cos(tilt) ≈ tilt²/2`` 作为倾斜平方的
    近似, 并施加 ``exp(-tilt²/std²)``.默认 std=0.1 rad ≈ 5.7°.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)  # ≈ 1 − cos(tilt); 小角度: tilt²/2
    return torch.exp(-tilt_sq / (std * std))


def upright_gaussian_at_height(
    env: ManagerBasedRlEnv,
    std: float,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """``body_upright_gaussian`` 按 trunk z 的 smoothstep 加权.

    当 ``z >= height_high`` 时给予完整 Gaussian 直立 reward, 当 ``z <= height_low``
    时为零, 之间 smoothstep.当直立激励应仅在目标站立高度生效时使用 — 否则策略
    可能找到 "蹲低且垂直" 的局部最优, 收集直立 reward 却从不上升.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_g = torch.exp(-tilt_sq / (std * std))
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright_g * smooth


def body_ang_vel_at_height(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    tilt_full_deg: float | None = None,
    tilt_zero_deg: float = 45.0,
) -> torch.Tensor:
    """trunk ``sum(ω_xy²)`` 惩罚, 由 trunk z (以及可选的倾斜) 门控.

    高度门控的到达阻尼: 在 ``height_low`` 以下为零 (地面 recovery —
    翻转/滚动需要大的躯干旋转, 必须保持自由), 在 ``height_high`` 以上为
    满.与 mjlab 的 body_angular_velocity_penalty 公式相同 (世界系 ω_xy,
    z 旋转自由), 但返回门控后的 POSITIVE cost; 使用负权重.

    ``tilt_full_deg`` (可选但强烈推荐): 额外按倾斜门控 — 仅当倾斜 ≤
    tilt_full_deg 时为满 cost, 当 ≥ tilt_zero_deg 时为零, 之间 smoothstep.
    教训 (2026-07 损坏前向 recovery 的 run): 仅用高度门控时, 弯腰起立
    的最后伸直 (倾斜 60°→0 发生在 z gate 内部) 本身就是一次大的躯干
    旋转 — 对其征税恰好在终点前筑起 reward 墙, 策略停在 gate 以下弯
    腰而非完成.加了倾斜门控后, 接近垂直的过程是自由的; 只有在垂直
    附近的残余摆动 (过冲→倾斜→重试振荡) 被阻尼.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    cost = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if tilt_full_deg is not None:
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        s = torch.clamp(
            (tilt_zero_deg - tilt_deg) / max(tilt_zero_deg - tilt_full_deg, 1e-6),
            0.0,
            1.0,
        )
        gate = gate * (s * s * (3.0 - 2.0 * s))
    return cost * gate


def standing_composite_score(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    target_overrides: dict | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """平滑的乘性目标状态评分 (三个 Gaussian 的乘积).

    返回 ``height_score * upright_score * pose_score``, 每项 ∈ [0, 1].由于
    各因子 *相乘*, 任何一项的不足都会使整个 reward 崩溃 — 策略无法通过 3 项中
    做对 2 项来获得 80% 的分数.梯度处处非零, 因此该评分在上升过程中也有效
    (不像二元 bonus 只在目标处生效).

    用于打破 Nash 均衡妥协 (例如, "在正确高度处的倾斜躯干" basin 满足
    加性 reward 的部分和).
    """
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    height_score = torch.exp(-(((z - target_height) / height_std) ** 2))

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    tilt_sq = 2.0 * (qx * qx + qy * qy)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err_sq = ((joint_pos - target) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    return height_score * upright_score * pose_score


def standing_success_bonus(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_tol: float,
    upright_threshold: float,
    pose_tol: float,
    joint_indices: list,
    target_overrides: dict | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """二元 bonus: 当且仅当高度、直立和姿态均在容差内时为 1.0.

    创建一个离散的目标状态吸引子, 这是基于梯度的姿态/直立/高度 reward 无法
    单独完全匹配的.周围的妥协 (倾斜躯干以平衡前倾 CoM、停在目标 z 以下 1cm 等)
    收集部分梯度信用但 bonus 为零 — bonus 仅在真正的目标状态可用, 因此一旦
    其余 reward 将策略带到附近, 它改变策略的相对偏好.
    """
    asset = env.scene[asset_cfg.name]

    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    height_ok = (z - target_height).abs() <= height_tol

    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    upright_ok = upright >= upright_threshold

    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    pose_err = (joint_pos - target).abs().max(dim=-1).values  # 最紧的关节
    pose_ok = pose_err <= pose_tol

    return (height_ok & upright_ok & pose_ok).float()


def com_upward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_height: float = 0.08,
    gate_z_below: float | None = None,
    gate_tilt_above_deg: float = 40.0,
    max_vz: float | None = None,
) -> torch.Tensor:
    """奖励向上 CoM 速度以激励动态站起动作.

    由高度门控: 仅在 CoM 低于 ``max_height`` (站立目标) 时激活.一旦站起,
    reward 为零, 因此机器人没有继续蹲着以 farm 向上速度 reward 的激励.

    ``max_vz`` (可选): 限制被奖励的速度.不加限时, reward 与 vz 成正比,
    这对爆炸性起跳每步付更多 — 一种暴力起跳激励.加了限制后, 任何达到
    max_vz 的上升都获得相同 reward, 因此达到上限的最温和上升是最优的
    (|a_z| 惩罚随后选择平滑的那一个).bootstrap 性质保留: 任何向上
    运动仍立即付费.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # nan_to_num: MuJoCo 在接触不稳定时可能产生 NaN; 视为 z=0
    com_z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    below_target = (com_z < max_height).float()
    reward = torch.clamp(vz, min=0.0, max=max_vz) * below_target
    if gate_z_below is not None:
        # recovery 门控 (velstand): 不加门控的话, 步态中 trunk 穿过 max_height
        # 时的下沉-上升都会付费 → bounce 激励.
        reward = reward * _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def fallen_too_long(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    gate_z_below: float = 0.10,
    gate_tilt_above_deg: float = 40.0,
    max_duration_s: float = 5.0,
) -> torch.Tensor:
    """终止持续摔倒超过 ``max_duration_s`` 的环境.

    对混合行走与摔倒 recovery 的环境 (velstand): fell_over 终止被 curriculum
    禁用以便策略可以尝试 recovery, 但没有兜底的话, 一次失败的 recovery 会在
    整个 20 s episode 中 farm recovery-reward, 使行走数据饥饿 (审计: ~25%
    行走份额).本函数给每次摔倒一个公平的 recovery 窗口, 然后回收环境.
    """
    asset: Entity = env.scene[asset_cfg.name]
    fallen = _fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg).bool()
    if not hasattr(env, "_fallen_timer_s"):
        env._fallen_timer_s = torch.zeros(env.num_envs, device=env.device)
    # 刚 reset 的环境以干净的计时器开始.
    env._fallen_timer_s[env.episode_length_buf <= 1] = 0.0
    env._fallen_timer_s = torch.where(fallen, env._fallen_timer_s + env.step_dt, torch.zeros_like(env._fallen_timer_s))
    return env._fallen_timer_s >= max_duration_s


def robot_state_is_nan(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """终止 MuJoCo 产生 NaN 关节位置的环境.

    MuJoCo 的接触求解器在极端穿透或冲量下 (例如机器人高速落地) 可能溢出为
    NaN.NaN 仿真状态会传播到观测, 腐蚀 policy 网络权重.

    立即终止在级联扩散之前 reset 环境:
    - 返回给 runner 的观测来自有效的 reset 状态.
    - 避免后续步的 NaN reward.

    注意: 本终止步的 reward 仍可能因仿真而为 NaN; mjlab 在 reset 之前
    计算 reward (见 manager_based_rl_env.py step()).我们的自定义 reward
    函数内部用 nan_to_num 防护 NaN, 但标准 mjlab reward 在此处仍可能
    为 NaN.一个 NaN reward 是可容忍的, 因为 done=True 阻止它通过 GAE
    向后传播.

    覆盖整个物理状态, 不仅仅是 joint_pos: 接触发散常使 base FREE-JOINT
    (位置/方向/速度) 或被动 ROUES 爆炸, 而非驱动关节.这些量喂给
    critic 的 obs 项 (base_lin_vel, base_ang_vel, projected_gravity,
    wheel_vel); 若不监视, 环境不 reset 而 NaN 到达 obs → rsl_rl 的
    check_nan 会杀死整个训练.测试非有限性 (NaN 和 inf, inf 在
    projected_gravity 归一化时下游变为 NaN).
    """
    asset: Entity = env.scene[asset_cfg.name]
    d = asset.data
    bad = ~torch.isfinite(d.joint_pos).all(dim=1)
    bad |= ~torch.isfinite(d.joint_vel).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_pos_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_quat_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_lin_vel_w).all(dim=1)
    bad |= ~torch.isfinite(d.root_link_ang_vel_w).all(dim=1)

    # 接触力可能比 qpos/qvel 早一步爆掉: MuJoCo 将退化接触解析为 inf/NaN
    # 冲量, 而积分状态仍然有限.该力喂给 critic-only 的 `foot_contact_forces`
    # obs (sign(F)*log1p(|F|)), 上面的状态检查不覆盖它 — 因此环境未 reset,
    # NaN 到达 runner 的 check_nan, 杀死整个 run (2026-08-21 崩溃,
    # Velocity2-Rough-Backlash 带 hfield 斜坡).
    for name in sensor_names:
        if name not in env.scene.sensors:
            continue
        force = getattr(env.scene.sensors[name].data, "force", None)
        if force is not None:
            bad |= ~torch.isfinite(force).flatten(start_dim=1).all(dim=1)
    return bad


def root_height_below(
    env: ManagerBasedRlEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """当 trunk 在世界 z 下降低于 ``min_height`` 时终止.

    由 roller_slope 用作 "坠入虚空": 地形在坡底有出口平地, 因此正常下坡
    永远不会低于最低出口平地的水平.选择 min_height 低于该水平 => 终止仅在
    机器人脱离实地坠入虚空时触发.与机器人的方向无关, 也与坡道的
    精确几何 (长度/坡度) 无关.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < min_height


def descent_speed_reward(
    env: ManagerBasedRlEnv,
    cap: float = 0.8,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励沿坡向下的前进速度 (世界 +x).

    坡道沿 +x 下行, 因此世界 x 方向的线速度衡量下坡进度.上限为 ``cap``
    m/s: 鼓励顺势滑行而非加速冲下.当机器人后退/上坡 (vx < 0) 时为零.
    没有此 reward, 最优解是原地不动保持直立 (机器人 "刹车" 而非滑行).
    NaN-safe.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(vx, min=0.0, max=cap)


def reset_rolling_entry(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    speed_range: tuple = (0.25, 0.45),
    wheel_radius: float = 0.0175,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """无滑移的滚动起步 (给轮子初速度).

    每个环境抽取一个前向速度 v; 设定 base 的线速度 (世界 x) = v 且
    4 个被动轮的旋转速度 = v / r, 因此 ω·r = v => 接触点零滑移.避免
    旧版仅推 base 的冲击 (base 动、轮不动 = 第一步的剧烈滑移).
    应在 reset_base 之后执行 (后者放置 base; 不再给它 velocity_range).
    """
    asset: Entity = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = int(env_ids.shape[0])
    lo, hi = speed_range
    v = torch.rand(n, device=env.device) * (hi - lo) + lo  # (n,) 前向速度

    # base 速度 (世界系): 仅 +x.
    root_vel = torch.zeros(n, 6, device=env.device)
    root_vel[:, 0] = v
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

    # 4 个被动轮的旋转 = v / r (正 = 前进, 见 wheel_speed).
    wheel_ids = []
    for name in (
        "passive_LF_?wheel",
        "passive_LR_?wheel",
        "passive_RF_?wheel",
        "passive_RR_?wheel",
    ):
        ids, _ = asset.find_joints(name)
        wheel_ids.append(ids[0])
    wheel_ids_t = torch.tensor(wheel_ids, device=env.device)
    omega = (v / wheel_radius).unsqueeze(1).repeat(1, len(wheel_ids))  # (n, 4)
    asset.write_joint_velocity_to_sim(omega, joint_ids=wheel_ids_t, env_ids=env_ids)


def wheel_glide_reward(
    env: ManagerBasedRlEnv,
    cap_speed: float = 0.35,
    wheel_radius: float = 0.0175,
) -> torch.Tensor:
    """奖励向前的轮子滚动 (滑行), 带上限.

    与 descent_speed (BASE 速度, 可通过 "跑"/推获得) 不同, 这里奖励
    被动轮的旋转 = 真正的滚动滑行.与任何 command 无关 (坡道任务的
    command 为零: 滑行来自重力).上限为 ``cap_speed`` (滚动速度 m/s)
    -> 超过此速度无任何激励; 轮子后退 (上坡) 时为零.NaN-safe.
    """
    asset: Entity = env.scene["robot"]
    lf, _ = asset.find_joints("passive_LF_?wheel")
    lr, _ = asset.find_joints("passive_LR_?wheel")
    rf, _ = asset.find_joints("passive_RF_?wheel")
    rr, _ = asset.find_joints("passive_RR_?wheel")
    vel = asset.data.joint_vel
    # 4 个轮子正转表示前进 (见 wheel_speed_reward).
    omega = (vel[:, lf[0]] + vel[:, lr[0]] + vel[:, rf[0]] + vel[:, rr[0]]) / 4.0
    speed = torch.nan_to_num(omega * wheel_radius, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(speed, min=0.0, max=cap_speed)


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """保持存活 (未终止) 的 reward.

    Args:
        env: 环境

    Returns:
        形状为 (num_envs,) 的 reward 张量 — 所有环境为 1
    """
    return torch.ones(env.num_envs, device=env.device)


def com_height_target(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    target_height_min: float = 0.1,
    target_height_max: float = 0.15,
) -> torch.Tensor:
    """保持质心在目标高度范围内的 reward.

    在范围内返回正 reward, 超出范围返回负惩罚.

    Args:
        env: 环境
        asset_cfg: Asset 配置
        target_height_min: CoM 目标高度下限 (米)
        target_height_max: CoM 目标高度上限 (米)

    Returns:
        形状为 (num_envs,) 的 reward 张量
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 地形生成原点以上的高度 (世界 z 减去地形 z).
    # env_origins[:, 2] 对平地而言为 0, 因此无条件安全.
    # nan_to_num: MuJoCo 在接触不稳定时可能产生 NaN; 视为 z=0
    # 使惩罚有限 (较小, 因为 0 接近目标范围).
    com_height = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)

    # 在范围内时奖励, 超出范围时惩罚
    # 使用平滑惩罚, 随距离范围的距离二次增长
    below_min = com_height < target_height_min
    above_max = com_height > target_height_max
    in_range = ~(below_min | above_max)

    # 计算超出范围的惩罚
    penalty_below = torch.square(com_height - target_height_min) * below_min.float()
    penalty_above = torch.square(com_height - target_height_max) * above_max.float()

    # reward: 在范围内为 +1, 超出范围为 -平方距离
    reward = in_range.float() - (penalty_below + penalty_above)

    return reward


def crouch_height_target(
    phase: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
) -> torch.Tensor:
    """沿 phase [0,1) 的 "梯形" trunk 高度目标.

    phase ∈ [0, hold_lo)      : 下沉   height_high -> height_low
    phase ∈ [hold_lo, hold_hi): 平台   height_low   (蹲姿滑行)
    phase ∈ [hold_hi, 1.0)    : 上升   height_low  -> height_high

    Args:
        phase: (B,) 逐环境 phase, 在 [0, 1) 内.
        height_low: 蹲姿 trunk 高度 (m).
        height_high: 站姿 trunk 高度 (m).
        hold_lo: 低端平台的下界 (phase 分数).
        hold_hi: 低端平台的上界 (phase 分数).

    Returns:
        (B,) 目标高度, 单位米.
    """
    descend = phase < hold_lo
    hold = (phase >= hold_lo) & (phase < hold_hi)

    frac_d = phase / hold_lo
    t_descend = height_high + (height_low - height_high) * frac_d

    t_hold = torch.full_like(phase, height_low)

    frac_r = (phase - hold_hi) / (1.0 - hold_hi)
    t_rise = height_low + (height_high - height_low) * frac_r

    return torch.where(descend, t_descend, torch.where(hold, t_hold, t_rise))


def crouch_glide_reward_from_values(
    com_height: torch.Tensor,
    cmd_cos: torch.Tensor,
    cmd_sin: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
) -> torch.Tensor:
    """高度目标跟踪的 Gaussian reward (纯函数).

    从 [cos, sin] 解码 phase, 然后将实测高度与梯形目标比较.
    返回 exp(-((h - target)/std)^2) ∈ (0, 1].
    """
    phase = (torch.atan2(cmd_sin, cmd_cos) / (2 * torch.pi)) % 1.0
    target = crouch_height_target(phase, height_low, height_high, hold_lo, hold_hi)
    return torch.exp(-(((com_height - target) / std) ** 2))


def crouch_glide_height_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    height_low: float = 0.075,
    height_high: float = 0.11,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """主要 reward: 沿 phase 跟踪 trunk 高度目标.

    CoM 高度的计算与 `com_height_target` 相同 (世界 z 减去地形原点, nan->0).
    phase 来自 GroundPick command.
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_height = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    cmd = env.command_manager.get_command(command_name)
    return crouch_glide_reward_from_values(
        com_height,
        cmd[:, 0],
        cmd[:, 1],
        height_low,
        height_high,
        hold_lo,
        hold_hi,
        std,
    )


def forward_speed_reward(
    env: ManagerBasedRlEnv,
    vel_ref: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励 trunk 前向速度 (保持冲量 / 不刹车).

    与 command 无关 (command 携带 phase, 不携带速度).
    tanh(clamp(vx, 0)/vel_ref) → 在 ~1 处饱和, 从不奖励后退.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = asset.data.root_link_lin_vel_b[:, 0]
    return torch.tanh(torch.clamp(vx, min=0.0) / vel_ref)


def crouch_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """沿 phase [0,1) 的 0..1 混合 — 0 = 站姿, 1 = 蹲姿.

    [0, descent_end)      : 0 -> 1  (下蹲)
    [descent_end, hold_end): 1      (低 / 蹲)
    [hold_end, rise_end)  : 1 -> 0  (起立)
    [rise_end, 1.0)       : 0       (高 / 站, 静止)
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def _crouch_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    crouch_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    stand_pose: dict | None = None,
):
    """phase 插值蹲姿的 (cur, target) 关节张量.

    target 按关节在 STAND <-> crouch_pose 之间由 4 段混合 b(phase) ∈ [0,1]
    插值 (0 = 站, 1 = 蹲).STAND 为给定 `stand_pose`, 否则为模型 DEFAULT
    (HOME).关节按名称解析, 因此 roller 机器人上的被动轮交错不会
    偏移索引.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = crouch_pose_blend(phase, descent_end, hold_end, rise_end)  # (B,) 0..1

    names = list(crouch_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]  # (B,k)

    stand = default.clone()  # 源姿态
    if stand_pose:
        for j, n in enumerate(names):
            if n in stand_pose:
                stand[:, j] = stand_pose[n]
    crouch = torch.tensor([crouch_pose[n] for n in names], device=env.device, dtype=default.dtype).unsqueeze(0)  # (1,k)

    target = stand + blend.unsqueeze(-1) * (crouch - stand)  # (B,k)
    cur = asset.data.joint_pos[:, ids]  # (B,k)
    return cur, target


def crouch_glide_pose_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    crouch_pose: dict | None = None,
    stand_pose: dict | None = None,
    std: float = 0.4,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """与 phase 插值关节姿态 (站 <-> 蹲) 的 Gaussian 匹配.

    指令式 reward: 告诉机器人在每个 phase 应处的精确关节配置.重新站起
    (target = stand_pose) 与下蹲 (target = crouch_pose) 同样被奖励 — 构造上
    对称.
    """
    cur, target = _crouch_pose_error(
        env,
        asset_cfg,
        command_name,
        crouch_pose or {},
        descent_end,
        hold_end,
        rise_end,
        stand_pose,
    )
    return torch.exp(-(((cur - target) / std) ** 2)).mean(dim=-1)


def crouch_glide_pose_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    crouch_pose: dict | None = None,
    stand_pose: dict | None = None,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """向 phase 插值蹲姿的 L1 bootstrap (负惩罚).

    处处梯度恒定 — 即使在上方 Gaussian 因远离目标而饱和到 ~0 时, 仍给
    策略指向目标姿态的方向.
    """
    cur, target = _crouch_pose_error(
        env,
        asset_cfg,
        command_name,
        crouch_pose or {},
        descent_end,
        hold_end,
        rise_end,
        stand_pose,
    )
    return -(cur - target).abs().mean(dim=-1)


def crouch_forward_lean(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pitch: float = 0.08,
    std: float = 0.1,
    descent_end: float = 0.10,
    hold_end: float = 0.50,
    rise_end: float = 0.60,
    asset_cfg: SceneEntityCfg = _TRUNK_BASE_ASSET_CFG,
) -> torch.Tensor:
    """蹲姿期间 trunk 的轻微前倾 (由蹲姿 blend 门控).

    抵消髋部快速屈曲引发的后仰.pitch 近似 = projected_gravity_b[:,0]
    (正 = 向前, 已验证).gate (blend) 在下蹲+低位时为 1, 站立时为 0
    → 仅对蹲姿产生偏置.target_pitch 小 = "非常轻微".
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = crouch_pose_blend(phase, descent_end, hold_end, rise_end)
    lean = asset.data.projected_gravity_b[:, 0]
    return gate * torch.exp(-((lean - target_pitch) ** 2) / std**2)


def neck_joint_vel_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """惩罚颈部关节速度以保持头部稳定.

    颈部关节索引为 5-8 (共 4 个关节).

    Args:
            env: 环境
            asset_cfg: Asset 配置

    Returns:
            形状为 (num_envs,) 的惩罚张量
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 取颈部关节索引 (neck_pitch, head_pitch, head_yaw, head_roll).
    # servo 视图: passive_* 关节 (backlash, 轮子) 不会偏移索引.
    neck_joint_indices = list(range(5, 9))
    joint_vel = _servo_joint_vel(env, asset)
    neck_joint_vel = joint_vel[:, neck_joint_indices]

    # 返回颈部关节速度的 L2 平方范数
    return torch.sum(torch.square(neck_joint_vel), dim=1)


def leg_joint_vel_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """惩罚腿部关节速度以鼓励更平滑、更少动态的运动.

    腿部关节索引为 0-4 和 9-13 (共 10 个关节).

    Args:
        env: 环境
        asset_cfg: Asset 配置

    Returns:
        形状为 (num_envs,) 的惩罚张量
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 取腿部关节索引 (左髋-踝: 0-4, 右髋-踝: 9-13).
    # servo 视图: passive_* 关节 (backlash, 轮子) 不会偏移索引.
    leg_joint_indices = list(range(0, 5)) + list(range(9, 14))
    joint_vel = _servo_joint_vel(env, asset)
    leg_joint_vel = joint_vel[:, leg_joint_indices]

    # 返回腿部关节速度的 L2 平方范数
    return torch.sum(torch.square(leg_joint_vel), dim=1)


_NECK_JOINT_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(neck|head).*",))
_HIP_PITCH_KNEE_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*(hip_pitch|knee).*",))
_ROLLER_FEET_SITE_CFG = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def feet_flat_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG,
    sensor_name: str | None = None,
) -> torch.Tensor:
    """惩罚脚部 site 不与地面平行的程度.

    脚部 site 坐标系在平放时 Z+ 朝上.我们将单位重力向量 (朝下) 投影到
    每个脚部 site 的局部坐标系中.平放时, 重力映射为 site 系中的
    [0,0,-1] (xy=0, penalty=0).任何倾斜都会使 Z 偏离世界上方向, 产生
    非零 xy 分量.

    最大值 ≈ 每脚 2.0 (脚完全侧放), 总计 ≈ 4.0.

    给定 ``sensor_name`` 时, 每只脚的惩罚由该脚自身的地面接触 GATE:
    空中 (摆动) 脚可自由倾斜, 仅要求支撑刀片保持平放 (使其轮子持续
    抓地).没有这个 gate, 惩罚会惩罚步幅所需的 recovery 脚抬起 — 通过
    保持两刀片都平放 (即 swizzle) 来最小化.假设 site 顺序 (左, 右)
    与 sensor slot 顺序 (ankle_l_v1, ankle_r_v1) 一致 — 本模型中两者
    都是左在前.

    Bug 注意: 必须按环境用 dim=-1 归一化重力.不带 dim 的 torch.norm()
    会在所有 envs × 3 维上计算标量, 使向量量级 ~1/sqrt(num_envs)
    → 惩罚小 ~num_envs 倍.
    """
    import torch.nn.functional as F
    from mjlab.utils.lab_api.math import quat_apply_inverse

    asset: Entity = env.scene[asset_cfg.name]
    gravity_w_n = F.normalize(asset.data.gravity_vec_w, dim=-1)  # (B, 3), unit vector per env

    foot_quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, N_feet, 4)
    per_foot = torch.zeros(env.num_envs, foot_quats.shape[1], device=env.device)
    for i in range(foot_quats.shape[1]):
        proj = quat_apply_inverse(foot_quats[:, i, :], gravity_w_n)  # (B, 3)
        per_foot[:, i] = torch.sum(torch.square(proj[:, :2]), dim=1)  # xy² only

    if sensor_name is not None:
        from mjlab.sensor import ContactSensor

        sensor: ContactSensor = env.scene[sensor_name]
        contact_time = sensor.data.current_contact_time  # (B, N_feet)
        assert contact_time is not None
        per_foot = per_foot * (contact_time > 0.0).float()

    return per_foot.sum(dim=1)


def feet_tiptoe_alignment(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _ROLLER_FEET_SITE_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """奖励每个脚部 site 的局部 x 轴朝下 — 踮脚站姿.

    平放时, 脚部 site 的 x 轴大致朝前 (水平).脚向前俯 (脚跟起、脚尖
    落) 会使 x 转向世界 -Z.我们奖励脚 x 轴的 z 分量为 -1 (完全朝下).

    每只脚: alignment ∈ [-1, 1], 双脚求和 ∈ [-2, 2].

    由 |vel_cmd_xy| > command_threshold 门控, 使策略无需静止时踮脚 —
    仅在行走时.本任务不使用 feet_flat_penalty; 两者会冲突.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quats = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # (B, N, 4) [w, x, y, z]
    w, qx, qy, qz = quats[:, :, 0], quats[:, :, 1], quats[:, :, 2], quats[:, :, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)  # (B, N) — z-component of local x-axis in world
    alignment = (-x_axis_z).sum(dim=-1)  # +1 per foot when pointing straight down

    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.linalg.norm(cmd[:, :2], dim=1)
    active = (cmd_mag > command_threshold).float()
    return alignment * active


def hip_pitch_knee_vel_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _HIP_PITCH_KNEE_CFG,
) -> torch.Tensor:
    """惩罚 hip_pitch 和 knee 关节速度 (L2 平方).

    行走需要这些矢状面关节的快速振荡.滑行在横向上使用 hip_roll, 矢状运动
    最小.本函数惩罚振荡但不阻止静态平衡调整.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def neck_joint_pos_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _NECK_JOINT_CFG,
    pattern: str = r".*(neck|head).*",
) -> torch.Tensor:
    """惩罚颈/头部关节位置偏离默认值 (L2 平方).

    每次调用都使用 find_joints() 以避免在跨不同关节布局的机器人 (例如
    walk 机器人 vs rollers 机器人, 后者被动轮会偏移颈部索引) 复用同一
    SceneEntityCfg 单例时出现陈旧缓存索引.

    ``pattern`` 选择计入的关节 (默认: 整个颈部 + 头部).spin 任务传入
    一个 EXCLUDE `head_yaw` 的模式, 让头部在旋转启动时充当惯性飞轮.
    """
    asset: Entity = env.scene[asset_cfg.name]
    # 排除 passive_* 关节 (backlash 铰链也含 "neck"/"head").
    if not pattern.startswith(r"^(?!passive_)"):
        pattern = r"^(?!passive_)" + pattern.lstrip("^")
    joint_ids, _ = asset.find_joints(pattern)
    error = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.sum(torch.square(error), dim=1)


def joint_torques_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """惩罚 actuator 力 (力矩) 以鼓励能量高效运动.

    Args:
        env: 环境
        asset_cfg: Asset 配置

    Returns:
        形状为 (num_envs,) 的惩罚张量 — actuator 力平方和
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 取 actuator 力 (actuation 空间中的标量驱动)
    actuator_forces = asset.data.actuator_force

    # 返回 L2 平方范数
    return torch.sum(torch.square(actuator_forces), dim=1)


def joint_torque_rate_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚 actuator 力矩的变化率 (齿轮箱冲击的近似).

    机器人撞击地面、actuator 抵抗冲量时会出现突发力矩尖峰.惩罚该变化率
    鼓励软着陆和保护齿轮箱的平滑力过渡.

    返回与上一帧力矩差的平方和.
    """
    asset: Entity = env.scene[asset_cfg.name]
    current = asset.data.actuator_force  # (num_envs, num_actuators)

    if not hasattr(env, "_prev_actuator_forces"):
        env._prev_actuator_forces = current.clone()
        return torch.zeros(env.num_envs, device=env.device)

    rate = current - env._prev_actuator_forces
    env._prev_actuator_forces = current.clone()
    return torch.sum(torch.square(rate), dim=1)


def feet_grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """脚部接触地面的正 reward (0, +0.5 或 +1.0).

    使用接触 sensor 的 `found` 字段.对 feet_ground_contact sensor (有 2 个
    主脚 geom), `found` 形状为 (num_envs, 2), 每脚二元接触.求和并归一化
    到 [0, 1].
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (num_envs, num_feet) or (num_envs, 1)
    if found.dim() > 1:
        found = found.sum(dim=-1)  # collapse foot dimension
    return torch.clamp(found, 0.0, 2.0) / 2.0


def body_impact_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 1.0,
) -> torch.Tensor:
    """惩罚受保护身体部件在地面上超过阈值的接触力.

    用于阻止摔倒时将躯干壳或头部撞向地面.sensor 应覆盖相关 body 或
    subtree 且 reduce='netforce'.低于阈值的力免费; 超过后惩罚线性
    增长.

    Args:
        env: RL 环境.
        sensor_name: 一个 ContactSensorCfg 的名称, fields=("force",),
            reduce="netforce".
        threshold: 接触力 (N), 低于此值不施加惩罚.

    Returns:
        惩罚张量 (num_envs,) — 每步超阈值的 N.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force  # (num_envs, N_bodies, 3)
    total_force = forces.sum(dim=1)  # sum over bodies in the subtree
    force_mag = torch.norm(total_force, dim=1)
    return torch.clamp(force_mag - threshold, min=0.0)


def wheel_speed_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    wheel_radius: float = 0.0175,
    vel_scale: float = 0.5,
    bidirectional: bool = False,
) -> torch.Tensor:
    """奖励与命令推力成正比的轮子旋转.

    4 个轮子均正转表示前进 (视觉验证).vel_scale m/s 等效处的 tanh
    饱和防止失控.

    - ``bidirectional=False`` (默认): 仅前进 — 对 cmd_x > 0 奖励正转,
      否则为零 (cmd_x < 0 由制动 reward 处理).
    - ``bidirectional=True``: 奖励命令方向的轮子旋转 — cmd_x > 0 前进,
      cmd_x < 0 后退 — 幅度为 |cmd_x|.让 cmd_x < 0 表示 "后退" 而非
      "刹车".
    """
    cmd_x = env.command_manager.get_command(command_name)[:, 0]  # (B,)

    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    # 4 个轮子均正转表示前进 (由 test_wheel_direction.py 验证)
    forward_omega = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]] + vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 4.0

    omega_scale = vel_scale / wheel_radius
    if bidirectional:
        # 旋转与命令符号对齐 (+ 前进, - 后退)
        aligned = torch.sign(cmd_x) * forward_omega
        return torch.abs(cmd_x) * torch.tanh(torch.clamp(aligned, min=0.0) / omega_scale)
    return torch.clamp(cmd_x, min=0.0) * torch.tanh(torch.clamp(forward_omega, min=0.0) / omega_scale)


def coasting_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
    stillness_std: float = 5.0,
    asset_cfg: SceneEntityCfg = _LEG_JOINTS_ASSET_CFG,
) -> torch.Tensor:
    """奖励 coasting: 在目标速度下保持低 leg-joint velocity.

    返回 exp(-vel_error / vel_std²) × exp(-sum(joint_vel²) / stillness_std²).
    两 factor 必须同时高 — 机器人在目标速度且保持腿静止 (gliding) 时获奖,
    而非任一单独达标.

    coasting 良好时典型值: ~0.7–1.0. 主动 stomping 速度时 joint_vel 项将 reward
    抑制向 0.
    """
    cmd = env.command_manager.get_command(command_name)
    vel_b = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    vel_error = torch.sum(torch.square(cmd[:, :2] - vel_b), dim=1)
    at_speed = torch.exp(-vel_error / vel_std**2)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    stillness = torch.exp(-joint_vel_sq / stillness_std**2)

    return at_speed * stillness


def braking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    vel_std: float = 0.3,
) -> torch.Tensor:
    """奖励在 cmd_x < 0 (brake commanded) 时停下.

    返回 clamp(-cmd_x, 0) * exp(-fwd_vel² / vel_std²).
    - cmd_x ≥ 0 (coast 或 push) 时静默.
    - cmd_x = -1 且 vel = 0: reward = 1.0 (完全停下).
    - cmd_x = -1 且 vel = vel_std: reward ≈ 0.37 (强梯度).
    vel_std=0.3 m/s 在步行速度下仍给出有意义的梯度.
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_x = cmd[:, 0]
    braking_strength = torch.clamp(-cmd_x, min=0.0)
    fwd_vel = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    stopped = torch.exp(-(fwd_vel.clamp(min=0.0) ** 2) / (vel_std**2))
    return braking_strength * stopped


def contact_frequency_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    max_contact_changes_per_sec: float = 4.0,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """惩罚高频接触变化以鼓励更慢的步频.

    追踪每秒接触状态变化数, 超过阈值时惩罚.

    Args:
        env: 环境
        sensor_name: 接触传感器名称
        max_contact_changes_per_sec: 每秒最大允许的接触变化次数
        command_threshold: 应用惩罚的最小 command magnitude

    Returns:
        shape 为 (num_envs,) 的 penalty tensor - 超过阈值时为负
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)

    # 检查 command 是否超过阈值
    if "twist" in env.command_manager._terms:
        cmd = env.command_manager.get_command("twist")
        cmd_vel = cmd[:, :3]
        cmd_norm = torch.linalg.norm(cmd_vel, dim=1)
        active_mask = cmd_norm > command_threshold
    else:
        active_mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    sensor = env.scene.sensors[sensor_name]
    contacts = sensor.data.found[:, :2]  # (num_envs, 2)

    # 需要时初始化追踪状态
    if not hasattr(env, "_contact_change_count"):
        env._contact_change_count = torch.zeros(env.num_envs, device=env.device)
        env._contact_change_timer = torch.zeros(env.num_envs, device=env.device)
        env._prev_contacts_for_freq = contacts.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # 检测任意接触变化 (任一脚)
    contact_changed = torch.any(contacts != env._prev_contacts_for_freq, dim=1)

    # 递增变化计数
    env._contact_change_count += contact_changed.float()

    # 更新计时器
    env._contact_change_timer += env.step_dt

    # 计算当前频率 (每秒变化次数)
    # 避免除零
    freq = env._contact_change_count / torch.clamp(env._contact_change_timer, min=0.01)

    # 每 1 秒重置计数和计时器
    reset_mask = env._contact_change_timer >= 1.0
    env._contact_change_count[reset_mask] = 0.0
    env._contact_change_timer[reset_mask] = 0.0

    # 频率超过最大值时惩罚
    # 对超过阈值的频率使用二次惩罚
    excess_freq = torch.clamp(freq - max_contact_changes_per_sec, min=0.0)
    penalty = -torch.square(excess_freq)

    # 更新前一次接触
    env._prev_contacts_for_freq = contacts.clone()

    # 应用 command 阈值掩码
    penalty = penalty * active_mask.float()

    return penalty


# ==============================================================================
# Ground Pick 奖励
# ==============================================================================


def mouth_ground_proximity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _MOUTH_TIP_ASSET_CFG,
    std: float = 0.03,
    target_height: float = 0.0,
    command_name: str = "twist",
) -> torch.Tensor:
    """奖励 mouth tip 接近地面, 按 approach phase 加权.

    ground pick task 的 command 为 [cos(2π*phase), sin(2π*phase), 0].
    approach phase 是前半周期 (sin > 0, phase ∈ [0, 0.5]),
    以 max(0, sin(2π*phase)) 平滑加权.

    Args:
        env: RL 环境.
        asset_cfg: mouth tip asset 的 scene entity 配置.
        std: mouth_tip 高度的 Gaussian std (m). 0.03 m 给出强梯度.
        target_height: mouth tip 的目标 z-height (m). 0 = 地面.
        command_name: phase command 项的名称 (例如 "twist").
    """
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]  # (num_envs,)
    proximity = torch.exp(-(((mouth_z - target_height) / std) ** 2))

    # Approach 权重: max(0, sin(2π*phase)) — 在 phase=0.25 处峰值为 1, 在 0 和 0.5 处为零
    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * proximity


def mouth_perpendicular_to_ground(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _MOUTH_TIP_ASSET_CFG,
    command_name: str = "twist",
) -> torch.Tensor:
    """奖励 mouth tip x-axis 在 approach phase 期间垂直 (指向下方).

    完全垂直接触 alignment=1; 水平为 0; 指向上为 -1. 以 max(0,
    sin(2π*phase)) 加权, 因此仅在下降期间生效.
    """
    asset = env.scene[asset_cfg.name]
    # site_quat_w: (num_envs, num_sites, 4) as [w, x, y, z]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]  # (num_envs, 4)
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # z-component of the site x-axis in world frame (first column of rotation matrix)
    x_axis_z = 2.0 * (qx * qz - w * qy)
    # dot with [0, 0, -1]: 1 = perfectly downward, -1 = upward
    alignment = -x_axis_z

    cmd = env.command_manager.get_command(command_name)
    approach_weight = torch.clamp(cmd[:, 1], min=0.0)

    return approach_weight * alignment


def sit_grounded(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    sin_threshold: float = 0.7,
    min_progress_frac: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    upright_cos_threshold: float = 0.5,
) -> torch.Tensor:
    """直立时 trunk-ground 接触的 positive reward.

    额外 gate 在 trunk 的 body-frame +Z 轴大致指向 world-up 方向 (cosine >=
    ``upright_cos_threshold``, 默认 0.5 → 接受至 60° 倾斜). 无此 gate, policy 可通过
    侧倒或前倾获得接触奖励 — trunk 在这些怪异姿态下触地, sit_grounded
    触发, policy 收敛到与实际 sit 姿态竞争的 "fallen" 模式.

    当提供 ``command_name`` 时, reward 被门控到 phase command 的 sit 窗口. 否则
    always-on, 可选通过 ``min_progress_frac`` 门控到 episode 后段.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    has_contact = (found > 0).float()

    # 直立检查: trunk body 的 +Z (世界系, 由 trunk 四元数导出的旋转矩阵第三列)
    # 与世界上方向的点积 = trunk 的 body-up · world-up.
    # 等价于: 对单位四元数 (w, x, y, z) 为 1 - 2*(qx² + qy²).
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) = (w, x, y, z)
    qx, qy = quat[:, 1], quat[:, 2]
    upright_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
    is_upright = (upright_cos >= upright_cos_threshold).float()

    contact_upright = has_contact * is_upright

    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * contact_upright
        return contact_upright
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * contact_upright


def sit_stability(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str | None = None,
    ang_vel_std: float = 0.5,
    sin_threshold: float = 0.7,
    min_progress_frac: float = 0.0,
) -> torch.Tensor:
    """对低 body 角速度的奖励.

    设置 ``command_name`` 时按相位 gate (相位命令的 sit 窗口); 否则常开, 可选通过
    ``min_progress_frac`` 限制到 episode 后段.鼓励稳定的静止姿态.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_norm = asset.data.root_link_ang_vel_w.norm(dim=-1)
    stillness = torch.exp(-((ang_vel_norm / ang_vel_std) ** 2))
    if command_name is None:
        if min_progress_frac > 0.0:
            progress = env.episode_length_buf.float() / float(env.max_episode_length)
            late_enough = (progress >= min_progress_frac).float()
            return late_enough * stillness
        return stillness
    cmd = env.command_manager.get_command(command_name)
    in_sit_window = (cmd[:, 1] > sin_threshold).float()
    return in_sit_window * stillness


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """关节位置偏离默认值 (HOME) 的 L1 惩罚.

    返回所选关节上 |joint_pos - default| 之和.与高斯 `pose` 奖励 (对任意小
    偏差都饱和到 ~1.0) 不同, 这在所有偏差量级下给出 *线性* 梯度 — 适用于
    对部分关节 (如 hip_yaw / hip_roll) 的聚焦惩罚, 防止它们漂移到宽支撑
    站姿, 即使其他关节接近 HOME.
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    err = asset.data.joint_pos[:, jnt_ids] - asset.data.default_joint_pos[:, jnt_ids]
    return torch.sum(torch.abs(err), dim=-1)


def joint_pos_limit_proximity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    margin: float = 0.15,
) -> torch.Tensor:
    """关节位置进入 *硬* 限位旁 ``margin`` (rad) 带的 L1 惩罚.

    基础 ``joint_pos_limits`` 奖励仅在越过 *软* 限位 (全局
    ``soft_joint_pos_limit_factor`` = 0.9 → 约最后 7.5% 行程) 后触发, 且仅按
    弧度过冲量计, 因此对停在硬限位上的关节几乎无用.此项直接读取 *硬*
    限位, 让每个奖励设置自己的宽 margin, 并限定到特定关节.

    动机: 低 kp 位置伺服加宽 ctrlrange 下, 策略可以 "免费" 命令远超关节限位
    (无命令侧代价) 并将关节停在硬限位上 — 例如 hip_yaw 撞到 ±limit 使脚
    滑动/枢转.过冲是 *有意为之* 的 (低 kp 伺服就是这样到达目标的), 因此
    威慑必须在 qpos 侧, 且在到达限位前就生效.

    对每个硬限位为 ``[lo, hi]`` 的选中关节::

        soft_lo = lo + margin,  soft_hi = hi - margin
        penalty = relu(soft_lo - q) + relu(q - soft_hi)

    按关节求和: 内部为零, 向每个限位线性递增.
    """
    asset = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, jnt_ids]
    hard = asset.data.joint_pos_limits[:, jnt_ids]  # (num_envs, num_sel_joints, 2)
    soft_lo = hard[..., 0] + margin
    soft_hi = hard[..., 1] - margin
    below = (soft_lo - q).clip(min=0.0)
    above = (q - soft_hi).clip(min=0.0)
    return torch.sum(below + above, dim=-1)


def phase_height_track(
    env: ManagerBasedRlEnv,
    command_name: str,
    stand_z: float,
    sit_z: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励 trunk_z 跟踪 stand 和 sit 高度之间由 sin 插值的目标.

    用于 sitstand 任务, 替代 sit 姿态的关节角度匹配 — 奖励 END STATE
    (低 trunk) 而不规定机器人如何到达.策略可自由选择运动策略 (深蹲,
    头部支撑下降等).

    命令 (来自 GroundPickPhaseCommand): cmd[:, 1] = sin(2π·phase).
    sin = +1 在 phase 0.25 (sit 峰值) → target = sit_z.
    sin = -1 在 phase 0.75 (stand 峰值) → target = stand_z.
    sin = 0 在过渡点 → target = 中点.
    """
    cmd = env.command_manager.get_command(command_name)
    sin_phase = cmd[:, 1]
    target_z = (stand_z + sit_z) * 0.5 - (stand_z - sit_z) * 0.5 * sin_phase
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return torch.exp(-(((z - target_z) / std) ** 2))


def interpolated_pose_target_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: list | None = None,
    source_overrides: dict | None = None,
    target_overrides: dict | None = None,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """关节位置 vs 时间插值目标姿态的高斯奖励.

    跟踪一个在 episode 内从源姿态线性插值到目标姿态的目标, 插值区间为
    progress fraction ``ramp_start_frac`` 到 ``ramp_end_frac``.在 ramp 之前/
    之后目标分别 clamp 到源 / 最终目标.

    重点是强制平滑下降: 提前 snap 到最终目标会让机器人相对于插值目标当前
    位置 *偏离目标*, 在不匹配期间持续损失 pose 奖励.

    Args:
        env: RL 环境.
        asset_cfg: 机器人 asset 的 scene entity 配置.
        std: 每关节的高斯 std (rad).
        joint_indices: 可选, 评估的关节子集.
        source_overrides: ``{joint_index: angle_rad}`` 定义源姿态 (ramp 起点).
            ``None`` = 默认/HOME 姿态.
        target_overrides: 同上, 用于目标姿态 (ramp 终点).
        ramp_start_frac: ramp 起始的 episode-progress fraction, [0, 1].
        ramp_end_frac: ramp 结束的 episode-progress fraction, [0, 1].
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return torch.exp(-(((joint_pos - interp) / std) ** 2)).mean(dim=-1)


def interpolated_pose_l1_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: list | None = None,
    source_overrides: dict | None = None,
    target_overrides: dict | None = None,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """与时间插值目标姿态的 L1 距离 (负值 — 用作惩罚).

    与 ``interpolated_pose_target_match`` 相同的插值计划, 但返回
    ``-mean(|joint_pos - interp|)`` 而非高斯.L1 梯度处处恒定 — 当高斯
    变体在远离目标处饱和为零, 策略无法发现目标方向时, 适合作为 bootstrap
    信号.
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    source = _servo_default_joint_pos(env, asset).clone()
    target = _servo_default_joint_pos(env, asset).clone()
    if source_overrides:
        for idx, val in source_overrides.items():
            source[:, idx] = val
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0).unsqueeze(-1)
    interp = source * (1.0 - tau) + target * tau

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        interp = interp[:, joint_indices]
    return -torch.abs(joint_pos - interp).mean(dim=-1)


def interpolated_height_l1_penalty(
    env: ManagerBasedRlEnv,
    start_height: float,
    end_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """与时间插值目标高度的 L1 距离 (负值 — 惩罚).

    与 ``interpolated_pose_l1_penalty`` 同角色但作用于 trunk z.无论当前 z
    偏差多大, 都提供朝目标高度的恒定梯度, 补充高斯版 ``interpolated_height_target``.
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return -torch.abs(z - target_z)


def interpolated_height_target(
    env: ManagerBasedRlEnv,
    start_height: float,
    end_height: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ramp_start_frac: float = 0.0,
    ramp_end_frac: float = 1.0,
) -> torch.Tensor:
    """trunk z vs 时间插值目标高度的高斯奖励.

    ``interpolated_pose_target_match`` 的配套 — 同样的时间插值逻辑应用到 trunk 高度.
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    span = max(ramp_end_frac - ramp_start_frac, 1e-6)
    tau = ((progress - ramp_start_frac) / span).clamp(0.0, 1.0)
    target_z = start_height * (1.0 - tau) + end_height * tau

    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return torch.exp(-(((z - target_z) / std) ** 2))


def bilateral_symmetry_penalty(
    env: ManagerBasedRlEnv,
    left_indices: list,
    right_indices: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """左右腿不对称的 L1 惩罚.

    对双侧对称的机器人, 腿的 HOME 和任何对称目标 (FOLD, SIT) 在每个匹配关节
    对上满足 ``q_left + q_right == 0`` (因为左右关节使用镜像符号约定).此项
    惩罚对该约束的偏离.

    当 pose-target 奖励的 ``mean()`` 让策略逃脱到单腿正确的解时 (你免费获得
    ~一半奖励, 修复第二条腿的梯度太弱无法逃出该局部极小), 此项有用.
    惩罚具有恒定 L1 梯度, 无论偏差多大, 任何不对称都付出代价, 唯一零点为
    完全对称构型.

    返回 ``-sum_i |q[left_i] + q[right_i]|``, 对 N 对取平均.
    """
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos
    left = pos[:, left_indices]
    right = pos[:, right_indices]
    return -torch.abs(left + right).mean(dim=-1)


def _multistage_target_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    waypoints,
) -> torch.Tensor:
    """计算跨 N 个 waypoint 的时间插值关节目标.

    waypoints: 有序列表 of dict {"frac": float in [0,1],
                                   "overrides": dict[int,float] | None}.
    第一个 waypoint 应有 frac=0.0 (通常为 HOME, overrides=None).
    后续 waypoint 定义里程碑.两个 waypoint 之间目标线性插值.第一个之前/
    最后一个之后 clamp.

    返回 (num_envs, num_joints) 的目标关节角度 tensor.
    """
    asset = env.scene[asset_cfg.name]
    default = _servo_default_joint_pos(env, asset)

    def build_pose(overrides):
        pose = default.clone()
        if overrides:
            for idx, val in overrides.items():
                pose[:, idx] = val
        return pose

    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    # 确定当前处于哪个段 (跨 envs 广播).
    out = build_pose(waypoints[0]["overrides"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0).unsqueeze(-1)
        prev_pose = build_pose(waypoints[i - 1]["overrides"])
        next_pose = build_pose(waypoints[i]["overrides"])
        seg = prev_pose * (1.0 - tau) + next_pose * tau
        # 当 progress 在 [f0, f1] 或超过时取该段的值.
        mask = (progress >= f0).float().unsqueeze(-1)
        out = torch.where(mask > 0, seg, out)
    return out


def _multistage_target_height(
    env: ManagerBasedRlEnv,
    waypoints,
) -> torch.Tensor:
    """与 _multistage_target_pose 相同逻辑, 但用于 trunk z 高度.

    waypoints: [{"frac": float, "height": float}, ...].
    """
    progress = env.episode_length_buf.float() / float(env.max_episode_length)
    out = torch.full_like(progress, waypoints[0]["height"])
    for i in range(1, len(waypoints)):
        f0 = waypoints[i - 1]["frac"]
        f1 = waypoints[i]["frac"]
        span = max(f1 - f0, 1e-6)
        tau = ((progress - f0) / span).clamp(0.0, 1.0)
        seg = waypoints[i - 1]["height"] * (1.0 - tau) + waypoints[i]["height"] * tau
        mask = (progress >= f0).float()
        out = torch.where(mask > 0, seg, out)
    return out


def multistage_pose_target_match(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: list | None = None,
) -> torch.Tensor:
    """interpolated_pose_target_match 的多 waypoint 变体.

    waypoints: [{"frac": 0.0, "overrides": None},
                {"frac": 0.4, "overrides": FOLD_OVERRIDES},
                {"frac": 0.7, "overrides": SIT_OVERRIDES}]

    用于强制通过一个或多个中间姿态的课程式轨迹 (如 stand → fold → sit).
    与单阶段版本相同的每关节高斯语义.
    """
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)


def multistage_pose_l1_penalty(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: list | None = None,
) -> torch.Tensor:
    """multistage_pose_target_match 的 L1 配套项."""
    asset = env.scene[asset_cfg.name]
    target = _multistage_target_pose(env, asset_cfg, waypoints)
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def multistage_height_target(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.03,
) -> torch.Tensor:
    """trunk z 的多 waypoint 高斯奖励."""
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return torch.exp(-(((z - target_z) / std) ** 2))


def multistage_height_l1_penalty(
    env: ManagerBasedRlEnv,
    waypoints: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """multistage_height_target 的 L1 配套项."""
    target_z = _multistage_target_height(env, waypoints)
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return -torch.abs(z - target_z)


def pose_target_match(
    env: ManagerBasedRlEnv,
    target_overrides: dict | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    joint_indices: list | None = None,
) -> torch.Tensor:
    """对单一固定目标的高斯 pose 匹配.

    target = ``default_joint_pos`` 并按索引应用 overrides.无 waypoint,
    无 episode-progress 插值 — 从 t=0 到 episode 结束都奖励同一目标.
    """
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)


def pose_l1_penalty(
    env: ManagerBasedRlEnv,
    target_overrides: dict | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_indices: list | None = None,
) -> torch.Tensor:
    """``pose_target_match`` 的 L1 配套项 (朝目标的恒定梯度)."""
    asset = env.scene[asset_cfg.name]
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = _servo_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def height_target_gaussian(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.02,
) -> torch.Tensor:
    """trunk z 对单一固定目标的高斯奖励."""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return torch.exp(-(((z - target_height) / std) ** 2))


def height_l1_penalty(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """``height_target_gaussian`` 的 L1 配套项."""
    asset = env.scene[asset_cfg.name]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return -torch.abs(z - target_height)


def trunk_vertical_accel_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """与 trunk ``|a_z|`` 成正比的惩罚 (v_z 的有限差分).

    捕捉硬冲击 (落地时大减速度尖峰) 并激励平滑准静态下降 (恒定速度 →
    a_z ≈ 0).静止时 a_z 为零, 坐姿机器人不付代价.

    状态保存在 env 的 ``_prev_trunk_vz`` 上; episode reset 时 accel 清零,
    以避免前一 episode 最终状态的瞬态泄漏到新的 episode 中.
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    prev = getattr(env, "_prev_trunk_vz", None)
    if prev is None or prev.shape[0] != vz.shape[0]:
        prev = vz.detach().clone()
    a_z = (vz - prev) / env.step_dt
    # 在 reset 步将 a_z 清零以消除跨 episode 瞬态.
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        a_z = torch.where(reset_mask, torch.zeros_like(a_z), a_z)
    env._prev_trunk_vz = vz.detach().clone()
    return -torch.abs(a_z)


def trunk_downward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_down_vel: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对超过 ``max_down_vel`` 的向下 trunk 速度的惩罚.

    限制下降 *速度*, 这是 ``trunk_vertical_accel_penalty`` 单独做不到的:
    快速恒速下降全程 a_z ≈ 0, 仅在底部付一次冲击尖峰 — 相对于更快到达
    目标姿态来说很便宜.此项让过快下降的每一步都损失奖励, 因此保持在
    上限之下的最缓下降是最优的.静止时和任何低于上限的运动 (包括所有
    上升运动) 均为零.
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(-vz - max_down_vel, min=0.0)


def seated_stillness(
    env: ManagerBasedRlEnv,
    height_full: float = 0.06,
    height_zero: float = 0.08,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对坐姿直立时 trunk 静止的奖励: |v| 高斯, 由 z 和 tilt gate 控制.

    exp(-(|v|/vel_std)²) · smoothstep(z) · smoothstep(tilt).z gate 在
    ``height_full`` 以下为满, ``height_zero`` 以上为零 (下降期间不激活).
    tilt gate 在 ``tilt_full_deg`` 以下为满, ``tilt_zero_deg`` 以上为零 —
    *没有* 它, "仰躺静止" 与 "正坐静止" 得分一样 (trunk 仰躺在 seated z
    带内且完全不动), 这正是 run 2 收敛到的 exploit.使 "在坐姿高度安静、
    直立地静止" 成为唯一获得奖励的静止.
    """
    asset = env.scene[asset_cfg.name]
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    t = torch.clamp((height_zero - z) / max(height_zero - height_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)
    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)
    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate


def upright_while_tall(
    env: ManagerBasedRlEnv,
    height_low: float,
    height_high: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """由 trunk z 的 smoothstep 加权的线性直立奖励.

    返回 ``body_upright_linear * smoothstep((z - low)/(high - low))``, 使
    机器人在高处站立时直立激励为满, 一旦进入低位 sit 构型则衰减到零
    (那里臀着地的朝向是可接受的).防止策略学会在高处后仰 (否则会通过
    受控跌落来刷下降奖励).
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    qx = quat[:, 1]
    qy = quat[:, 2]
    upright = 1.0 - 2.0 * (qx * qx + qy * qy)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    t = torch.clamp((z - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return upright * smooth


def phase_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """沿 phase [0,1) 混合 0..1 — 0 = STAND 姿态, 1 = DOWN 姿态.

    [0, descent_end)       : 0 -> 1  (下降)
    [descent_end, hold_end): 1       (低位)
    [hold_end, rise_end)   : 1 -> 0  (上升)
    [rise_end, 1.0)        : 0       (高位 / 静止)
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b


def kick_pose_target(
    phase: torch.Tensor,
    stand: torch.Tensor,
    back: torch.Tensor,
    forward: torch.Tensor,
    windup_end: float,
    kick_end: float,
    return_end: float,
) -> torch.Tensor:
    """4 keyframe shoot 手势的插值关节目标.

    phase (B,) ∈ [0,1). stand/back/forward (k,) 或 (1,k). 返回 (B,k).

    [0, windup_end)        STAND   -> BACK     (蓄力)     [windup_end, kick_end) BACK    -> FORWARD  (击打)
    [kick_end, return_end) FORWARD -> STAND    (回收)     [return_end, 1.0)      STAND             (静止)
    """
    p = phase.unsqueeze(-1)  # (B,1)

    def interp(a, b, s):
        return a + s * (b - a)

    s1 = (p / windup_end).clamp(0.0, 1.0)
    s2 = ((p - windup_end) / (kick_end - windup_end)).clamp(0.0, 1.0)
    s3 = ((p - kick_end) / (return_end - kick_end)).clamp(0.0, 1.0)

    seg1 = interp(stand, back, s1)
    seg2 = interp(back, forward, s2)
    seg3 = interp(forward, stand, s3)  # 在 s3=1 (phase>=return_end) 时 => STAND

    out = seg1
    out = torch.where(p >= windup_end, seg2, out)
    out = torch.where(p >= kick_end, seg3, out)
    return out


def _kick_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    stand_pose: dict,
    back_pose: dict,
    forward_pose: dict,
    windup_end: float,
    kick_end: float,
    return_end: float,
    joint_names: list | None = None,
):
    """shoot 手势的 (cur, target), 关节按名称解析.

    3 个姿态共享相同的键 (14 关节).名称顺序由 `stand_pose` 给出 (或由
    `joint_names` 提供 — 键的子集, 例如一侧右腿 + 颈, 另一侧左腿, 用于对
    手势 vs 支撑腿施加不同 std).
    """
    if not stand_pose:
        raise ValueError("_kick_pose_error requires a non-empty stand_pose dict")
    asset: Entity = env.scene[asset_cfg.name]
    names = list(joint_names) if joint_names is not None else list(stand_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]

    def vec(d):
        return torch.tensor([d[n] for n in names], device=env.device, dtype=asset.data.joint_pos.dtype)

    stand_v, back_v, fwd_v = vec(stand_pose), vec(back_pose), vec(forward_pose)

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    target = kick_pose_target(phase, stand_v, back_v, fwd_v, windup_end, kick_end, return_end)  # (B,k)
    cur = asset.data.joint_pos[:, ids]  # (B,k)
    return cur, target


def kick_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: dict | None = None,
    back_pose: dict | None = None,
    forward_pose: dict | None = None,
    std: float = 0.4,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_names: list | None = None,
) -> torch.Tensor:
    """关节姿态 vs 插值 shoot 目标的高斯奖励.

    直接且对称的奖励: 每个阶段施加精确的关节构型.按名称解析.`joint_names`
    限定到子集评估 (例如右腿 + 颈紧跟踪, 支撑左腿松跟踪让其保持平衡).
    """
    cur, target = _kick_pose_error(
        env,
        asset_cfg,
        command_name,
        stand_pose or {},
        back_pose or {},
        forward_pose or {},
        windup_end,
        kick_end,
        return_end,
        joint_names,
    )
    return torch.exp(-(((cur - target) / std) ** 2)).mean(dim=-1)


def kick_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    stand_pose: dict | None = None,
    back_pose: dict | None = None,
    forward_pose: dict | None = None,
    windup_end: float = 0.35,
    kick_end: float = 0.45,
    return_end: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_names: list | None = None,
) -> torch.Tensor:
    """朝插值目标的 bootstrap L1 (恒定梯度, 惩罚<=0)."""
    cur, target = _kick_pose_error(
        env,
        asset_cfg,
        command_name,
        stand_pose or {},
        back_pose or {},
        forward_pose or {},
        windup_end,
        kick_end,
        return_end,
        joint_names,
    )
    return -(cur - target).abs().mean(dim=-1)


def kick_engagement(
    phase: torch.Tensor,
    windup_end: float,
    return_end: float,
) -> torch.Tensor:
    """手势 engagement gate ∈ [0,1] (纯值) — 用于加权单脚平衡奖励,
    仅在 STAND 静止之外应用.

    [0, windup_end)         : 0 -> 1  (蓄力期间上升)
    [windup_end, return_end): 1       (击打阶段 = 预期单脚支撑)
    [return_end, 1.0)       : 0       (STAND 静止, 双脚支撑, CoM 居中 OK)
    """
    g = torch.zeros_like(phase)
    ramp = phase < windup_end
    g = torch.where(ramp, phase / windup_end, g)
    hold = (phase >= windup_end) & (phase < return_end)
    g = torch.where(hold, torch.ones_like(phase), g)
    return g


def com_over_support_foot(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
    std: float = 0.04,
    windup_end: float = 0.35,
    return_end: float = 0.75,
) -> torch.Tensor:
    """高斯奖励: CoM 水平投影靠近支撑脚, 由击打相位 gate 控制.

    (kick_engagement).

    学习向支撑脚 (support) 的侧向重心转移.没有它, 源自双脚支撑起始姿态的
    单脚手势会保持 CoM 居中于两脚之间 → 另一只脚一抬起就翻倒.
    STAND 静止时 gate 为 0 (双脚支撑, CoM 居中允许).

    `asset_cfg` 须指向支撑脚 site (如 site_names=["left_foot"]).`std` 单位
    为米 (CoM↔脚的容忍半径, ~脚的尺寸).
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_xy = asset.data.root_com_pos_w[:, :2]
    foot_id = asset_cfg.site_ids[0]
    foot_xy = asset.data.site_pos_w[:, foot_id, :2]
    dist2 = ((com_xy - foot_xy) ** 2).sum(dim=-1)
    reward = torch.exp(-dist2 / (std**2))

    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0
    gate = kick_engagement(phase, windup_end, return_end)
    return gate * reward


def _phase_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    target_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    source_pose: dict | None = None,
):
    """相位插值姿态的 (cur, target), 按名称解析.

    目标 = source + blend(phase)·(target_pose - source), source = STAND
    (`source_pose` 若提供, 否则为模型的 DEFAULT/HOME).blend ∈ [0,1]
    (0 = STAND, 1 = target_pose), 通过 `phase_pose_blend`.
    """
    if not target_pose:
        raise ValueError("_phase_pose_error requires a non-empty target_pose dict")

    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = phase_pose_blend(phase, descent_end, hold_end, rise_end)  # (B,)

    names = list(target_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]  # (B,k)

    source = default.clone()
    if source_pose:
        for j, n in enumerate(names):
            if n in source_pose:
                source[:, j] = source_pose[n]
    target_vec = torch.tensor([target_pose[n] for n in names], device=env.device, dtype=default.dtype).unsqueeze(
        0
    )  # (1,k)

    target = source + blend.unsqueeze(-1) * (target_vec - source)  # (B,k)
    cur = asset.data.joint_pos[:, ids]  # (B,k)
    return cur, target


def phase_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: dict | None = None,
    source_pose: dict | None = None,
    std: float = 0.3,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """关节姿态 vs 插值 STAND<->DOWN 目标的高斯奖励.

    直接奖励: 指示每个阶段的精确关节构型.上升 (目标 → STAND) 与下降
    (目标 → DOWN) 获得相同奖励 — 构造上对称.按名称解析.
    """
    cur, target = _phase_pose_error(
        env,
        asset_cfg,
        command_name,
        target_pose or {},
        descent_end,
        hold_end,
        rise_end,
        source_pose,
    )
    return torch.exp(-(((cur - target) / std) ** 2)).mean(dim=-1)


def phase_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: dict | None = None,
    source_pose: dict | None = None,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """朝插值目标的 bootstrap L1 (负惩罚).

    处处恒定梯度 — 即使上方高斯在远离目标处饱和到 ~0 时也给出朝目标的
    方向.
    """
    cur, target = _phase_pose_error(
        env,
        asset_cfg,
        command_name,
        target_pose or {},
        descent_end,
        hold_end,
        rise_end,
        source_pose,
    )
    return -(cur - target).abs().mean(dim=-1)


def phase_pose_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: list | None = None,
    target_overrides: dict | None = None,
    phase: str = "approach",
) -> torch.Tensor:
    """匹配目标姿态的奖励, 由相位周期命令加权.

    相位条件任务的通用 helper (如 sit/stand).命令将相位编码为
    [cos(2π·phase), sin(2π·phase), 0]:
      - "approach" 权重 = max(0, sin(2π·phase)) — 在 phase 0.25 峰值.
      - "return"   权重 = max(0,-sin(2π·phase)) — 在 phase 0.75 峰值.

    Args:
        env: RL 环境.
        asset_cfg: 机器人 asset 的 scene entity 配置.
        std: 每关节的高斯 std (rad).
        command_name: 相位命令项的名称 (如 "twist").
        joint_indices: 可选, 评估的关节子集 (其余忽略).
        target_overrides: {joint_index: angle_rad}.未列出的关节默认为
            asset.data.default_joint_pos (home/standing 姿态).
        phase: "approach" 或 "return".
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    target = _servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        target = target[:, joint_indices]
    pose_reward = torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)

    cmd = env.command_manager.get_command(command_name)
    weight = torch.clamp(cmd[:, 1], min=0.0) if phase == "approach" else torch.clamp(-cmd[:, 1], min=0.0)
    return weight * pose_reward


def ground_pick_return_pose(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: list | None = None,
) -> torch.Tensor:
    """ground pick 后回到 standing 姿态的奖励, 由 return 相位加权.

    return 相位为下半周期 (sin < 0, phase ∈ [0.5, 1.0]), 由
    max(0, -sin(2π*phase)) 平滑加权.

    Args:
        env: RL 环境.
        asset_cfg: 机器人 asset 的 scene entity 配置.
        std: 每关节的高斯 std (rad).
        command_name: 相位命令项的名称 (如 "twist").
        joint_indices: 评估的关节子集.用于对腿关节 vs 颈/头关节施加不同
            std (调用此奖励两次).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)  # (num_envs, n_servo_joints)
    default_pos = _servo_default_joint_pos(env, asset)

    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]

    pose_reward = torch.exp(-(((joint_pos - default_pos) / std) ** 2)).mean(dim=-1)

    # Return 权重: max(0, -sin(2π*phase)) — 在 phase=0.75 峰值为 1, 在 0.5 和 1 为零
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)

    return return_weight * pose_reward


def ground_pick_return_upright(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.4,
    command_name: str = "twist",
) -> torch.Tensor:
    """奖励 trunk 垂直度, 由 RETURN 相位加权 (起立辅助).

    与 ``ground_pick_return_pose`` 相同的 return 权重
    (``max(0, -sin(2π·phase))``), 因此仅在起立期间奖励直立, 绝不与 approach
    的前倾对抗.Verticality = ``exp(-tilt²/std²)``, 使用与
    ``body_upright_gaussian`` 相同的 tilt 代理 (``2*(qx²+qy²) ≈ 1-cos(tilt)``).
    较宽的 std (0.4 rad ≈ 23°) 即使在相当倾斜的蹲姿下也给出梯度.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)  # qx² + qy²
    upright = torch.exp(-tilt_sq / (std * std))
    cmd = env.command_manager.get_command(command_name)
    return_weight = torch.clamp(-cmd[:, 1], min=0.0)
    return return_weight * upright


# --------------------------------------------------------------------------- #
# Ground-pick: 分段相位 gate (下降/保持/上升/静止时长独立,                       #
# 替代正弦加权 max(0,±sin)).                                                    #
#   down-gate  = phase_pose_blend(phase, descent_end, hold_end, rise_end)        #
#               0 (高位) -> 1 (下降) -> 1 (低位保持) -> 0 (上升/静止)           #
#   up-gate    = phase_rise_gate(phase, hold_end, rise_end)                       #
#               0 上升前 -> 0..1 (上升) -> 1 (站立静止)                          #
# --------------------------------------------------------------------------- #
def phase_rise_gate(phase: torch.Tensor, hold_end: float, rise_end: float) -> torch.Tensor:
    """RETURN 的上升 gate: hold_end 前为 0, [hold_end, rise_end) 上 0->1, 之后为 1 (站立静止)."""
    g = torch.zeros_like(phase)
    rising = (phase >= hold_end) & (phase < rise_end)
    g = torch.where(rising, (phase - hold_end) / (rise_end - hold_end), g)
    g = torch.where(phase >= rise_end, torch.ones_like(phase), g)
    return g


def _gp_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def mouth_ground_proximity_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _MOUTH_TIP_ASSET_CFG,
    std: float = 0.10,
    target_height: float = 0.0,
    command_name: str = "twist",
    descent_end: float = 0.25,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """mouth_ground_proximity 由分段 down-gate 控制 (下降+保持)."""
    asset = env.scene[asset_cfg.name]
    mouth_z = asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]
    proximity = torch.exp(-(((mouth_z - target_height) / std) ** 2))
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * proximity


def mouth_perpendicular_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _MOUTH_TIP_ASSET_CFG,
    command_name: str = "twist",
    descent_end: float = 0.25,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """mouth_perpendicular_to_ground 由分段 down-gate 控制."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.site_quat_w[:, asset_cfg.site_ids[0], :]
    w, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x_axis_z = 2.0 * (qx * qz - w * qy)
    alignment = -x_axis_z  # 1 = 嘴正朝下
    gate = phase_pose_blend(_gp_phase(env, command_name), descent_end, hold_end, rise_end)
    return gate * alignment


def ground_pick_return_pose_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.3,
    command_name: str = "twist",
    joint_indices: list | None = None,
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """ground_pick_return_pose 由分段 up-gate 控制 (上升+静止)."""
    asset = env.scene[asset_cfg.name]
    joint_pos = _servo_joint_pos(env, asset)
    default_pos = _servo_default_joint_pos(env, asset)
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        default_pos = default_pos[:, joint_indices]
    pose_reward = torch.exp(-(((joint_pos - default_pos) / std) ** 2)).mean(dim=-1)
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * pose_reward


def ground_pick_return_upright_phased(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.4,
    command_name: str = "twist",
    hold_end: float = 0.35,
    rise_end: float = 0.60,
) -> torch.Tensor:
    """ground_pick_return_upright 由分段 up-gate 控制."""
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright = torch.exp(-tilt_sq / (std * std))
    gate = phase_rise_gate(_gp_phase(env, command_name), hold_end, rise_end)
    return gate * upright


def neck_vel_descent_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    joint_indices: list | None = None,
    hold_end: float = 0.35,
) -> torch.Tensor:
    """惩罚下降+保持期间颈部关节的速度 (减速头部下俯).

    代价 = mean(joint_vel²), phase < hold_end 时 gate 为 1 (下降 + 低位保持),
    之后为 0 (上升 + 静止) -> *不* 阻碍颈部抬起.返回正代价; 用负权重使用.
    """
    asset = env.scene[asset_cfg.name]
    vel = _servo_joint_vel(env, asset)
    if joint_indices is not None:
        vel = vel[:, joint_indices]
    cost = (vel**2).mean(dim=-1)
    phase = _gp_phase(env, command_name)
    gate = (phase < hold_end).to(vel.dtype)  # 仅下降 + 低位保持
    return gate * cost


def sample_mouth_payload(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    min_kg: float = 0.01,
    max_kg: float = 0.04,
) -> None:
    """reset 事件: 每个 env 抽取 "口中含物" 的质量 (kg), 存于 env._mouth_payload_kg.

    由 apply_mouth_payload_force 使用.
    """
    buf = getattr(env, "_mouth_payload_kg", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, device=env.device)
        env._mouth_payload_kg = buf
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    buf[env_ids] = torch.rand(len(env_ids), device=env.device) * (max_kg - min_kg) + min_kg


def apply_mouth_payload_force(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _JAW_SOFT_ASSET_CFG,
    command_name: str = "twist",
    hold_end: float = 0.35,
    ramp: float = 0.05,
    gravity: float = 9.81,
) -> torch.Tensor:
    """每步 hook (用作权重 0 的 reward): 将口中所持物的 *重量* 作为外部
    垂直力施加到 mouth_tip, 由上升阶段 gate 控制 (phase >= hold_end, 在
    "抓取"时刻快速 ramp).

    模拟起立期间嘴端的质点: 力 m·g 施加到 body 的 CoM + 力矩
    (p_mouth - p_com) × F, 等价于施加到 mouth_tip (对颈部有正确力臂).
    返回 0 (这不是真正的 reward — 仅是应用 hook).
    """
    asset: Entity = env.scene[asset_cfg.name]
    payload = getattr(env, "_mouth_payload_kg", None)
    if payload is None:
        return torch.zeros(env.num_envs, device=env.device)
    phase = _gp_phase(env, command_name)
    gate = ((phase - hold_end) / ramp).clamp(0.0, 1.0)  # grab 前为 0 -> 之后为 1
    fz = -(gate * payload) * gravity  # (N,) 垂直力 (向下)

    bid = int(asset_cfg.body_ids[0])
    sid = int(asset_cfg.site_ids[0])
    p_mouth = asset.data.site_pos_w[:, sid, :]  # (N,3)
    p_com = asset.data.body_com_pos_w[:, bid, :]  # (N,3)
    F = torch.zeros((env.num_envs, 3), device=env.device, dtype=p_mouth.dtype)
    F[:, 2] = fz
    tau = torch.cross(p_mouth - p_com, F, dim=-1)  # 将 F 施加到 mouth_tip
    asset.write_external_wrench_to_sim(
        forces=F.unsqueeze(1),
        torques=tau.unsqueeze(1),
        body_ids=[bid],
    )
    return torch.zeros(env.num_envs, device=env.device)


# ==============================================================================
# 域随机化事件
# ==============================================================================


def randomize_delayed_actuator_gains(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    operation: str = "scale",
):
    """每 episode 随机化固件 PD 增益 (非累积).

    在标准 BAM actuator (``bam.mjlab.BamActuator``) 下, 增益通过
    ``set_gains``/``reset_gains`` 按 env 缩放 (actuator 持有 ``kp_scale``/
    ``kd_scale``), 因此我们绝不触碰 MuJoCo model — 无累积风险.采样的每关节
    因子取平均为每个 env 的单个标量 (actuator 在其关节上施加一个 scale),
    与之前行为一致.非 BAM actuator 被跳过 (如 roller XmlActuator, 不暴露
    set_gains).

    Args:
        env: 环境
        env_ids: 要随机化的 env ID (None = 全部)
        kp_range: kp 随机化的 (min, max)
        kd_range: kd 随机化的 (min, max)
        asset_cfg: asset 配置
        operation: 未使用 (保留以兼容 cfg; 始终应用 scaling)
    """
    del operation
    from bam.mjlab import BamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    for actuator in asset.actuators:
        if not isinstance(actuator, BamActuator):
            continue
        n_joints = len(actuator.ctrl_ids)
        kp_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kp_range[1] - kp_range[0]) + kp_range[0]
        kd_samples = torch.rand(len(env_ids), n_joints, device=env.device) * (kd_range[1] - kd_range[0]) + kd_range[0]
        # 先恢复标称值 (防止累积), 再施加新的 scale.
        actuator.reset_gains(env_ids)
        actuator.set_gains(
            env_ids,
            kp_scale=kp_samples.mean(dim=1, keepdim=True),
            kd_scale=kd_samples.mean(dim=1, keepdim=True),
        )


@requires_model_fields("dof_frictionloss", "dof_damping")
def expand_bam_friction_fields(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
):
    """空操作 startup 事件, 唯一目的是上方的装饰器.

    bam 的 BamActuator (mjlab_frictionloss 分支) 每步将 per-env 摩擦预算写入
    MuJoCo 的 dof_frictionloss/dof_damping, 要求这些 model 字段按 world 展开.
    mjlab 仅展开事件函数通过 requires_model_fields 声明的字段, 因此每个使用
    BAM actuator 的 env 必须注册此事件.
    """


def randomize_bam_friction(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """BAM actuator 的每 episode 关节摩擦随机化 (非累积).

    在 BAM 下, MuJoCo 的 dof_frictionloss 被清零 (BAM 在 compute() 中计算摩擦),
    因此 stock dr.dof_frictionloss 是空操作.此项改为在 ``scale_range`` 中采样
    per-env 标量并应用到 FrictionDRBamActuator 的 ``friction_scale``, 后者乘以
    BAM 的速度无关摩擦预算 (Coulomb + Stribeck + load).先恢复标称值 (1.0)
    以避免累积.对没有 friction_scale hook 的 actuator 为空操作.
    """
    from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    lo, hi = scale_range
    for actuator in asset.actuators:
        if isinstance(actuator, FrictionDRBamActuator):
            actuator.reset_friction_scale(env_ids)
            samples = torch.rand(len(env_ids), 1, device=env.device) * (hi - lo) + lo
            actuator.set_friction_scale(env_ids, samples)


def randomize_mass_and_inertia(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """用同一缩放因子同时随机化 body 质量和惯性.

    保持物理一致性 — 质量和惯性必须一起缩放, 以避免产生导致仿真不稳定
    的无效惯性张量.

    Args:
        env: 环境
        env_ids: 要随机化的 env ID
        scale_range: (min, max) 缩放因子, 同时应用于质量和惯性
        asset_cfg: 指定随机化哪些 body 的 asset 配置
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]

    # 获取 body 索引
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    # 每个 env 采样一个随机 scale (同时应用于质量和惯性)
    num_envs = len(env_ids)
    num_bodies = len(body_indices)
    scales = torch.rand(num_envs, num_bodies, device=env.device) * (scale_range[1] - scale_range[0]) + scale_range[0]

    # 首次调用时存储原始值
    if not hasattr(env, "_original_mass_inertia"):
        env._original_mass_inertia = {
            "mass": env.sim.model.body_mass[0, body_indices].clone(),
            "inertia": env.sim.model.body_inertia[0, body_indices].clone(),
        }

    # 先恢复原始值 (防止累积)
    original = env._original_mass_inertia
    env.sim.model.body_mass[env_ids[:, None], body_indices] = original["mass"].unsqueeze(0).expand(num_envs, -1)
    env.sim.model.body_inertia[env_ids[:, None], body_indices] = (
        original["inertia"].unsqueeze(0).expand(num_envs, -1, -1)
    )

    # 对质量和惯性施加相同 scale
    env.sim.model.body_mass[env_ids[:, None], body_indices] *= scales
    env.sim.model.body_inertia[env_ids[:, None], body_indices] *= scales.unsqueeze(-1)  # 缩放全部 3 个惯性分量


def standing_envs_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    standing_stages: list[dict],
) -> torch.Tensor:
    """根据训练进度更新站立 env 的相对数量.

    Args:
        env: RL 环境
        env_ids: env ID (未使用, 但课程接口要求)
        command_name: 速度命令项的名称
        standing_stages: 含 'step' 和 'rel_standing_envs' 键的 dict 列表
            例: [
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 1000, "rel_standing_envs": 0.1},
                {"step": 2000, "rel_standing_envs": 0.2},
            ]

    Returns:
        当前 rel_standing_envs 值 (tensor)
    """
    del env_ids  # 未使用

    from typing import cast

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # 根据当前 step 更新 rel_standing_envs
    for stage in standing_stages:
        if env.common_step_counter > stage["step"]:
            cfg.rel_standing_envs = stage["rel_standing_envs"]

    return torch.tensor([cfg.rel_standing_envs])


def velocity_tracking_std_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    std_stages: list[dict],
) -> torch.Tensor:
    """根据训练进度更新速度跟踪 std 参数.

    从宽松的 std 开始 (容易获得奖励) 学习基本行走, 然后逐步收紧以提升
    速度跟踪精度.

    Args:
        env: RL 环境
        env_ids: env ID (未使用, 但课程接口要求)
        reward_name: 奖励项名称 (如 "track_linear_velocity")
        std_stages: 含 'step' 和 'std' 键的 dict 列表
            例: [
                {"step": 0, "std": 0.5},      # 宽松起步 - 学走路
                {"step": 250, "std": 0.3},     # 中等 - 改进步态
                {"step": 500, "std": 0.2},     # 严格 - 精确跟踪
            ]

    Returns:
        当前 std 值 (tensor)
    """
    del env_ids  # 未使用

    # 获取奖励项配置
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)

    # 根据当前 step 更新 std
    current_std = std_stages[0]["std"]  # 默认第一阶段

    for stage in std_stages:
        if env.common_step_counter > stage["step"]:
            current_std = stage["std"]

    # 更新奖励项的 std 参数
    reward_term_cfg.params["std"] = current_std

    return torch.tensor([current_std])


def push_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    push_stages: list[dict],
) -> torch.Tensor:
    """根据训练进度更新推力速度范围.

    从无/小推力开始学习干净行走, 然后逐步增大以建立鲁棒性, 不干扰早期学习.

    Args:
        env: RL 环境
        env_ids: env ID (未使用, 但课程接口要求)
        event_name: 推力事件项名称 (如 "push_robot")
        push_stages: 含 'step' 和 'velocity_range' 键的 dict 列表
            例: [
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 250, "velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
                {"step": 500, "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
            ]

    Returns:
        当前最大推力幅值 (tensor)
    """
    del env_ids  # 未使用

    # 注意: 必须更新活跃的 EventManager term_cfg, 而非 env.cfg.events —
    # EventManager.__init__ 会 deepcopy(cfg), 因此修改 env.cfg.events 是空操作.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    # 根据当前 step 更新 velocity_range
    current_range = push_stages[0]["velocity_range"]  # 默认第一阶段

    for stage in push_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["velocity_range"]

    # 更新事件配置的 velocity_range 参数
    event_cfg.params["velocity_range"] = current_range

    # 返回最大幅值用于日志
    max_push = max(abs(current_range["x"][0]), abs(current_range["x"][1]))
    return torch.tensor([max_push])


def wheel_friction_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    ranges_stages: list[dict],
) -> torch.Tensor:
    """根据训练 step 阶段更新轮子摩擦."""
    del env_ids  # 未使用

    current_ranges = ranges_stages[0]["ranges"]
    for stage in ranges_stages:
        if env.common_step_counter > stage["step"]:
            current_ranges = stage["ranges"]

    env.event_manager.get_term_cfg(event_name).params["ranges"] = current_ranges
    return torch.tensor([current_ranges[0]])


def reward_weight(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    weight_stages: list[dict],
) -> torch.Tensor:
    """按 step 分阶段的奖励权重课程.

    mjlab 1.3.0 移除了内置的 ``mdp.reward_weight`` helper, 因此 microduck 提供
    自己的版本.``weight_stages`` 是 ``{"step": int, "weight": float}`` dict
    列表; 应用其 step 已过的最新阶段的权重.修改活跃的 RewardManager
    term cfg (而非 env.cfg, 后者在 manager init 时已 deepcopy).
    """
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in weight_stages:
        if env.common_step_counter > stage["step"]:
            term_cfg.weight = stage["weight"]
    return torch.tensor([term_cfg.weight])


def com_range_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    range_stages: list[dict],
) -> torch.Tensor:
    """根据训练进度更新 CoM 随机化范围.

    逐步增大 CoM 偏移范围, 使机器人先在小 CoM 不确定性下学会行走, 再
    渐进增大.

    Args:
        env: RL 环境
        env_ids: env ID (未使用)
        event_name: CoM 随机化事件名称 (如 "randomize_com")
        range_stages: 含 'step' 和 'range' 键的 dict 列表 (range 单位米)
            例: [
                {"step": 0,          "range": 0.003},
                {"step": 1000 * 24,  "range": 0.005},
                {"step": 2000 * 24,  "range": 0.008},
            ]

    Returns:
        当前范围值 (tensor, 用于日志)
    """
    del env_ids

    # 注意: 必须更新活跃的 EventManager term_cfg, 而非 env.cfg.events —
    # EventManager.__init__ 会 deepcopy(cfg), 因此修改 env.cfg.events 是空操作.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_range = range_stages[0]["range"]
    for stage in range_stages:
        if env.common_step_counter > stage["step"]:
            current_range = stage["range"]

    event_cfg.params["ranges"] = (-current_range, current_range)
    return torch.tensor([current_range])


def slope_move_masks(distance: "torch.Tensor", size_x: float):
    """坡度课程的升级/降级 mask.

    move_up   : 行进超过 40% tile → 已冲下坡道, 增大坡度.与
                terrain_edge_reached termination 对齐 (~3.8 m,
                threshold_fraction=0.95 默认, size_x=8.0), 它在半程
                阈值 (4.0 m) 前结束 episode — 不对齐则成功的穿越者
                永远不会被升级.
    move_down : 几乎没前进 (< 20% tile) → 早期摔倒/卡住, 减小坡度.
    """
    move_up = distance > size_x * 0.4
    move_down = (distance < size_x * 0.2) & (~move_up)
    return move_up, move_down


def terrain_levels_slope(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """roller_slope 的坡度课程 (无命令速度).

    基于从 spawn 原点行进的 x 距离的进度.
    """
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    distance = asset.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up, move_down = slope_move_masks(distance, terrain_generator.size[0])
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def velocity_command_ranges_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[dict],
    update_lin_vel_y: bool = True,
    update_ang_vel_z: bool = True,
    forward_only: bool = False,
) -> torch.Tensor:
    """根据训练进度更新速度命令范围.

    逐步增大命令速度范围, 使机器人渐进学习更高速度.从小范围开始稳定学习,
    然后扩展到更具挑战性的速度.

    Args:
        env: RL 环境
        env_ids: env ID (未使用, 但课程接口要求)
        command_name: 速度命令项名称 (如 "twist")
        velocity_stages: 含 'step', 'lin_vel_range' 和 'ang_vel_range' 键的 dict 列表
            例: [
                {"step": 0, "lin_vel_range": 0.3, "ang_vel_range": 1.5},
                {"step": 500 * 24, "lin_vel_range": 0.4, "ang_vel_range": 1.75},
                {"step": 1000 * 24, "lin_vel_range": 0.5, "ang_vel_range": 2.0},
            ]
        update_lin_vel_y: 是否更新侧向 (y) 速度范围.
        update_ang_vel_z: 是否更新偏航 (z) 角速度范围.
        forward_only: 若为 True, 仅更新前进 (x) 速度范围.

    Returns:
        当前最大线速度 (tensor)
    """
    del env_ids  # 未使用

    from typing import cast

    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None, f"Command term '{command_name}' not found"

    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

    # 根据当前 step 更新速度范围
    current_lin_vel = velocity_stages[0]["lin_vel_range"]
    current_ang_vel = velocity_stages[0]["ang_vel_range"]

    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            current_lin_vel = stage["lin_vel_range"]
            current_ang_vel = stage["ang_vel_range"]

    # 更新命令范围
    if forward_only:
        cfg.ranges.lin_vel_x = (0.0, current_lin_vel)
    else:
        cfg.ranges.lin_vel_x = (-current_lin_vel, current_lin_vel)
    if update_lin_vel_y:
        cfg.ranges.lin_vel_y = (-current_lin_vel, current_lin_vel)
    if update_ang_vel_z:
        cfg.ranges.ang_vel_z = (-current_ang_vel, current_ang_vel)

    return torch.tensor([current_lin_vel])


def projected_gravity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """body 系下的投影重力向量.

    返回投影到机器人 body 系的重力向量, 表示纯朝向不含线加速度.
    比原始 accelerometer 更简单, 仅依赖朝向.

    Returns:
        torch.Tensor: body 系投影重力 (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def _imu_misalignment_quat(env: ManagerBasedRlEnv, max_angle_rad: float) -> torch.Tensor:
    """per-env 恒定的 IMU 安装失准旋转 (采样一次).

    建模每个机器人 IMU 的固定小安装/校准误差.首次使用时惰性采样并缓存 —
    每个 env 在整个 run 中恒定 (如同 startup 随机化), 因此是 *系统性 per-robot
    bias*, 而非逐步噪声.替代旧的 randomize_imu_orientation 事件, 后者写入
    site_quat (在 mjlab 1.3.0 下未按 env 展开, 且 projected_gravity /
    base_ang_vel 观测也不读取它).

    返回 (num_envs, 4) 单位四元数 (w, x, y, z).
    """
    q = getattr(env, "_imu_misalign_quat", None)
    if q is None:
        n = env.num_envs
        axis = torch.randn(n, 3, device=env.device)
        axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
        angle = torch.rand(n, device=env.device) * max_angle_rad  # [0, max]
        q = quat_from_angle_axis(angle, axis)
        env._imu_misalign_quat = q
    return q


def projected_gravity_imu_misaligned(
    env: ManagerBasedRlEnv,
    max_angle_deg: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """带 per-env 恒定 IMU 安装失准的 projected_gravity."""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.projected_gravity_b)


def base_ang_vel_imu_misaligned(
    env: ManagerBasedRlEnv,
    max_angle_deg: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """带与 gravity 相同 per-env IMU 失准的 base 角速度."""
    asset: Entity = env.scene[asset_cfg.name]
    q = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(q, asset.data.root_link_ang_vel_b)


def raw_accelerometer(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """原始 accelerometer 读数 (含重力 + 线加速度).

    返回归一化的原始 accelerometer 读数, 模拟真实 IMU 测量值.
    与仅反映朝向的纯 projected_gravity 不同.读取 MuJoCo accelerometer
    传感器 "imu_accel".

    Returns:
        torch.Tensor: 归一化原始 accelerometer 读数 (num_envs, 3)
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 访问 model 以找到 sensor 地址
    # accelerometer 传感器是 robot.xml 中第 5 个 sensor (index 4)
    # Sensors 列表: framequat, gyro, gyro, velocimeter, accelerometer, subtreeangmom
    mj_model = asset.data.model

    # 从 model 数组获取 sensor 地址 (sensor_adr 为 torch tensor)
    sensor_adr_array = mj_model.sensor_adr  # TorchArray/tensor
    sensor_id = 4  # imu_accel 是第 5 个 sensor (0-indexed)
    sensor_adr = int(sensor_adr_array[sensor_id].item())  # 转为 Python int

    # 读取 accelerometer 数据 (sensor 测量的比力)
    # Shape: (num_envs, 3)
    accel_raw = asset.data.data.sensordata[:, sensor_adr : sensor_adr + 3]

    # MuJoCo accelerometer 测量比力 (如同真实 sensor)
    # 取反以匹配约定: 静止直立时应朝下
    accel_negated = -accel_raw

    # 归一化为单位向量
    accel_norm = torch.norm(accel_negated, dim=-1, keepdim=True)
    accel_normalized = torch.where(
        accel_norm > 0.1,
        accel_negated / accel_norm,
        asset.data.projected_gravity_b,  # 回退到投影重力
    )

    return accel_normalized


def randomize_imu_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_angle_deg: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """用小角度随机化 IMU 传感器安装朝向.

    模拟真实机器人中轻微的安装误差或校准偏移.IMU 朝向通过绕随机轴旋转
    最多 max_angle_deg 来随机化.

    Args:
        env: 环境
        env_ids: 要随机化的 env ID
        max_angle_deg: 最大旋转角度 (度, 默认 2.0°)
        asset_cfg: asset 配置
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    env.scene[asset_cfg.name]

    # IMU site 是 robot.xml 中第一个 site (index 0)
    # Sites: imu (0), left_foot (1), right_foot (2)
    site_id = 0

    # 首次调用时存储原始朝向
    if not hasattr(env, "_original_imu_quat"):
        env._original_imu_quat = env.sim.model.site_quat[0, site_id].clone()

    # 为每个环境生成随机旋转
    num_envs = len(env_ids)
    max_angle_rad = max_angle_deg * torch.pi / 180.0

    # 每个轴的随机旋转角度 [-max_angle, +max_angle]
    angles = (torch.rand(num_envs, 3, device=env.device) * 2 - 1) * max_angle_rad

    # 欧拉角转四元数 (小角度近似以提高效率)
    # 小角度: quat ≈ [1, θx/2, θy/2, θz/2]
    half_angles = angles / 2.0
    quats_delta = torch.zeros(num_envs, 4, device=env.device)
    quats_delta[:, 0] = 1.0  # w 分量
    quats_delta[:, 1:] = half_angles  # x, y, z 分量

    # 归一化四元数
    quats_delta = quats_delta / torch.norm(quats_delta, dim=1, keepdim=True)

    # 获取原始四元数并施加 delta 旋转
    original_quat = env._original_imu_quat.unsqueeze(0).expand(num_envs, -1)

    # 四元数乘法: q_new = q_delta * q_original
    # q1 * q2 = [w1*w2 - dot(v1,v2), w1*v2 + w2*v1 + cross(v1,v2)]
    w1, x1, y1, z1 = (
        quats_delta[:, 0],
        quats_delta[:, 1],
        quats_delta[:, 2],
        quats_delta[:, 3],
    )
    w2, x2, y2, z2 = (
        original_quat[:, 0],
        original_quat[:, 1],
        original_quat[:, 2],
        original_quat[:, 3],
    )

    new_quat = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
        ],
        dim=1,
    )

    # 应用到选中的环境
    env.sim.model.site_quat[env_ids, site_id] = new_quat


def standing_phase(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """standing 任务的简单时间相位.

    返回基于时间在 0 到 1 之间循环的标量相位值.让策略在站立时也有
    时间推移的感觉.

    Args:
        env: RL 环境
        asset_cfg: 未使用, 保留以保持 API 一致性

    Returns:
        相位值 [0, 1], shape (num_envs, 1) 的 tensor
    """
    # 基于时间的简单相位, 每 2 秒循环一次
    # 给策略一个时变信号
    phase_period = 2.0  # 秒
    time = env.episode_length_buf * env.step_dt
    phase = (time % phase_period) / phase_period

    return phase.unsqueeze(-1)  # Shape: (num_envs, 1)


def air_time_adaptive(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,  # below this: no reward (standing)
    running_threshold: float = 0.5,  # above this: use running air-time window
    walk_threshold_min: float = 0.10,
    walk_threshold_max: float = 0.25,
    run_threshold_min: float = 0.05,
    run_threshold_max: float = 0.25,
) -> torch.Tensor:
    """带行走 vs 跑步分离摆动时间窗口的 air-time 奖励.

    - command < command_threshold  → 0 (站立, 无奖励)
    - command_threshold–running_threshold → 行走窗口 [walk_min, walk_max]
    - command > running_threshold  → 跑步窗口 [run_min,  run_max]

    使行走步态保持其 100–250 ms 的从容摆动, 而跑步可使用更快的
    50–250 ms 节奏.
    """
    sensor = env.scene.sensors[sensor_name]
    current_air_time = sensor.data.current_air_time  # (num_envs, num_feet)
    assert current_air_time is not None

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

    is_walking = ((total_speed >= command_threshold) & (total_speed < running_threshold)).float()  # (num_envs,)
    is_running = (total_speed >= running_threshold).float()

    # per-env 阈值广播到各脚
    tmin = (is_walking * walk_threshold_min + is_running * run_threshold_min).unsqueeze(1)
    tmax = (is_walking * walk_threshold_max + is_running * run_threshold_max).unsqueeze(1)

    in_range = (current_air_time > tmin) & (current_air_time < tmax)
    reward = torch.sum(in_range.float(), dim=1)  # 按脚求和

    # 站立时零奖励
    active = (total_speed >= command_threshold).float()
    return reward * active


def stillness_at_zero_command(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    vel_std: float = 0.1,
) -> torch.Tensor:
    """命令接近零时奖励保持静止.

    当 command < threshold 时返回 exp(-body_vel² / vel_std²), 否则 0.该值
    随 body 速度单调递减 — 移动越快奖励越低.没有机器人可以越过以 "逃脱"
    的阈值, 不像基于 gate 的步进惩罚.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    stillness = torch.exp(-(body_vel**2) / vel_std**2)

    return is_standing_cmd * stillness


def joint_vel_l2_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """仅当命令接近零时惩罚腿关节速度.

    针对站立抖动问题: 站立时策略在 home 姿态周围做快速振荡修正.按命令
    gate, 因此完全不影响行走步态.
    """
    asset: Entity = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    leg_indices = list(range(0, 5)) + list(range(9, 14))
    joint_vel = asset.data.joint_vel[:, leg_indices]
    vel_sq = torch.sum(joint_vel**2, dim=-1)

    return is_standing_cmd * vel_sq


def foot_step_penalty_when_standing(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    body_vel_threshold: float = 0.2,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """零命令且 body 未被推时惩罚迈步.

    air_time 奖励的对称对应项:
    - air_time 在 command > threshold 时给 +奖励 (行走)
    - 此项在 command < threshold 时给 -奖励 (站立)

    body 速度 gate 防止惩罚推力后的恢复迈步: 若机器人已在快速移动 (被推),
    不施加惩罚, 使其仍可迈步恢复平衡.

    返回 [0, 1] 内的值 (配置中使用负权重).
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors["feet_ground_contact"]

    # 是否有脚最近抬起? (上次完成的空中相位 > threshold)
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2)
    any_foot_stepped = (air_time > air_time_threshold).any(dim=1).float()

    # 是否处于站立模式? (命令接近零)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing = (total_speed < command_threshold).float()

    # body 是否静止? (未被推)
    body_vel = torch.norm(asset.data.root_link_vel_w[:, :2], dim=1)
    is_still = (body_vel < body_vel_threshold).float()

    return any_foot_stepped * is_standing * is_still


def recovery_stepping_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    velocity_threshold: float = 0.3,
    air_time_threshold: float = 0.05,
) -> torch.Tensor:
    """仅当零命令且机器人高速移动 (推力恢复) 时奖励脚 air time.

    鼓励机器人在被推时迈步恢复平衡, 但在正常行走 (command > threshold)
    时不触发.

    Args:
        env: RL 环境
        asset_cfg: asset 配置 (未使用, 保留以保持 API 一致性)
        command_name: command manager 中速度命令的名称
        command_threshold: 低于此速度认为机器人处于站立模式
        velocity_threshold: 激活迈步奖励的线速度阈值 (m/s)
        air_time_threshold: 计为一步的最小 air time (秒)

    Returns:
        shape (num_envs,) 的奖励 tensor
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 仅对站立 env 触发 (命令接近零)
    command = env.command_manager.get_command(command_name)
    total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    is_standing_cmd = (total_speed < command_threshold).float()

    # 获取 base 线速度幅值
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # 仅 XY 平面

    # 仅在速度高时 (被推) 奖励迈步
    should_step = vel_magnitude > velocity_threshold

    # 从 contact sensor 获取脚 air time
    contact_sensor = env.scene.sensors["feet_ground_contact"]
    air_time = contact_sensor.data.last_air_time[:, :2]  # (num_envs, 2) - 左右脚

    # 若任一脚最近在空中则奖励
    foot_in_air = (air_time > air_time_threshold).any(dim=1)  # (num_envs,)

    # 仅在: 站立命令 AND 高 body 速度 AND 脚迈步 时给奖励
    reward = is_standing_cmd * should_step.float() * foot_in_air.float()

    return reward


def adaptive_pose_weight(
    env: ManagerBasedRlEnv,
    base_pose_reward: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    velocity_threshold: float = 0.3,
    min_weight: float = 0.3,
) -> torch.Tensor:
    """机器人高速移动 (推力恢复) 时降低姿态跟踪权重.

    使机器人在恢复迈步时可偏离站立姿态, 而静止时保持严格姿态跟踪.

    Args:
        env: RL 环境
        base_pose_reward: 原始 pose 奖励 (加权前)
        asset_cfg: asset 配置 (未使用, 保留以保持 API 一致性)
        velocity_threshold: 开始降低权重的线速度阈值 (m/s)
        min_weight: 高速时的最小权重乘子 (0-1)

    Returns:
        shape (num_envs,) 的加权奖励 tensor
    """
    asset: Entity = env.scene[asset_cfg.name]

    # 获取 base 线速度幅值
    base_lin_vel = asset.data.root_link_vel_w[:, :3]  # (num_envs, 3)
    vel_magnitude = torch.norm(base_lin_vel[:, :2], dim=1)  # 仅 XY 平面

    # 计算权重: 静止时 1.0, 高速时 min_weight
    # 通过类 sigmoid 函数平滑过渡
    weight = min_weight + (1.0 - min_weight) * torch.exp(
        -(((vel_magnitude - velocity_threshold) / velocity_threshold).clamp(min=0.0) ** 2)
    )

    return base_pose_reward * weight


def randomize_base_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    max_pitch_deg: float = 10.0,
    max_roll_deg: float = 5.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """episode 开始时随机化 base 朝向以强制反应式行为.

    在每个 episode 开始时给机器人 base 朝向添加随机 pitch 和 roll.防止策略
    记忆单一初始状态, 迫使其使用反馈来适应不同朝向.

    Args:
        env: 环境
        env_ids: 要随机化的 env ID
        max_pitch_deg: 最大 pitch 角 (度, 前后倾斜)
        max_roll_deg: 最大 roll 角 (度, 侧向倾斜)
        asset_cfg: asset 配置
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    env.scene[asset_cfg.name]
    num_envs = len(env_ids)

    # 生成随机 pitch 和 roll 角
    max_pitch_rad = max_pitch_deg * torch.pi / 180.0
    max_roll_rad = max_roll_deg * torch.pi / 180.0

    pitch = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_pitch_rad
    roll = (torch.rand(num_envs, device=env.device) * 2 - 1) * max_roll_rad
    yaw = torch.zeros(num_envs, device=env.device)  # 保持 yaw 为 0

    # 欧拉角 (roll, pitch, yaw) 转四元数
    # 使用标准航空航天序列 (ZYX)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    quat_w = cr * cp * cy + sr * sp * sy
    quat_x = sr * cp * cy - cr * sp * sy
    quat_y = cr * sp * cy + sr * cp * sy
    quat_z = cr * cp * sy - sr * sp * cy

    new_quat = torch.stack([quat_w, quat_x, quat_y, quat_z], dim=1)

    # 归一化四元数
    new_quat = new_quat / torch.norm(new_quat, dim=1, keepdim=True)

    # 获取 root 位置索引 (freejoint 从 qpos index 0 开始)
    # Freejoint: [x, y, z, qw, qx, qy, qz]
    root_quat_idx = 3  # 四元数从 index 3 开始

    # 将随机化朝向应用到选中的环境
    env.sim.data.qpos[env_ids, root_quat_idx : root_quat_idx + 4] = new_quat


def set_face_down_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """将机器人设为俯卧 (腹朝下) 朝向, 用于 stand-up 训练.

    绕 pitch 轴 (Y) 向前旋转 90°, 使正面/腹部朝地、腿朝上.叠加随机 yaw.

    四元数推导:
        quat_pitch90 = [s, 0, s, 0]   其中 s = sqrt(2)/2  (绕 Y 旋转 90°)
        quat_yaw     = [cy, 0, 0, sy]
        combined     = quat_yaw * quat_pitch90 = [s*cy, -s*sy, s*cy, s*sy]
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0**-0.5  # sqrt(2)/2

    new_quat = torch.stack(
        [
            s * cy,  # w
            -s * sy,  # x
            s * cy,  # y
            s * sy,  # z
        ],
        dim=1,
    )

    # Freejoint qpos: [x, y, z, qw, qx, qy, qz, ...]
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0  # 清零速度


def set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.5,
):
    """随机将每个 env 初始化为面朝下 (腹) 或面朝上 (背), 带随机 yaw.

    Face-down:  +90° pitch → quat = [s*cy, -s*sy,  s*cy,  s*sy]
    Face-up:    -90° pitch → quat = [s*cy,  s*sy, -s*cy,  s*sy]

    Args:
        env: RL 环境.
        env_ids: 要 reset 的 env 索引 tensor.
        asset_cfg: 机器人 asset 的 scene entity 配置.
        face_down_prob: 采样 face-down (vs face-up) 的概率.课程可从高初始值
            (较易任务) 渐降到 0.5.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0**-0.5  # sqrt(2)/2

    face_down = torch.stack([s * cy, -s * sy, s * cy, s * sy], dim=1)
    face_up = torch.stack([s * cy, s * sy, -s * cy, s * sy], dim=1)

    mask = torch.rand(num, device=env.device) < face_down_prob  # True → face-down
    new_quat = torch.where(mask.unsqueeze(1), face_down, face_up)

    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0  # 清零速度


def set_random_ground_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    face_down_prob: float = 0.4,
    face_up_prob: float = 0.4,
    sitting_prob: float = 0.2,
    standing_prob: float = 0.0,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    sitting_z_min: float = 0.07,
    sitting_z_max: float = 0.09,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    sitting_joint_overrides: dict | None = None,
    sitting_joint_noise_std: float = 0.0,
    sitting_tilt_max: float = 0.0,
    face_up_roll_max: float = 0.0,
):
    """reset 到随机地面状态: face-down, face-up, sitting 或 standing.

    比 ``set_random_prone_orientation`` 更广 — stand-up env 用它让策略学会
    从任何合理姿态恢复, 包括 sitting keyframe (sit policy 的静止状态) 和
    已站立姿态 (也学会 *保持* 站立, 而非仅起立).

    模式 (概率归一化; 无需和为 1.0):
      - face-down (腹朝地): +90° pitch, 随机 yaw, z ∈ [prone_z_min, prone_z_max].
      - face-up   (背朝地): -90° pitch, 随机 yaw, z ∈ [prone_z_min, prone_z_max].
      - sitting:              直立 (±sitting_tilt_max), 随机 yaw, z 低,
                             关节设为 ``sitting_joint_overrides``.
      - standing:             直立 (±sitting_tilt_max), 随机 yaw, z ∈
                             [standing_z_min, standing_z_max], 关节保持 HOME
                             (``reset_robot_joints`` 所设).

    Args:
        env: RL 环境.
        env_ids: 要 reset 的 env 索引 tensor.
        asset_cfg: 机器人 asset 的 scene entity 配置.
        face_down_prob: 采样 face-down (腹) 桶的概率.
        face_up_prob: 采样 face-up (背) 桶的概率.
        sitting_prob: 采样 sitting 桶的概率.
        standing_prob: 采样 standing 桶的概率.
        prone_z_min: prone (face-down/up) 桶的 trunk z 下界 (m).
        prone_z_max: prone 桶的 trunk z 上界 (m).
        sitting_z_min: sitting 桶的 trunk z 下界 (m).
        sitting_z_max: sitting 桶的 trunk z 上界 (m).
        standing_z_min: standing 桶的 trunk z 下界 (m).
        standing_z_max: standing 桶的 trunk z 上界 (m).
        sitting_joint_overrides: ``{qpos_joint_index: angle_rad}``, 写入
            sitting 桶 env 的 ``qpos[7+idx]``.``None`` 保持关节为
            ``reset_robot_joints`` 已设的值.
        sitting_joint_noise_std: 在 ``sitting_joint_overrides`` 之上添加到
            sitting 桶关节的高斯噪声 std (rad).
        sitting_tilt_max: sitting/standing 桶的最大 pitch/roll 倾斜 (rad);
            0 = 完全直立.
        face_up_roll_max: face-up 桶的最大 roll (rad); 0 = 完全仰躺.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.int)
    num = len(env_ids)

    total = face_down_prob + face_up_prob + sitting_prob + standing_prob
    p_fd = face_down_prob / total
    p_fu = (face_down_prob + face_up_prob) / total
    p_sit = (face_down_prob + face_up_prob + sitting_prob) / total

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    s = 2.0**-0.5  # sqrt(2)/2

    face_down = torch.stack([s * cy, -s * sy, s * cy, s * sy], dim=1)
    face_up = torch.stack([s * cy, s * sy, -s * cy, s * sy], dim=1)
    # 直立 sitting: 默认仅 yaw, 可选 ±sitting_tilt_max 的 pitch/roll 噪声,
    # 使策略不过拟合到完全直立的起始姿态.
    if sitting_tilt_max > 0.0:
        pitch = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        roll = (torch.rand(num, device=env.device) * 2 - 1) * sitting_tilt_max
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        # ZYX intrinsic 欧拉角 → 四元数 (yaw * pitch * roll).
        sit_w = cr * cp * cy + sr * sp * sy
        sit_x = sr * cp * cy - cr * sp * sy
        sit_y = cr * sp * cy + sr * cp * sy
        sit_z = cr * cp * sy - sr * sp * cy
        sitting = torch.stack([sit_w, sit_x, sit_y, sit_z], dim=1)
    else:
        sitting = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=1)

    u = torch.rand(num, device=env.device)
    is_fu = (u >= p_fd) & (u < p_fu)
    is_sit = (u >= p_fu) & (u < p_sit)
    is_stand = u >= p_sit

    # face-up 部分 roll 噪声: 将仰躺姿态绕 body 长轴旋转均匀 ±face_up_roll_max.
    # 原因 (2026-07, back-recovery 曾依赖 seed 运气): 仰躺与俯卧之间的奖励
    # 地貌是平坦的 — upright_linear (cos tilt) 在整个 roll 中 ≈0, 高度不变 —
    # 因此从背上翻起仅通过后续的前升起路径获益, 这是一种长 horizon 依赖,
    # 噪声探索很少从完全平坦的仰躺起始发现.加入 roll 噪声后, 一部分
    # face-up spawn 从接近侧躺 (roll 路径中途) 开始: 策略从容易的起始学会
    # roll 完成并泛化回平坦仰躺 — 内置反向课程.均匀采样使每个难度都有
    # 代表 (平坦背 |roll|<15° 在 ±90° 时 ≈17%), 无需退火计划, 且多样的
    # 摔倒后姿态本身就是部署的现实 DR.
    if face_up_roll_max > 0.0:
        theta = (torch.rand(num, device=env.device) * 2 - 1) * face_up_roll_max
        ct = torch.cos(theta * 0.5)
        st = torch.sin(theta * 0.5)
        # Log-roll = 绕 body 长轴旋转, 即 body z (脊柱: 站立时 trunk z 朝上
        # → 躺下时水平).不是 body x — 仰躺时 body x 朝上, x-roll 仅会使
        # 机器人原地旋转, 如同已有的 yaw 噪声.
        # body 系旋转 → 右乘: q_fu ⊗ [ct, 0, 0, st].
        w, x, y, z = face_up[:, 0], face_up[:, 1], face_up[:, 2], face_up[:, 3]
        face_up = torch.stack(
            [
                w * ct - z * st,
                x * ct + y * st,
                y * ct - x * st,
                w * st + z * ct,
            ],
            dim=1,
        )

    # sitting 和 standing 共享相同的直立朝向 (identity + 可选 ±sitting_tilt_max);
    # 仅在 trunk 高度和关节姿态上不同.
    new_quat = face_down.clone()
    new_quat[is_fu] = face_up[is_fu]
    new_quat[is_sit] = sitting[is_sit]
    new_quat[is_stand] = sitting[is_stand]

    # 每个 env 的随机 z: face-down/up 用 prone 高度, sit 用低高度, stand 用 ~站立高度.
    z_prone = torch.rand(num, device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
    z_sit = torch.rand(num, device=env.device) * (sitting_z_max - sitting_z_min) + sitting_z_min
    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    new_z = z_prone.clone()
    new_z = torch.where(is_sit, z_sit, new_z)
    new_z = torch.where(is_stand, z_stand, new_z)

    env.sim.data.qpos[env_ids, 2] = new_z
    env.sim.data.qpos[env_ids, 3:7] = new_quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    # sitting 桶关节 override (如膝/踝弯曲到 keyframe).
    # override 键是 SERVO 索引 (14 关节布局); 转为 entity 关节索引, 使带交错
    # passive_* 关节 (backlash) 的 model 写入正确的关节.qpos 列 = 7 + entity
    # 关节索引 (robot free joint 在前, 所有铰链 1-dof).
    asset: Entity = env.scene[asset_cfg.name]
    servo_ids = _servo_joint_ids(env, asset)
    if sitting_joint_overrides:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            for jnt_idx, angle in sitting_joint_overrides.items():
                env.sim.data.qpos[sit_env_ids, 7 + servo_ids[jnt_idx]] = angle

    # sitting env 的关节噪声: 每个驱动关节上的高斯噪声, 使策略看到一系列
    # 合理 "sit" 起始而非单一标准姿态.覆盖真实世界迁移, standup policy
    # 从 sit policy 接管时关节角度不会精确匹配 SIT keyframe.
    if sitting_joint_noise_std > 0.0:
        sit_env_ids = env_ids[is_sit]
        if len(sit_env_ids) > 0:
            # 仅 servo 关节: passive_* 关节 (backlash 铰链) 行程极小,
            # reset 时必须保持为 0.
            n_sit = len(sit_env_ids)
            cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
            noise = torch.randn(n_sit, len(cols), device=env.device) * sitting_joint_noise_std
            env.sim.data.qpos[sit_env_ids.unsqueeze(1).long(), cols.unsqueeze(0)] += noise


# 深蹲锚定姿态 (velstand run-5): "卡住" 的恢复中途 basin —
# 膝盖折叠于身体下方, trunk 前倾, 脚平放.数值通过延伸 HOME zig-zag
# (hip fwd / knee back / ankle fwd, 符号约定按 SIT keyframe 折叠方向)
# 到深屈曲, 在 ±1.57 关节限位内.hip_yaw/hip_roll/neck 保持 HOME.
_CROUCH_ANCHOR_BY_NAME = {
    "left_hip_pitch": -1.15,
    "left_knee": 1.25,
    "left_ankle": 1.05,
    "right_hip_pitch": 1.15,
    "right_knee": -1.25,
    "right_ankle": -1.05,
}


def set_random_crouch_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    depth_min: float = 0.35,
    depth_max: float = 1.0,
    pitch_max_deg: float = 55.0,
    joint_noise: float = 0.12,
    z_stand: float = 0.115,
    z_deep: float = 0.06,
):
    """将选中 env reset 到随机恢复中途的蹲姿.

    恢复最后一英里的反向课程 (velstand run-5 教训): prone-init episode
    花费大部分摔倒预算到达深蹲, 到达后不久被回收, 因此蹲→站这段几乎
    无在策略数据 — 策略收敛到停在那里.跨该段 seed reset (深度 λ ∈
    [depth_min, depth_max] 在站立与深蹲锚点之间, trunk pitch 和 z 按 λ
    缩放) 使 episode 从 step 0 起前沿就密集.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]

    lam = torch.rand(num, device=env.device) * (depth_max - depth_min) + depth_min

    # 关节: 在腿 pitch 链上 lerp HOME → anchor, 仅 servo 关节加均匀噪声
    # (passive_* backlash 铰链 ±1° 行程 — 那里加噪声会 spawn 到限位外).
    joints = asset.data.default_joint_pos[env_ids].clone()
    for name, anchor in _CROUCH_ANCHOR_BY_NAME.items():
        ids, _ = asset.find_joints(f"^{name}$")
        j = ids[0]
        joints[:, j] = joints[:, j] + lam * (anchor - joints[:, j])
    noise_mask = torch.zeros(joints.shape[1], device=joints.device)
    noise_mask[_servo_joint_ids(env, asset)] = 1.0
    joints += (torch.rand_like(joints) * 2 - 1) * joint_noise * noise_mask

    # base 朝向: 前倾 pitch 随深度缩放 (stuck basin 从两个摔倒方向都是前蹲),
    # 随机 yaw, 小 roll 噪声.
    pitch = lam * math.radians(pitch_max_deg) + (torch.rand(num, device=env.device) * 2 - 1) * math.radians(10.0)
    pitch = torch.clamp(pitch, min=math.radians(5.0))
    roll = (torch.rand(num, device=env.device) * 2 - 1) * math.radians(8.0)
    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    # ZYX intrinsic 欧拉角 → 四元数 (yaw * pitch * roll), 同
    # set_random_ground_state 的 sitting 分支.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    # trunk 高度按深度缩放, 小幅上浮余量以干净落定.
    z = z_stand + lam * (z_deep - z_stand) + torch.rand(num, device=env.device) * 0.01

    env.sim.data.qpos[env_ids, 2] = z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qpos[env_ids, 7:] = joints
    env.sim.data.qvel[env_ids, :] = 0.0


def maybe_set_random_prone_orientation(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    prone_prob: float = 0.0,
    face_down_prob: float = 0.5,
    prone_z_min: float = 0.20,
    prone_z_max: float = 0.25,
    crouch_prob: float = 0.0,
):
    """以概率 `prone_prob` 将朝向 override 为 prone 的 reset 事件.

    以概率 `prone_prob` 将 (reset_base 已设的) 直立朝向替换为 prone 朝向;
    否则保持直立.在被 override 的 env 中, `face_down_prob` 选择 face-down
    (腹) vs face-up (背).

    同时将被 override env 的 z 提升到 [prone_z_min, prone_z_max] 以保证
    头/颈间隙 — vel-env reset z (~0.125) 在 90° pitch 时会使头穿地.

    prone_prob=2/3 且 face_down_prob=0.5 时得到平衡的 33/33/33 直立/
    face-down/face-up reset 分布, 这是学习摔倒恢复与正常直立起始的标准混合.

    ``crouch_prob`` > 0 时, 额外的独占 env 切片通过
    ``set_random_crouch_state`` reset 到随机恢复中途蹲姿 (恢复最后一英里
    的反向课程 — 见其 docstring).
    """
    if prone_prob <= 0.0 and crouch_prob <= 0.0:
        return
    # env_ids=None 表示 "全部 env" (初始全局 reset 传 None —
    # 旧的 early-return 静默跳过了那里的 prone init).
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    env_ids_t = (
        env_ids.to(env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor)
        else torch.tensor(env_ids, device=env.device, dtype=torch.long)
    )
    # 一次抽取将 env 划分为独占的 prone / crouch / 未触碰 切片.
    u = torch.rand(len(env_ids_t), device=env.device)
    selected = env_ids_t[u < prone_prob]
    crouch_selected = env_ids_t[(u >= prone_prob) & (u < prone_prob + crouch_prob)]
    if len(selected) > 0:
        set_random_prone_orientation(env, selected, asset_cfg=asset_cfg, face_down_prob=face_down_prob)
        # override z 使 prone body 落定时有头/颈间隙.
        z = torch.rand(len(selected), device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
        env.sim.data.qpos[selected, 2] = z
    if len(crouch_selected) > 0:
        set_random_crouch_state(env, crouch_selected, asset_cfg=asset_cfg)


def event_param_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    param_stages: list[dict],
) -> torch.Tensor:
    """在计划 step 处修改事件项的 params.

    termination_param_curriculum 的镜像, 但用于事件.通过 get_term_cfg
    使用活跃 EventManager term cfg, 因为 env.cfg.events 是 deepcopy.
    param_stages: {step: int, params: dict} 列表.在最新匹配阶段浅合并到
    活跃事件项的 params 中.
    """
    del env_ids
    event_cfg = env.event_manager.get_term_cfg(event_name)
    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    event_cfg.params.update(current)
    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


def face_down_prob_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    prob_stages: list[dict],
) -> torch.Tensor:
    """在训练过程中逐步调整 reset 事件的 face_down_prob.

    Args:
        env: RL 环境.
        env_ids: 要 reset 的 env 索引 tensor (未使用).
        event_name: 使用 set_random_prone_orientation 的事件项名称
        prob_stages: {step: int, prob: float} 列表.prob 越高 = face-down
            reset 越多 (较易任务); 随训练进行逐步到 0.5.
    """
    del env_ids

    # 注意: 必须更新活跃的 EventManager term_cfg, 而非 env.cfg.events —
    # EventManager.__init__ 会 deepcopy(cfg), 因此修改 env.cfg.events 是空操作.
    event_cfg = env.event_manager.get_term_cfg(event_name)

    current_prob = prob_stages[0]["prob"]
    for stage in prob_stages:
        if env.common_step_counter > stage["step"]:
            current_prob = stage["prob"]

    event_cfg.params["face_down_prob"] = current_prob
    return torch.tensor([current_prob])


class VelocityCommandCommandOnly(UniformVelocityCommand):
    """类似 UniformVelocityCommand 但仅绘制命令箭头 (无实际速度箭头)."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # 原地转弯练习: 对一部分 env, 线速度清零并强制有意义的 (远离零的)
        # yaw 命令.独立均匀采样几乎不产生 "lin≈0, |ang| 大" (~2% 样本),
        # 因此原地旋转实际上未训练 → 真实机器人转弯慢/不稳定.
        # 镜像基础 rel_forward_envs 机制.
        p = getattr(self.cfg, "rel_turn_in_place_envs", 0.0)
        if p <= 0.0:
            return
        r = torch.empty(len(env_ids), device=self.device)
        turn_ids = env_ids[r.uniform_(0.0, 1.0) < p]
        if len(turn_ids) == 0:
            return
        self.vel_command_b[turn_ids, 0] = 0.0
        self.vel_command_b[turn_ids, 1] = 0.0
        lo, hi = self.cfg.ranges.ang_vel_z
        maxr = max(abs(lo), abs(hi))
        rr = torch.empty(len(turn_ids), device=self.device)
        sign = torch.where(rr.uniform_(0.0, 1.0) < 0.5, -1.0, 1.0)
        mag = torch.empty(len(turn_ids), device=self.device).uniform_(0.4 * maxr, maxr)
        self.vel_command_b[turn_ids, 2] = sign * mag
        # 这些 env 必须实际转弯 — 取消其站立标记 (否则会清零命令)
        # 并刷新世界系参考副本.
        self.is_standing_env[turn_ids] = False
        self.vel_command_w[turn_ids] = self.vel_command_b[turn_ids]

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return

        cmds = self.command.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_w = self.robot.data.root_link_quat_w
        base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()

        base_pos_w = base_pos_ws[batch]
        base_mat_w = base_mat_ws[batch]
        cmd = cmds[batch]

        if np.linalg.norm(base_pos_w) < 1e-6:
            return

        def local_to_world(vec: np.ndarray) -> np.ndarray:
            return base_pos_w + base_mat_w @ vec

        scale = self.cfg.viz.scale * 2.0
        z_offset = self.cfg.viz.z_offset

        # 命令线速度箭头 (蓝色).
        cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
        cmd_lin_to = local_to_world((np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale)
        visualizer.add_arrow(cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015)


@_dataclass(kw_only=True)
class VelocityCommandCommandOnlyCfg(UniformVelocityCommandCfg):
    """VelocityCommandCommandOnly 的 cfg — 添加原地转弯重采样."""

    # 每次 resample 中被命令原地转弯的 env 比例 (lin=0, |ang| 强制到
    # [0.4·max, max]).0 = 禁用 (仅基础均匀采样).
    rel_turn_in_place_envs: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "VelocityCommandCommandOnly":
        """构建 VelocityCommandCommandOnly 命令项."""
        return VelocityCommandCommandOnly(self, env)


class RelativeHeadingVelocityCommand(VelocityCommandCommandOnly):
    """cmd[2] 为机器人 body 系下朝向误差的速度命令.

    cmd[0] = lin_vel_x  (油门: 0=滑行, +推, -刹)
    cmd[1] = lin_vel_y  (未用, 0)
    cmd[2] = heading_error  (+ = 目标在右/CW, - = 在左/CCW)
             0 → 直行, ±max = 目标在右/左 max_angle rad

    训练时: 每次 episode reset 采样随机世界系朝向.每步
    cmd[2] = clamp(wrap(current_yaw - target_yaw), ±max_angle).
    机器人指向目标 CCW (左) 时为正 → 需右转.

    推理时: 用户直接传入 cmd[2].保持 cmd[2] = 常数给出比例朝向修正 ≈
    恒定转弯速率.

    cfg 中设 heading_command=False 和 rel_heading_envs=0.0 (朝向内部处理).
    cfg 中的 ang_vel_z 范围用作 cmd[2] 的 clip 限值.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        """以 per-env target heading 初始化 heading command 项."""
        super().__init__(cfg, env)
        # 每环境采样的目标朝向, world frame (rad)
        self._target_heading_w = torch.zeros(self.num_envs, device=self.device)
        # cmd[2] 的 clip 上限: 用 cfg 中 ang_vel_z[1] (正向上限)
        ang_rng = cfg.ranges.ang_vel_z
        self._heading_max = float(ang_rng[1]) if ang_rng else 1.0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        n = len(env_ids)
        # 在 [-π, π] 内均匀采样随机 world-frame 目标朝向
        self._target_heading_w[env_ids] = torch.rand(n, device=self.device) * 2.0 * math.pi - math.pi
        # ang_vel 槽清零; _update_command 每步会填充它
        self.vel_command_b[env_ids, 2] = 0.0

    def _update_command(self) -> None:
        # 不要调用 super()._update_command() — 它会运行 heading
        # 比例控制器并用 yaw 速率覆盖 cmd[2].
        # 改为每步从头重新计算 heading error.
        quat = self.robot.data.root_link_quat_w  # (N, 4) [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # 正值 = 目标在机器人 CCW (左) 侧 → 左转. 标准约定.
        delta = self._target_heading_w - current_yaw
        heading_error = torch.atan2(torch.sin(delta), torch.cos(delta))
        self.vel_command_b[:, 2] = heading_error.clamp(-self._heading_max, self._heading_max)

    def _update_metrics(self) -> None:
        pass  # No velocity tracking metrics for heading command


class RelativeHeadingVelocityCommandCfg(UniformVelocityCommandCfg):
    """Cfg for RelativeHeadingVelocityCommand — heading-error velocity command 的 Cfg."""

    def build(self, env: ManagerBasedRlEnv) -> "RelativeHeadingVelocityCommand":
        """构造 RelativeHeadingVelocityCommand command 项."""
        return RelativeHeadingVelocityCommand(self, env)


def heading_tracking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.5,
) -> torch.Tensor:
    """奖励减少 heading error, 当 cmd[2] 编码 heading error 时.

    返回 exp(-cmd[2]² / std²).
    - error = 0 (在 heading 上): reward = 1.0.
    - error = std: reward ≈ 0.37 (强梯度).
    - error = 1.0 rad, std=0.5: reward ≈ 0.018 (近零).

    std=0.5 rad (≈28°) 在预期范围内给出有意义的梯度.
    """
    cmd = env.command_manager.get_command(command_name)
    heading_error = cmd[:, 2]
    return torch.exp(-(heading_error**2) / (std**2))


def skating_air_time_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.4,
    vel_gate_ref: float = 0.0,
) -> torch.Tensor:
    """仅当 pushing (cmd_x > 0) 时奖励 feet air time.

    鼓励机器人在 skating stroke 的 recovery phase 抬起每只脚, 而非在地面拖动.
    以 cmd_x 缩放, 因此激励随 push intensity 增长.

    当 ``vel_gate_ref`` > 0 时, reward 还乘以 forward-speed gate, 因此不推动 body
    (原地 tap-dancing) 而抬脚将无收益. ``threshold_min`` 设置计入的最短 swing —
    调高以禁止急促高频 flutter.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    reward = reward * torch.clamp(cmd_x, min=0.0)
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        reward = reward * gate
    return reward


def _forward_progress_gate(env: ManagerBasedRlEnv, v_ref: float) -> torch.Tensor | None:
    """机体前进速度的 0→1 ramp: 站立时为 0, 达到/超过 v_ref 时为 1.

    用于 gate 步态塑形奖励, 使不推进机体的迈步 (如原地踢踏舞) 一无所获 — 只有当
    步态真正完成其工作 (向前移动) 时才支付步态 FORM 的奖励. 禁用时返回 None (v_ref <= 0).
    """
    if v_ref <= 0.0:
        return None
    v_fwd = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return (v_fwd.clamp(min=0.0) / v_ref).clamp(max=1.0)


def single_support_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_gate_ref: float = 0.0,
    double_penalty: float = 0.25,
) -> torch.Tensor:
    """奖励单支撑 (滑冰步态), 轻微抑制 swizzle.

    真正的滑冰是 STRIDE: 一只刀片蹬地时另一只摆动, 即交替左右的单支撑.
    对称的 swizzle 始终保持两只刀片着地并仍能转动轮子, 所以仅靠 wheel_speed
    会收敛到它.

    每步统计接触刀片数:
      - 恰好 1 只着地 (stride)  → + clamp(cmd_x,0) · gate
      - 2 只着地 (双支撑)      → − double_penalty · clamp(cmd_x,0)
      - 0 只着地 (飞行/跳跃)    →  0

    正向的单支撑奖励由前进速度 gate (``vel_gate_ref``), 使原地踏步 (无推进) 一无所获 —
    消除 tap-dance hack. 双支撑惩罚较小且不 gate: 体重转移/蹬地期间的短暂双支撑是
    正常滑冰, 所以仅轻微抑制永久双支撑 (swizzle) 而非禁止. 真正的反 swizzle 信号是
    skating_air_time — swizzle 从不抬脚.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None

    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)  # (num_envs,)
    single = (n_contact == 1).float()
    double = (n_contact >= 2).float()

    cmd_x = torch.clamp(env.command_manager.get_command(command_name)[:, 0], min=0.0)
    single_r = single * cmd_x
    gate = _forward_progress_gate(env, vel_gate_ref)
    if gate is not None:
        single_r = single_r * gate
    return single_r - double_penalty * double * cmd_x


def glide_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    vel_ref: float = 0.2,
    stillness_std: float = 5.0,
    asset_cfg: SceneEntityCfg = _LEG_JOINTS_ASSET_CFG,
) -> torch.Tensor:
    """奖励步态的 GLIDE 阶段: 单刀滑行且腿部安静.

    没有其他项奖励滑行 — skating_air_time 为每次摆动付费, 所以策略最大化
    摆动 FREQUENCY (狂踢). 此项为单脚滑行付费, 给策略一个放慢并每次划水
    都到位的理由:

        reward = single_support · forward_gate · stillness · (cmd_x >= 0)

    - single_support: 恰好一只刀片接触. 必需 — 这是修复早期 broken glide 的
      关键, 之前缺少此项使双刀 swizzle-coast 能刷奖励并退化步态.
    - forward_gate = clamp(v_fwd,0,vel_ref)/vel_ref → 不前进时为 0.
    - stillness = exp(-Σ leg_joint_vel² / stillness_std²) → 仅当腿部安静时高;
      踢腿 (快速关节运动) ≈ 0, 所以只有真正的滑行才付费.
    - 仅在 push/coast 时激活 (cmd_x >= 0); 刹车时静默.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    single = (torch.sum((contact_time > 0.0).float(), dim=1) == 1).float()

    forward_gate = _forward_progress_gate(env, vel_ref)
    if forward_gate is None:
        forward_gate = torch.ones(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    stillness = torch.exp(-joint_vel_sq / stillness_std**2)

    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    active = (cmd_x >= 0.0).float()
    return single * forward_gate * stillness * active


def leg_symmetry_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_bases: tuple = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"),
) -> torch.Tensor:
    """奖励左右腿镜像 — swizzle 的标志性对称性.

    机器人使用镜像的 L/R 符号约定, 所以双侧对称配置满足 q_left + q_right ≈ 0
    (每对匹配关节). 返回 ``-mean_pairs |q_left + q_right|`` (L1, 常数梯度);
    使用 POSITIVE weight 使不对称被惩罚, 对称 swizzle 被青睐. L/R 索引对一次
    性按名称解析并缓存在 env 上.
    """
    asset: Entity = env.scene[asset_cfg.name]
    if not hasattr(env, "_leg_sym_ids"):
        left, right = [], []
        for base in joint_bases:
            li, _ = asset.find_joints([f"left_{base}"])
            ri, _ = asset.find_joints([f"right_{base}"])
            left.append(li[0])
            right.append(ri[0])
        env._leg_sym_ids = (
            torch.tensor(left, device=env.device),
            torch.tensor(right, device=env.device),
        )
    lids, rids = env._leg_sym_ids
    q = asset.data.joint_pos
    return -torch.abs(q[:, lids] + q[:, rids]).mean(dim=-1)


def grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
) -> torch.Tensor:
    """奖励双刀片接触 — 经典 swizzle 保持着地 (不抬脚).

    single_support_reward 的镜像, 但奖励双支撑 (n_contact >= 2), 按 |cmd_x| 缩放,
    使其在任一方向塑形 push 阶段 (前进或后退 — swizzle env 以 cmd_x < 0 表示 "后退").
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    cmd_x = torch.abs(env.command_manager.get_command(command_name)[:, 0])
    return grounded * cmd_x


def gait_symmetry_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """惩罚左右脚使用失衡 (一只刀片承担大部分工作).

    symmetry augmentation 关闭时, 没有什么阻止策略学习主要用一条腿蹬地的
    不对称步态 — 这会偏航并失稳 (尤其在起步时). 累积每脚的摆动时间并惩罚
    归一化失衡 |L - R| / (L + R):
      - 平衡交替步态          -> ~0 (无惩罚)
      - 一脚摆动远多于另一脚  -> ~1 (最大惩罚)
    仅惩罚 CUMULATIVE 失衡 — 真实步态瞬时单支撑不对称 (此刻一只脚在摆动) 是正常的.
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    air = sensor.data.current_air_time  # (N, num_feet)
    assert air is not None

    if not hasattr(env, "_swing_accum") or env._swing_accum.shape[0] != env.num_envs:
        env._swing_accum = torch.zeros(env.num_envs, air.shape[1], device=env.device)
    reset = env.episode_length_buf <= 1
    env._swing_accum[reset] = 0.0
    env._swing_accum += (air > 0.0).float() * env.step_dt

    L = env._swing_accum[:, 0]
    R = env._swing_accum[:, 1]
    return torch.abs(L - R) / (L + R + 1e-3)


def heading_hold_reward(
    env: ManagerBasedRlEnv,
    std: float = 0.4,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励保持 SPAWN 朝向 (直行) — 校正式, 基于角度.

    奖励 yaw ANGLE 保持在 reset 时捕获的朝向附近:
        reward = exp(-wrap(yaw - yaw_spawn)² / std²)

    这是直行的正确方式 (相对于惩罚 yaw-RATE, 那只是告诉策略 '永远不要转弯' →
    它无法回正并开环漂移). 这里漂移会降低奖励, 策略可自由 yaw 回正来恢复奖励.

    spawn 朝向在每个 env reset 后的最初几步 (episode_length_buf <= 1) 捕获,
    此时机器人仍在 spawn 姿态附近. 读取 root_link_quat_w, 它在 reward 时
    (物理步进后) 是新鲜的. Heading-invariant: 参考是每个 env 自己的随机
    spawn yaw, 所以与 reset 时的全圆 yaw 随机化兼容.
    """
    asset: Entity = env.scene[asset_cfg.name]
    quat = asset.data.root_link_quat_w  # (N, 4) [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    if not hasattr(env, "_heading_ref") or env._heading_ref.shape[0] != env.num_envs:
        env._heading_ref = yaw.clone()
    just_reset = env.episode_length_buf <= 1
    env._heading_ref = torch.where(just_reset, yaw, env._heading_ref)

    err = yaw - env._heading_ref
    err = torch.atan2(torch.sin(err), torch.cos(err))  # wrap to [-π, π]
    return torch.exp(-(err**2) / std**2)


def action_over_limit_penalty(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    overshoot: float = 0.3,
) -> torch.Tensor:
    """惩罚命令关节目标超过其硬限位 (+ overshoot).

    策略侧威慑, 防止过驱关节撞到机械限位: 例如 hip_roll 的限位是 ±0.38 rad 但
    ctrlrange 是 ±10 rad, 所以低-kp servo 可被命令远超限位以最大力矩撞击 —
    这是脆弱的 sim-only trick, 无法迁移.

    读取命令目标 (raw_action · scale + offset), 仅惩罚 BEYOND
    (hard_limit + overshoot) 的部分:

        penalty = Σ relu(target - (hi + overshoot)) + relu((lo - overshoot) - target)

    与 qpos-limit 惩罚不同, 此项在 COMMAND 上触发, 而非关节位置 — 所以关节仍可
    达到其全范围 (command ≈ limit), 不会偷走可用幅值. 因为约束的是策略的 OUTPUT,
    学习到的行为被烘焙进网络, 迁移到部署时无需任何 env 侧 action clip (那只会
    在 sim 中存在 → 不一致). ``overshoot`` 给低-kp servo 在负载下达到近限位目标
    的余量; 仅超出此的过驱被惩罚.
    """
    term = env.action_manager.get_term(action_name)
    target = term.raw_action * term.scale + term.offset  # (B, action_dim) abs targets
    jnt_ids = term.target_ids
    hard = env.scene["robot"].data.joint_pos_limits[:, jnt_ids]  # (B, action_dim, 2)
    lo = hard[..., 0] - overshoot
    hi = hard[..., 1] + overshoot
    over = (target - hi).clip(min=0.0) + (lo - target).clip(min=0.0)
    return torch.sum(over, dim=-1)


def forward_lean_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    target_pitch: float = 0.08,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = _TRUNK_BASE_ASSET_CFG,
) -> torch.Tensor:
    """奖励推进时轻微前倾, 以抵消滑冰划水产生的后向力矩.

    使用 projected_gravity_b 的 x 分量作为 pitch 代理:
      forward_lean = -gravity_b[:, 0]  (前倾时为正)

    仅在 cmd_x > 0 时触发. 在 target_pitch 弧度前倾时达到峰值.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd_x = env.command_manager.get_command(command_name)[:, 0]
    forward_lean = asset.data.projected_gravity_b[:, 0]
    push = torch.clamp(cmd_x, min=0.0)
    return push * torch.exp(-((forward_lean - target_pitch) ** 2) / (std**2))


class GroundPickPhaseCommand(UniformVelocityCommand):
    """ground pick / sit-stand 任务的相位编码命令.

    用循环相位信号替换速度命令:
        command = [cos(2π*phase), sin(2π*phase), 0]

    Phase ∈ [0, 0.5]: approach (下降).
    Phase ∈ [0.5, 1.0]: return (回升).

    Phase 在 episode reset 时按 env 随机化以解耦 envs.
    周期默认 4s; 通过 cfg.period 覆盖 (sitstand 用 8s 以更慢, 更温和的下蹲).
    """

    PERIOD: float = 4.0  # 默认; cfg.period 覆盖

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        """初始化 ground-pick 相位命令, 带 per-env 相位计数器."""
        super().__init__(cfg, env)
        self._gp_phase = torch.zeros(self.num_envs, device=self.device)
        self._period = float(getattr(cfg, "period", self.PERIOD))
        # 为 False 时, 每个 episode 从 phase 0 (站立) 开始, 而非随机相位.
        # 与 runtime 一致, runtime 中按钮从站立状态以 phase 0 启动循环.
        # 默认 True 保留历史 ground_pick 行为 (随机相位以解耦 envs).
        self._randomize_phase = bool(getattr(cfg, "randomize_phase", True))

    @property
    def command(self) -> torch.Tensor:
        """当前相位编码命令向量 [cos, sin, 0]."""
        return self.vel_command_b

    def compute(self, dt: float) -> None:
        """按 dt 推进相位并刷新命令向量."""
        self._gp_phase = (self._gp_phase + dt / self._period) % 1.0
        self.vel_command_b[:, 0] = torch.cos(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 1] = torch.sin(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 2] = 0.0

    def reset(self, env_ids: torch.Tensor | None) -> dict:
        """为给定 envs 重置相位计数器."""
        if env_ids is not None and len(env_ids) > 0:
            if self._randomize_phase:
                self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
            else:
                self._gp_phase[env_ids] = 0.0
        return {}

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass  # 相位连续; 无需重采样

    def _update_command(self) -> None:
        pass  # 在 compute() 中更新

    def _update_metrics(self) -> None:
        pass  # ground pick 无速度跟踪指标


@_dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    """GroundPickPhaseCommand 的 Cfg — 循环相位命令."""

    class_type: type = GroundPickPhaseCommand
    period: float = 4.0  # 周期长度 (秒); sitstand 用 8.0
    randomize_phase: bool = True  # False -> 每个 episode 从 phase 0 (站立) 开始

    def build(self, env: ManagerBasedRlEnv) -> "GroundPickPhaseCommand":
        """构建 GroundPickPhaseCommand 命令项."""
        return GroundPickPhaseCommand(self, env)


# --------------------------------------------------------------------------- #
# 统一 pose 命令机制                                                           #
# --------------------------------------------------------------------------- #
#
# 背景: 我们弃用了旧的 NeckOffsetJointPositionAction +
# disturbance-randomization 方案 (其中头/身体运动是策略应鲁棒对待的外部
# 扰动). 那训练的是弱, 间接信号 — 见 `project_neck_offset_decoupling.md`
# 的事后分析.
#
# 替代: head 和 body pose 现在是 *commands* — 直接, 密集的策略输入, 带
# 跟踪奖励. 部署时, runtime 用用户请求的 pose 填充这些 slot; 训练时,
# 它们从 per-dim 范围均匀采样 (从 step 0 起保持非零以使输入神经元存活) 并
# 通过课程 ramp.
#
# 布局, 跨所有 microduck 策略统一以兼容 runtime obs:
#   command vector (13D) = [vx, vy, vtheta,           ← "twist" (速度)
#                           neck_pitch, head_pitch,   ← "head_pose" (delta)
#                           head_yaw, head_roll,
#                           body_x, body_y, body_z,   ← "body_pose" (delta)
#                           body_roll, body_pitch, body_yaw]
# 策略 obs 总计 61D (51 - 3 + 13).
# --------------------------------------------------------------------------- #


class UniformPoseCommand(CommandTerm):
    """通用 N-dim 均匀 pose 命令.

    每维独立在 cfg.ranges[i] = (lo, hi) 均匀采样并在重采样间保持值. 无指标,
    无 debug viz — 保持轻量因为我们有很多这样的命令.
    """

    cfg: "UniformPoseCommandCfg"

    def __init__(self, cfg: "UniformPoseCommandCfg", env: ManagerBasedRlEnv):
        """初始化 uniform pose 命令, 带零命令缓冲区."""
        super().__init__(cfg, env)
        self.dim = len(cfg.ranges)
        self._command = torch.zeros(self.num_envs, self.dim, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """当前采样的 pose 命令向量."""
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        r = torch.empty(n, device=self.device)
        for i, (lo, hi) in enumerate(self.cfg.ranges):
            self._command[env_ids, i] = r.uniform_(lo, hi)
        # 显式零命令桶. 均匀采样基本不会产生全零命令, 所以部署 idle 情况
        # ("保持名义 pose") 否则在训练中缺失 (velocity body-control run-1 教训:
        # 策略仅在命令存在时才静止).
        if self.cfg.zero_command_prob > 0.0:
            zero_mask = torch.rand(n, device=self.device) < self.cfg.zero_command_prob
            self._command[env_ids[zero_mask]] = 0.0


@_dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    """Per-dim 均匀范围; 构建 UniformPoseCommand."""

    # 每维的 (lo, hi) 元组. 长度定义命令维度.
    ranges: tuple[tuple[float, float], ...] = ()
    # 重采样产生精确全零命令的概率.
    zero_command_prob: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "UniformPoseCommand":
        """构建 UniformPoseCommand 命令项."""
        return UniformPoseCommand(self, env)


def zero_command_padding(
    env: ManagerBasedRlEnv,
    dim: int,
) -> torch.Tensor:
    """宽度 `dim` 的常量零 obs 项.

    用于不主动跟踪 head/body 命令的 env (如 sitstand, ground_pick), 但仍需
    统一 61D obs 形状以使 runtime 能用相同缓冲区布局喂所有策略.
    """
    return torch.zeros(env.num_envs, dim, device=env.device)


def head_pose_tracking(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    std: float = 0.5,
    fine_std: float | None = None,
    fine_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """匹配命令的 neck/head delta 的 per-joint Gaussian 奖励.

    4 个 neck/head 关节的 exp(-(err/std)^2) 的均值. 结果是 (N,) ∈ [0, 1].
    Mean 形式 (vs sum-of-squares) 在仅一个关节偏差时保持梯度 — vs SOS 中
    单个大误差杀死整个奖励.

    `std` 是 per-joint 容差: err=std 时 per-joint 奖励为 1/e (~0.37). 选 std
    量级为命令范围, 使梯度在课程扩大时不死.

    `fine_std` (可选) 混入第二个窄 Gaussian:
    (1-fine_weight)·exp(-(err/std)²) + fine_weight·exp(-(err/fine_std)²).
    原因: 单一宽 std (0.5 rad ≈ 29°) 使小误差几乎免费 — 重头上 10° 重力
    下垂仅消耗 ~0.03 奖励, 所以策略任其下垂. 窄分量 (~0.1 rad) 为那些小
    误差定价, 宽分量在课程扩大时保持远命令处的梯度.

    cmd 形状 (N, 4) = 默认关节位置的 delta, 顺序为
    [neck_pitch, head_pitch, head_yaw, head_roll].

    在 backlash model 上, 测量角度是 qpos[servo] + qpos[backlash] — OUTPUT
    link, 也是 encoder obs (joint_pos_rel_backlash) 报告的. 仅测量 servo 会
    让头下垂 backlash 间隙奖励免费 AND 惩罚策略补偿它 (servo 偏上 = servo
    侧 "误差"). 在无 passive_*_backlash 关节的 model 上 mask 为 0, 此项
    退化为 servo.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 4)

    if not hasattr(env, "_head_pose_neck_ids"):
        ids, names = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
        env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        name_to_id = {n: i for i, n in enumerate(asset.joint_names)}
        bl = [name_to_id.get(f"passive_{n}_backlash") for n in names]
        env._head_pose_bl_ids = torch.tensor([0 if b is None else b for b in bl], device=env.device, dtype=torch.long)
        env._head_pose_bl_mask = torch.tensor([0.0 if b is None else 1.0 for b in bl], device=env.device)

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = joint_pos[:, neck_ids] + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    actual = measured - asset.data.default_joint_pos[:, neck_ids]
    err = actual - cmd
    per_joint = torch.exp(-((err / std) ** 2))
    if fine_std is not None:
        per_joint = (1.0 - fine_weight) * per_joint + fine_weight * torch.exp(-((err / fine_std) ** 2))
    return per_joint.mean(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 传感器导出的 critic obs 的 NaN-safe wrapper.
#
# `robot_state_is_nan` 覆盖关节 + root state, 所以从中导出的每个 obs 都受其
# 触发的 reset 保护. 下面三项不在此列: 它们读传感器数据 (raycast 高度,
# contact air-time, contact force), 这些 MuJoCo 可在集成的机器人 state 仍干净
# 时返回非有限值. 它们仅用于 critic, 所以单步清理对策略零成本, 而让值通过
# 会通过 rsl_rl 的 check_nan 杀死整个 run. 在此清理不掩盖真实的物理爆炸 —
# 那些仍通过 nan_state 终止并显示为 wandb 中的 Episode_Termination/nan_state.
# ─────────────────────────────────────────────────────────────────────────────


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def foot_contact_forces_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_contact_forces` (见上文说明)."""
    return _finite(_velocity_obs.foot_contact_forces(env, sensor_name))


def foot_height_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_height` (见上文说明)."""
    return _finite(_velocity_obs.foot_height(env, sensor_name))


def foot_air_time_safe(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """NaN-safe `foot_air_time` (见上文说明)."""
    return _finite(_velocity_obs.foot_air_time(env, sensor_name))


def head_pose_bias_penalty(
    env: ManagerBasedRlEnv,
    command_name: str = "head_pose",
    tau_s: float = 1.0,
    gate_height_low: float | None = None,
    gate_height_high: float = 0.11,
    gate_tilt_full_deg: float = 20.0,
    gate_tilt_zero_deg: float = 45.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚时间平均 (DC) 的 neck/head 跟踪误差: -mean(|EMA(err)|).

    ``head_pose_tracking`` 的伴侣, 后者评分 INSTANTANEOUS 误差. 为何用单独的 DC 项
    而非仅收紧那个 Gaussian 的 std: 走路不可避免地摇晃占机器人质量 38% 的头,
    所以瞬时紧容差项是对走路的永久税收, 没有策略能逃脱 — 实测 ~0.77/step, 对比
    air_time 奖励 ~1.01/step, 这正是让 velocity run 2026-08-20 完全放弃踏步的原因
    (wandb 5yay13u4). 稳态下垂 IS escapable: 策略可偏置 neck 命令向上以抵消重力下垂.
    在 ``tau_s`` 上平均使振荡抵消, 仅定价 bias.

    故意用 L1 (非 Gaussian): 大 bias 处梯度保持常数, 而 Gaussian 会变平并死.

    在 backlash model 上测量角度透过间隙读取, 与 head_pose_tracking 和 encoder obs 一致.

    ``gate_height_low`` (可选): recovery env (standup / velstand) 的直立 gate, 形状和
    语义同 body_ang_vel_at_height — 低于 gate_height_low 或超过 gate_tilt_zero_deg
    倾斜时为零, 高于 gate_height_high 且低于 gate_tilt_full_deg 时满. gate 乘以
    喂入 EMA 的 ERROR (不仅是输出): 摔倒/起立时 EMA 见零并衰减, 所以到达直立时
    bias 时钟从 ~0 开始, 而非在终点线收取整个地面阶段累积的误差 — 那将是 recovery
    完成前的奖励墙, 正是已退役 head_impact_penalty 的失败模式. 输出也 gate, 所以
    新摔立即停止累积.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 4)

    if not hasattr(env, "_head_pose_neck_ids"):
        # 与 head_pose_tracking 共享 id 缓存 (任一可能先运行).
        head_pose_tracking(env, command_name=command_name, asset_cfg=asset_cfg)

    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = joint_pos[:, neck_ids] + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    err = (measured - asset.data.default_joint_pos[:, neck_ids]) - cmd

    if gate_height_low is not None:
        z = torch.nan_to_num(
            asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2],
            nan=0.0,
        )
        t = torch.clamp(
            (z - gate_height_low) / max(gate_height_high - gate_height_low, 1e-6),
            0.0,
            1.0,
        )
        gate = t * t * (3.0 - 2.0 * t)
        quat = asset.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt.clamp(-1.0, 1.0)))
        st = torch.clamp(
            (gate_tilt_zero_deg - tilt_deg) / max(gate_tilt_zero_deg - gate_tilt_full_deg, 1e-6),
            0.0,
            1.0,
        )
        gate = gate * (st * st * (3.0 - 2.0 * st))
        err = err * gate.unsqueeze(-1)
    else:
        gate = None

    if not hasattr(env, "_head_bias_ema"):
        env._head_bias_ema = torch.zeros_like(err)
    # 刚 reset 的 env: 丢弃上一 episode 的累积 bias.
    fresh = env.episode_length_buf <= 1
    env._head_bias_ema[fresh] = 0.0

    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    env._head_bias_ema = (1.0 - alpha) * env._head_bias_ema + alpha * err
    out = -env._head_bias_ema.abs().mean(dim=-1)
    if gate is not None:
        out = out * gate
    return out


def body_pose_tracking_6d(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.095,
    xy_std: float = 0.02,
    z_std: float = 0.01,
    angle_std: float = math.radians(8),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """跟踪命令 body pose 的 6 个 per-axis Gaussian 奖励的均值.

    cmd 形状 (N, 6) = [x, y, z, roll, pitch, yaw], 均为相对名义站立 pose 的
    delta (xy 为相对 spawn origin 的 delta, z 为相对 nominal_height 的 delta,
    角度为相对直立 = 0 的 delta).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 6)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    # 相对 env spawn origin 的位置. 用 nan_to_num 因为 MuJoCo 在接触不稳定时
    # 可能产生 NaN, 我们不想污染奖励.
    pos_w = asset.data.root_link_pos_w
    origin = env.scene.terrain.env_origins
    rel = torch.nan_to_num(pos_w - origin, nan=0.0)
    x_err = rel[:, 0] - dx
    y_err = rel[:, 1] - dy
    z_err = rel[:, 2] - (nominal_height + dz)

    # 从四元数取 ZYX 欧拉角.
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    roll_err = roll - droll
    pitch_err = pitch - dpitch
    yaw_err = wrap_to_pi(yaw - dyaw)

    r_x = torch.exp(-((x_err / xy_std) ** 2))
    r_y = torch.exp(-((y_err / xy_std) ** 2))
    r_z = torch.exp(-((z_err / z_std) ** 2))
    r_r = torch.exp(-((roll_err / angle_std) ** 2))
    r_p = torch.exp(-((pitch_err / angle_std) ** 2))
    r_w = torch.exp(-((yaw_err / angle_std) ** 2))

    return (r_x + r_y + r_z + r_r + r_p + r_w) / 6.0


def termination_param_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    term_name: str,
    param_stages: list[dict],
) -> torch.Tensor:
    """在计划的 step 处修改 termination 项的参数.

    TerminationManager 保留自己的 cfg dict 副本, 所以必须直接编辑 live
    term_cfgs 列表 — env.cfg.terminations 是 no-op. 用于在训练后期禁用
    termination (例如在 iter N 处把 bad_orientation 的 limit_angle 设为 pi,
    使机器人可以摔倒而不结束 episode 并学会恢复).

    param_stages: {step: int, params: dict} 列表. dict 浅合并到最新匹配阶段的
    live term_cfg.params 中.
    """
    del env_ids
    tm = env.termination_manager
    if term_name not in tm._term_names:
        # 项已移除 (如 play 模式完全禁用 fell_over).
        return torch.tensor(0.0)
    idx = tm._term_names.index(term_name)
    term_cfg = tm._term_cfgs[idx]

    current = param_stages[0]["params"]
    for stage in param_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["params"]
    term_cfg.params.update(current)

    first_val = next(iter(current.values()))
    return torch.tensor(float(first_val) if isinstance(first_val, (int, float)) else 0.0)


def body_pose_tracking_locomotion(
    env: ManagerBasedRlEnv,
    command_name: str = "body_pose",
    nominal_height: float = 0.105,
    xy_std: float = 0.02,
    z_std: float = 0.03,
    angle_std: float = math.radians(30),
    axis_weights: tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ),
    vel_gate_command_name: str | None = None,
    vel_gate_std: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_cfg: SceneEntityCfg = _FEET_ASSET_CFG,
) -> torch.Tensor:
    """Locomotion-aware 的 6D body pose 跟踪.

    形状同 body_pose_tracking_6d (6D cmd, 6 个 Gaussian 的均值), 但 x/y/yaw
    相对 *feet support polygon* 测量, 而非 spawn origin. 这使奖励在机器人走路
    (或站立) 时有意义:

      x, y  : trunk 位置 − feet-centroid, 旋转到 trunk body frame.
              dx = +0.02 表示 "trunk 相对脚质心前倾 2 cm."
      z     : trunk 世界高度 (− nominal_height) — locomotion-neutral.
      roll  : trunk 世界 roll                — locomotion-neutral.
      pitch : trunk 世界 pitch               — locomotion-neutral.
      yaw   : trunk 世界 yaw − circular-mean(feet site yaw). dyaw = +0.3 rad
              表示 "trunk 相对脚指向方向扭转 17°."

    body_pose_tracking_6d 奖励相对 spawn origin / world yaw 测量 x/y/yaw, 这
    在机器人平移或转弯时立即杀死梯度. 本版本无论机器人在世界何处都保持有意义.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (N, 6)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]

    pos_w = asset.data.root_link_pos_w
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    trunk_yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

    # 世界坐标系下的 feet centroid.
    foot_pos = asset.data.site_pos_w[:, feet_cfg.site_ids]  # (N, 2, 3)
    foot_quat = asset.data.site_quat_w[:, feet_cfg.site_ids]  # (N, 2, 4)
    feet_centroid = foot_pos.mean(dim=1)  # (N, 3)

    # body frame 下相对 feet centroid 的 trunk xy (按 −yaw 旋转 world Δxy).
    dx_w = pos_w[:, 0] - feet_centroid[:, 0]
    dy_w = pos_w[:, 1] - feet_centroid[:, 1]
    cos_y = torch.cos(trunk_yaw)
    sin_y = torch.sin(trunk_yaw)
    x_body = cos_y * dx_w + sin_y * dy_w
    y_body = -sin_y * dx_w + cos_y * dy_w

    # 相对 spawn-origin 地形高度的 Z (仍在世界系).
    origin = env.scene.terrain.env_origins
    z_world = torch.nan_to_num(pos_w[:, 2] - origin[:, 2], nan=0.0)

    # Feet yaw → circular mean. 注意: 这依赖 site 朝向与脚指向方向一致;
    # 若 site frame 被旋转, 此 yaw 参考可能有偏移 (per-env 常数, 所以 dyaw=0
    # 仍映射到 "feet-aligned").
    fqw, fqx, fqy, fqz = (
        foot_quat[..., 0],
        foot_quat[..., 1],
        foot_quat[..., 2],
        foot_quat[..., 3],
    )
    foot_yaws = torch.atan2(2.0 * (fqw * fqz + fqx * fqy), 1.0 - 2.0 * (fqy * fqy + fqz * fqz))  # (N, 2)
    mean_foot_yaw = torch.atan2(torch.sin(foot_yaws).mean(dim=1), torch.cos(foot_yaws).mean(dim=1))

    x_err = x_body - dx
    y_err = y_body - dy
    z_err = z_world - (nominal_height + dz)
    roll_err = roll - droll
    pitch_err = pitch - dpitch
    yaw_err = wrap_to_pi(trunk_yaw - mean_foot_yaw - dyaw)

    r_x = torch.exp(-((x_err / xy_std) ** 2))
    r_y = torch.exp(-((y_err / xy_std) ** 2))
    r_z = torch.exp(-((z_err / z_std) ** 2))
    r_r = torch.exp(-((roll_err / angle_std) ** 2))
    r_p = torch.exp(-((pitch_err / angle_std) ** 2))
    r_w = torch.exp(-((yaw_err / angle_std) ** 2))

    # Per-axis 加权均值. 传 axis_weights=(0,0,1,1,1,1) 可禁用 xy
    # 跟踪 — 当 xy lean 在机器人上与 pitch/roll 机械耦合时有用, 使独立 xy
    # 命令成为噪声源而非可学习目标.
    wx, wy, wz, wr, wp, wyaw = axis_weights
    total_w = wx + wy + wz + wr + wp + wyaw
    reward = (wx * r_x + wy * r_y + wz * r_z + wr * r_r + wp * r_p + wyaw * r_w) / max(total_w, 1e-6)

    # 可选 gate: 当 vel_gate_command_name 设置时, 用速度命令幅值的 Gaussian 缩放
    # 奖励. vel_gate_std ≈ 0.1 时, gate 在命令速度为 0 时 ~1, 在 |vel_cmd|≥0.3 时
    # 衰减到 ~exp(-9)≈0 — body tracking 仅在机器人应静止时才有意义贡献. 避免
    # tracking vs walking 冲突, 该冲突曾阻止上一 run 学好任一项.
    if vel_gate_command_name is not None:
        # 仅 gate 命令的 LINEAR 速度 (xy) — 原地转弯时 body pose 仍有意义,
        # 但前/侧向行走时不是.
        vel_cmd = env.command_manager.get_command(vel_gate_command_name)  # (N, 3)
        vel_mag = torch.linalg.vector_norm(vel_cmd[:, :2], dim=-1)
        gate = torch.exp(-((vel_mag / vel_gate_std) ** 2))
        reward = reward * gate

    return reward


def pose_command_range_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    range_stages: list[dict],
) -> torch.Tensor:
    """在训练过程中 ramp UniformPoseCommand 的 per-dim 范围.

    range_stages: {step: int, ranges: tuple[(lo, hi), ...]} 列表.
    第一阶段在其 step 之前应用; 最新通过的阶段胜出.
    始终用 live CommandManager term cfg (而非 env.cfg.commands) 使更新生效 —
    CommandManager 保留自己的 term ref 并在每次重采样时读 `term.cfg.ranges`.
    """
    del env_ids

    term = env.command_manager.get_term(command_name)
    assert term is not None, f"Command term '{command_name}' not found"
    cfg = term.cfg  # type: ignore[assignment]

    current = range_stages[0]["ranges"]
    for stage in range_stages:
        if env.common_step_counter >= stage["step"]:
            current = stage["ranges"]

    cfg.ranges = tuple(current)
    # 返回 max abs range 作为标量供 wandb 可见.
    max_abs = max((max(abs(lo), abs(hi)) for lo, hi in current), default=0.0)
    return torch.tensor(max_abs)


# ─────────────────────────────────────────────────────────────────────────────
# 从 mjlab_microban (microban velocity recipe) 移植的步态塑形惩罚.
# ─────────────────────────────────────────────────────────────────────────────
def no_stepping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """当命令速度低于阈值时惩罚悬空脚.

    抑制机器人应静止时的原地踏步. 返回每个 env 的悬空脚数 (用 negative weight).
    从 mjlab_microban 移植.
    """
    command = env.command_manager.get_command(command_name)  # (N, 3)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    below_threshold = cmd_speed < command_threshold

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)  # (N, num_feet)
    in_air = ~found.bool()

    return in_air.float().sum(dim=-1) * below_threshold.float()


def feet_distance_penalty(
    env: ManagerBasedRlEnv,
    min_dist: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """惩罚双脚在水平面内过于靠近.

    每 env 返回 ``clamp(min_dist - d, min=0)`` (用 negative weight), 其中 ``d`` 是
    两个 foot site 的水平 (xy) 距离. 从 mjlab_microban 移植. 尚未接入 velocity —
    留待后续.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # (N, 2, 2)
    dist = torch.norm(foot_pos_xy[:, 0] - foot_pos_xy[:, 1], dim=-1)  # (N,)
    return torch.clamp(min_dist - dist, min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 非累积 domain randomization (恢复名义-再-施加).
#
# 原版 mdp.randomize_field 用 operation="add"/"scale" + mode="reset" 读取
# CURRENT model 值并对其施加操作, 不恢复名义 — 所以每次 episode reset 时
# 扰动叠加在上一次上, 参数在训练中随机游走偏离名义. 对 body_ipos (CoM)
# 这是长期存在的 microduck 不稳定性: CoM 在数百次 reset 中漂移数厘米偏离中心
# → 逐渐失衡的机器人 → 更多摔倒 → 早期峰值后 reward/episode-length 崩溃.
# 这些函数镜像 randomize_mass_and_inertia: 一次缓存名义, 每次抽取前恢复, 然后
# 施加新采样的扰动 — 所以每 episode 重新采样但从不累积.
# ─────────────────────────────────────────────────────────────────────────────
def randomize_com(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    ranges: tuple[float, float],
    field: str = "body_ipos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """每 episode 随机化 body CoM (body_ipos) 且不累积.

    对 buggy mdp.randomize_field(add, body_ipos, reset) 的直接替代. ``ranges`` 是
    (lo, hi) 施加到全部 3 个 CoM 轴; com_range 课程更新此 ``ranges`` 参数.
    ``field`` 声明使 event 能以 ``domain_randomization=True`` 运行 (mjlab 读
    params["field"] 来 per-env 展开该 model field).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    body_indices = asset.indexing.body_ids[body_ids]

    mf = getattr(env.sim.model, field)
    # 按 (field, body set) 缓存 key: 多个 randomize_com event 可共享
    # 同一 field (如 trunk + head 都随机化 body_ipos), 且不能在单一
    # _original_body_ipos attr 上冲突 — 它们的 body 数不同.
    _bidx = body_indices.tolist() if hasattr(body_indices, "tolist") else list(body_indices)
    cache_attr = f"_original_{field}_" + "_".join(str(int(i)) for i in _bidx)
    # 首次调用时缓存名义 (model[0] 此时仍是名义).
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, body_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_bodies = len(body_indices)

    # 先恢复名义 (防止累积), 再加新鲜 offset.
    mf[env_ids[:, None], body_indices] = nominal.unsqueeze(0).expand(num_envs, -1, -1)
    lo, hi = ranges
    offsets = torch.rand(num_envs, num_bodies, 3, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], body_indices] += offsets
    return torch.tensor(float(hi))


def randomize_dof_field_scaled(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    field: str,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """每 episode 缩放 per-dof model field (如 dof_frictionloss/dof_damping).

    不累积: 恢复名义, 再施加新鲜缩放.

    ``field`` 兼作 domain_randomization field 名. 注意: 在 BAM actuator 下,
    dof_frictionloss 和 dof_damping 在 edit_spec 中被置零 (BAM 自建模摩擦),
    所以缩放它们是 no-op — 仅在 XML position actuator 时有意义. 保持正确
    以防重新启用时的累积 footgun.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset: Entity = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(len(asset.indexing.joint_ids)))[joint_ids]
    dof_indices = asset.indexing.joint_v_adr[joint_ids]

    mf = getattr(env.sim.model, field)
    cache_attr = f"_original_{field}"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, mf[0, dof_indices].clone())
    nominal = getattr(env, cache_attr)

    num_envs = len(env_ids)
    num_dofs = len(dof_indices)

    mf[env_ids[:, None], dof_indices] = nominal.unsqueeze(0).expand(num_envs, -1)
    lo, hi = scale_range
    scales = torch.rand(num_envs, num_dofs, device=env.device) * (hi - lo) + lo
    mf[env_ids[:, None], dof_indices] *= scales
    return torch.tensor(float(hi))


# =============================================================================
# BallKick 任务 — 球 reset event, 踢球奖励, critic-only 球观测
# =============================================================================


def _ball_kick_dir(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Per-env 的世界系踢球方向 (XY 单位向量), 惰性分配.

    由 ``reset_ball_in_front_of_foot`` 在 episode reset 时设为机器人前进方向.
    episode 内冻结, 所以策略不能在踢球后通过转弯重新定义 "前进".
    """
    if not hasattr(env, "_ball_kick_dir_w"):
        env._ball_kick_dir_w = torch.zeros(env.num_envs, 2, device=env.device)
        env._ball_kick_dir_w[:, 0] = 1.0
    return env._ball_kick_dir_w


def reset_ball_in_front_of_foot(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    offset: tuple = (0.09, -0.042),
    noise_xy: float = 0.015,
    ball_radius: float = 0.035,
    asset_name: str = "ball",
):
    """把球放在 (右) 脚前方; 存储踢球方向.

    ``offset`` 是机器人 yaw frame 下的名义球心位置: HOME 时右脚中心在 (0, -0.042),
    趾尖在 x≈0.034, 所以 (0.08, -0.042) 把半径 35mm 的球放在趾前约 1cm 处.
    ``noise_xy`` (per 轴均匀 ±) 是放置 DR: 策略对球 BLIND, 所以这是强制
    摆动在真实世界放置误差下可用的手段.

    直接从 qpos 读机器人 root (root_link_pos_w 滞后到下一次 forward()); 必须注册在
    reset_base / set_ground_state 之后 (event 按 dict 插入顺序运行) 以确保机器人
    pose 已就位.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]

    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    qw, qx, qy, qz = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    n = len(env_ids)
    off = torch.tensor(offset, device=env.device, dtype=torch.float).repeat(n, 1)
    off += (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * noise_xy

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = root[:, 0] + cos_y * off[:, 0] - sin_y * off[:, 1]
    pose[:, 1] = root[:, 1] + sin_y * off[:, 0] + cos_y * off[:, 1]
    pose[:, 2] = env.scene.terrain.env_origins[env_ids, 2] + ball_radius
    pose[:, 3] = 1.0  # identity quat
    ball.write_root_link_pose_to_sim(pose, env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids)

    kick_dir = _ball_kick_dir(env)
    kick_dir[env_ids, 0] = cos_y
    kick_dir[env_ids, 1] = sin_y


def ball_forward_velocity(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    max_speed: float = 5.0,
) -> torch.Tensor:
    """沿 per-env 踢球方向的球 XY 速度, clamp 到 [0, max].

    到 ``max_speed`` 为止密集且线性: 球前进的每一额外速度在球滚动的每步都多付费,
    所以探索微动即可 bootstrap 踢球, 无需峰值检测机制. 后退/侧向球运动得 0 而非
    惩罚 — 误击不应吓退策略去接触球.

    当 ``max_speed`` 设为 TARGET 速度 (而非大 cap), 与 ``ball_speed_overshoot_penalty``
    配对: 奖励在目标处饱和本身不能消除 "更用力更好" — 更用力的踢使球在/超过 cap
    的步数更多, 所以滚动时间积分仍随击球速度增长. overshoot 惩罚才是使目标成为
    实际最优的关键.
    """
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    return torch.nan_to_num(fwd, nan=0.0).clamp(0.0, max_speed)


def ball_speed_overshoot_penalty(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
    target_speed: float = 1.0,
    max_penalty: float = 5.0,
) -> torch.Tensor:
    """球前进速度超出 ``target_speed`` 的部分 (线性, ≥ 0).

    目标速度踢球的 ``ball_forward_velocity`` 伴侣: 目标以下此项为 0 (capped 线性
    奖励提供向上梯度); 以上每 m/s overshoot 在持续的每步线性计费. 使此项的
    |weight| 低于 capped 奖励的 weight, 以让组合 landscape 在目标处达峰且
    overshoot 侧斜率更缓 — 稍微过力必须比完全不踢更便宜.
    """
    ball: Entity = env.scene[asset_name]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]
    fwd = (vel_xy * _ball_kick_dir(env)).sum(dim=1)
    over = torch.nan_to_num(fwd, nan=0.0) - target_speed
    return over.clamp(0.0, max_penalty)


def single_foot_grounded_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """二元奖励: 感测脚接触地形时为 1.

    ``feet_grounded_reward`` 的单脚变体 — 用于在踢球时钉住 SUPPORT 脚 (anti-hop):
    摆右腿免费, 抬左脚每步消耗此奖励.
    """
    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() > 1:
        found = found.sum(dim=-1)
    return torch.clamp(found, 0.0, 1.0)


def ball_pos_in_base(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
) -> torch.Tensor:
    """球相对机器人 root 的位置, 在机器人 base frame 中.

    CRITIC-ONLY 观测 (asymmetric actor-critic): 部署的策略无球感测, 所以 actor
    必须对球 BLIND — critic 仍可用它预测踢球收益.
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rel = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    return torch.bmm(rot.transpose(1, 2), rel.unsqueeze(-1)).squeeze(-1)


def ball_vel_in_base(
    env: ManagerBasedRlEnv,
    asset_name: str = "ball",
) -> torch.Tensor:
    """球在机器人 base frame 中的线速度.

    CRITIC-ONLY (见上文).
    """
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene[asset_name]
    rot = matrix_from_quat(robot.data.root_link_quat_w)
    vel = ball.data.root_link_lin_vel_w
    return torch.bmm(rot.transpose(1, 2), vel.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# SPIN 任务 — 在 rollers 上快速原地旋转                                          #
# --------------------------------------------------------------------------- #
# 相位包络: 按钮 slot 命令携带相位, 驱动梯形目标 YAW 角速度 (而非 pose 如 crouch).
#   [0, accel_end)        0.5 s   0 -> rate_max    (启动)
#   [accel_end, hold_end) 1.6 s   rate_max         (稳态)
#   [hold_end, brake_end) 0.5 s   rate_max -> 0    (制动)
#   [brake_end, 1.0)      1.4 s   0                (站立休息)
# 一个周期内包络下面积 = 2.1 * SPIN_RATE_MAX rad. 在 3.0 rad/s:
# 2.1 * 3.0 = 6.3 rad ~ 1 圈 (而非 ~2, 如旧目标 6.0 那样).
SPIN_PERIOD = 4.0
SPIN_RATE_MAX = 3.0
SPIN_ACCEL_END = 0.125
SPIN_HOLD_END = 0.525
SPIN_BRAKE_END = 0.650


def spin_rate_by_phase(
    phase: torch.Tensor,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """沿相位的目标 yaw 角速度 (rad/s, 正 = 逆时针)."""
    w = torch.zeros_like(phase)
    accel = phase < accel_end
    w = torch.where(accel, rate_max * phase / accel_end, w)
    hold = (phase >= accel_end) & (phase < hold_end)
    w = torch.where(hold, torch.full_like(phase, rate_max), w)
    brake = (phase >= hold_end) & (phase < brake_end)
    w = torch.where(brake, rate_max * (1.0 - (phase - hold_end) / (brake_end - hold_end)), w)
    return w


def spin_gate_by_phase(
    phase: torch.Tensor,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """[0,1] 内的塑形 gate = 归一化包络.

    在整个休息段为 0: 启动器 (腿剪式, 轮差速) 仅在启动 + 稳态时施加, 所以
    机器人在交还 roller policy 前回到中性站姿.
    """
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end) / rate_max


def spin_phase_from_command(cmd: torch.Tensor) -> torch.Tensor:
    """从 slot 的 [cos(2πφ), sin(2πφ), 0] 命令提取相位 [0,1)."""
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def _spin_target_rate(
    env: ManagerBasedRlEnv,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def _spin_gate(
    env: ManagerBasedRlEnv,
    command_name: str,
    rate_max: float,
    accel_end: float,
    hold_end: float,
    brake_end: float,
) -> torch.Tensor:
    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    return spin_gate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)


def spin_rate_reward_from_values(omega_z: torch.Tensor, omega_target: torch.Tensor, std: float) -> torch.Tensor:
    """对 yaw 角速度误差的 Gaussian (纯函数, 可测试)."""
    return torch.exp(-(((omega_z - omega_target) / std) ** 2))


def spin_rate_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    std: float = 1.5,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """spin 主目标: 跟踪目标 yaw 角速度 ω*(φ).

    ω_z 取 body frame (IMU 陀螺仪所见, 即 policy 所观测). 反向旋转比静止
    受更多惩罚, 因 Gaussian 中心在正目标.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_rate_reward_from_values(omega_z, target, std)


def spin_rate_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bootstrap L1: 即使 `spin_rate_track` 的 Gaussian 在远离目标处饱和,
    仍提供常数梯度朝向目标.

    用 POSITIVE weight (返回值已为负).
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_z = asset.data.root_link_ang_vel_b[:, 2]
    target = _spin_target_rate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return -torch.abs(omega_z - target)


SPIN_LAUNCH_DRIFT_SCALE = 0.2  # 启动期间漂移成本衰减


def spin_stay_in_place(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    launch_scale: float = SPIN_LAUNCH_DRIFT_SCALE,
    accel_end: float = SPIN_ACCEL_END,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """trunk 的 ‖v_xy‖² 成本: 原地旋转, 并消除入场动量.

    无参考状态 (不同于从 reset 测量的漂移), 所以在 episode 的 5 个周期内有效.
    用 NEGATIVE weight.

    启动期间衰减: 在 `[0, accel_end)` 机器人需蹬地注入角动量, 入场状态给
    它最多 0.3 m/s 它应 CONVERT 成旋转. 在此时按全价收翻译直接与目标对立.
    成本在此段乘 `launch_scale`, 之后 (稳态, 制动, 休息) 按全价, 此时 "原地"
    才是真标准.

    与 spin 其他启动器不同, 此项不被 `spin_gate_by_phase` 关闭: 休息期间
    我们恰恰要它保持满, 因为那时机器人应静止.
    """
    asset: Entity = env.scene[asset_cfg.name]
    v_xy = asset.data.root_link_lin_vel_b[:, :2]
    cost = torch.sum(torch.square(v_xy), dim=1)

    phase = spin_phase_from_command(env.command_manager.get_command(command_name))
    scale = torch.where(
        phase < accel_end,
        torch.full_like(cost, launch_scale),
        torch.ones_like(cost),
    )
    return cost * scale


# 在 rollers model 上测量的半轮距 (HOME pose, left_foot/right_foot site):
# 0.0499 m, 对比 spec 估计的 0.03 m. SPIN_RATE_MAX (A1) 的机械推论:
# 预期差速 = 2*SPIN_RATE_MAX*半轮距/r, r = 0.0175 m.
# 在旧目标 6.0 rad/s: 2*6.0*0.0499/0.0175 = 34.2 rad/s (取为 34.0, 即比 spec
# 估计的 20.0 多 71% -> 超 30% 阈值).
# 在新目标 3.0 rad/s: 2*3.0*0.0499/0.0175 = 17.1 rad/s. 在此处留 34.0 会将
# tanh(17.1/34) = 0.47 钳在自身最大值, 恰好削弱我们要强化的塑形
# (见 spin_stay_in_place).
SPIN_WHEEL_OMEGA_SCALE = 17.0  # rad/s; 按测量半轮距和 SPIN_RATE_MAX = 3.0 重新校准


def spin_wheel_differential_from_values(diff: torch.Tensor, gate: torch.Tensor, omega_scale: float) -> torch.Tensor:
    """纯函数: 轮差速的 tanh, 由 gate 承载, clamp ≥ 0."""
    return gate * torch.tanh(torch.clamp(diff, min=0.0) / omega_scale)


def spin_wheel_differential(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    omega_scale: float = SPIN_WHEEL_OMEGA_SCALE,
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """奖励滚动旋转 (而非打滑).

    对逆时针 spin, 左刃后退右刃前进; 4 轮正向前进时, 这给 ω_D − ω_G > 0.
    tanh 在 `omega_scale` 饱和以避免轮速军备竞赛.
    """
    asset: Entity = env.scene["robot"]
    lf_ids, _ = asset.find_joints("passive_LF_?wheel")
    lr_ids, _ = asset.find_joints("passive_LR_?wheel")
    rf_ids, _ = asset.find_joints("passive_RF_?wheel")
    rr_ids, _ = asset.find_joints("passive_RR_?wheel")

    vel = asset.data.joint_vel
    omega_left = (vel[:, lf_ids[0]] + vel[:, lr_ids[0]]) / 2.0
    omega_right = (vel[:, rf_ids[0]] + vel[:, rr_ids[0]]) / 2.0
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return spin_wheel_differential_from_values(omega_right - omega_left, gate, omega_scale)


def spin_grounded(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """spin 期间双刃着地 — 防止 "跳起扭身".

    swizzle `grounded_reward` 的变体, 不可在此复用: 它按 cmd_x 加权, 而相位
    命令下 cmd_x = cos(2πφ).
    """
    from mjlab.sensor import ContactSensor

    sensor: ContactSensor = env.scene[sensor_name]
    contact_time = sensor.data.current_contact_time  # (num_envs, num_feet)
    assert contact_time is not None
    n_contact = torch.sum((contact_time > 0.0).float(), dim=1)
    grounded = (n_contact >= 2).float()
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return grounded * gate


def leg_antisymmetry(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_bases: tuple = ("hip_pitch", "knee"),
    rate_max: float = SPIN_RATE_MAX,
    accel_end: float = SPIN_ACCEL_END,
    hold_end: float = SPIN_HOLD_END,
    brake_end: float = SPIN_BRAKE_END,
) -> torch.Tensor:
    """spin 期间启动腿剪式 (一前一后).

    机器人用镜像 L/R 符号约定: 对称 pose 满足 q_G + q_D ≈ 0 (见
    `leg_symmetry_reward`), 所以剪式满足 q_G ≈ q_D. 返回
    `gate(φ) · (−mean|q_G − q_D|)` — 用 POSITIVE weight, 由课程递减: 启动器
    退场让 policy 精炼自己的动作.
    """
    asset: Entity = env.scene[asset_cfg.name]
    left, right = [], []
    for base in joint_bases:
        li, _ = asset.find_joints([f"left_{base}"])
        ri, _ = asset.find_joints([f"right_{base}"])
        left.append(li[0])
        right.append(ri[0])
    lids = torch.tensor(left, device=env.device)
    rids = torch.tensor(right, device=env.device)

    q = asset.data.joint_pos
    scissor = -torch.abs(q[:, lids] - q[:, rids]).mean(dim=-1)
    gate = _spin_gate(env, command_name, rate_max, accel_end, hold_end, brake_end)
    return gate * scissor


# =============================================================================
# Backlash model — 透过 backlash 的 encoder 关节观测
# =============================================================================
# Backlash 模型 (robot_groundcontact_backlash.xml) 在每个 servo 关节串联一个
# 未驱动 ``passive_<joint>_backlash`` 铰链. link 角度为 qpos[servo] + qpos[backlash],
# 真实 encoder 位于 play 的 OUTPUT 侧 — 它读的就是和. 这些 obs 在 backlash 任务中
# 替换 joint_pos_rel / joint_vel_rel (见 tasks/backlash.py), 使 policy 看到的
# 正是 runtime 将喂给它的内容. asset_cfg regex 应只选 servo 关节 (常用 ``^(?!passive_).*``).


def _backlash_encoder_ids(
    env: "ManagerBasedRlEnv",
    asset: Entity,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(main_ids, backlash_ids, mask) — 按 (entity, joint 选择) 缓存.

    mask 在匹配的 passive_<name>_backlash 关节存在处为 1.0, 使相同 obs 函数在
    无 backlash 关节的 model 上不变运行.
    """
    key = (asset_cfg.name, str(asset_cfg.joint_ids))
    cache = env.__dict__.setdefault("_backlash_encoder_cache", {})
    hit = cache.get(key)
    if hit is not None:
        return hit

    names = asset.joint_names
    jnt_ids = asset_cfg.joint_ids
    main_ids = list(range(len(names)))[jnt_ids] if isinstance(jnt_ids, slice) else [int(i) for i in jnt_ids]
    name_to_id = {n: i for i, n in enumerate(names)}
    bl_ids, mask = [], []
    for i in main_ids:
        bl = name_to_id.get(f"passive_{names[i]}_backlash")
        bl_ids.append(0 if bl is None else bl)
        mask.append(0.0 if bl is None else 1.0)

    device = asset.data.joint_pos.device
    out = (
        torch.tensor(main_ids, dtype=torch.long, device=device),
        torch.tensor(bl_ids, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )
    cache[key] = out
    return out


def joint_pos_rel_backlash(
    env: "ManagerBasedRlEnv",
    biased: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """encoder 透过 backlash 铰链读取的 joint_pos_rel.

    返回 (qpos[servo] + qpos[backlash]) - default[servo]. biased=True 时,
    per-env encoder-calibration bias 施加到 servo 读数 (每个 servo 一个 encoder
    → 每关节一个 bias; backlash 加项保持原始).
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    pos = joint_pos[:, main_ids] + asset.data.joint_pos[:, bl_ids] * mask
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    return pos - default_joint_pos[:, main_ids]


def joint_vel_rel_backlash(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """encoder 透过 backlash 铰链读取的 joint_vel_rel.

    firmware 从 encoder 位置导出 present_velocity, 所以它也见 backlash 运动:
    qvel[servo] + qvel[backlash].
    """
    asset: Entity = env.scene[asset_cfg.name]
    main_ids, bl_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    vel = asset.data.joint_vel[:, main_ids] + asset.data.joint_vel[:, bl_ids] * mask
    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return vel - default_joint_vel[:, main_ids]


# ─────────────────────────────────────────────────────────────────────────────
# Sit↔Stand posture 命令 + posture-conditioned 奖励 (sitstand env).
#
# 一个策略, 双向: 命令是单一 sit/stand flag, 由 twist slot 携带
# (cmd = [sit_flag, 0, 0], 所以 "stand" 是全零命令 — 与每个其他策略的部署
# idle 相同). 下面所有任务奖励从 live 命令 per-env 选择其目标
# (SIT keyframe + SIT_Z vs HOME + STAND_Z), 所以同一奖励栈驱动下蹲,
# 坐姿休息, 起立和站立休息. 用 _servo_* helper → 兼容 backlash-model.
# ─────────────────────────────────────────────────────────────────────────────


class SitStandCommand(UniformVelocityCommand):
    """Posture 命令: cmd = [sit_flag, 0, 0], 带 dwell-time 重采样和
    SLEWED 内部目标 blend.

    sit_flag ∈ {0.0, 1.0}. 由 command manager 在 cfg 的 resampling_time_range
    (每个 posture 的 dwell time) 上重采样, 并在 episode reset 时重采样.
    cfg.sit_prob 是重采样命令 SIT 的概率; 配合 reset-state mix 训练全部
    四种 (start-state × command) 组合, 包括 "保持你已在做的".

    ``alpha`` (0 = STAND 目标, 1 = SIT 目标) 以常数速率朝 flag slew
    (完整过渡在 cfg.ramp_s 秒内), 是 posture_* 奖励跟踪的. THE 防崩机制:
    用二元目标, 早到每省一步就付全 goal-state jackpot, 而线性速度 cap 惩罚
    积分为有界 excess-distance 成本 — 瞬间跌落以 ~7× 击败 1 s 下蹲. 用 slewed
    目标, 超前 ramp 在 height/composite stack 上得分 ~0 (z 远离命令高度), 所以
    跟踪慢设定点 IS argmax; cap 留作 overshoot/bounce 的后盾. OBS 保持原始
    二元 flag (部署: runtime 写 0/1; 训练对 flip 的响应是 ~ramp_s 滑行).

    episode reset 时, alpha 从机器人 ACTUAL trunk 高度初始化, 而非 flag —
    坐姿 spawn 不能被 stand 初始化的 ramp 向上拖 (反之亦然).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        """初始化 sit/stand 命令, 带 slewed 目标 blend."""
        super().__init__(cfg, env)
        self._sit_prob = float(getattr(cfg, "sit_prob", 0.5))
        self._ramp_s = float(getattr(cfg, "ramp_s", 2.0))
        self._sit_z = float(getattr(cfg, "sit_z", 0.060))
        self._stand_z = float(getattr(cfg, "stand_z", 0.115))
        self._env_ref = env
        self._alpha = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """sit/stand posture flag 命令 [sit_flag, 0, 0]."""
        return self.vel_command_b

    @property
    def alpha(self) -> torch.Tensor:
        """Slewed 目标 blend: 0 = STAND 目标, 1 = SIT 目标."""
        return self._alpha

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        sit = (torch.rand(n, device=self.device) < self._sit_prob).float()
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = sit

    def _alpha_from_height(self) -> torch.Tensor:
        z = torch.nan_to_num(
            self.robot.data.root_link_pos_w[:, 2] - self._env_ref.scene.terrain.env_origins[:, 2],
            nan=self._stand_z,
        )
        return torch.clamp((self._stand_z - z) / max(self._stand_z - self._sit_z, 1e-6), 0.0, 1.0)

    def compute(self, dt: float) -> None:
        """刷新 alpha: 新 episode 从高度重新初始化, 然后朝 flag slew."""
        super().compute(dt)
        # episode 开始时从 ACTUAL trunk 高度重新初始化 blend.
        # 在此做 (而非 reset()), 因 command manager 在 set_ground_state event
        # 传送机器人之前 reset, 所以 reset() 会读传送前的高度. episode 的首次
        # compute 时 spawn state 已就位.
        fresh = self._env_ref.episode_length_buf <= 1
        if fresh.any():
            self._alpha = torch.where(fresh, self._alpha_from_height(), self._alpha)
        # 目标 blend 朝命令 flag 的常数速率 slew.
        step = dt / max(self._ramp_s, 1e-6)
        delta = self.vel_command_b[:, 0] - self._alpha
        self._alpha += torch.clamp(delta, -step, step)

    def _update_command(self) -> None:
        pass  # 无 heading controller / standing-env 机制.

    def _update_metrics(self) -> None:
        pass  # posture flag 无速度跟踪指标.


@_dataclass(kw_only=True)
class SitStandCommandCfg(UniformVelocityCommandCfg):
    """SitStandCommand 的 Cfg — 二元 sit/stand posture flag 带 slewed blend."""

    class_type: type = SitStandCommand
    # 重采样命令 SIT (vs STAND) 的概率.
    sit_prob: float = 0.5
    # 内部目标 blend 完整遍历 STAND↔SIT 的秒数.
    ramp_s: float = 2.0
    # 休息高度, 用于从 spawn state 初始化 blend.
    sit_z: float = 0.060
    stand_z: float = 0.115

    def build(self, env: ManagerBasedRlEnv) -> "SitStandCommand":
        """构建 SitStandCommand 命令项."""
        return SitStandCommand(self, env)


def _posture_blend(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """posture 奖励的目标 blend ∈ [0, 1] (0 = STAND, 1 = SIT).

    用 SitStandCommand 的 slewed ``alpha`` (移动设定点) 当 term 暴露它时;
    否则回退到原始二元 flag.
    """
    term = env.command_manager.get_term(command_name)
    alpha = getattr(term, "alpha", None)
    if alpha is not None:
        return alpha
    return env.command_manager.get_command(command_name)[:, 0]


def _posture_targets(
    env: ManagerBasedRlEnv,
    asset: Entity,
    command_name: str,
    sit_overrides: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """命令 posture 的 (目标 blend, per-env 关节目标).

    STAND 目标 = default_joint_pos (HOME); SIT 目标 = HOME 应用 keyframe override;
    SLEWED blend 在两者间插值, 所以 ramp 中段被奖励的 pose 与下降高度同步折叠.
    """
    blend = _posture_blend(env, command_name)
    stand_target = _servo_default_joint_pos(env, asset)
    sit_target = stand_target.clone()
    for idx, val in sit_overrides.items():
        sit_target[:, idx] = val
    target = stand_target + blend.unsqueeze(-1) * (sit_target - stand_target)
    return blend, target


def _posture_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """per-env 的 (slewed 目标 trunk z, 实际 trunk z)."""
    blend = _posture_blend(env, command_name)
    target_z = stand_z + blend * (sit_z - stand_z)
    asset = env.scene["robot"]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    return target_z, z


def posture_pose_match(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对命令 posture 的目标 pose 的 Gaussian pose-match."""
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return torch.exp(-(((joint_pos - target) / std) ** 2)).mean(dim=-1)


def posture_pose_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """``posture_pose_match`` 的 L1 伴侣 (常数梯度朝目标)."""
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    target = target[:, joint_indices]
    return -torch.abs(joint_pos - target).mean(dim=-1)


def posture_height_gaussian(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对命令 posture 目标高度的 trunk z Gaussian."""
    del asset_cfg  # trunk z read via _posture_height
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return torch.exp(-(((z - target_z) / std) ** 2))


def posture_height_l1(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """``posture_height_gaussian`` 的 L1 伴侣 — 过渡驱动器.

    当机器人休息在 *错误* posture 时, 此项每步收常数成本 (~|Δz| = 55 mm),
    这使 "忽略命令" 在两个方向都是净负策略.
    """
    del asset_cfg
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    return -torch.abs(z - target_z)


def posture_composite(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_overrides: dict,
    joint_indices: list,
    sit_z: float,
    stand_z: float,
    height_std: float = 0.03,
    upright_std: float = 0.40,
    pose_std: float = 0.40,
    head_std: float | None = None,
    head_command_name: str = "head_pose",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """对命令 posture 的乘积 goal score (height·upright·pose [·head]).

    ``standing_composite_score`` 的 posture-conditioned 版本: 任一因子不足
    都使整个项崩溃, 所以部分和妥协 (plank, flop, lean) 永不付费. 两个休息
    state 都要求直立 trunk, 所以直立因子与 posture 无关.

    ``head_std`` (可选): 在 neck/head 关节对 ``head_pose`` 命令加第四个因子
    (误差约定同 head_pose_tracking). 无此项时 goal state 对头盲: 训练的策略
    休息时头垂到地面 — trunk 直立, 腿在 pose, z 在目标全保持而头悬挂, 仅消耗
    轻跟踪项. 有此因子时, "到达" 要求头在其命令 pose, 所以 head assist 在
    过渡中免费 (composite 在那里 ≈0) 但必须收回才能收 goal 奖励.
    """
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets(env, asset, command_name, sit_overrides)
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)

    height_score = torch.exp(-(((z - target_z) / height_std) ** 2))

    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    upright_score = torch.exp(-tilt_sq / (upright_std * upright_std))

    joint_pos = _servo_joint_pos(env, asset)[:, joint_indices]
    pose_err_sq = ((joint_pos - target[:, joint_indices]) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))

    score = height_score * upright_score * pose_score

    if head_std is not None:
        if not hasattr(env, "_head_pose_neck_ids"):
            ids, _ = asset.find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
            env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        neck_ids = env._head_pose_neck_ids
        head_cmd = env.command_manager.get_command(head_command_name)
        actual = asset.data.joint_pos[:, neck_ids] - asset.data.default_joint_pos[:, neck_ids]
        head_err_sq = ((actual - head_cmd) ** 2).mean(dim=-1)
        score = score * torch.exp(-head_err_sq / (head_std * head_std))

    return score


def posture_stillness(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_z: float,
    stand_z: float,
    band_full: float = 0.012,
    band_zero: float = 0.03,
    vel_std: float = 0.05,
    tilt_full_deg: float = 25.0,
    tilt_zero_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励在命令 posture 处直立时的 trunk 静止.

    将 ``seated_stillness`` 推广到两个休息 state: exp(-(|v|/std)²) 由 |z − 命令 z|
    上的 smoothstep gate (``band_full`` 内满, ``band_zero`` 外零 → 过渡期间不活跃)
    和 trunk 倾斜 (倾斜休息 — 背/面/侧 — 一无所获). 额外由目标 ramp 完成
    (|flag − alpha| 小) gate, 所以静止在过渡中段不付费. 使 "安静休息, 直立,
    在命令高度" 成为栈峰.
    """
    asset = env.scene[asset_cfg.name]
    target_z, z = _posture_height(env, command_name, sit_z, stand_z)
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)

    flag = env.command_manager.get_command(command_name)[:, 0]
    blend = _posture_blend(env, command_name)
    ramp_done = ((flag - blend).abs() < 0.02).float()

    err = torch.abs(z - target_z)
    t = torch.clamp((band_zero - err) / max(band_zero - band_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)

    quat = asset.data.root_link_quat_w
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)

    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate * ramp_done


def posture_rise_bootstrap(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_height: float,
    max_vz: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """向上 vz 奖励, 仅在命令 STAND 且 z < max_height 时活跃.

    standup-env 教训: 仅终点奖励在零运动处梯度为零, 所以 "保持坐姿吃 L1"
    是局部最优 — 为起立 *motion* 本身付费使任何尝试立即为正. 在
    ``max_height`` 以上关闭 (设在 stand 目标之上一点以使最后 cm 仍付费;
    恰在 STAND_Z gate 会让策略停在短处). 命令 SIT 时为零, 所以永不与下蹲
    对抗. ``max_vz`` cap 奖励速度 (任何 ≥ cap 的起立得相同, 所以爆发性
    起立不能超越温和起立).
    """
    asset = env.scene[asset_cfg.name]
    sit = env.command_manager.get_command(command_name)[:, 0]
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return torch.clamp(vz, min=0.0, max=max_vz) * (z < max_height).float() * (1.0 - sit)


def trunk_upward_velocity_penalty(
    env: ManagerBasedRlEnv,
    max_up_vel: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """超过 ``max_up_vel`` 的向上 trunk 速度惩罚.

    ``trunk_downward_velocity_penalty`` 对起立的镜像: 对过快 (暴力) 起立的每步
    计费, 所以爆发性起立不能分摊到到达站立奖励. 静止时为零, 任何慢于 cap 的
    起立为零, 所有向下运动为零. 在起立被发现后通过课程引入 (attempt-tax 教训).
    """
    asset = env.scene[asset_cfg.name]
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    return -torch.clamp(vz - max_up_vel, min=0.0)


# ==============================================================================
# Roulade (前滚翻) 任务 — episode 动态动作
# ==============================================================================
#
# 第三次尝试 roulade. 前两次的教训:
#   • origin/roulade (相位时钟 + 时间窗口奖励阶段): 在 face-down ~90° 平台化 —
#     时间窗口是 keyframes-in-time, 可 camp 的局部最优 (正是 sit/standup 教训).
#     还把 -ω_y 积分为前进, 按本 codebase 自己的约定 (face-down = +90° pitch
#     = 绕 +y 旋转, 见 set_random_ground_state) 这是 WRONG SIGN — 进度奖励为
#     反向旋转付费.
#   • origin/roulade 后续 commit (keyframe 模仿): 同一 waypoint-camping 家族,
#     按 feedback-episodic-pose-landing 弃用.
#
# 此设计改用已验证的 episode 食谱:
#   • 一个密集进度信号: 为 max-so-far 累积前向旋转的 INCREMENT 付费
#     (potential-based — camping 策略每步赚 0, 完整滚翻无论路径或速度
#     恰好赚 2π 价值).
#   • 落地奖励 (composite 乘积, upright, height, rise velocity) 由 ROLL
#     COMPLETION (max rotation ≥ 阈值) gate — 基于 state 的 gate, 非基于时钟.
#     "什么都不做" 一无所获; 在 spawn 站立一无所获; 只有滚动打开站立吸引子
#     年金.
#   • 通过 mid-roll spawn 反向课程 (修复 back-recovery 的 face-up partial-roll
#     trick): 一部分 episode 从滚翻 50°–185° 处开始, 抱团, 可选带前向角动量,
#     旋转累加器初始化为 spawn 角度以保持进度记账一致.
#
# RUN-1 教训 (2026-08): 不支持的旋转计数和未 cap 的付费率下, 最优策略是
# 暴力弹道甩 ("breakdance") — 同 2π, 更快完成, 折扣更多年金. 无法迁移. 修复:
#   • SUPPORT GATE: 累加器仅在机器人某 geom 接触地形时积分
#     (robot_ground_contact sensor) — 真实 roulade 永不离地; 空中旋转一无所获
#     且不能打开完成 gate.
#   • HEAD LATCH: 落地年金额外要求头-地接触在 accum 处于第一象限窗口时发生 —
#     "过头顶" 是要求, 非 0.5-weight 建议.
#   • PAID-RATE CAP: 进度 increment 在 max_paid_rate cap; 快于 cap 的旋转
#     FORFEIT 超额 (非递延), 所以速度不再付费. 显式 overspeed 惩罚作后盾.
#
# env 对象上的 per-env state (惰性创建, 由 reset_roulade_state 重置):
#   env._roulade_accum      — 仅支持的前向 pitch rate 积分 (rad)
#   env._roulade_max        — 本 episode 至今的 max(accum) (进度前沿)
#   env._roulade_paid       — roulade_progress 已付的前沿
#   env._roulade_head_latch — 头在第一象限中段触地后为 True

# 前滚翻符号: face-down 是 +90° pitch = 绕 body +y 旋转
# (set_random_ground_state 约定), 所以前滚翻 = POSITIVE body-frame ω_y.
# 经验验证 (见 claude_experiments smoke test): 绕 +y 的正 qvel 使机器人鼻朝下/前
# 并驱动 accum 上升.
_ROULADE_FWD_SIGN = 1.0

# 累加器更新读取的 sensor 名 (必须与 env cfg 匹配).
_ROULADE_SUPPORT_SENSOR = "robot_ground_contact"
_ROULADE_HEAD_SENSOR = "head_ground_contact"

# Head-latch 窗口: accum 在此窗口内头触地标记 episode 为真正过头顶滚翻.
# 真实 roulade 中头在 ~60–120° body 旋转时触地; 窗口在其周围宽松.
_HEAD_LATCH_LO = math.radians(20.0)
_HEAD_LATCH_HI = math.radians(170.0)

# jaw_soft LOCAL frame 中的头顶轴 (2026-08-13 经验测量: 机器人在 HOME 安放时
# world-up 在 jaw_soft frame 中的表示). latch 要求此轴在接触时指向下 —
# "头顶平面在地面", 而非脸或壳侧 (run-5 修复: run-4 策略翻过肩膀, 仍触
# jaw_soft).
_HEAD_TOP_AXIS = (0.882, 0.0, 0.471)
# dot(top_axis_world, -z) 阈值. 测量地标 (trunk pitched 110°):
# 被动 face-plant (neck at HOME) 读 +0.6, 完全下巴收 (neck_pitch −1,
# head_pitch +1) 读 −0.99 — 0.3 接受部分收下巴而远离任何脸/侧接触.
_HEAD_TOP_DOWN_MIN = 0.3

# 累加器上的矢状平面 gate (run-5): 干净前滚翻中 body 的 LATERAL 轴全程保持
# 水平 — 其 world-z 分量 2(q_y·q_z + q_w·q_x) 对任意纯 pitch ≈ 0, 并在翻过
# 肩膀时趋向 ±1. lateral 轴在水平 ~30° 内时全旋转积分, ~60° 外为零:
# 侧滚不算旋转, 不赚进度, 且永不打开落地 gate.
_FLAT_FULL = 0.5  # |lateral_axis_z| = sin(30°): 以下全积分
_FLAT_ZERO = 0.866  # sin(60°): 以上零积分


def _lateral_axis_z(quat: torch.Tensor) -> torch.Tensor:
    """body 的 lateral (y) 轴的 world-z 分量. 0 = flat/矢状."""
    return 2.0 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])


def _head_top_down(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """头顶轴指向地面 (dot with -z > min) 处为 True."""
    if not hasattr(env, "_roulade_head_body_id"):
        ids, _ = asset.find_bodies("jaw_soft")
        env._roulade_head_body_id = ids[0]
    q = asset.data.body_link_quat_w[:, env._roulade_head_body_id]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a, b, c = _HEAD_TOP_AXIS
    # R(q) @ axis_local 的 z 分量
    axis_world_z = 2.0 * (x * z - w * y) * a + 2.0 * (y * z + w * x) * b + (1.0 - 2.0 * (x * x + y * y)) * c
    return axis_world_z < -_HEAD_TOP_DOWN_MIN


def _sensor_any_contact(env: ManagerBasedRlEnv, name: str) -> torch.Tensor | None:
    if name not in env.scene.sensors:
        return None
    found = env.scene.sensors[name].data.found
    return (found.view(found.shape[0], -1) > 0).any(dim=-1)


def _roulade_state(env: ManagerBasedRlEnv) -> tuple:
    if not hasattr(env, "_roulade_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._roulade_accum = z.clone()
        env._roulade_max = z.clone()
        env._roulade_paid = z.clone()
        env._roulade_head_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._roulade_last_update_step = -1
    return env._roulade_accum, env._roulade_max, env._roulade_paid


def _update_roulade_accum(env: ManagerBasedRlEnv, asset: Entity) -> None:
    """将前向 pitch rate 积分到 per-env 旋转累加器.

    step-guarded 使同一 control step 中多个读累加器的奖励项不双重积分.
    前沿 (max) 只前进; 后向摇晃 (蓄力) 既不付费也不退付.

    SUPPORT GATE (run-1 修复): 旋转仅在机器人接触地形时积分 — roulade 是支持
    motion; 弹道翻转累积为零, 所以它们既不付费也不打开完成 gate.

    也在 accum 处于第一象限窗口时头触地 latch env._roulade_head_latch —
    落地年金要求此, 使 "过头顶" 成为任务的硬要求.
    """
    _roulade_state(env)
    step = int(env.common_step_counter)
    if step != env._roulade_last_update_step:
        omega_fwd = _ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
        delta = torch.nan_to_num(omega_fwd, nan=0.0) * env.step_dt
        supported = _sensor_any_contact(env, _ROULADE_SUPPORT_SENSOR)
        if supported is not None:
            delta = delta * supported.float()
        # 矢状平面 gate (run-5): 侧/肩滚不算.
        y_z = torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=1.0).abs()
        t = torch.clamp((_FLAT_ZERO - y_z) / (_FLAT_ZERO - _FLAT_FULL), 0.0, 1.0)
        delta = delta * (t * t * (3.0 - 2.0 * t))
        env._roulade_accum = env._roulade_accum + delta
        env._roulade_max = torch.maximum(env._roulade_max, env._roulade_accum)

        head_contact = _sensor_any_contact(env, _ROULADE_HEAD_SENSOR)
        if head_contact is not None:
            in_window = (env._roulade_accum > _HEAD_LATCH_LO) & (env._roulade_accum < _HEAD_LATCH_HI)
            # Run-5: 接触必须是头的 FLAT TOP (顶轴指向地面) — 脸/壳侧接触不 latch.
            env._roulade_head_latch = env._roulade_head_latch | (head_contact & in_window & _head_top_down(env, asset))
        env._roulade_last_update_step = step


def _roulade_completion_gate(
    env: ManagerBasedRlEnv,
    gate_lo: float,
    gate_hi: float,
    require_head: bool = False,
) -> torch.Tensor:
    """进度前沿上的 smoothstep: gate_lo rad 以下为 0, gate_hi 以上为 1.

    旧相位时钟落地窗口的 state-based 替代 — 只能通过实际旋转 (且 SUPPORTED —
    累加器 contact-gated) 打开, 所以滚前站立和弹道翻转都收不到.
    require_head=True 时 gate 额外要求 head latch — episode 必须翻过头顶才能
    解锁落地年金.
    """
    _, max_accum, _ = _roulade_state(env)
    t = torch.clamp((max_accum - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    if require_head:
        gate = gate * env._roulade_head_latch.float()
    return gate


def reset_roulade_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    standing_prob: float = 0.5,
    midroll_prob: float = 0.5,
    standing_z_min: float = 0.11,
    standing_z_max: float = 0.12,
    standing_tilt_max: float = 0.0,
    forward_vel_range: tuple = (0.0, 0.0),
    midroll_pitch_min: float = math.radians(50.0),
    midroll_pitch_max: float = math.radians(185.0),
    midroll_z_min: float = 0.05,
    midroll_z_max: float = 0.10,
    midroll_omega_range: tuple = (0.0, 0.0),
    tuck_overrides: dict | None = None,
    tuck_factor_range: tuple = (0.3, 1.0),
    joint_noise_std: float = 0.0,
):
    """reset 到站立开始或 mid-roll state (反向课程).

    站立桶: 直立 (±standing_tilt_max pitch/roll 噪声), 随机 yaw,
    HOME 关节 (reset_robot_joints 留下), z ∈ [standing_z_min, _max].
    ``forward_vel_range`` 是冲力钩子: per-env 前向 base 速度 (body x, 通过
    spawn yaw 映射到 world) 均匀采样 — 0 为静止滚翻, 之后扩大以训练从行走
    中滚翻.

    Mid-roll 桶: 滚入 ``midroll_pitch_min..max`` (90° = 头上, 180° = 背上),
    随机 yaw, 腿由 per-env 因子在 ``tuck_factor_range`` 内 HOME→tuck lerp,
    z ∈ [midroll_z_min, _max], 可选来自 ``midroll_omega_range`` 的前向角动量.
    旋转累加器初始化为 spawn pitch 以保持进度记账 (和完成 gate) 一致:
    170° spawn 只为剩余 ~190° 付费.
    """
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    num = len(env_ids)
    asset: Entity = env.scene[asset_cfg.name]
    accum, max_accum, paid = _roulade_state(env)

    total = standing_prob + midroll_prob
    is_mid = torch.rand(num, device=env.device) < (midroll_prob / max(total, 1e-6))

    yaw = torch.rand(num, device=env.device) * 2 * np.pi - np.pi
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    # 每桶的 pitch: 站立为小噪声, mid-roll 为角度.
    pitch = (torch.rand(num, device=env.device) * 2 - 1) * standing_tilt_max
    mid_pitch = torch.rand(num, device=env.device) * (midroll_pitch_max - midroll_pitch_min) + midroll_pitch_min
    pitch = torch.where(is_mid, mid_pitch, pitch)
    roll = (torch.rand(num, device=env.device) * 2 - 1) * max(standing_tilt_max, math.radians(5.0))

    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    # ZYX intrinsic Euler → 四元数 (yaw * pitch * roll), 同 set_random_ground_state.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = torch.stack([qw, qx, qy, qz], dim=1)

    z_stand = torch.rand(num, device=env.device) * (standing_z_max - standing_z_min) + standing_z_min
    z_mid = torch.rand(num, device=env.device) * (midroll_z_max - midroll_z_min) + midroll_z_min
    new_z = torch.where(is_mid, z_mid, z_stand)

    env.sim.data.qpos[env_ids, 2] = new_z
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    servo_ids = _servo_joint_ids(env, asset)

    # Mid-roll 关节: 在 override 关节上 lerp HOME → tuck, 所有 servo 关节加噪声
    # (passive_* backlash 铰链必须保持 0).
    mid_env_ids = env_ids[is_mid]
    if len(mid_env_ids) > 0 and tuck_overrides:
        u = (
            torch.rand(len(mid_env_ids), device=env.device) * (tuck_factor_range[1] - tuck_factor_range[0])
            + tuck_factor_range[0]
        )
        for jnt_idx, angle in tuck_overrides.items():
            col = 7 + servo_ids[jnt_idx]
            home = env.sim.data.qpos[mid_env_ids, col]
            env.sim.data.qpos[mid_env_ids, col] = home + u * (angle - home)
    if len(mid_env_ids) > 0 and joint_noise_std > 0.0:
        cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
        noise = torch.randn(len(mid_env_ids), len(cols), device=env.device) * joint_noise_std
        env.sim.data.qpos[mid_env_ids.unsqueeze(1), cols.unsqueeze(0)] += noise

    # Mid-roll 前向角动量: 绕 body +y 旋转. MuJoCo free joint qvel[3:6] 是
    # BODY frame 中的角速度, 所以 [0, ω, 0] 是前滚轴, 与 spawn yaw 无关
    # (在 smoke test 中验证 — yawed spawn 仍在自己 frame 中直向前滚).
    if len(mid_env_ids) > 0 and midroll_omega_range[1] > 0.0:
        omega = (
            torch.rand(len(mid_env_ids), device=env.device) * (midroll_omega_range[1] - midroll_omega_range[0])
            + midroll_omega_range[0]
        )
        env.sim.data.qvel[mid_env_ids, 4] = _ROULADE_FWD_SIGN * omega

    # 冲力钩子: STANDING spawn 的前向 base 速度, body x → world xy
    # 通过 spawn yaw. (0, 0) = 静止开始, 禁用.
    stand_env_ids = env_ids[~is_mid]
    if len(stand_env_ids) > 0 and forward_vel_range[1] > 0.0:
        vx = (
            torch.rand(len(stand_env_ids), device=env.device) * (forward_vel_range[1] - forward_vel_range[0])
            + forward_vel_range[0]
        )
        yaw_s = yaw[~is_mid]
        env.sim.data.qvel[stand_env_ids, 0] = vx * torch.cos(yaw_s)
        env.sim.data.qvel[stand_env_ids, 1] = vx * torch.sin(yaw_s)

    # 进度记账: 站立从 0 开始, mid-roll 从 spawn pitch 开始.
    spawn_angle = torch.where(is_mid, mid_pitch, torch.zeros_like(mid_pitch))
    accum[env_ids] = spawn_angle
    max_accum[env_ids] = spawn_angle
    paid[env_ids] = spawn_angle
    # Head latch: mid-roll spawn 视为已过头阶段 (反向课程教滚翻 COMPLETION;
    # 要求它们从未有机会赚到的 latch 会使其落地 gate 永久关闭).
    # 站立 spawn 必须通过实际翻过头顶来赚取.
    env._roulade_head_latch[env_ids] = is_mid


def roulade_progress(
    env: ManagerBasedRlEnv,
    target_angle: float = 2 * math.pi,
    max_paid_rate: float = 3.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """为进度前沿的 increment 付费, 最多一次完整滚翻.

    reward = Δ(min(max_accum, target)) / (step_dt · target), 在
    max_paid_rate rad/s 付费旋转处 CAPPED. 无可 farm: face-down camping (0/step),
    前沿下摇晃 (0/step), 或超 2π 旋转 (clamp). 累加器 support-gated, 所以
    空中旋转也不付费.

    max_paid_rate (run-1 修复): 快于 cap 的旋转 FORFEIT 超额 — 付费指针仍跳到
    前沿, 只是付 capped 量. 暴力甩因此比受控 ≤cap 滚翻收 LESS 总进度奖励,
    而非更快收相同总量.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    _, max_accum, paid = _roulade_state(env)
    new_paid = torch.clamp(max_accum, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    env._roulade_paid = torch.maximum(paid, new_paid)
    return delta / (env.step_dt * target_angle)


def roulade_head_pivot(
    env: ManagerBasedRlEnv,
    sensor_name: str = "head_ground_contact",
    angle_lo: float = math.radians(30.0),
    angle_hi: float = math.radians(240.0),
    rate_norm: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """奖励前滚中头-地接触.

    contact × window(accum ∈ [angle_lo, angle_hi]) × clamp(ω_fwd/rate_norm, 0, 1)
    × (0.3 + 0.7·top_down).
    rate factor 是反 camping 守卫: 脸朝下趴地、头贴地的机器人 ω_fwd ≈ 0, 收益为零 —
    该项只为 pivoting OVER the head 付费. top_down factor (run-5) 将此 dense shaping 与
    latch 对齐: 滚翻中任何头部接触付费 30%, 在 FLAT TOP (下巴内收) 上接触付全费 —
    即教会 tuck 的梯度.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    accum, _, _ = _roulade_state(env)

    if sensor_name not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    found = env.scene.sensors[sensor_name].data.found
    contact = (found.view(found.shape[0], -1) > 0).any(dim=-1).float()

    in_window = ((accum > angle_lo) & (accum < angle_hi)).float()
    omega_fwd = _ROULADE_FWD_SIGN * asset.data.root_link_ang_vel_b[:, 1]
    rate = torch.clamp(torch.nan_to_num(omega_fwd, nan=0.0) / rate_norm, 0.0, 1.0)
    top = 0.3 + 0.7 * _head_top_down(env, asset).float()
    return contact * in_window * rate * top


def roulade_landing_composite(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    target_overrides: dict | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """standing_composite_score × completion gate.

    大额年金: 一旦滚翻 (近) 完成, 在 HOME 姿态下站立的每一步都付费 —
    以脚落地并保持稳定主导一切部分结果. 旋转量达 gate_lo 之前为零, 因此
    standing spawn 不能靠什么都不做而 farm 它.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    score = standing_composite_score(
        env,
        target_height=target_height,
        height_std=height_std,
        upright_std=upright_std,
        pose_std=pose_std,
        joint_indices=joint_indices,
        target_overrides=target_overrides,
        asset_cfg=asset_cfg,
    )
    return score * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_upright_after_roll(
    env: ManagerBasedRlEnv,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Linear cos(tilt) × completion gate — bootstrap 拉向垂直.

    从 ANY orientation 都有梯度 (composite 在远离目标时近零), 但只在滚翻之后: 在
    gate_lo 之前严格为零, 因此不会像旧的 always-on upright 项那样反作用于翻转.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    upright = 1.0 - 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    return torch.clamp(upright, min=0.0) * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_height_after_roll(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float = 0.04,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Broad height Gaussian × completion gate — 拉升至站立高度."""
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    g = torch.exp(-(((z - target_height) / std) ** 2))
    return g * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_landing_sharp(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float = 0.015,
    upright_std: float = 0.3,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Tight-std upright × height Gaussians × completion gate — 最后一英里.

    run-4 修复 27°-lean / 1-cm-crouch 终态 basin: 宽 landing composite (upright_std 0.40) 在
    该姿态下约 0.5, 因此 policy 停在那里. 这就是 standup 的双层教训 — 宽层触及, 锐层
    完成. 在 27° 倾斜下该项约 0.1 (真实梯度); 垂直时约 1.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    quat = asset.data.root_link_quat_w
    tilt_sq = 2.0 * (quat[:, 1].pow(2) + quat[:, 2].pow(2))
    upright_g = torch.exp(-tilt_sq / (upright_std * upright_std))
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    height_g = torch.exp(-(((z - target_height) / height_std) ** 2))
    gate = _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)
    return upright_g * height_g * gate


def roulade_stand_tax(
    env: ManagerBasedRlEnv,
    target_height: float,
    gate_lo: float = math.radians(260.0),
    gate_hi: float = math.radians(330.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """SELF-NEGATING height L1 below target, active only after roll completion.

    返回 −max(0, target − z) × completion_gate — 使用 POSITIVE weight (penalty sign convention).
    run-3 修复 post-roll crumple-camping: gated landing rewards 使站立优于瘫倒, 但瘫倒本身
    是 FREE 的 — 仅靠 positive gated 项, "保持瘫倒" 每步约 0, 是一个舒适 basin (standup
    static-sit 教训: basin 必须净 NEGATIVE 才能强制 rise). gate 保证滚翻本身不被征税, 且
    要求 head latch, 因此无滚翻 episode 不会被惩罚出怪异回避行为.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    shortfall = torch.clamp(target_height - z, min=0.0)
    return -shortfall * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_rise_velocity(
    env: ManagerBasedRlEnv,
    max_height: float = 0.125,
    gate_lo: float = math.radians(180.0),
    gate_hi: float = math.radians(260.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """com_upward_velocity × late-roll gate — bootstrap 退出 rise.

    roulade 后半段 (supine → sitting-up → standing) 即 face-up recovery 问题,
    standup env 证明 end-state rewards 在零运动处零梯度: 直接为 rising vz 付费.
    gate 从 ~180° (仰卧) 开启, 因此滚前摇摆无收益; 在 max_height 以上关闭,
    因而不能被 hopping farm.
    """
    asset: Entity = env.scene[asset_cfg.name]
    _update_roulade_accum(env, asset)
    z = torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    reward = torch.clamp(vz, min=0.0) * (z < max_height).float()
    return reward * _roulade_completion_gate(env, gate_lo, gate_hi, require_head=True)


def roulade_overspeed_penalty(
    env: ManagerBasedRlEnv,
    omega_max: float = 4.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Max(0, |ω_y| − omega_max)² — 对 whip-speed 旋转的二次惩罚.

    正值; 使用 negative weight. 补充 roulade_progress 中的 paid-rate cap:
    cap 移除了快于 ~3 rad/s 旋转的 INCENTIVE, 此项在 omega_max 之上加显式 COST,
    使 "violent" 严格劣于 "controlled" 而不仅仅是不好. 受控的完整滚翻
    (平均 ~2–3 rad/s) 永不触及此项.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_y = torch.nan_to_num(asset.data.root_link_ang_vel_b[:, 1], nan=0.0)
    excess = torch.clamp(omega_y.abs() - omega_max, min=0.0)
    return excess.pow(2)


def roulade_flatness_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """(lateral-axis world-z)² — 朝 sagittal roll 的 dense gradient.

    正值; 使用 negative weight. 站立时为零, 任意深度的 CLEAN forward roll 也为零
    (纯 pitch 使 lateral axis 保持水平), 完全侧倒到肩时达 1. 累加器的 flatness gate
    使侧滚无利可图; 此项加 per-step gradient, 将方向导回平面.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(_lateral_axis_z(asset.data.root_link_quat_w), nan=0.0).pow(2)


def roulade_sagittal_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Rotation out of the sagittal plane: body-frame ω_x² + ω_z² (正值; 使用 negative weight).

    ω_y 是滚翻轴, 保持自由.
    """
    asset: Entity = env.scene[asset_cfg.name]
    omega_b = asset.data.root_link_ang_vel_b
    return torch.nan_to_num(omega_b[:, 0].pow(2) + omega_b[:, 2].pow(2), nan=0.0)


def roulade_lateral_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Body-frame lateral (y) linear velocity² — 保持滚翻走直线."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_link_lin_vel_b[:, 1].pow(2), nan=0.0)
