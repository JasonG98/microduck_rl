"""Backlash 任务变体 — 替换为 backlash 机器人模型 + 编码器 obs.

``make_backlash_variant(cfg)`` 将任意 microduck env cfg 转换为对应的
backlash 版本 (task id ``Mjlab-<Task>-<Flat|Rough>-Backlash-MicroDuck``):

1. Robot → 匹配的 backlash 机器人 cfg (与 14 个 servo 关节各串联一个未驱动的
   ``passive_<joint>_backlash`` 铰链, ±1° 间隙), 由 BacklashEncoderBamActuator
   驱动, 其固件 PD 闭合于穿越 backlash 的编码器读数 — 与真实舵机一致, 编码器
   位于齿轮间隙的输出侧. 传入与基础任务模型镜像的 robot cfg: Velocity 用
   MICRODUCK_WALK_BACKLASH_ROBOT_CFG (robot_walk_backlash.xml), 默认
   MICRODUCK_BACKLASH_ROBOT_CFG 用于 VelStand/StandUp
   (robot_groundcontact_backlash.xml).
2. joint_pos / joint_vel obs → joint_pos_rel_backlash / joint_vel_rel_backlash:
   策略观测 qpos[servo] + qpos[backlash] (编码器视角), 保留编码器偏差 DR 路径
   (``biased`` 参数) 不变. obs 和 action 维度不变 (仍为 14 个关节), runtime/export
   无需改动.
3. dof_pos_limits reward 仅作用于 servo 关节: backlash 关节一生都紧贴在 ±1°
   限位 (这正是 backlash 的本意), 否则会持续喂入一个永久的超出软限位惩罚.

其他一切 (rewards, DR events, curricula) 原样保留 — backlash 关节上的
``passive_`` 前缀意味着每个既有的 ``^(?!passive_).*`` 正则 (actuators,
pose reward, joint obs 选择) 已自动将它们排除在外.
"""

from copy import deepcopy

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_BACKLASH_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp

_SERVO_JOINTS_ONLY = (r"^(?!passive_).*",)


def make_backlash_variant(
    cfg: ManagerBasedRlEnvCfg,
    robot_cfg: EntityCfg = MICRODUCK_BACKLASH_ROBOT_CFG,
) -> ManagerBasedRlEnvCfg:
    """将 microduck env cfg (velocity/velstand/standup/...) 转换为 backlash 版本."""
    cfg.scene.entities = {**cfg.scene.entities, "robot": robot_cfg}

    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        for term_name, func in (
            ("joint_pos", microduck_mdp.joint_pos_rel_backlash),
            ("joint_vel", microduck_mdp.joint_vel_rel_backlash),
        ):
            term = terms.get(term_name)
            if term is None:
                continue
            term.func = func
            # 从未收窄选择的 env 会把 backlash 关节本身喂入 obs
            # (维度错 + 重复计数).
            if "asset_cfg" not in term.params:
                term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=_SERVO_JOINTS_ONLY)

    # Backlash 关节合法地压在硬限位上 — 把它们从软限位惩罚中排除
    # (其默认 asset_cfg 覆盖所有关节).
    dof_limits = cfg.rewards.get("dof_pos_limits")
    if dof_limits is not None and "asset_cfg" not in dof_limits.params:
        dof_limits.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=_SERVO_JOINTS_ONLY)

    # pose (variable_posture) reward 会基于选定的 joint names 解析其 std
    # 字典, 并在歧义匹配时报错 — 在 backlash 模型上 "passive_left_hip_yaw_backlash"
    # 同时匹配 ".*hip_yaw.*" 和 roller env 的 ".*passive_.*" std 项.在选择
    # 前面加上 backlash 排除前缀即可; 既有的 lookahead (velocity 的
    # passive/neck/head 排除) 仍能正常组合, 保留 wheels 的 env 也能继续选中
    # 它们.
    pose = cfg.rewards.get("pose")
    if pose is not None and "asset_cfg" in pose.params:
        # 先 deepcopy — 基础模板在多个 make() 调用间共享 SceneEntityCfg 对象;
        # 原地修改会泄漏到基础任务中.
        ac = deepcopy(pose.params["asset_cfg"])
        ac.joint_names = tuple(
            p if "_backlash" in p else r"^(?!passive_.*_backlash)" + p.lstrip("^") for p in ac.joint_names
        )
        pose.params["asset_cfg"] = ac

    return cfg
