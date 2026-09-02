"""Microduck *sitstand* 任务 (v1.5, mjlab 1.3.0) — 指令 坐 ↔ 站, 平缓.

单 policy, 双向, 由姿态指令驱动:
    cmd (twist 槽位) = [sit_flag, 0, 0]   sit_flag ∈ {0 = 站, 1 = 坐}
"Stand" 是全零指令 — 与其他所有 policy 相同的部署空闲. 指令在 episode 中
翻转, 停留数秒, 所以每个 episode 训练下降, 坐姿休息, 上升和站立休息, 加上
"保持你已经在做的" (reset 状态 × 指令独立).

2026-08 从头重建 (旧相位循环 env 早于 1.3.0 迁移和每个 sit/standup 教训).
设计综合:
  - 姿态条件单目标奖励 (mdp posture_*): sit 环境的最小可行 "有机发现" 栈,
    但目标 (SIT 关键帧 + SIT_Z vs HOME + STAND_Z) 按每环境从活跃指令选择.
    无轨迹, 无 waypoints, 无相位时序 — policy 发现自己的过渡路径, 用任意多步
    (膝先下, 头辅助等都被允许: 全碰撞模型, 无头部地面惩罚, 无摔倒终止).
  - 双向平缓: 下降速度上限 (sit 环境验证的配方, 从 step 0 起的 -10) AND
    镜像上升速度上限 (由课程在上升被发现后引入 — standup 尝试税教训),
    加上全程 |a_z| 冲击惩罚.
  - 休息质量: posture_stillness (在指令高度的 velocity-Gaussian, 倾斜门控)
    + posture_composite (相对指令目标的乘性 高度·直立·姿态 — 像 plank/flop/
    lean 等部分和利用塌缩到 ~0).
  - 头部在两姿态下均可指令 (head_pose 指令 + 跟踪, 完全像 velocity/standup),
    body_command 槽位零填充 → 61D obs 一致.
  - Sim2real: velocity 一致的 DR / obs 噪声 / 延迟 / 正则项 (转移配方), sit
    环境的接触求解器强化 (nconmax=200, iters 30/50 — 坐姿接触 NaN 修复),
    延迟推力渐升 (早期推力使 sit 环境忘却坐姿).

关键帧 (稳定性验证, 与 sit/standup 环境保持同步):
  SIT  = 膝 ±1.35, hip_pitch ∓0.4079, 踝/hip_roll 0, 躯干 z 0.060
         (2026-07-27 扫掠 — 旧关键帧倾倒; 更改此姿态前在 sim 中验证倾斜).
  STAND = HOME 关节, 躯干 z 0.115 (实测站立平衡).

关节布局 (14 个驱动关节):
    0-4 : 左腿 (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : 颈/头 (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: 右腿 (hip_yaw, hip_roll, hip_pitch, knee, ankle)
"""

import math
from copy import deepcopy

# 对称性
ENABLE_SYMMETRY = False

# ── Domain randomisation (与 velocity 环境对齐以保证 sim2real 一致) ────
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # 匹配 velocity: 随机化头部组件 CoM
ENABLE_KP_RANDOMIZATION = False  # 匹配 velocity (OFF)
ENABLE_KD_RANDOMIZATION = False  # 匹配 velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # 匹配 velocity: dr.pseudo_inertia (mass+inertia)
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # 匹配 velocity: FrictionDRBamActuator.friction_scale
ENABLE_ARMATURE_RANDOMIZATION = True  # 匹配 velocity: 反映转子惯量
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # 匹配 velocity: obs 级每环境错位
ENABLE_ENCODER_BIAS = True  # 匹配 velocity: 每环境关节编码器偏移 (actor obs)

# ── 范围 (与 velocity 环境对齐) ──────────────────────────────────────────────
COM_RANDOMIZATION_RANGE = 0.003  # 通过 com_range 课程渐升到 0.015
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # 通过 head_com_range 课程渐升到 0.01
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # 未使用 (kp DR off)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # 未使用 (kd DR off)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
# 最终量级匹配 velocity 的 ±0.3 但渐升被延迟 (见 push_magnitude 课程):
# sit 环境的教训 — 在过渡运动巩固前中下降时推力使 policy 忘却它们并收敛到
# "只站立什么都不做".
VELOCITY_PUSH_RANGE = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # 匹配 velocity (obs 级, 零中心随机轴)

# Episode 长度: 容纳 2-3 个姿态段 (每段停留 3.5-6.5 s), 即每个 episode 至少
# 一个完整 坐 → 休息 → 上升 → 休息 循环.
EPISODE_LENGTH_S = 12.0
# 每个指令姿态的停留时间, 之后重采样可能翻转它. 下限必须舒适超过平缓过渡
# (~1.5 s) 加一些休息, 所以 "到达, 然后保持静止" 总是被训练.
POSTURE_DWELL_S = (3.5, 6.5)
# 重采样指令 SIT 的概率 (vs STAND). 0.5 → (reset 状态 × 指令) 的所有四种
# 组合获得相等覆盖, 包括两个保持.
SIT_PROB = 0.5

# ── SIT 关键帧 (joint_pos 索引 → 角度, rad). 单一固定目标. ─────
# 稳定性验证 2026-07-27 (sit 环境, scratchpad sweep_sit_pose2.py):
# 膝 ±1.35, hip_pitch = HOME ∓ 0.05 倾, 踝 0, hip_roll 0 在 95-100% 噪声
# reset 中以 3-5° 倾斜稳定. 旧关键帧 (膝 ±1.0472, hip_pitch HOME) 静态
# 不稳定 — 1 s 内倾倒到 ~88°, 静默驱动 sit 环境的整个 hop/back-flop/plank
# 利用链. 若机器人或关键帧改变, 重新运行扫掠 — 验证倾斜, 非 z.
# 与 microduck_sit_env_cfg.SITTING_TARGET_OVERRIDES 和
# microduck_standup_env_cfg.SITTING_JOINT_OVERRIDES 保持同步.
SITTING_TARGET_OVERRIDES = {
    1: 0.0,  # 左  hip_roll   (HOME -0.0873)
    2: -0.4079,  # 左  hip_pitch  (HOME -0.4579; +0.05 = 微前倾)
    3: 1.35,  # 左  knee       (HOME -0.0049)
    4: 0.0,  # 左  ankle      (HOME +0.4530)
    # 颈/头有意省略 → 由 head_pose 指令驱动.
    10: 0.0,  # 右 hip_roll   (HOME +0.0873)
    11: 0.4079,  # 右 hip_pitch  (HOME +0.4579)
    12: -1.35,  # 右 knee       (HOME +0.0049)
    13: 0.0,  # 右 ankle      (HOME -0.4530)
}

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# 躯干高度目标 (m) — 两者在 sim 中实测, 从不在机器人或关键帧更改间沿用
# (sit run-1 / standup 教训).
STAND_Z = 0.115
SIT_Z = 0.060

# ``upright_while_tall`` 的直立门控窗口: STAND_UPRIGHT_Z 以上满直立激励,
# SIT_UPRIGHT_Z 处衰减到 0 (承诺坐姿). 阻止 "高位时仍后倾" 下降利用;
# 常开 upright_linear 下限覆盖坐姿区域.
STAND_UPRIGHT_Z = 0.10
SIT_UPRIGHT_Z = 0.075

# 目标渐升持续时间 (s): 指令项在此时间内斜坡内部目标混合 STAND↔SIT, 姿态
# 奖励跟踪移动目标. 反碰撞机制 (run-1 失败: 近瞬时过渡). 二值目标下, 早到
# 每节省一步支付完整目标悬赏 (~7/step), 而线速度上限积分到有界超额距离代价
# (~50 总, 瞬时下降) — 碰撞赢得 ~7×. 用斜坡, 超前于设定点使高度/组合栈
# 在斜坡剩余部分归零, 所以跟踪慢设定点是 argmax. 55 mm 超过 2 s ≈ 0.028
# m/s, 舒适地在下方两上限内.
POSTURE_RAMP_S = 2.0

# 垂直速度上限 (m/s) — 现在是斜坡目标周围过冲/弹跳的后备 (见
# POSTURE_RAMP_S), 非主要平缓机制. 上升上限更松 (对抗重力上升需要一些动量
# 翻过脚跟) 且由课程仅在上升运动被发现后引入 — 见 rise_speed_weight 课程.
MAX_DESCENT_SPEED = 0.05
MAX_RISE_SPEED = 0.08

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
    MICRODUCK_ROUGH_TERRAINS_CFG,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg


def make_microduck_sitstand_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """创建 Microduck sitstand 环境配置."""
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # 注意: 此处无头部地面接触惩罚 (与 sit 环境不同). 过渡期间使用头部作为
    # 第三支撑点被明确允许 — plank 作为终端休息的利用被 posture_composite
    # + posture_stillness 反选 (两者在 plank 倾斜/高度处 ≈0).

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── 基础配置 ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    # Standup 机器人变体: 全碰撞网格 — 身体必须在坐姿时物理接触地面, 膝/头
    # 可在过渡中触碰.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── 动作 ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── 奖励: 丢弃行走专用项 ──────────────────────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── 奖励: 姿态条件单目标栈 ──────────────────────
    # 下方每个任务项读取指令姿态并按环境选择其目标 (SIT 关键帧 + SIT_Z vs
    # HOME + STAND_Z). 权重镜像 sit 环境验证的栈 (正任务质量 ≈ velocity 尺度,
    # 使共享 sim2real 正则项以相同相对强度作用 — standup 转移教训).

    # 姿态目标 — 仅腿 (头部由指令驱动). 宽松 std 保持任一端的梯度 (~1.35 rad
    # 膝增量).
    cfg.rewards["posture_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_match,
        weight=4.0,
        params={
            "command_name": "twist",
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": SITTING_TARGET_OVERRIDES,
        },
    )

    # 头部姿态跟踪 (可指令头部控制, 同 velocity/standup) — 在两姿态下活跃.
    # 权重保持轻, 使过渡中瞬时头部辅助仅支付小跟踪代价.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.75,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # L1 bootstrap — 朝指令姿态的恒定梯度.
    cfg.rewards["posture_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_l1,
        weight=1.0,
        params={
            "command_name": "twist",
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": SITTING_TARGET_OVERRIDES,
        },
    )

    # 躯干高度 — 双层高斯 (standup 配方: 宽层用于跨 55 mm 行程的 bootstrap
    # 拉, 尖峰层使最后 cm 有真实梯度而非饱和平台) + L1 过渡驱动.
    cfg.rewards["posture_height"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "twist",
            "sit_z": SIT_Z,
            "stand_z": STAND_Z,
            "std": 0.04,
        },
    )
    cfg.rewards["posture_height_sharp"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={
            "command_name": "twist",
            "sit_z": SIT_Z,
            "stand_z": STAND_Z,
            "std": 0.015,
        },
    )
    # L1 权重 6.0: 在 sit 的 5.0 和 standup 的 7.5 之间 — 在错误姿态下休息
    # 必须在两个方向都明显净负 (站立指令下保持坐姿是 standup 环境在低 L1 时
    # 的停滞模式).
    cfg.rewards["posture_height_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_height_l1,
        weight=6.0,
        params={
            "command_name": "twist",
            "sit_z": SIT_Z,
            "stand_z": STAND_Z,
        },
    )

    # 上升 bootstrap — 在指令 STAND 且躯干低于 0.125 (略高于目标使最后 cm 仍
    # 支付) 时为向上运动本身付费. 仅目的地奖励在零运动时零梯度; 无此项
    # standup 环境停在坐姿. 在 SIT 指令下为零.
    cfg.rewards["rise_bootstrap"] = RewardTermCfg(
        func=microduck_mdp.posture_rise_bootstrap,
        weight=0.75,
        params={
            "command_name": "twist",
            "max_height": 0.125,
            "max_vz": MAX_RISE_SPEED,  # 爆炸发射不能超越平缓上升
        },
    )

    # ── 平缓 (此环境的目的) — 三个互补信号 ─────
    #  - ``descent_speed``: 下降 vz 超过 0.05 m/s 的每步惩罚. sit 的反暴力项:
    #    快速下降在每步付费, 不能摊销. 从 step 0 起的 -10 (sit 教训: 在 -5 时
    #    碰撞坐姿净正), 由课程收紧到 -20.
    #  - ``rise_speed``: 站起的镜像上限 (0.08 m/s). 从权重 0 开始, 由课程在
    #    iter 750 引入 — standup 教训: 技能被发现时活跃的运动税使探索尝试净负
    #    且技能从未被发现. 坐姿关键帧起点容易 (无俯卧翻转), 所以 750 足够晚.
    #  - ``gentle_motion``: |a_z| 冲击惩罚, 双向, 常开.
    #
    # ⚠️ 正权重, 有意: 这三个函数已返回负值 (-clamp(...), -|a_z|), 与
    # *_l1_penalty 助手相同约定 (此处用 +1/+6). run 7ev90yd9 (2026-08-12) 在
    # 负权重下 — 双重否定使其成为暴力奖励 (wandb: Episode_Reward/descent_speed
    # +4.6, rise_speed +2.1, gentle_motion +0.57, 三个最大正项) 并训练出
    # 臀跳, 碰撞坐姿 policy. roller_standup 在 gentle_rise 中发现的相同 bug 类.
    # 任何奖励更改后, 检查 wandb Episode_Reward/<penalty> 保持 ≤ 0.
    cfg.rewards["descent_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_downward_velocity_penalty,
        weight=10.0,
        params={
            "max_down_vel": MAX_DESCENT_SPEED,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["rise_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_upward_velocity_penalty,
        weight=0.0,
        params={
            "max_up_vel": MAX_RISE_SPEED,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # 双层直立压力 (sit 环境值 — 反 flop 校准):
    #  - 常开线性下限: 在两休息处保持躯干垂直; 在 2.5 时 "躺背" 比直立休息
    #    落后 ~4.5/step (sit run-2 修复).
    #  - 高度门控助推: 阻止 "高位时后倾" 下降利用; 上升期间兼作到达直立拉.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_while_tall"] = RewardTermCfg(
        func=microduck_mdp.upright_while_tall,
        weight=1.5,
        params={
            "height_low": SIT_UPRIGHT_Z,
            "height_high": STAND_UPRIGHT_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # 在指令姿态下的静止 — "到达, 然后安静直立休息" 作为明确正峰. z 门控是
    # 指令高度周围的带 (过渡时不活跃); 倾斜门控对倾斜休息不支付 (背/面/侧
    # flop 得零 — sit run-2 利用).
    cfg.rewards["posture_stillness"] = RewardTermCfg(
        func=microduck_mdp.posture_stillness,
        weight=2.0,
        params={
            "command_name": "twist",
            "sit_z": SIT_Z,
            "stand_z": STAND_Z,
            "band_full": 0.012,
            "band_zero": 0.03,
            "vel_std": 0.05,
            "tilt_full_deg": 25.0,
            "tilt_zero_deg": 60.0,
        },
    )

    # 相对指令目标的乘性目标评分 — 杀死两姿态下的部分和刷 (plank, flop,
    # lean, park-1cm-short). 宽 std 保持远离目标的可见梯度 (standup 验证的
    # 校准). head_std 添加颈/头在指令处的因子: 第一次符号修复的 run 以头部
    # 悬垂到地面休息 (躯干/腿/z 都在目标 → 完整组合, 仅丢失 0.75 跟踪项,
    # 悬垂头添加被动稳定性). 用头部因子, 目标状态本身要求头在其指令姿态;
    # 过渡中瞬时头部辅助保持免费 (组合那里反正 ≈0).
    cfg.rewards["posture_composite"] = RewardTermCfg(
        func=microduck_mdp.posture_composite,
        weight=3.0,
        params={
            "command_name": "twist",
            "sit_overrides": SITTING_TARGET_OVERRIDES,
            "joint_indices": _LEG_JOINTS,
            "sit_z": SIT_Z,
            "stand_z": STAND_Z,
            "height_std": 0.03,
            "upright_std": 0.40,  # ≈ 23° 有效 — plank (~70°+) 得分 ~0
            "pose_std": 0.40,
            "head_std": 0.40,  # 头完全下垂 (~1.2 rad) → 因子 ~0.01
        },
    )

    # ── Sim2real 正则项 — 与 velocity 匹配 ─────────────────────────
    # velocity 的精确集和绝对权重:
    #   • action_rate_l2: stage 0 处 -0.1, 由 iter 1500 渐升 -0.1 → -1.0
    #   • body_ang_vel -0.05, angular_momentum -0.02
    #   • soft_landing 丢弃; joint_torques_l2 / neck_action_rate_l2 未添加
    # 加 joint_torque_rate_l2 (抗抖动), 在过渡运动存在后渐入. 两上限 + |a_z|
    # 已推动慢-谨慎运动; 按正则项类型教训, 这些平滑项阻尼抖动而不阻止慢
    # 大运动, 所以比 velocity 更重也可辩护 — 从一致开始, 仅在真机抖动时收紧.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=0.0)

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05  # velocity 值
    cfg.rewards["angular_momentum"].weight = -0.02  # velocity 值
    cfg.rewards.pop("soft_landing", None)  # velocity 移除它

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # 丢弃基础 "upright" 高斯 — 由上方双层直立替代.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── 观测 (与行走 / sit / standup policies 布局相同) ───
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )
    # mjlab 1.3.0 基础模板添加基于传感器的 foot_height + height_scan obs.
    # Sitstand 无地形高度传感器 (并丢弃行走足部奖励), 所以移除这些项.
    # foot_air_time/foot_contact(_forces) 使用 feet_ground_contact 传感器,
    # sitstand 确实定义了它, 所以它们保留.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(cfg.observations["actor"].terms[gravity_term_name])
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])

    # IMU obs 延迟: max_lag 1 — velocity 的 2026-07 审计值 (真实 dxl IMU 路径
    # 快, ±20 ms 包络).
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # obs 噪声匹配 velocity 环境.
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    # IMU 安装错位 DR (匹配 velocity): IMU 派生 actor obs 的每环境恒定旋转;
    # critic 保留真值.
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # joint_vel 上 1-ctrl-step 滞后 (Dynamixel present_velocity 旧 ~1 周期).
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # 按组 deepcopy joint_pos/joint_vel (它们共享基础模板对象), 使下方
    # encoder-bias `biased` 标志仅应用于 actor.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR (匹配 velocity): actor 看到 joint_pos + 每环境偏移;
    # critic 保留真值 joint pos.
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Head pose 指令 (可指令头部控制, 同 velocity/standup) ──
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),  # neck_pitch
            (-0.05, 0.05),  # head_pitch
            (-0.07, 0.07),  # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )

    # 指令 obs 槽位. head_command 是真实 head_pose 指令;
    # body_command 保持零填充 (此处不使用身体控制).
    # 与 velocity/standup 布局对齐: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    # ── 指令: twist 槽位中的 sit/stand 姿态标志 ────────────────────
    # cmd = [sit_flag, 0, 0]; 停留时间重采样在 episode 中翻转姿态.
    # "Stand" 是全零指令 (部署空闲对齐). runtime 通过向指令缓冲的 vx 槽位
    # 写入 0/1 驱动此过程. 内部项在 POSTURE_RAMP_S 时间内斜坡目标混合,
    # 姿态奖励跟踪此混合 (见常量注释); obs 保持原始二值标志.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = POSTURE_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.SitStandCommandCfg(
        **{
            **vars(command),
            "sit_prob": SIT_PROB,
            "ramp_s": POSTURE_RAMP_S,
            "sit_z": SIT_Z,
            "stand_z": STAND_Z,
        }
    )

    # ── 终止 ──────────────────────────────────────────────────────────
    # 无摔倒终止: 过渡期间的摇摆/倾斜必须完整演出, 使 policy 经历冲击/直立
    # 代价而非被截断 episode.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── 事件 ────────────────────────────────────────────────────────────
    # BAM (mjlab_frictionloss 分支) 每步写入每环境 dof_frictionloss/dof_damping;
    # 此 no-op 事件注册这些字段以进行每世界扩展.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # 匹配 velocity

    # 基座 reset: 站立, 略高于实测平衡 (STAND_Z=0.115).
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.11, 0.12)

    # Reset 状态混合: 50% 站立 / 50% 已坐姿 (SIT 关键帧加关节/倾斜噪声).
    # 结合独立 50/50 姿态指令, 这训练所有四种情况 — 从站立坐下, 从坐姿站起,
    # 保持站立, 保持坐姿 — 并直接给 policy 两端目标状态的值 (sit 环境的
    # 发现-bootstrap 教训, 扩展到两端).
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.5,
            "standing_prob": 0.5,
            "sitting_joint_overrides": SITTING_TARGET_OVERRIDES,
            "sitting_joint_noise_std": 0.10,  # ≈ 6° 每关节
            "sitting_tilt_max": math.radians(8),
            "sitting_z_min": 0.06,  # 稳定到 0.060 休息姿态
            "sitting_z_max": 0.075,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )

    # MuJoCo 物理鲁棒性 (sit 环境的接触 NaN 修复). standup XML 在每个 body
    # 上有全碰撞; 坐姿将躯干 + 弯曲腿 + 头部全部置于近距离地面/自接触.
    # 默认 nconmax=35 和求解器 iters=10 在坐姿尝试时溢出接触求解器 → NaN →
    # nan_state 终止惩罚下降本身 ("学习后到 iter 500 忘却" 模式).
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 30
    cfg.sim.mujoco.ls_iterations = 50

    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {
                    "x": VELOCITY_PUSH_RANGE,
                    "y": VELOCITY_PUSH_RANGE,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if ENABLE_COM_RANDOMIZATION:
        # mjlab 1.3.0: 原生 dr.body_ipos (operation="add") 在每次 reset 时
        # 读取编译时默认值 → 原生非累积.
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        # 匹配 velocity: 通过 pseudo_inertia 的物理一致 mass+inertia
        # (alpha 通过 e^(2α) 缩放两者, CoM 不变). Startup 模式.
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        # 匹配 velocity: 通过 FrictionDRBamActuator 钩子按环境缩放 BAM 的摩擦
        # 预算 (BAM 下 dof_frictionloss 被清零).
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # 注意: IMU 安装错位在上方 OBSERVATION 层应用 (匹配 velocity) — 旧的
    # 基于事件的 randomize_imu_orientation 写入 site_quat, 在 mjlab 1.3.0 下
    # 既非每环境也未被 obs 读取.

    # ── 地形 ───────────────────────────────────────────────────────────────
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG
        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # ── 课程 ────────────────────────────────────────────────────────────
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Head pose 指令范围课程 — 与 velocity/standup 环境相同的每关节渐宽
    # (5% → 100% 每关节可达 delta).
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {
                    "step": 0,
                    "ranges": (
                        (-0.05, 0.05),
                        (-0.05, 0.05),
                        (-0.07, 0.07),
                        (-0.015, 0.015),
                    ),
                },
                {
                    "step": 500 * 24,
                    "ranges": (
                        (-0.17, 0.17),
                        (-0.17, 0.17),
                        (-0.21, 0.21),
                        (-0.047, 0.047),
                    ),
                },
                {
                    "step": 1000 * 24,
                    "ranges": (
                        (-0.39, 0.39),
                        (-0.39, 0.39),
                        (-0.49, 0.49),
                        (-0.11, 0.11),
                    ),
                },
                {
                    "step": 1500 * 24,
                    "ranges": (
                        (-0.72, 0.72),
                        (-0.72, 0.72),
                        (-0.91, 0.91),
                        (-0.20, 0.20),
                    ),
                },
                {
                    "step": 2000 * 24,
                    "ranges": (
                        (-1.10, 1.10),
                        (-1.10, 1.10),
                        (-1.40, 1.40),
                        (-0.31, 0.31),
                    ),
                },
            ],
        },
    )

    # CoM 随机化范围课程 — 匹配 velocity (躯干上限 ±15 mm, 头部 ±10 mm,
    # 按 2026-07 审计).
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    # 推力课程 — 显著延迟 (sit 环境教训): 过渡中的推力使机器人翻入运动巩固前
    # 无法恢复的构型; 早期推力使 sit policy 忘却坐姿并收敛到 "只站立什么都不做".
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                    {
                        "step": 1000 * 24,
                        "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    },
                    {
                        "step": 1500 * 24,
                        "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                    },
                    {
                        "step": 2000 * 24,
                        "velocity_range": {"x": (-0.20, 0.20), "y": (-0.20, 0.20)},
                    },
                    {
                        "step": 2500 * 24,
                        "velocity_range": {
                            "x": VELOCITY_PUSH_RANGE,
                            "y": VELOCITY_PUSH_RANGE,
                        },
                    },
                ],
            },
        )

    # action_rate 课程 — velocity 的精确渐升 (-0.1 → -1.0 到 iter 1500).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * 24, "weight": -0.2},
                {"step": 750 * 24, "weight": -0.4},
                {"step": 1000 * 24, "weight": -0.6},
                {"step": 1250 * 24, "weight": -0.8},
                {"step": 1500 * 24, "weight": -1.0},
            ],
        },
    )

    # 下降速度上限收紧: 在量级 10 下发现坐姿 (碰撞坐姿已净负), 然后收紧到 20.
    # 正权重 — 函数自否定 (见奖励定义处的符号约定警告).
    cfg.curriculum["descent_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "descent_speed",
            "weight_stages": [
                {"step": 0, "weight": 10.0},
                {"step": 500 * 24, "weight": 20.0},
            ],
        },
    )

    # 上升速度上限 — 仅在上升运动存在后引入 (standup 尝试税教训: 发现期间
    # 任何运动税使探索尝试净负且技能从未被发现).
    # 推迟到 1500/2500 (从 750/1250): 上升需要一个短暂动态爆发以翻过脚跟
    # (vz > 0.08 持续数步), 第一次符号修复的 run 停滞在头朝下前折 — 半完成
    # 的上升 — 与上限在最后重心转移仍在巩固时征税一致. 坐方向平缓不依赖
    # 此上限 (descent_speed 已覆盖), 所以后引入代价低. 若上升在此启动时
    # 退化, 软化最终阶段 — 绝不提前.
    cfg.curriculum["rise_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "rise_speed",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1500 * 24, "weight": 5.0},
                {"step": 2500 * 24, "weight": 10.0},
            ],
        },
    )

    # 力矩速率抗抖动 — 在两个过渡运动都存在后渐入.
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 750 * 24, "weight": -5e-4},
                {"step": 1250 * 24, "weight": -1e-3},
            ],
        },
    )

    return cfg


# ── RL runner 配置 ──────────────────────────────────────────────────────────

MicroduckSitStandRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # 匹配 velocity; normalizer 必须由 export.py 烘焙到 ONNX
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_sitstand",
    run_name="microduck_sitstand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
