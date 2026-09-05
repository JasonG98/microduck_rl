"""Microduck *站立* 任务 (v1.5) — 专门: 坐姿 → 站立.

Episodic policy 从坐姿关键帧平缓上升到站立关键帧. 与 sit 环境配对 — 共同
构成干净的 坐↔站 对, 每个 policy 负责一个方向.

Reset: 坐姿关键帧 (躯干 z ≈ 0.07, 膝/踝弯曲, 头部在 HOME). 目标: 站立关键帧
(躯干 z ≈ 0.12, HOME 关节). 奖励设计 (sit 环境的镜像): 单一固定目标从 t=0
奖励到 episode 结束; 平缓性仅通过 |a_z| 强制; 平滑性由常规 sim2real 正则项
强制. 无轨迹 waypoints, 无 episode 进度门控 — policy 自由发现其自己的上升
路径.

身体控制 (2026-07-29 重新引入): 站立后, policy 跟踪来自名义站立的指令躯干
delta [z, roll, pitch] (前零填充 6D obs 槽位中的真实 body_pose 指令). 在 iter
2500 通过本文件底部的身体控制课程启动, 在 ground_state_mix 恢复课程完成渐升
之后.
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
COM_RANDOMIZATION_RANGE = 0.003  # 通过 com_range 课程渐升到 0.015 (velocity 2026-07 审计上限; 此处原为 0.02)
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # 通过 head_com_range 课程渐升到 0.01
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # 未使用 (kp DR off)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # 未使用 (kd DR off)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
# 匹配 velocity 的 ±0.3 (velocity 在 2026-07 审计中从 ±0.5 软化). 下方推力
# 课程仍按 0 → ±0.08 → 此最终值渐升, 使坐起 bootstrap 不从 step 0 起被推
# 来推去 (velocity 从 step 0 起全强度推力, 但它从站立开始, 非坐/俯卧).
VELOCITY_PUSH_RANGE = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = (
    6.0  # 匹配 velocity (原为 2.0 — 审计前值; 真实 IMU 有 ~5° 系统俯仰误差 + 估计器漂移, 2° 训练的带太窄)
)

# Episode 长度: 足够用于平缓上升 + 短暂稳定.
EPISODE_LENGTH_S = 6.0

# ── 坐姿源姿态 (asset.data.joint_pos 索引 → 角度, rad) ───────────
# 必须匹配 sit policy 的 *实际终态*. 镜像 sit 环境的 SITTING_TARGET_OVERRIDES
# (microduck_sit_env_cfg.py) — 扫掠稳定平衡姿态 (膝 ±1.35 ≈ 77°,
# hip_pitch ∓0.4079 = 微前倾, 踝 0). 保持两者同步: 此 reset 就是 sit→站
# 交接. 颈/头有意省略 → reset 保持 HOME, 使 standup policy 从 sit policy 收敛
# 的精确位置开始.
# mjlab 1.3.0 + 规范 BAM 下的关节索引. 被动颚关节不再是 articulation 一部分
# (从 qpos 排除), 所以布局是干净的 14 关节顺序: 0-4 左腿, 5-8 颈/头, 9-13 右腿.
# (此前 passive_1/passive_2 位于 9,10, 使右腿移到 11-15.)
SITTING_JOINT_OVERRIDES = {
    1: 0.0,  # 左  hip_roll   (HOME -0.0873)
    2: -0.4079,  # 左  hip_pitch  (HOME -0.4579; +0.05 = 微前倾)
    3: 1.35,  # 左  knee       (HOME -0.0049)
    4: 0.0,  # 左  ankle      (HOME +0.4530)
    10: 0.0,  # 右 hip_roll   (HOME +0.0873)
    11: 0.4079,  # 右 hip_pitch  (HOME +0.4579)
    12: -1.35,  # 右 knee       (HOME +0.0049)
    13: 0.0,  # 右 ankle      (HOME -0.4530)
}

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# 躯干高度目标 (m).
# SIT_Z 匹配 sit 环境实测坐姿平衡 (上述扫掠稳定姿态中静止时的躯干 z).
# 原为 0.07 (旧机器人); 与 microduck_sit_env_cfg.py 保持同步.
SIT_Z = 0.060
# STAND_Z = 自然站立平衡处实测的躯干 z (HOME 关节姿态, 垂直躯干). 此前为
# 0.120 — 比 HOME 处机械可达高 5 mm — 这迫使 policy 进入后倾妥协以满足不可能
# 的高度目标. 通过 velocity policy 在零指令下保持机器人静止实测: 115 mm.
STAND_Z = 0.115

# ── 身体姿态指令 (2026-07-29 重新引入) ───────────────────────────────
# 主开关. OFF 精确恢复先前环境: 无 body_pose 指令, 零填充 body_command obs
# 槽位 (obs 仍为 61D), 无跟踪奖励, 无身体控制课程 (包括
# height_stand_sharp / upright_sharp / standing_composite 上的冲突放松阶段).
ENABLE_BODY_CONTROL = True
# 6D 指令槽位 [x, y, z, roll, pitch, yaw] 用于与 velocity/velstand 的 obs
# 一致, 但仅 z/roll/pitch 被跟踪 (下方 axis_weights) — 与原 standup 身体控制
# 和 runtime 接口相同的 3 轴. x/y/yaw 永远保持微小 "alive" 范围: policy 学会
# 忽略它们 (它们是奖励不相关噪声) 而非留下死权重.
# z 范围非对称: STAND_Z 是 HOME 处的自然平衡, 所以其下方有大量蹲姿, 但其上方
# 仅 ~1 cm 腿伸展. 角度上限 ±15°: velocity 身体控制 run 1 显示 ±20° 训练出
# 抽搐/过驱倾斜.
BODY_CMD_MAX_Z_DOWN = 0.04  # m, STAND_Z 下方蹲
BODY_CMD_MAX_Z_UP = 0.030  # m, STAND_Z 上方伸展
BODY_CMD_MAX_ANGLE = math.radians(15)  # rad, 躯干 pitch/roll
BODY_CMD_ALIVE_XY = 0.005  # m, 永久 x/y 噪声范围
BODY_CMD_ALIVE_ANGLE = 0.05  # rad, stage-0 / 永久 yaw 范围
# 重采样时的精确零指令概率: 保持部署空闲状态 ("名义站立, 无指令") 被训练
# (velocity run-1 教训 — 均匀采样从不产生全零指令).
BODY_CMD_ZERO_PROB = 0.3

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
    BODY_POSE_CMD_RESAMPLE_S,
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
    MICRODUCK_ROUGH_TERRAINS_CFG,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg


def make_microduck_standup_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """创建 Microduck 站立环境配置 (坐姿关键帧起点)."""
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

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── 基础配置 ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

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

    # ── 奖励: 有机 standup policy 的最小可行集 ────────────
    # 单一固定目标 (STAND = HOME 姿态 + STAND_Z), 从 t=0 起活跃. 无轨迹, 无
    # waypoints, 无 episode 进度门控. policy 自由发现任何满足以下条件的上升
    # 路径:
    #   (1) 终态匹配 HOME 姿态 + STAND_Z
    #   (2) 上升平缓 (全程低 |a_z|)
    #   (3) 躯干全程保持直立 (失败模式: 伸腿时后倾; 不像 sit 有 "低 z 安全"
    #       区)
    #   (4) 关节/动作运动保持平滑 (sim2real 正则项)
    #
    # 2026-07 转移修复 (真机上暴力/抖动): 下方所有任务权重除以 4 (8→2,
    # 30→7.5, 15→3.75, …), 使总任务质量 (~12) 匹配 velocity 的 (~11), 共享的
    # sim2real 正则项以与转移良好的 velocity 环境中相同的相对强度作用. 此前
    # 任务质量 ~49, 所以名义相同的正则权重在此处实际 ~4× 更弱 → 站立点附近
    # 的抖动/极限环几乎免费. 任务项之间的内部比例不变 (均匀缩放), 所以下方
    # 每项的理由注释仍成立 — 只是绝对奖励数值读 ×4. PPO 归一化优势, 所以全局
    # 尺度本身不重要; 只有任务↔正则比例重要.

    # 姿态目标 — 腿+髋+膝+踝. target_overrides=None → HOME.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,  # HOME = 站立
        },
    )

    # 头部姿态跟踪 (可指令头部控制, 同 velocity 环境). 替换旧的
    # pose_stand_neck 奖励 (将颈/头钉在 HOME) — 颈/头现在由 head_pose 指令
    # 驱动. 出于同样原因从下方 pose_stand_l1 / standing_composite 中移除, 使
    # 没有奖励与 head_pose_tracking 的梯度对抗.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.75,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # 头部 DC 下垂惩罚 (velocity 的修复, standup 适配). 头部跟踪误差的 1 s EMA
    # 上的 L1 — 仅对 policy 可通过向上偏置颈部指令抵消的持续重力下垂收费;
    # 瞬态运动平均掉. 两个 standup 专用安全措施, 两者此处都是强制:
    #  - 直立门控 (同 arrival_damping 值): 门控乘以喂入 EMA 的误差, 所以地面/
    #    上升阶段累积为零 — 终点线无奖励墙, 头部枢轴翻转无税 (退役的
    #    head_impact_penalty 正是以此方式冻结 policy).
    #  - 从 0 开始, 由下方课程在 iter 3000 引入 — 同 arrival_damping/torque_rate
    #    的发现 vs 精炼时序.
    cfg.rewards["head_pose_bias"] = RewardTermCfg(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # 由 head_pose_bias_weight 课程渐升
        params={
            "command_name": "head_pose",
            "tau_s": 1.0,
            "gate_height_low": 0.09,
            "gate_height_high": 0.11,
            "gate_tilt_full_deg": 20.0,
            "gate_tilt_zero_deg": 45.0,
        },
    )

    # L1 bootstrap — 即使远离 HOME 也有恒定梯度.
    # 从 2 提到 5: 收敛时 policy 停在 ~0.18 rad 偏离 HOME (主要是弯曲膝盖),
    # 在权重 2 时仅代价 -0.35/step — 足够便宜可忽略. 在权重 5 该误差代价
    # -0.9/step, 迫使 policy 实际关闭剩余关节的差距.
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=1.25,
        params={
            # 仅腿 — 颈/头由 head_pose_tracking 驱动.
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )

    # 躯干高度目标 — 双层高斯, 同时获得 bootstrap 距离 AND STAND_Z 处的尖峰.
    #  - ``height_stand``: 宽 std (0.04), 用于从 sit 的 bootstrap 拉.
    #  - ``height_stand_sharp``: 窄 std (0.015), 在最后 cm 处创建强梯度.
    #    早期 run 在 z ≈ 0.109 收敛, 因为宽 std 高斯已饱和 (0.93/1.0) — 无梯度
    #    拉最后 cm. 尖峰层在该范围添加 0.36→1.0 奖励跳跃, ~3× 边际拉力.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std": 0.04,
            "target_height": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std": 0.015,
            "target_height": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # L1 从 10 提到 30: 先前 run 因坐姿静止而平台化, 因为静止坐姿盆 (-0.5 L1
    # 奖励 + 其他全正) 净正. 权重 30 时坐姿静止代价 -1.5/step — "保持坐姿"
    # 的净代价迫使探索.
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=7.5,
        params={
            "target_height": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # 奖励 STAND_Z 下方向上 CoM 速度 — 为上升的 *运动* 付费, 而非仅目的地.
    # 关键 bootstrap: 仅目的地奖励下, "保持坐姿直立收集大部分姿态 + 直立"
    # 是主导局部最优. 直接奖励 vz > 0 使任何上升尝试立即变正. 在 max_height
    # 以上关闭门控, 使 policy 不能通过上下弹跳刷. max_height 设在略高于
    # STAND_Z (0.12 → 0.125), 使奖励在最后 cm 上升中仍活跃. 早期 0.11 使
    # policy 停在 ~0.108 (门控关闭高度) 且从不完成爬升.
    # 无 max_vz 上限 (2026-07-24 回退, 第二次失败 run): 上限化奖励上升速度 —
    # 即使在慷慨的 0.30 — 缩小发现阶段嘈杂恢复 *尝试* 的回报, 面/面下恢复
    # 从未学到. 两次失败 run 共享相同 wandb 签名, 无论上限值 (0.15 或 0.30)
    # 和门控调整: 站立指标在 ground_state_mix 阶段 (1500/2500) 下降, 而非像
    # 参考 run 那样恢复. 现在由下方后期阶段的惩罚课程替代完成平滑 (见
    # arrival_damping / smoothness_polish 注释).
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=0.75,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": 0.125,
        },
    )

    # 平缓上升 — |a_z| 惩罚. 与 com_upward_velocity 兼容: 恒定正 vz 收集
    # 向上速度奖励且有 a_z = 0, 所以两压力共同选择平滑恒速上升.
    # 注意此项是 GLOBAL (非相位门控): 俯卧翻转全额付费 (冲击 + 推离是 |a_z|
    # 尖峰). 2026-07-24 将其翻倍到 -0.01 促成面朝上冻结; -0.005 是上限, 除非
    # 它获得像 arrival_damping 那样的高度/倾斜门控.
    # ⚠️ 正权重: trunk_vertical_accel_penalty 已返回 -|a_z|. 先前 -0.005 双
    # 重否定为垂直冲击的 (小) 奖励 — roller_standup 在其 gentle_rise 中发现
    # 并修复的相同符号 bug, 在 sitstand run 7ev90yd9 上再次确认 (其
    # Episode_Reward/gentle_motion 记录为正). 保持小量: |a_z| 在俯卧翻转中
    # 不可避免, 大权重是运动阻断剂.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # 到达阻尼器 — 躯干 ω_xy², 门控在高度 AND 倾斜 (45° 倾斜以上 / 0.09 m
    # 以下为零, 20° 倾斜以下 / 0.11 m 以上满). 针对真机失败循环: 上升 →
    # 过冲垂直 → 倾倒 → 重试.
    #
    # 从权重 0 开始 — 由下方 arrival_damping 课程在 iter 3000 引入. 两次失败
    # run (2026-07-24) 证明任何在恢复发现阶段 (ground_state_mix 渐升面朝下/
    # 面朝上直到 iter 2500) 活跃的尝试税阻止翻转被发现: 困难姿态的探索是嘈杂
    # 抽打, 对其收费使尝试净负, "什么都不做"赢. 门控精炼 (倾斜门控, 减半权重,
    # 慷慨 vz 上限) 不改变失败签名 — 修复是时序, 而非量级. 从 iter 3000 起技能
    # 已存在并持续被俯卧 reset 练习, 所以阻尼精调其执行而非阻止其发现.
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low": 0.09,
            "height_high": 0.11,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # 直立 — 双层, 同高度奖励.
    #  - ``upright_linear``: cos(tilt). 高倾斜处强梯度 (如恢复开始时倒置),
    #    接近垂直时弱. 提供从任何朝向的 bootstrap 拉.
    #  - ``upright_sharp``: exp(-tilt²/std²), std ≈ 6°. 线性版本用尽气力的近
    #    垂直区域梯度最强. 先前 run 在 ~37° 后倾收敛, 因为小倾斜处线性拉力变
    #    弱; 此项惩罚该精确区域.
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # 尖峰高斯直立, 由躯干 z 门控. 仅当机器人实际在站立高度时支付 — 阻止
    # "蹲低并垂直"利用. std 从 0.1 加宽到 0.3 (≈17°): 此前太尖, 在倾斜盆中
    # 接近零分 (无梯度). 用 0.3, z=0.111 处倾斜盆 (smoothstep ~0.91) 和倾斜
    # 37° (高斯 ~0.11) 得分 ~0.1 = 拉向垂直的可见梯度.
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=1.5,
        params={
            "std": 0.3,
            "height_low": SIT_Z,
            "height_high": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # 平滑乘性目标状态评分 (宽 std).
    # 先前紧 std (height=0.015, upright=0.15, pose=0.20) 使组合在倾斜盆 ~5e-5
    # — 对 policy 不可见, 零梯度. 加宽使倾斜盆得分 ~0.2 (可见梯度) 而目标仍
    # 得分 ~1.0 (清晰吸引子).
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=microduck_mdp.standing_composite_score,
        weight=3.75,
        params={
            "target_height": STAND_Z,
            "height_std": 0.04,  # 4cm — 宽, 覆盖爬升
            "upright_std": 0.40,  # ≈ 23° — 倾斜盆得分 ~0.3
            "pose_std": 0.40,  # joint-RMS, 足够宽以覆盖部分姿态
            "joint_indices": _LEG_JOINTS,  # 颈/头由 head_pose_tracking 驱动
            "target_overrides": None,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # 身体姿态跟踪 — 仅 z/roll/pitch (axis_weights), runtime 身体控制轴.
    # Locomotion 变体 (非 body_pose_tracking_6d), 使未用 x/y 轴不引用生成原点
    # (机器人在俯卧翻转中离开该点). 权重从 0 开始; body_pose_tracking_weight
    # 从 iter 2500 (ground_state_mix 完成后) 渐入, 使恢复发现不受干扰. 俯卧/
    # 上升时奖励在所有跟踪轴上 ≈0, 所以机器人站起前它只是另一个站立吸引子 —
    # 与运动惩罚不同, 它不能对翻转/上升尝试征税.
    # 有意紧 std (standup 阶段 2 教训): 1 cm z 误差在 z_std=0.01 时轴奖励降到
    # 0.37 (真实梯度); 0.02 → 0.78.
    if ENABLE_BODY_CONTROL:
        cfg.rewards["body_pose_tracking"] = RewardTermCfg(
            func=microduck_mdp.body_pose_tracking_locomotion,
            weight=0.0,
            params={
                "command_name": "body_pose",
                "nominal_height": STAND_Z,
                "z_std": 0.01,
                "angle_std": math.radians(5),
                "axis_weights": (0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
                "vel_gate_command_name": None,
            },
        )

    # ── Sim2real 正则项 — 与 velocity 匹配 (2026-07) ───────────────
    # velocity 的精确集和绝对权重:
    #   • action_rate_l2: stage 0 处 -0.1, 由 iter 1500 渐升 -0.1 → -1.0
    #     (下方 action_rate_weight 课程, velocity 的精确阶段)
    #   • body_ang_vel -0.05, angular_momentum -0.02
    #   • microduck 专用额外项丢弃, 同 velocity:
    #     neck_action_rate_l2, joint_torques_l2, joint_torque_rate_l2, soft_landing
    # 一致性由上方 ÷4 任务栈缩放变为真实 — 此前相同绝对权重相对于 ~49 任务
    # 质量实际 ~4× 更弱.
    #
    # 历史/风险: 在旧任务尺度下, 将 body_ang_vel 提到 -0.15 和 action_rate
    # 终点到 -1.2 杀死背部恢复 (两者都是翻转的运动阻断剂). 在新 ÷4 尺度下,
    # body_ang_vel -0.05 ≈ -0.2 旧单位 — 关注 ground_state_mix 渐入 (iters
    # 600–2500) 时的面朝下/面朝上恢复. 若恢复冻结: 先将 body_ang_vel 减半到
    # -0.025, 再软化 action_rate 课程终点到 -0.6.
    #
    # 2026-07 平滑度精修 (÷4 重缩放后真机上上升暴力 + 过冲重试循环):
    # joint_torque_rate_l2 (抗抖动: 惩罚力矩变化, 非幅度/旋转) + arrival_damping
    # (上方奖励块). 两者都从权重 0 开始, 由下方平滑度精修课程在 iter 3000 引入
    # — 奖励集到那时与工作中的 2026-07-23 run 相同. 见 arrival_damping 注释,
    # 为何时序 (发现 vs 精调), 而非量级, 决定这些项是否破坏恢复.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=0.0)

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05  # 运动阻断剂: 保持轻 (velocity 值)
    cfg.rewards["angular_momentum"].weight = -0.02  # velocity 值
    cfg.rewards.pop("soft_landing", None)  # velocity 移除它

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # 仅丢弃基础 "upright" 高斯 — standup 使用自己的 upright_linear/
    # upright_sharp 替代. (angular_momentum 保留上方以匹配 velocity;
    # soft_landing/hip_yaw_roll_deviation 丢弃以匹配 velocity.)
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── 观测 (与行走 / sit policies 布局相同) ─────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )
    # mjlab 1.3.0 基础模板添加基于传感器的 foot_height + height_scan obs.
    # Standup 无地形高度传感器 (并丢弃行走足部奖励), 所以移除这些项.
    # foot_air_time/foot_contact(_forces) 使用 feet_ground_contact 传感器,
    # standup 确实定义了它, 所以它们保留.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    # 保留的传感器派生 critic 项获得 NaN 安全包装: 非有限接触力绕过
    # robot_state_is_nan (它仅检查关节 + 根状态), 单个 NaN 通过 rsl_rl 的
    # check_nan 杀死 run — 2026-08-21 Velocity2-Rough-Backlash 崩溃. Standup
    # 持续着陆和翻转, 所以退化接触在此更可能.
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(cfg.observations["actor"].terms[gravity_term_name])
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])

    # IMU obs 延迟: max_lag 1 (原为 3 = 60 ms 最坏情况) — 匹配 velocity 的
    # 2026-07 审计值; 真实 dxl IMU 路径快 (±20 ms 包络).
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

    # 编码器偏置 DR (匹配 velocity): actor 看到 joint_pos + 每环境偏置;
    # critic 保留真值关节位置. 需要基础模板 encoder_bias 事件.
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── 头部姿态指令 (可指令头部控制, 同 velocity 环境) ───
    # 颈/头关节的 4D 相对 HOME 增量: [neck_pitch, head_pitch, head_yaw,
    # head_roll]. 由下方 head_pose_tracking 跟踪; 范围由 head_pose_range 课程
    # 加宽. 与 velocity 环境相同的每关节上限.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),  # neck_pitch
            (-0.05, 0.05),  # head_pitch
            (-0.07, 0.07),  # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )

    # ── 身体姿态指令 (相对名义站立的 6D 增量) ───────────────────
    # [x, y, z, roll, pitch, yaw]. 仅 z/roll/pitch 被跟踪 (见下方
    # body_pose_tracking); x/y/yaw 是永久 alive 范围噪声. 范围从微小开始;
    # body_pose_range 课程在恢复技能存在后加宽 z/roll/pitch (ground_state_mix
    # 在 2500 完成).
    if ENABLE_BODY_CONTROL:
        cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
            resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
            zero_command_prob=BODY_CMD_ZERO_PROB,
            ranges=(
                (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY),  # x (m)
                (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY),  # y (m)
                (-0.005, 0.005),  # z (m)
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # roll
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # pitch
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # yaw
            ),
        )

    # 指令 obs 槽位. head_command 是真实 head_pose 指令; body_command 槽位
    # 在启用身体控制时携带真实 body_pose 指令, 否则零填充 (obs 形状两种方式
    # 相同). 与 velocity/velstand 布局一致: [twist(3), head_pose(4), body_pose(6)].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        if ENABLE_BODY_CONTROL:
            cfg.observations[group].terms["body_command"] = ObservationTermCfg(
                func=mdp.generated_commands,
                params={"command_name": "body_pose"},
            )
        else:
            cfg.observations[group].terms["body_command"] = ObservationTermCfg(
                func=microduck_mdp.zero_command_padding,
                params={"dim": 6},
            )

    # ── 指令: 零附近微小噪声 (保留以维持 obs 形状一致) ──────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── 终止 ──────────────────────────────────────────────────────────────────
    # 机器人从坐姿开始 — 基于倾斜的摔倒终止不适用此处.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )

    # ── 事件 ────────────────────────────────────────────────────────────────
    # BAM (mjlab_frictionloss 分支) 每步写入每环境 dof_frictionloss/dof_damping;
    # 此 no-op 事件注册这些字段以供每世界扩展.
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

    # 从坐姿关键帧开始, 带关节 + 躯干倾斜噪声. 真实部署从 sit policy 交接
    # 不会精确重现 SIT 关键帧 — standup policy 必须对一组合理的 "类坐姿"
    # 起点稳健. 无噪声时 policy 过拟合到精确规范 SIT 姿态.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            # 从任何姿态初始化, 各 25%: 前 (面朝下), 后 (面朝上), 坐姿关键帧,
            # 和已站立 (使 policy 也学会 *保持* 站立, 不仅上升).
            # 初始混合 = 课程 stage 0 (易); ground_state_mix 课程在训练中渐升
            # 这些 易→难. 面朝上 (后) 从 0 开始, 晚期引入 (最难恢复).
            "face_down_prob": 0.20,  # 腹部贴地 (+90° pitch)
            "face_up_prob": 0.00,  # 背部贴地 (-90° pitch) — 晚期引入
            "sitting_prob": 0.40,  # 坐姿关键帧 (部署交接)
            "standing_prob": 0.40,  # 已在站立高度直立
            # 俯卧 reset 高度: 躯干面朝下静止在 ~0.044 m (实测), 所以在地面稍
            # 上方生成而非 0.20–0.25 默认 (会在着陆前自由落体 ~15 cm).
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            # 面朝上生成的部分滚动噪声 (绕身体长轴 ±90°): 背部恢复是种子运气
            # (1 成功 / 3 失败, 等价奖励), 因为从平卧到俯卧的奖励景观平坦 —
            # 滚动完成前无梯度. 近侧边生成将起点放在滚动中途 → 内置反向课程.
            # 见 mdp.py 中的 set_random_ground_state.
            "face_up_roll_max": math.radians(90),
            "sitting_joint_overrides": SITTING_JOINT_OVERRIDES,
            "sitting_joint_noise_std": 0.12,  # ≈ 7° 每关节
            "sitting_tilt_max": math.radians(10),  # ±10° pitch/roll
            # 坐姿平衡是 SIT_Z=0.060 — 围绕它的 −1cm/+3cm 带 (平衡为 0.07 时
            # 0.06–0.10 的相同分布).
            "sitting_z_min": 0.05,
            "sitting_z_max": 0.09,
            # 站立初始化: 躯干略高于实测平衡 (STAND_Z=0.115).
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )

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
        # mjlab 1.3.0: 原生 dr.body_ipos (operation="add") 每次 reset 读取
        # 编译时默认 → 原生非累积.
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
        # 匹配 velocity: 随机化头部组件体的 CoM.
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
        # 匹配 velocity: 反映转子惯量 (非累积, 影响 BAM).
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
        # (alpha 通过 e^(2α) 缩放两者, CoM 不变). Startup 模式. 旧的自定义
        # randomize_mass_and_inertia 在 mjlab 1.3.0 下是 no-op.
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
        # 匹配 velocity: 通过 FrictionDRBamActuator hook 每环境缩放 BAM 摩擦
        # 预算 (dof_frictionloss 在 BAM 下归零).
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # 注意: IMU 安装错位在下方 OBSERVATION 级应用 (匹配 velocity) — 旧基于
    # 事件的 randomize_imu_orientation 写 site_quat, 在 mjlab 1.3.0 下既非每
    # 环境也不被 obs 读取.

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

    # 初始姿态课程: 将 set_ground_state 混合从 易→难 渐升而非从 step 0 起扁平
    # 25/25/25/25. 扁平划分下 policy 优化易多数 (保持站立 + 坐起) 并使困难姿态
    # 训练不足 — 前面仅部分起, 面朝上 (后) 冻结成 "什么都不做". 此项先引入
    # 站立/坐姿, 再面朝下, 最后面朝上, 并在后期偏向困难姿态使其获得最多练习.
    # (event_param_curriculum 浅合并这些键到活跃 set_ground_state 事件;
    # z 范围 / 关节覆盖保持不变.)
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                # step,          standing, sitting, face_down(front), face_up(back)
                {
                    "step": 0,
                    "params": {
                        "standing_prob": 0.40,
                        "sitting_prob": 0.40,
                        "face_down_prob": 0.20,
                        "face_up_prob": 0.00,
                    },
                },
                {
                    "step": 600 * 24,
                    "params": {
                        "standing_prob": 0.25,
                        "sitting_prob": 0.30,
                        "face_down_prob": 0.35,
                        "face_up_prob": 0.10,
                    },
                },
                {
                    "step": 1500 * 24,
                    "params": {
                        "standing_prob": 0.20,
                        "sitting_prob": 0.25,
                        "face_down_prob": 0.30,
                        "face_up_prob": 0.25,
                    },
                },
                {
                    "step": 2500 * 24,
                    "params": {
                        "standing_prob": 0.15,
                        "sitting_prob": 0.20,
                        "face_down_prob": 0.30,
                        "face_up_prob": 0.35,
                    },
                },
            ],
        },
    )

    # 头部姿态指令范围课程 — 与 velocity 环境相同的每关节加宽 (~2000 iters
    # 内每关节相对 HOME 可达增量的 5% → 100%).
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

    # 注意: 先前的 head_pose_std / head_pose_weight 课程 (头部下垂的创可贴)
    # 已移除 — 下垂是后向 CoM 平衡拐杖, 由 STAND2 前移站立姿态在源处修复.
    # head_pose 跟踪保持基线 (权重 3.0, std 0.5) + head_pose_range.

    # CoM 随机化范围课程 — 匹配 velocity (前 ~1500 / ~1000 iters 渐升
    # 0.003 → 0.015 躯干, 0.003 → 0.01 头部). 躯干上限 ±15 mm 按 velocity 的
    # 2026-07 审计: 超过该值随机化 CoM 可完全离开足部支撑多边形, 训练出
    # 超反应校正. (此处旧 0.02 最终阶段超过该上限.)
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

    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                    {
                        "step": 500 * 24,
                        "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                    },
                    {
                        "step": 1000 * 24,
                        "velocity_range": {
                            "x": VELOCITY_PUSH_RANGE,
                            "y": VELOCITY_PUSH_RANGE,
                        },
                    },
                ],
            },
        )

    # action_rate 课程 — velocity 的精确渐升 (-0.1 → -1.0 到 iter 1500).
    # 比旧 -0.4/-0.8/-1.0-by-500 渐升的早期阶段更温和: 上升技能在轻平滑下
    # 被发现, 然后阻尼收紧.
    # (旧注, 仍相关: -1.2 终点曾阻止背部恢复; -1.0 是上限. 在 ÷4 任务尺度
    # 下此 -1.0 现在实际咬合.)
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

    # 平滑度精修课程 — 仅在恢复技能存在后引入反暴力项. ground_state_mix
    # 在 iter 2500 完成困难姿态渐升; 从 3000 起, 俯卧 reset 持续练习学到的
    # 翻转, 同时这些惩罚精调其执行 (到达时刹车, 更少抖动). 两次 run 证明
    # 相同权重从 step 0 活跃阻止翻转被发现 (对探索的尝试税). 若恢复在 3000
    # 后退化, 软化最后阶段, 不要将引入提前.
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "arrival_damping",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 3000 * 24, "weight": -0.025},
                {"step": 4000 * 24, "weight": -0.05},
            ],
        },
    )
    # head_pose_bias: 与 arrival_damping 相同的引入时序 (见其注释 — 时序, 非
    # 量级, 保护恢复发现). 剂量: standup 在 head_pose_tracking 用 0.75 vs
    # velocity 的 2.0 (任务权重 ÷4 重平衡), 所以下垂落在 1.5 vs velocity 的
    # 3.0. 在 1.5 时 15° 站立下垂代价 0.39/step, 5° 代价 0.13/step. 若一次
    # run 后站立头部仍低, 提高最后阶段 — 不要将引入提前.
    cfg.curriculum["head_pose_bias_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 3000 * 24, "weight": 0.5},
                {"step": 4000 * 24, "weight": 1.5},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 3000 * 24, "weight": -1e-3},
            ],
        },
    )

    # ── 身体控制课程 ────────────────────────────────────────────────────────
    # 下方一切都是身体控制专用 — 注意提前返回; 在此行上方添加任何无关 cfg.
    if not ENABLE_BODY_CONTROL:
        return cfg

    # 跟踪权重在 2500 渐入 — 恰在 ground_state_mix 达到其最终 (最难) 混合时,
    # 使恢复发现阶段在无身体指令压力下训练. 最终权重 4.0: 在全指令下固定
    # 站立项以 ~2/step 反对跟踪, 在下方放松阶段之后, 跟踪的边际增益是
    # ~0.65/step 每单位权重 → 4.0 以余量获胜. (无放松阶段反对是 ~4.3/step,
    # 即使旧设计的权重 5 也输 — 阶段 2 教训.)
    cfg.curriculum["body_pose_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "body_pose_tracking",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 2500 * 24, "weight": 1.5},
                {"step": 3000 * 24, "weight": 3.0},
                {"step": 4000 * 24, "weight": 4.0},
            ],
        },
    )

    # 指令范围加宽, 与权重渐升同步. x/y/yaw 保持其 alive 范围 (未跟踪);
    # 仅 z/roll/pitch 加宽. z 非对称 — 见 BODY_CMD 常量块.
    _alive_xy = (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY)
    _alive_ang = (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE)
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                # ranges = (x, y, z, roll, pitch, yaw)
                {
                    "step": 0,
                    "ranges": (
                        _alive_xy,
                        _alive_xy,
                        (-0.005, 0.005),
                        _alive_ang,
                        _alive_ang,
                        _alive_ang,
                    ),
                },
                {
                    "step": 2500 * 24,
                    "ranges": (
                        _alive_xy,
                        _alive_xy,
                        (-0.010, 0.005),
                        (-math.radians(8), math.radians(8)),
                        (-math.radians(8), math.radians(8)),
                        _alive_ang,
                    ),
                },
                {
                    "step": 3000 * 24,
                    "ranges": (
                        _alive_xy,
                        _alive_xy,
                        (-0.018, 0.008),
                        (-math.radians(12), math.radians(12)),
                        (-math.radians(12), math.radians(12)),
                        _alive_ang,
                    ),
                },
                {
                    "step": 4000 * 24,
                    "ranges": (
                        _alive_xy,
                        _alive_xy,
                        (-BODY_CMD_MAX_Z_DOWN, BODY_CMD_MAX_Z_UP),
                        (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
                        (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
                        _alive_ang,
                    ),
                },
            ],
        },
    )

    # 冲突放松 — standup 阶段 2 教训应用于此奖励集: 尖锐固定站立吸引子直接
    # 出价超过指令偏差 (在 Δz=−2cm/15° 倾斜: height_stand_sharp −0.83,
    # upright_sharp −0.79, standing_composite −1.9 每 step). 它们的 bootstrap/
    # 精修工作在 3000 完成; body_pose_tracking 在 cmd=0 (30% 重采样) 接管
    # "名义站立处的尖峰"角色, 带更紧 std. 宽 bootstrap 层 (height_stand,
    # upright_linear, height_stand_l1, pose_stand_*) 不动 — 它们是恢复依赖的,
    # 其在全指令下反对温和 (~0.9/step 总). 站立吸引子质量大致守恒: 之前
    # 6.25 → 2.2 + 跟踪 4.0.
    cfg.curriculum["height_stand_sharp_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "height_stand_sharp",
            "weight_stages": [
                {"step": 0, "weight": 1.0},
                {"step": 3000 * 24, "weight": 0.5},
                {"step": 4000 * 24, "weight": 0.2},
            ],
        },
    )
    cfg.curriculum["upright_sharp_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "upright_sharp",
            "weight_stages": [
                {"step": 0, "weight": 1.5},
                {"step": 3000 * 24, "weight": 1.0},
                {"step": 4000 * 24, "weight": 0.5},
            ],
        },
    )
    cfg.curriculum["standing_composite_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "standing_composite",
            "weight_stages": [
                {"step": 0, "weight": 3.75},
                {"step": 3000 * 24, "weight": 2.5},
                {"step": 4000 * 24, "weight": 1.5},
            ],
        },
    )

    return cfg


# ── RL runner 配置 ──────────────────────────────────────────────────────────

MicroduckStandUpRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_stand",
    run_name="microduck_stand",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
