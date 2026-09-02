"""Microduck BallKick 任务 — 用一只脚向前踢球 (KICK_FOOT 标志).

Episodic policy: 机器人从站立开始 (HOME 姿态 + 噪声), 一个 70mm / 15g
球放在踢球脚前方 (下方 KICK_FOOT — 右脚和左脚训练为两个独立 run).
目标是向前踢球 (reset 时机器人朝向) 到 BALL_TARGET_SPEED 同时保持平衡
并对外部推力保持鲁棒, 然后回到干净站立.

关键设计决策:
  - policy 对球盲 (actor 中无球 obs): 真机无球感知 — 操作员将机器人对准球.
    对放置误差的鲁棒性来自 reset 时的 ±2cm 球位置 DR. critic 看到球位置/速度
    (非对称 actor-critic) 使价值函数能预判踢球收益.
  - 无相位指令: 踢球奖励从 t=0 起可用, 更早踢球收集更多球滚动奖励,
    所以 policy 立即踢球. 部署时: 硬 ONNX 切换到此 policy (像 jump/ground-pick),
    踢球, 然后约 2s 后自动切回.
  - 右脚踢球通过几何 + 经济方式强制: 球生成在右脚尖, 且常开左脚接地奖励
    使左腿成为支撑腿 (抬起它每步付费; 防跳).
  - 踢球奖励与球前向速度线性 (上限 5 m/s), 非饱和 tanh — "尽可能用力"
    需要高速时有梯度.
  - obs 布局是统一 61D actor 布局 (twist + 零填充 head/body 指令槽位),
    使 runtime 可用一个缓冲区硬切换 ONNX 文件.

DR / 噪声 / 正则项: velocity 一致, 从 standup 环境复制 (后者本身匹配 velocity —
经过验证的转移配方). 任务奖励质量 ~10 ≈ velocity 的 ~11, 使共享正则项权重
以相同相对强度作用.
"""

import math
from copy import deepcopy

# ── 踢球脚: "right" 或 "left" ───────────────────────────────────────────
# 翻转球生成侧和支撑脚 (防跳) 传感器. 其他一切左右对称 (HOME 姿态有镜像
# 符号). 两个 policy 作为独立 run 训练 — wandb experiment/run 名跟随此标志.
KICK_FOOT = "right"
assert KICK_FOOT in ("right", "left")

# 对称性 — 必须保持 OFF: 踢球任务本质上是单脚的.
ENABLE_SYMMETRY = False

# ── Domain randomisation (匹配 velocity / standup) ─────────────────────
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = False
ENABLE_KD_RANDOMIZATION = False
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

# ── 范围 (匹配 velocity / standup) ───────────────────────────────────
COM_RANDOMIZATION_RANGE = 0.003  # 通过课程渐升到 0.015
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # 通过课程渐升到 0.01
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # 未使用 (kp DR off)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # 未使用 (kd DR off)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # 通过推力课程渐升
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# ── 任务常量 ────────────────────────────────────────────────────────────
# 足够长以容纳踢球 + 数秒球滚动奖励 + 回到稳定.
EPISODE_LENGTH_S = 5.0

# 70mm 直径 / 15g 球 (见 ball.xml).
BALL_RADIUS = 0.035
# 机器人 yaw 坐标系中名义球心偏移. HOME 处实测: 脚中心在 (0, ±0.042),
# 脚尖 x≈0.034. 半径 0.035 加 ±0.015 噪声下球后表面最差 x=0.040 → 总是
# 离脚尖 ≥6mm. (0.08 ± 0.02 允许与脚尖的生成穿透: 求解器在 reset 时
# 弹出球 — 无踢的免费 "踢球" 奖励.)
# 横向符号跟随踢球脚 (right = -y, left = +y).
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
# 每轴均匀 ± 放置噪声. 这是使盲 policy 的挥腿对真实瞄准误差鲁棒的 DR.
BALL_POS_NOISE_XY = 0.015

# 目标踢球速度 (m/s). 第一个训练的 policy (线性奖励上限 5 m/s) 踢得太重 —
# 这将踢球驯服为轻柔可控的轻触. 注意: 下方踢球奖励权重按保持目标处收益
# ≈ +3/step 缩放 (上限项 weight ≈ 3/target) — 若更改目标, 需同步缩放权重.
BALL_TARGET_SPEED = 1.0

# 躯干站立高度 (HOME 处实测自然平衡 — 见 standup 环境).
STAND_Z = 0.115

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

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

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BALL_CFG,
    MICRODUCK_STANDUP_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg


def make_microduck_ball_kick_env_cfg(
    play: bool = False,
    kick_foot: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """创建 Microduck BallKick 环境配置.

    ``kick_foot`` 覆盖模块级 KICK_FOOT 标志 (测试使用); 正常训练只需设置此
    文件顶部的标志.
    """
    kick_foot = kick_foot or KICK_FOOT
    assert kick_foot in ("right", "left")
    support_foot = "left" if kick_foot == "right" else "right"

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

    # 支撑脚传感器: 非踢球脚必须在踢球过程中保持接地.
    support_foot_ground_cfg = ContactSensorCfg(
        name="support_foot_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=rf"^{support_foot}_foot_collision$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
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

    # 全碰撞机器人 (与 standup/ground-pick 相同规格): 球必须能接触整条腿,
    # 非仅行走模型的脚垫. 机器人必须保持为第一个 entity
    # (set_random_ground_state 和基座 reset 事件在 qpos[:, 0:7] 写入机器人
    # 根状态).
    cfg.scene.entities = {
        "robot": MICRODUCK_STANDUP_ROBOT_CFG,
        "ball": MICRODUCK_BALL_CFG,
    }
    cfg.scene.sensors = (feet_ground_cfg, support_foot_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # 球的额外接触余量 (球-地形 + 球-机器人接触叠加在全碰撞机器人预算之上).
    cfg.sim.nconmax = 50

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
        "pose",  # 步态条件; 由下方 pose_target_match 替代
        "soft_landing",  # velocity 移除它
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── 奖励: 踢球目标 — 目标速度, 非最大速度 ────────────────
    # 双侧景观在 BALL_TARGET_SPEED 处达峰 (0.25 m/s — 轻柔轻触):
    #   • ball_forward_velocity, 线性且上限到目标: 从首次接触的密集
    #     bootstrap 梯度. Weight 12.0 = 3.0/target 使目标处收益保持 ≈ +3/step
    #     (旧 weight 3.0 下收益 0.75/step — 对 ~7/step 站立栈太弱, 不足以
    #     支付挥腿的瞬态姿态/直立代价).
    #   • ball_speed_overshoot_penalty (weight -4.0): 每超过目标 1 m/s 持续
    #     每步 -4. 需要, 因为仅上限无法驯服踢球 — 更重的踢球使球在上限
    #     停留更多步, 所以总 (每步 × 滚动时间) 奖励仍随击球速度增长.
    # 斜率保持非对称 (下方 +12/(m/s), 上方 -4/(m/s)): 最优在目标处, 但
    # 用力过猛比不踢便宜得多 (净奖励仅在 ~1.0 m/s 处达 0, 4× 目标).
    cfg.rewards["ball_forward_velocity"] = RewardTermCfg(
        func=microduck_mdp.ball_forward_velocity,
        weight=12.0,
        params={"asset_name": "ball", "max_speed": BALL_TARGET_SPEED},
    )
    cfg.rewards["ball_speed_overshoot"] = RewardTermCfg(
        func=microduck_mdp.ball_speed_overshoot_penalty,
        weight=-4.0,
        params={"asset_name": "ball", "target_speed": BALL_TARGET_SPEED},
    )

    # 支撑脚: 非踢球脚接地时二值 +1. 常开防跳 — 挥动踢球脚免费, 抬起支撑脚
    # 每步付费. 也抑制行走/盘带利用 (任何步态一半时间失去此奖励).
    cfg.rewards["support_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.single_foot_grounded_reward,
        weight=2.0,
        params={"sensor_name": support_foot_ground_cfg.name},
    )

    # ── 奖励: 踢球前后干净站立 ──────────────────────────────────
    # 腿在 HOME. std=0.5 有意宽松: 踢球本身是大瞬态腿偏差, 必须保持可负担.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,  # HOME = standing
        },
    )

    # 颈/头在 HOME (此任务无头部指令; 更紧 std — 头部不参与踢球).
    cfg.rewards["pose_stand_neck"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=1.0,
        params={
            "std": 0.3,
            "joint_indices": _NECK_JOINTS,
            "target_overrides": None,
        },
    )

    # 直立 — velocity 的精确配方 (weight 2.0, std²=0.05).
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # 躯干在站立高度 — 阻止蹲/屈作为踢球准备.
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std": 0.04,
            "target_height": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # ── Sim2real 正则项 — velocity 一致 (见 standup 环境说明) ──
    cfg.rewards["action_rate_l2"].weight = -0.1  # stage-0; 课程渐升到 -1.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── 观测 (统一 61D actor 布局, 球盲) ───────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )
    # 此环境无地形高度传感器 (仅平面) — 丢弃基础模板的基于传感器的项,
    # 像 standup/ground-pick 一样.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(cfg.observations["actor"].terms[gravity_term_name])
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])

    # IMU obs 延迟 — 匹配 velocity 的 2026-07 审计值.
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # obs 噪声 — 匹配 velocity 环境.
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    # IMU 安装错位 DR (obs 级, 仅 actor).
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # joint_vel 上 1-ctrl-step 滞后 (Dynamixel 移动平均, 见 velocity 环境).
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # 按组 deepcopy joint_pos/joint_vel 使下方 encoder-bias `biased` 标志
    # 仅应用于 actor.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # 指令 obs 槽位 — 统一布局一致: [twist(3), head(4), body(6)],
    # head/body 零填充 (此任务无 head/body 姿态控制).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    # 仅 critic 的球状态 (非对称 actor-critic): actor 对球盲 (真机无球感知),
    # critic 用它预测踢球收益.
    cfg.observations["critic"].terms["ball_position"] = ObservationTermCfg(
        func=microduck_mdp.ball_pos_in_base,
        params={"asset_name": "ball"},
    )
    cfg.observations["critic"].terms["ball_velocity"] = ObservationTermCfg(
        func=microduck_mdp.ball_vel_in_base,
        params={"asset_name": "ball"},
    )

    # ── 指令: 零周围微小噪声 (仅 obs-shape 一致) ───────────────
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

    # ── 终止 ──────────────────────────────────────────────────────────
    # fell_over 保留 (机器人从站立开始, 必须在踢球过程中保持站立).
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── 事件 ────────────────────────────────────────────────────────────
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

    # 站立起点的关节噪声: 部署从 walk / velstand policy 交接, 其稳定站立
    # 不会精确匹配 HOME.
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)

    # 仅站立起点 (复用 standup 环境的 ground-state 机制做噪声直立生成:
    # 随机 yaw ± 倾斜噪声, z 接近平衡).
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.0,
            "standing_prob": 1.0,
            "sitting_tilt_max": math.radians(5),  # 站立时 ±5° pitch/roll
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )

    # 球放置 — 必须在 set_ground_state 之后 (事件按字典插入顺序运行;
    # 球位置从最终机器人姿态派生). 也存储每环境踢球方向 (reset 时机器人朝向).
    ball_offset_y = -BALL_OFFSET_ABS_Y if kick_foot == "right" else BALL_OFFSET_ABS_Y
    cfg.events["reset_ball"] = EventTermCfg(
        func=microduck_mdp.reset_ball_in_front_of_foot,
        mode="reset",
        params={
            "offset": (BALL_OFFSET_X, ball_offset_y),
            "noise_xy": BALL_POS_NOISE_XY,
            "ball_radius": BALL_RADIUS,
            "asset_name": "ball",
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
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── 地形: 仅平面 (粗糙地形上的球是不同任务) ──────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── 课程 ────────────────────────────────────────────────────────────
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # action_rate 渐升 — velocity 的精确阶段 (-0.1 → -1.0 到 iter 1500).
    # 注意: 踢球是快速一次性挥腿; 若收敛踢球太弱, 软化渐升末端
    # (-1.0 → -0.6) 是第一个尝试的旋钮 (运动阻断项 vs 动态任务权衡,
    # 见 standup 正则项注释).
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
        # 在踢球技能开始形成后渐入推力: iter 0 单腿击球阶段的全力推力会
        # 对挥腿本身的发现征税 (与 standup 相同时序教训).
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

    return cfg


# ── RL runner 配置 ──────────────────────────────────────────────────────────

MicroduckBallKickRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer 必须由 export.py 烘焙到 ONNX
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
    experiment_name=f"ball_kick_{KICK_FOOT}",
    run_name=f"ball_kick_{KICK_FOOT}",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
