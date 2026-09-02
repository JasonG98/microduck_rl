"""Microduck 地面拾取任务.

Episodic policy 蹲下使其嘴尖尽可能接近地面但不接触 (正确朝向, 嘴朝下),
然后回到干净站立姿态 — 全程保持稳定并对推力鲁棒. obs/action 空间与行走
policy 相同, 使两者可在 runtime 通过单键切换.

任务空间目标 (无 DOWN 姿态): mouth_ground_proximity 将嘴拉向地面,
head_impact_penalty (强) 禁止接触 -> 平衡 = 嘴刚好在地面上方;
mouth_perpendicular_to_ground 将其朝下.

相位编码 (在指令槽位中, 3-D):
    command = [cos(2π·phase), sin(2π·phase), 0]
    phase ∈ [0, 0.5]  → 接近 (奖励嘴下降)
    phase ∈ [0.5, 1]  → 返回 (奖励回到站立姿态)

相位在 episode reset 时按环境随机化以解耦环境并避免同步振荡.
PERIOD = 4 s (2 s 下 + 2 s 上).

── mjlab 1.3.0 + 规范 BAM ────────────────────────────────────────────────
迁移以匹配 velocity 环境的 sim2real 机制: 固定 (非累积) CoM / head-CoM /
mass-inertia / friction / armature DR, obs 级 IMU 错位, encoder-bias,
obs normalization. 任务专用正则项有意保持比 velocity 更重 (慢速谨慎的
伸展比行走需要更多阻尼) — 见正则项块.
"""

import math
from copy import deepcopy

# 对称性 — v1.5 禁用: SYMMETRY_CFG 的 _OBS_PERM 为旧 51D obs 布局硬编码,
# 在新 61D obs (包含 head_command/body_command 填充) 上会破坏. 所有 v1.5
# 环境在 SYMMETRY_CFG 为新 obs 结构重写前以对称性关闭运行.
ENABLE_SYMMETRY = False

# ── Domain randomisation 开关 (匹配 velocity 环境) ────────────────
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = False  # off, 像 velocity
ENABLE_KD_RANDOMIZATION = False
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # 按环境缩放 BAM 摩擦预算
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # 反映转子惯量 (影响 BAM)
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # 在 obs 级应用 (每环境旋转)
ENABLE_ENCODER_BIAS = True  # actor obs 看到 joint_pos + 每环境偏移
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False
ENABLE_NECK_OFFSET_RANDOMIZATION = False  # 禁用 — 头部用于任务

# ── 范围 (匹配 velocity 环境) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm 初始, 通过课程渐升
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ±3mm 初始, 通过课程渐升
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (
    -0.15,
    0.15,
)  # 准静态动作 -> 轻柔推力 (±0.3 即使直立也使其摔倒)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # 匹配 velocity (原为 1.0)
ENCODER_BIAS_RANGE = (-0.015, 0.015)

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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_GROUND_PICK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    HEAD_BODY_NAMES,
    MICRODUCK_ROUGH_TERRAINS_CFG,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg

# ── 分段相位轮廓 (独立时长) ──────────────────────────
# 代替正弦加权 (耦合下降/保持/上升), 用 4 段轮廓门控奖励: 下降和上升慢,
# 低保持短, 站立休息长.
# GP_PERIOD = 4 s 时长:
#   下降   [0, DESCENT_END)        1.5 s  STAND->低过渡
#   低保持 [DESCENT_END, HOLD_END) 0.2 s  轻触 (短)
#   上升   [HOLD_END, RISE_END)    1.5 s  低->STAND 过渡
#   休息   [RISE_END, 1)           0.8 s  站立
# ⚠️ RISE_END=0.80 > infer_policy 脚本的 φ=0.7 截止: 上升仅在槽位播放到
# φ~1.0 (整个周期) 时完整. 检查 runtime 的实际窗口.
# ⚠️ 部署时 --ground-pick-period = 4.0.
GP_PERIOD = 4.0
DESCENT_END = 0.375
HOLD_END = 0.425
RISE_END = 0.80


def make_microduck_ground_pick_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """创建 Microduck 地面拾取环境配置."""
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

    # 头部触地冲击传感器 — 覆盖 neck 子树 (head_plate, head_shell 等).
    # 由 head_impact_penalty 奖励使用以阻止 policy 在接近时将头部撞向地面.
    head_impact_cfg = ContactSensorCfg(
        name="head_impact_contact",
        primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── 基础配置 ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_GROUND_PICK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, head_impact_cfg)
    cfg.viewer.body_name = "trunk_base"

    # ── 动作 ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # 无 NeckOffsetJointPositionAction — 头部关节是任务运动的一部分

    # ── 奖励: 移除行走专用项 ────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",  # 由相位条件 ground_pick_return_pose 替代
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── 奖励: 主要地面拾取目标 ──────────────────────────────────

    # 接近阶段: 奖励嘴尖尽可能接近地面. target_height=0 将嘴拉向地面;
    # std=0.10 从 ~20 cm (从站立) 起给梯度. "不接触" 由下方 head_impact_penalty
    # (强) 保证 -> 平衡是嘴刚好在地面上方. 权重从 2.0 升到 3.0 以拉更近.
    cfg.rewards["mouth_ground_proximity"] = RewardTermCfg(
        func=microduck_mdp.mouth_ground_proximity_phased,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "std": 0.10,
            "target_height": 0.0,
            "command_name": "twist",
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # 接近阶段: 奖励嘴尖 x 轴朝下 (垂直于地面).
    # alignment ∈ [-1, 1]: 1 = x 轴完全垂直, 0 = 水平, -1 = 朝上.
    # 朝向: 嘴轴朝下 (垂直于地面). 权重从 1.0 升到 2.0 -> "正确朝向"
    # 是显式目标.
    cfg.rewards["mouth_perpendicular_to_ground"] = RewardTermCfg(
        func=microduck_mdp.mouth_perpendicular_phased,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "command_name": "twist",
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # 返回阶段 — 腿. mjlab 1.3.0 + 规范 BAM 下被动颚关节不再是 articulation
    # 一部分, 所以 joint_pos 是干净 14 关节布局: 0-4 左腿, 5-8 颈/头, 9-13
    # 右腿. (原为旧 16 关节布局 [0-4, 11-15], passive_1/passive_2 在 9,10.)
    _LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    cfg.rewards["ground_pick_return_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=6.0,  # 4->6 : 加强起立时腿部伸展
        params={
            "std": 0.3,
            "command_name": "twist",
            "joint_indices": _LEG_JOINTS,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # 返回阶段 — 颈/头 (关节 5-8): 紧 std 防止后向过冲和头-身体碰撞
    # (头部 geom 无碰撞网格, 所以 self_collisions 抓不到 — 姿态奖励是唯一守卫).
    _NECK_JOINTS = [5, 6, 7, 8]
    cfg.rewards["ground_pick_return_pose_neck"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=6.0,
        params={
            "std": 0.15,
            "command_name": "twist",
            "joint_indices": _NECK_JOINTS,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # 起立辅助: 躯干垂直仅在上升时奖励 (加权 max(0,-sin) 像姿态返回).
    # 仅姿态返回不保证起立时的动态平衡; 此项推动躯干在伸展时保持垂直.
    # 门控在返回 -> 不干扰接近时前倾 (常开 upright 保持弱, 0.2).
    cfg.rewards["return_upright"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_upright_phased,
        weight=4.0,  # 2->4 : 更强地辅助起立时躯干平衡
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.4,
            "command_name": "twist",
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )

    # 防俯冲: 在下降+保持期间惩罚颈部速度 (上升时门控=0 -> 不阻碍起立).
    # 减缓头部俯冲但不阻止其返回.
    cfg.rewards["neck_vel_descent"] = RewardTermCfg(
        func=microduck_mdp.neck_vel_descent_penalty,
        weight=-0.1,
        params={
            "command_name": "twist",
            "joint_indices": _NECK_JOINTS,
            "hold_end": HOLD_END,
        },
    )

    # 起立时 "嘴里" 随机重量 (举起的物体, 10-40 g/episode).
    # 权重 0 的奖励: 作为 per-step 钩子将物体重量作为外力应用到 mouth_tip,
    # 门控在上升 (phase >= HOLD_END). payload 本身由 sample_mouth_payload
    # 事件在 reset 时采样.
    cfg.rewards["mouth_payload_force"] = RewardTermCfg(
        func=microduck_mdp.apply_mouth_payload_force,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["jaw_soft"], site_names=["mouth_tip"]),
            "command_name": "twist",
            "hold_end": HOLD_END,
        },
    )

    # ── 奖励: 稳定性 (从 velocity 环境保留, 权重按此任务调优)

    # 直立: 降低权重 — 机器人需要在接近时前倾.
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 0.2

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    cfg.rewards["angular_momentum"].weight = -0.02

    cfg.rewards["soft_landing"].weight = -1e-5

    # 拾取全程保持双脚接地 (脚不离地).
    # 注意: 这仅是接触; 脚在踝上的翻转由下方 feet_flat 处理 (非此项).
    cfg.rewards["feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.feet_grounded_reward,
        weight=3.0,
        params={"sensor_name": feet_ground_cfg.name},
    )

    # 脚平放. feet_grounded 仅看接触 (每脚 found): 一个在踝上翻转 (在边/尖
    # 上翻) 保持一个接触点的脚会穿过 -> "脚翻转了". feet_flat_penalty 将
    # 重力投影到脚 site 坐标系: 平放时 site Z 垂直 (xy²≈0); 任何翻转 ->
    # xy²>0. 因此禁止脚在踝轴上的翻转.
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["left_foot", "right_foot"]),
        },
    )

    # ── 奖励: 正则项 (比 velocity 更重 — 慢速谨慎伸展) ─
    # 有意保持比 velocity 环境更重: 地面拾取运动慢且精确, 所以强平滑性
    # 有助转移 (与动态 standup 恢复不同, 后者重正则项阻止了运动).

    # 动作平滑 — 平重权重 (通过下方课程渐入, 终值 -2.0 而非 velocity 的 -1.0).
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-2.0)

    # 颈/头平滑 — 更高权重因为头部大量使用.
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(func=microduck_mdp.neck_action_rate_l2, weight=-1.0)

    # 关节力矩惩罚 — 增加以进一步惩罚快速/用力动作.
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=microduck_mdp.joint_torques_l2, weight=-5e-3)

    # 自碰撞 — 深蹲时头和颈可能碰到腿.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # 无接触强制: 不想要接触 (嘴应保持刚好在上方). 强惩罚低阈值 ->
    # 任何地面接触代价高. 此项对抗 mouth_ground_proximity, 固定
    # "尽可能近但不接触" 的平衡.
    cfg.rewards["head_impact_penalty"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-2.0,
        params={"sensor_name": head_impact_cfg.name, "threshold": 1.0},
    )

    # ── 观测 (与行走 policy 相同的 61D 布局) ──────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )
    # mjlab 1.3.0 基础模板添加基于传感器的 foot_height + height_scan obs.
    # Ground-pick 无地形高度传感器 (并丢弃行走足部奖励), 所以移除这些项.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(cfg.observations["actor"].terms[gravity_term_name])
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])

    # 传感器延迟 — 匹配 velocity 环境
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 3
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 3
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # 观测噪声 — 匹配 velocity 环境
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    # IMU 安装错位 DR (匹配 velocity): IMU 派生 actor obs 的每环境恒定旋转;
    # critic 保留真值. 替代旧的基于事件的 randomize_imu_orientation
    # (site_quat 写入 — 1.3.0 下是 no-op).
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # joint_vel 上 1-ctrl-step 滞后: Dynamixel 固件通过前一位置采样窗口的
    # 移动平均计算 present_velocity, 所以 policy 实际读取的值约为 1 控制
    # 周期前的.
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # 按组 deepcopy joint_pos/joint_vel (它们共享基础模板对象) 使下方
    # encoder-bias `biased` 标志仅应用于 actor. 被动排除正则现在是无害 no-op
    # (articulation 中无被动关节) 但为与其他环境一致而保留.
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # Encoder-bias DR (匹配 velocity): actor 看到 joint_pos + 每环境偏移;
    # critic 保留真值 joint pos. 需要基础模板 encoder_bias 事件.
    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── 填充指令向量到统一 13D 布局 ──────────────────────────
    # Ground-pick 不使用 head/body 姿态指令 (头部由任务的相位运动驱动), 但
    # 所有 microduck policy 共享相同 61D obs 形状, 使 runtime 可输入单一
    # 指令缓冲. 10 个尾随槽位 (head 4 + body 6) 为常数零.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    # ── 指令: 循环相位编码 ────────────────────────────────────────
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    # 周期 = GP_PERIOD (6 s). 分段轮廓 (文件顶部常量) 解耦下降/保持/上升/休息:
    # 下降和上升 ~1.5 s (慢 -> 无失稳), 低保持 ~0.6 s (短), 休息 ~2.4 s.
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": GP_PERIOD,
        }
    )

    # ── 终止 ──────────────────────────────────────────────────────────
    # NaN 物理时终止 (极端接触冲量) 以防损坏 obs.
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

    # "嘴里" 随机重量: 每 episode 采样 (10-40 g), 由 mouth_payload_force 钩子
    # 在起立时应用. 想象机器人举起一个物体.
    cfg.events["sample_mouth_payload"] = EventTermCfg(
        func=microduck_mdp.sample_mouth_payload,
        mode="reset",
        params={"min_kg": 0.01, "max_kg": 0.04},
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)  # 匹配 velocity
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    if ENABLE_VELOCITY_PUSHES:
        # Play: 间隔宽松 (2-4 s) 以在现实行为上判断动作, 非火力压制下
        # (0.5-1 s 是激进压力测试, 即使直立也 "摔倒").
        interval = (2.0, 4.0) if play else VELOCITY_PUSH_INTERVAL_S
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
        # mjlab 1.3.0: 原生 dr.body_ipos (operation="add") 在每次 reset 时读取
        # 编译时默认值 → 原生非累积. 替代旧的 mdp.randomize_field/body_ipos
        # 路径 (1.3.0 下是 no-op).
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
        # 匹配 velocity: 随机化头部组件 body 的 CoM.
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        # 休眠 (KP/KD off, 像 velocity). 注意: randomize_delayed_actuator_gains
        # 早于规范 BAM; 仅在移植到 BamActuator.set_gains 后启用.
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
        # (alpha 通过 e^(2α) 缩放两者, CoM 不变). Startup 模式. 旧的
        # 自定义 randomize_mass_and_inertia 在 mjlab 1.3.0 下是 no-op.
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

    # 注意: IMU 安装错位在上方 OBSERVATION 层应用 (匹配 velocity) — 旧的
    # 基于事件的 randomize_imu_orientation 写入 site_quat, mjlab 1.3.0 下是 no-op.

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
    # 移除此处不适用的基础课程项
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # action_rate 课程: 轻起步使粗伸展运动形成, 然后硬收紧 (-2.0, 比 velocity
    # 的 -1.0 更重) 以保平滑.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.8},
                {"step": 250 * 24, "weight": -1.5},
                {"step": 500 * 24, "weight": -2.0},
            ],
        },
    )

    # CoM 随机化范围课程 — 匹配 velocity (渐升 0.003 → 0.02 躯干, 0.003 → 0.01 头部).
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
                    {"step": 2000 * 24, "range": 0.02},
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

    return cfg


# ── RL runner 配置 ──────────────────────────────────────────────────────────

MicroduckGroundPickRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # 匹配 velocity; normalizer 由 export.py 烘焙到 ONNX
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
    experiment_name="ground_pick",
    run_name="ground_pick",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=20_000,
)
