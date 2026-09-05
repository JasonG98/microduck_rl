"""Microduck 前滚 (roulade) 任务 — 尝试 3, run 2.

Episodic policy: 机器人从站立开始, 经过头部平坦顶部向前翻滚, 然后双脚落地.
部署时触发, 像 sit/standup 一样 (policy 切换 = 翻滚立即开始; 无相位时钟,
无参考运动).

RUN-2 重做 (run 1 学到了暴力的弹道 "breakdance" 鞭打 — run-1 奖励下的最优:
相同 2π, 更早, 无代价): 旋转现在仅在机器人接触地面时计数 (支撑门控累加器 —
roulade 永不离开地面), 落地年金需要过头顶接触锁, 付费进度速率上限 3 rad/s
(更快则放弃超出部分), 超速惩罚对 |ω| > 4 rad/s 征税, 冲击/平滑惩罚从 step 0
起活跃 (此环境中发现容易; 风格是稀缺资源, 非探索).

设计 (完整历史见 mdp.py 的 roulade 部分):
  • 一个密集进度信号 — 支付最大累计前进旋转的增量 (基于势函数: 完整翻滚
    总共支付 2π 价值, 任何位置停留每步支付零).
  • 落地奖励门控在翻滚完成 (旋转前沿 ≥ ~260°), 非时钟 — "什么都不做" 无收益,
    站立生成无法刷取, 无直立/高度压力反对翻转.
  • 通过翻滚中途生成做反向课程 (修复 standup 仰面恢复的技巧): 部分 episode
    从翻滚 50°-185° 处开始, 蜷缩, 带前向角动量, 累加器预设到生成角度.
    roulade 的后半程就是仰面恢复问题, 我们已知可学.
  • 后续 Élan 钩子: reset_roulade_state.forward_vel_range 给站立生成一个初始
    前向基座速度 — 设置 ROULADE_FORWARD_VEL_RANGE 到例如 (0.0, 0.3) 以训练
    从行走中进入的翻滚. (0, 0) = 仅静止.

DR / obs / 正则项镜像 standup 环境 (velocity sim2real 一致), 运动阻断项
(body_ang_vel, |a_z|, arrival damping) 在发现期间保持近零, 由课程后引入 —
翻滚是大角速度, 大冲击事件; 对尝试征税阻止发现 (standup 上证明两次).
"""

import math
from copy import deepcopy

# 对称性 — 翻滚是矢状/左右对称的; 镜像损失直接对抗 run 2 中看到的侧向塌缩
# 失败. 在 symmetry.py 迁移到 61 维布局后启用 (2026-08-13, 包含
# "policy" → "actor" 输出键修复; roulade 是第一个使用它的环境).
ENABLE_SYMMETRY = True

# ── Domain randomisation (匹配 standup/velocity 以保证 sim2real 一致) ───
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = False  # 匹配 velocity (OFF)
ENABLE_KD_RANDOMIZATION = False  # 匹配 velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = False  # 翻滚中途推力不连贯
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

# ── 范围 (匹配 standup 环境) ───────────────────────────────────────────────
COM_RANDOMIZATION_RANGE = 0.003  # 通过课程渐升到 0.015
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # 通过课程渐升到 0.01
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # 未使用 (kp DR off)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # 未使用 (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: 一个受控翻滚 ~2 s + 起立 ~1.5 s + 稳定. Run-3: 4 → 5 s
# (4 s 在节奏翻滚后无空间起立).
EPISODE_LENGTH_S = 5.0

# 实测站立躯干高度 (standup 教训: 不要猜测).
STAND_Z = 0.115

# ── Élan (run-up) 钩子 ────────────────────────────────────────────────────────
# (0, 0) = 从静止翻滚 (run 1). 展宽到例如 (0.0, 0.3) 以训练带前向动量进入的
# 翻滚 — 站立生成获得随机初始前向基座速度, 近似从行走 policy 交接而不模拟
# 行走本身.
ROULADE_FORWARD_VEL_RANGE = (0.0, 0.0)

# ── 翻滚中途生成 (反向课程) ───────────────────────────────────────
# 90° = 头顶平衡, 180° = 仰卧, 270° = 仰面, ~340° = 后倾坐姿,
# >260° 打开落地门控. Run-3 变更: MAX 展宽 185° → 340° — run-2 wandb 显示
# 翻滚后半程 (仰面 → 坐姿 → 起立) 从未被生成也从未学会; ~300° 后的生成
# 出生即打开落地门控, 给 crouch→stand 最后一英里密集 on-policy 数据
# (velstand run-5 crouch-basin 教训).
MIDROLL_PITCH_MIN = math.radians(50.0)
MIDROLL_PITCH_MAX = math.radians(340.0)
MIDROLL_OMEGA_RANGE = (0.0, 3.0)  # rad/s 生成时前向动量
# 蜷缩锚点: 腿折叠 (velstand crouch reset 的 crouch-anchor 值) + 下巴收紧
# (run-5: neck_pitch −1 / head_pitch +1 使平坦头顶正对地面 — 实测 axis_z
# −0.99 vs 被动 face-plant 的 +0.6; 头顶锁需要此姿态, 所以翻滚中途生成必须
# 呈现蜷缩构型). servo-index 键控; 翻滚中途生成按每环境因子 lerp HOME→tuck.
TUCK_OVERRIDES = {
    2: -1.15,  # left  hip_pitch
    3: 1.25,  # left  knee
    4: 1.05,  # left  ankle
    5: -1.0,  # neck_pitch  (chin tuck)
    6: 1.0,  # head_pitch  (chin tuck)
    11: 1.15,  # right hip_pitch
    12: -1.25,  # right knee
    13: -1.05,  # right ankle
}

# 旋转阈值 (rad), 用于状态门控.
LANDING_GATE_LO = math.radians(260.0)
LANDING_GATE_HI = math.radians(330.0)
RISE_GATE_LO = math.radians(180.0)
RISE_GATE_HI = math.radians(260.0)

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

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg


def make_microduck_roulade_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """创建 Microduck 前滚环境配置."""
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

    # 头部-地面接触 — 翻滚的枢轴信号. jaw_soft 是承载头部碰撞 geom
    # (top_head_shell = 平坦顶部, jaw, bottom_head_shell) 的 body, 在
    # robot_groundcontact.xml 中. 名称是关键载荷:
    # _update_roulade_accum 读取它用于过头顶锁.
    head_ground_cfg = ContactSensorCfg(
        name="head_ground_contact",
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # 全机器人地面接触 — 支撑门控 (run-2 修复): 旋转累加器仅在某个机器人 geom
    # 接触地形时积分, 所以弹道翻转 ("breakdance") 无进度收益且永不完成.
    # 名称是关键载荷: _update_roulade_accum 读取它.
    robot_ground_cfg = ContactSensorCfg(
        name="robot_ground_contact",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── 基础配置 ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (
        feet_ground_cfg,
        self_collision_cfg,
        head_ground_cfg,
        robot_ground_cfg,
    )
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

    # ── 奖励: roulade 任务集 ─────────────────────────────────────────────
    # 进度增量 — 翻滚期间唯一的密集任务信号. 1.5 s 翻滚期间平均 ~0.7/step;
    # 从站立生成每个完整翻滚的总支付 ≈ weight × (所用 episode 步数) × 均值
    # ≈ weight × 50.
    cfg.rewards["roulade_progress"] = RewardTermCfg(
        func=microduck_mdp.roulade_progress,
        weight=8.0,
        # max_paid_rate: run-4 提升 3 → 5 rad/s. 实测物理 (run-3 checkpoint
        # eval): 顶部翻转以 3.5-5.5 rad/s 运行 — 此机器人 10 cm 高, 自然翻滚
        # 时间尺度快, 3 rad/s 上限放弃了大部分物理必要的旋转. 风格压力在
        # |a_z| / action_rate / 支撑门控中, 而非对抗重力时钟.
        params={"target_angle": 2 * math.pi, "max_paid_rate": 5.0},
    )

    # 鞭打速度税 — run-4 阈值 4 → 7 rad/s (高于实测 p90 翻转速度 ~5.5):
    # 对真正的鞭打征税, 非自然翻滚.
    cfg.rewards["roulade_overspeed"] = RewardTermCfg(
        func=microduck_mdp.roulade_overspeed_penalty,
        weight=-0.1,
        params={"omega_max": 7.0},
    )

    # 头部作枢轴整形: 接触 × 翻滚中途窗口 × 前向速率因子
    # (速率因子杀死 "面朝下头贴地休息" 的刷取).
    cfg.rewards["roulade_head_pivot"] = RewardTermCfg(
        func=microduck_mdp.roulade_head_pivot,
        weight=0.5,
        params={
            "sensor_name": head_ground_cfg.name,
            "angle_lo": math.radians(30.0),
            "angle_hi": math.radians(240.0),
            "rate_norm": 2.0,
        },
    )

    # 完成门控站立年金 — 主吸引子. 宽松 std (standup 组合教训: 部分落地
    # 必须可见地得分, ~0.2+).
    cfg.rewards["roulade_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.roulade_landing_composite,
        weight=4.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
            "target_overrides": None,
        },
    )

    # 完成门控 bootstrap 层 (远离目标的梯度, 此处组合乘积 ≈0):
    # 线性直立 + 宽松高度高斯.
    cfg.rewards["roulade_upright_after_roll"] = RewardTermCfg(
        func=microduck_mdp.roulade_upright_after_roll,
        weight=1.5,
        params={"gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )
    cfg.rewards["roulade_height_after_roll"] = RewardTermCfg(
        func=microduck_mdp.roulade_height_after_roll,
        weight=1.0,
        params={
            "target_height": STAND_Z,
            "std": 0.04,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
        },
    )

    # 尖锐落地层 (run-4): 紧 std 直立 × 高度乘积叠加在宽松组合之上.
    # Run-3 eval 显示每个完成 episode 停在相同 z≈0.105 / 27° 倾斜姿态 —
    # 宽松 std 在那里得分 ~0.5, 无完成梯度. 尖锐层: 盆地处 ~0.1, 直立处
    # ~1.0 — 最后一英里 10× 差分.
    cfg.rewards["roulade_landing_sharp"] = RewardTermCfg(
        func=microduck_mdp.roulade_landing_sharp,
        weight=2.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.015,
            "upright_std": 0.3,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
        },
    )

    # 完成门控站立税 (run-3, standup 核心教训): 旋转完成后, 每步低于 STAND_Z
    # 付费 — "翻滚后蜷缩成一堆" 从免费翻转为净负, 与打破 standup 静态坐姿
    # 盆地的修复相同 (其高度 L1 在 ÷4 缩放权重 7.5). 翻滚期间门控关闭,
    # 翻滚本身不被征税; 翻滚中/后期生成出生即激活此税, 这正是目的.
    cfg.rewards["roulade_stand_tax"] = RewardTermCfg(
        func=microduck_mdp.roulade_stand_tax,
        weight=5.0,
        params={
            "target_height": STAND_Z,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
        },
    )

    # 退出起立 bootstrap: 向上 CoM 速度, 门控到翻滚后期区域 (仰面 → 起立
    # 是仰面恢复问题; 终态奖励在零运动处零梯度 — standup 教训 #2).
    cfg.rewards["roulade_rise_velocity"] = RewardTermCfg(
        func=microduck_mdp.roulade_rise_velocity,
        weight=0.75,
        params={
            "max_height": STAND_Z + 0.01,
            "gate_lo": RISE_GATE_LO,
            "gate_hi": RISE_GATE_HI,
        },
    )

    # 直线性 — run-5: run-4 policy 从肩膀翻滚 (比直过头顶能量更低的路径 —
    # 避免完全倒立构型, 与人类初学者默认欺骗相同). 结构修复是累加器的
    # 平坦度门控 + 头顶锁 (侧翻不再计为旋转); 这些惩罚提供朝平面的密集
    # 每步梯度, 权重从 run-2 值提升 5× (后者对 progress@8 是噪声).
    cfg.rewards["roulade_sagittal"] = RewardTermCfg(
        func=microduck_mdp.roulade_sagittal_penalty,
        weight=-0.1,
    )
    cfg.rewards["roulade_lateral_vel"] = RewardTermCfg(
        func=microduck_mdp.roulade_lateral_velocity_penalty,
        weight=-0.5,
    )
    cfg.rewards["roulade_flatness"] = RewardTermCfg(
        func=microduck_mdp.roulade_flatness_penalty,
        weight=-0.5,
    )

    # ── Sim2real 正则项 ─────────────────────────────────────────────────
    # 运动阻断项在发现期间保持近零 (翻滚是大角速度 + 冲击事件); 稳定/打磨
    # 压力来自后引入的下方门控项 (arrival_damping, |a_z|, 力矩速率) —
    # standup 时序教训.
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(func=microduck_mdp.joint_torque_rate_l2, weight=0.0)

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002  # 必须保持 ≈0: 翻滚就是 ω
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    # 到达阻尼器 — 躯干 ω_xy² 门控在站立高度 AND 低倾斜, 所以翻滚本身
    # 永不被征税; 从 0 引入并由课程渐升.
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

    # |a_z| 冲击整形 — 从 step 0 起活跃 (run-2 变更: run 1 在零冲击代价下发现
    # 暴力解并锁定; 此环境中发现容易, 所以从一开始就整形风格是优先).
    # 课程进一步渐升.
    # 注意: trunk_vertical_accel_penalty 自否定 (返回 -|a_z|) → 正权重
    # (惩罚符号约定; 负权重在此会奖励暴力 — run-2 smoke test 中捕获, 总和为正).
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # 自碰撞 — 轻: 蜷缩翻滚需要身体对身体接触 (膝对躯干); standup 的
    # -1.0 会对抗蜷缩.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # 常开直立会反对翻转 (旧尝试的核心失败); 落地直立由上方完成门控项处理.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── 观测 (与行走 / standup policies 布局相同) ─────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(cfg.observations["actor"].terms[gravity_term_name])
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

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

    # 指令 obs 槽位: head (4) 和 body (6) 均零填充 — 头部是任务一部分
    # (它是枢轴), 所以此处无 head_pose 指令, 但保持与 velocity/standup 的
    # 61D obs 布局一致, 使 runtime 栈无需更改即可工作 (发送零).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    # ── 指令: 零周围微小噪声 (保持 obs-shape 一致) ──────────
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
    # 摔倒就是任务 — 仅保留 NaN 守卫 + 超时.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
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
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # 站立起点 + 翻滚中途反向课程生成; 也重置旋转累加器 (必须在
    # reset_robot_joints 之后运行 — 字典插入顺序 — 因为翻滚中途 tuck 从
    # 其写入的 HOME 姿态 lerp).
    cfg.events["set_roulade_state"] = EventTermCfg(
        func=microduck_mdp.reset_roulade_state,
        mode="reset",
        params={
            "standing_prob": 0.5,
            "midroll_prob": 0.5,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
            "standing_tilt_max": math.radians(5.0),
            "forward_vel_range": ROULADE_FORWARD_VEL_RANGE,
            "midroll_pitch_min": MIDROLL_PITCH_MIN,
            "midroll_pitch_max": MIDROLL_PITCH_MAX,
            "midroll_z_min": 0.05,
            "midroll_z_max": 0.10,
            "midroll_omega_range": MIDROLL_OMEGA_RANGE,
            "tuck_overrides": TUCK_OVERRIDES,
            "tuck_factor_range": (0.3, 1.0),
            "joint_noise_std": 0.08,
        },
    )

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

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

    # ── 地形 ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── 课程 ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # 反向课程混合: 早期重翻滚中途 (完成子任务从 day 0 可学 — 与仰面恢复
    # 重叠), 随完整翻滚被发现转向站立起点. 翻滚中途永不到零: 它保持后半程
    # 练习且反正也是现实 DR.
    # Run-3: 阶段推迟 1500/3000 → 3000/6000 — run 2 在站立生成翻滚掌握前
    # 转离翻滚中途 (iter 1876 时进度 episode-sum 是完整翻滚的 ~20%;
    # 课程节奏失败, 与 2026-07-28 standup 回归同类).
    cfg.curriculum["roulade_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_roulade_state",
            "param_stages": [
                {"step": 0, "params": {"standing_prob": 0.50, "midroll_prob": 0.50}},
                {
                    "step": 3000 * 24,
                    "params": {"standing_prob": 0.65, "midroll_prob": 0.35},
                },
                {
                    "step": 6000 * 24,
                    "params": {"standing_prob": 0.80, "midroll_prob": 0.20},
                },
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

    # action_rate 渐升 — run-4: 上限软化 -0.6 → -0.4 且 -0.4 阶段推迟
    # 2000 → 3000. Run-3 落地指标在 ~iter 2700 达峰后下降, 跟踪 -0.4/-0.6
    # 阶段 — 收紧正在挤压起立. (Run-2 注释仍有效: step 0 起 -0.1 最低,
    # run 1 在近零平滑下滋生暴力.)
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 1500 * 24, "weight": -0.2},
                {"step": 3000 * 24, "weight": -0.4},
            ],
        },
    )

    # 平滑打磨 — 仅在翻滚技能存在后引入 (standup 时序教训: 发现期间活跃的
    # 任何尝试税阻止动作被发现; 修复是时序, 非量级).
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "arrival_damping",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 2500 * 24, "weight": -0.025},
                {"step": 3500 * 24, "weight": -0.05},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 2500 * 24, "weight": -5e-4},
                {"step": 3500 * 24, "weight": -1e-3},
            ],
        },
    )
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            # 正权重: 函数自否定 (返回 -|a_z|).
            "reward_name": "gentle_landing",
            "weight_stages": [
                {"step": 0, "weight": 0.002},
                {"step": 2500 * 24, "weight": 0.005},
            ],
        },
    )

    return cfg


# ── RL runner 配置 ──────────────────────────────────────────────────────────

MicroduckRouladeRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_roulade",
    run_name="microduck_roulade",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
