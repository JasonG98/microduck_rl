"""Microduck 滚轮蹲滑任务.

通过 runtime 的 --ground-pick 槽位由按钮 A 触发的一次性动作:
机器人蹲下并依靠惯性滑行 (保持 ~1 s), 然后站起并将控制权交还给 roller policy.

混合体:
  - 物理 / 滚轮机器人  ← microduck_velocity_rollers_env_cfg.py
  - 一次性相位机制 ← microduck_ground_pick_env_cfg.py
    (指令 GroundPickPhaseCommand : [cos(2πφ), sin(2πφ), 0], 周期 4 s)

通过 crouch_glide_height_by_phase 实现 "梯形" 高度目标 (高→低→保持 1 s→高).
统一 61D 观测 → 可在 runtime 互换.
"""

import math
from copy import deepcopy

ENABLE_SYMMETRY = False

# DR — 沿用 roller 环境
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_WHEEL_FRICTION_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

COM_RANDOMIZATION_RANGE = 0.003
HEAD_COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.2, 0.2)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE = (-0.015, 0.015)

ENTRY_VELOCITY_X = (0.2, 0.5)  # m/s: 机器人以滚动方式进入

# 循环时序 (相位), 5 s 周期上分 4 段:
#   下降   [0, DESCENT_END]        = 0.10*5 = 0.5 s  (下蹲)
#   低/蹲  [DESCENT_END, HOLD_END] = 0.40*5 = 2.0 s  (蹲姿滑行)
#   上升   [HOLD_END, RISE_END]    = 0.10*5 = 0.5 s  (起立)
#   高/站  [RISE_END, 1.0]         = 0.40*5 = 2.0 s  (站立休息)
# 注意: 周期必须与部署时的 --ground-pick-period 匹配 (5.0).
CROUCH_PERIOD = 5.0
DESCENT_END = 0.10
HOLD_END = 0.50
RISE_END = 0.60

# 蹲姿目标 (rad, 按关节名) — 在 scripts/crouch_pose_editor.py 中编辑.
# 奖励按相位在 站立(HOME) <-> 此姿态 间插值. 按名解析 -> 对交错滚轮鲁棒.
# 站立姿态 (动作的起点/终点). 默认 = sim 的 HOME (已验证与机器人读取一致).
# 如需其他站姿, 用 read_pose.py 读取机器人站姿替换这些值.
# ⚠️ 部署时, 动作结束后 runtime 将控制权交还给 roller policy, 它从 HOME 重启 —
# 保持 STAND_POSE 接近 HOME 以确保干净切换.
STAND_POSE = {
    # 在真机上读取 (read_pose.py) — 动作目标站姿.
    "left_hip_yaw": -0.0476,
    "left_hip_roll": -0.0629,
    "left_hip_pitch": -0.2869,
    "left_knee": 0.9618,
    "left_ankle": 1.1674,
    "neck_pitch": 0.6029,
    "head_pitch": 0.543,
    "head_yaw": -0.069,
    "head_roll": -0.0414,
    "right_hip_yaw": -0.0337,
    "right_hip_roll": -0.0061,
    "right_hip_pitch": 0.1534,
    "right_knee": -0.9725,
    "right_ankle": -1.0646,
}

CROUCH_POSE = {
    # 在真机上读取 (Dynamixel XL330, read_pose.py) — 可保持的姿态.
    "left_hip_yaw": -0.0184,
    "left_hip_roll": 0.0307,
    "left_hip_pitch": 1.4082,
    "left_knee": 1.5248,
    "left_ankle": -0.0675,
    "neck_pitch": 1.0937,
    "head_pitch": 1.2149,
    "head_yaw": -0.0184,
    "head_roll": -0.0368,
    "right_hip_yaw": 0.0184,
    "right_hip_roll": -0.0169,
    "right_hip_pitch": -1.4757,
    "right_knee": -1.5907,
    "right_ankle": 0.0568,
}
CROUCH_POSE_STD = 0.4  # 每关节高斯容差 (rad)
CROUCH_LEAN_PITCH = 0.08  # 蹲下时轻微前倾 (rad ≈ 4.6°)

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
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG, PpoWithSymmetryCfg


def make_microduck_roller_crouch_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """滚轮上的蹲滑环境, 由 ground-pick 槽位的相位驱动."""
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(ankle_l_v1|ankle_r_v1)$",
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

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === 奖励 ===
    keep = {"upright", "body_ang_vel", "angular_momentum", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -1.0

    # 主奖励: 按相位插值的姿态 (站立 <-> 蹲).
    # 指令: 告诉机器人每一时刻的精确关节构型. "起立" (phase->1, 目标 = HOME)
    # 与 "蹲下" (保持段, 目标 = CROUCH_POSE) 以完全相同的方式奖励 — 对称.
    _pose_params = {
        "command_name": "twist",
        "crouch_pose": CROUCH_POSE,
        "stand_pose": STAND_POSE,
        "descent_end": DESCENT_END,
        "hold_end": HOLD_END,
        "rise_end": RISE_END,
    }
    cfg.rewards["crouch_glide_pose"] = RewardTermCfg(
        func=microduck_mdp.crouch_glide_pose_by_phase,
        weight=6.0,
        params={**_pose_params, "std": CROUCH_POSE_STD},
    )
    # Bootstrap L1: 即使高斯在远离姿态时饱和, 也提供恒定梯度指向目标.
    cfg.rewards["crouch_glide_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.crouch_glide_pose_l1,
        weight=2.0,
        params=_pose_params,
    )
    # 保持动量 (不刹车) — 与指令无关.
    cfg.rewards["forward_speed"] = RewardTermCfg(
        func=microduck_mdp.forward_speed_reward,
        weight=1.0,
        params={"vel_ref": 0.2},
    )
    # 蹲下时轻微前倾 -> 对抗真机快速下降时观察到的后仰. 由 blend 门控 (仅蹲段).
    cfg.rewards["crouch_forward_lean"] = RewardTermCfg(
        func=microduck_mdp.crouch_forward_lean,
        weight=1.0,
        params={
            "command_name": "twist",
            "target_pitch": CROUCH_LEAN_PITCH,
            "std": 0.1,
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )
    # 滑行稳定性
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(func=microduck_mdp.neck_action_rate_l2, weight=-0.5)
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=microduck_mdp.joint_torques_l2, weight=-1e-3)

    # === 终止条件 ===
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # === 事件 ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    del cfg.events["foot_friction"]

    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)
    # 入口速度: 机器人以向前滚动方式启动 (在蹲下期间需保持的动量). 通过
    # reset_root_state_uniform 注入 (CLEAN 默认状态 + range), 而非通过 reset 模式下
    # 的 push_by_setting_velocity (后者累加到当前 root 速度, 可能发散并使 base
    # free-joint 爆炸 -> NaN). 见 ENTRY_VELOCITY_X 的注释.
    cfg.events["reset_base"].params["velocity_range"] = {"x": ENTRY_VELOCITY_X}

    if ENABLE_WHEEL_FRICTION_RANDOMIZATION:
        cfg.events["randomize_wheel_friction"] = EventTermCfg(
            func=dr.dof_frictionloss,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
                "operation": "abs",
                "ranges": (0.000, 0.000),
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
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # === 观测 (统一 61D 布局) ===
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

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

    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        scale=1.0,
        params={"asset_cfg": wheel_cfg},
    )

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    # === 指令: 相位 (同 ground_pick) ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    # period=CROUCH_PERIOD (下降更慢); randomize_phase=False -> 每个 episode 从站姿
    # 启动 (相位 0), 与部署一致 (按钮从相位 0 启动循环). 避免学到 "保持蹲姿"
    # 这种从已低位启动的行为.
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": CROUCH_PERIOD,
            "randomize_phase": False,
        }
    )

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === 课程 ===
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.5},
                {"step": 250 * 24, "weight": -0.8},
                {"step": 500 * 24, "weight": -1.0},
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


MicroduckRollerCrouchRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
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
    experiment_name="roller_crouch",
    run_name="roller_crouch",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8_000,
)
