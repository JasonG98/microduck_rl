"""Microduck 滚轮下坡 — 被动平衡下滑.

机器人生成在平地上 (向前一个冲量), 沿下坡斜面滚动并保持站立自由滑行. 无任何
操控: twist 指令被中和 (rel_standing_envs=1.0). 自定义 平地+斜面 地形
(FlatRampTerrainCfg), 坡度课程 (terrain_levels_slope). 统一 61D 观测 → runtime
可互换 (--new-cmd-obs) — 直接沿用 make_microduck_velocity_rollers_env_cfg
(DR/obs/reset 未在此处修改).
"""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.slope_terrain import RAMP_DEG_MAX, FlatRampTerrainCfg
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# 平地+斜面+出口的地形几何.
FLAT_LENGTH = 2.0
RAMP_LENGTH_RANGE = (
    3.0,
    8.0,
)  # 斜面的水平长度, 由每瓦片随机抽取
RUNOUT_LENGTH = 4.0  # 底部的平地出口
SPAWN_ON_RAMP = 0.3  # 在斜面上生成该米数 (重力 -> 滚动, 无打滑)
ENTRY_VELOCITY_X = (0.25, 0.45)  # 向前/向下的初始小动量 (m/s)
TILE_SIZE = (15.0, 4.0)  # >= 平地 + 最大斜面 + 出口 (= 14) + 余量
SPAWN_YAW = (0.0, 0.0)  # 面朝下坡方向 (+x), 固定

# PLAY 时的坡度: None = 随机 (与训练一致). 设为 0..1 的值可强制特定坡度
# (1.0 = 最陡 ~20°, 0.5 = 中等). 可通过环境变量 SLOPE_PLAY_DIFFICULTY
# 覆盖, 无需改代码 (例: SLOPE_PLAY_DIFFICULTY=1.0 uv run play ...; "none"/"random" = 随机).
PLAY_DIFFICULTY = None


def _resolve_play_difficulty():
    """play 难度: 优先读 SLOPE_PLAY_DIFFICULTY 环境变量, 否则用常量."""
    raw = os.environ.get("SLOPE_PLAY_DIFFICULTY")
    if raw is None:
        return PLAY_DIFFICULTY
    raw = raw.strip().lower()
    if raw in ("", "none", "random"):
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        print(f"[roller_slope] SLOPE_PLAY_DIFFICULTY='{raw}' 无效 -> 默认 {PLAY_DIFFICULTY}")
        return PLAY_DIFFICULTY


# "掉入虚空" 终止: 低于最低出口平地 (最陡且最长的斜面), 带余量 => 在正常下坡时
# 永不触发, 仅当机器人脱离实体地面时触发.
_MAX_DROP = RAMP_LENGTH_RANGE[1] * math.tan(math.radians(RAMP_DEG_MAX))
VOID_FLOOR = -_MAX_DROP - 0.5


def make_microduck_roller_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构建滚轮下坡环境 cfg (平地 + 斜面 + 出口, twist 中和)."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    # === 地形: 平地 + 斜面 (随机长度) + 出口平地 ===
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=TILE_SIZE,
            curriculum=True,
            num_rows=10,  # 10 个坡度级别
            num_cols=1,
            difficulty_range=(0.0, 1.0),
            sub_terrains={
                "flat_ramp": FlatRampTerrainCfg(
                    flat_length=FLAT_LENGTH,
                    ramp_length_range=RAMP_LENGTH_RANGE,
                    runout_length=RUNOUT_LENGTH,
                    spawn_on_ramp=SPAWN_ON_RAMP,
                )
            },
        ),
        max_init_terrain_level=0,  # 课程: 从最缓的斜面开始
    )

    # play 时: 展示不同坡度. 难度 None -> 随机坡度 (在所有行中抽取级别);
    # 0..1 的值则强制特定坡度 (1.0 = 最陡). 通过 SLOPE_PLAY_DIFFICULTY 控制.
    if play:
        play_difficulty = _resolve_play_difficulty()
        if play_difficulty is not None:
            cfg.scene.terrain.terrain_generator.difficulty_range = (
                play_difficulty,
                play_difficulty,
            )
        else:
            cfg.scene.terrain.max_init_terrain_level = None

    # === 指令中和 (纯平衡) ===
    command = cfg.commands["twist"]
    command.rel_standing_envs = 1.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    if getattr(command.ranges, "ang_vel_z", None) is not None:
        command.ranges.ang_vel_z = (0.0, 0.0)

    # === RESET: 始终面朝下坡 (+x), 无 base 推力 ===
    # 继承的 yaw 是随机的 (-180°/+180°) -> 固定为 0 (面朝坡底). 不注入任何 base
    # 速度: 机器人生成在斜面上 (见 spawn_on_ramp), 重力使轮子滚动 (动量在轮子上,
    # 无打滑). 旧的 base 推力 (base 快, 轮子不动) 会打滑 -> 接触尖峰 -> NaN 发散,
    # 且机器人 "走几步去停下" 而非滚动.
    cfg.events["reset_base"].params["pose_range"]["yaw"] = SPAWN_YAW
    # 此处不施加 base 推力 (base 动 + 轮子静 = 第一步打滑冲击). 初始动量以一致的
    # 滚动方式 (base + 轮子, ω·r = v) 由下方 reset_rolling_entry 注入 -> 干净起步.
    cfg.events["reset_base"].params["velocity_range"] = {}

    # === 奖励: 自由平衡 (机器人自行放置重心) ===
    # 不设固定姿态奖励: 不再向其规定平地上的站姿 (那会阻止其屈曲/倾斜). 它可自由
    # 移动 CoM (髋/膝, 倾斜) 来保持坡度. 只奖励: 站立, 存活, 滑行, 直行 — 并通过
    # 终止条件避免摔倒.
    keep = {"action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"] = RewardTermCfg(
        func=microduck_mdp.body_upright_gaussian,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "std": 0.2,
        },
    )
    cfg.rewards["alive"] = RewardTermCfg(func=microduck_mdp.is_alive, weight=1.0)
    # 自由滑行 (滚动), 而非加速/奔跑: 奖励轮子向下滚动, 上限 cap_speed.
    # 上限 => 无动力加速; 基于轮子 => "奔跑" (推 base 不滚) 无收益. 无滑行奖励时
    # 最优解是静止; 加上后只要能保持平衡就会任由滚动.
    cfg.rewards["wheel_glide"] = RewardTermCfg(
        func=microduck_mdp.wheel_glide_reward,
        weight=2.0,
        params={"cap_speed": 0.35},
    )
    # 直行: 保持生成时的 yaw (= 0 = 面朝下坡). 修正型 (机器人可自行调整) 是直行
    # 的正确方式. 注意: PPO 对称 (SYMMETRY_CFG) 为旧的 51D obs 编写 -> 此处不可用.
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=1.5,
        params={"std": 0.4},
    )
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2,
        weight=-0.5,
    )
    # 保持头部正立: 惩罚颈/头关节偏离 home 位置. 已移除腿部固定姿态 (为了自由
    # 平衡), 但头部无处可托 -> 会随意乱摆. 此项只约束头/颈, 不影响腿部.
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_joint_pos_l2,
        weight=-0.75,
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2,
        weight=-1e-3,
    )
    cfg.rewards["action_rate_l2"].weight = -1.0

    # === 终止: 摔倒 + 掉入虚空 ===
    # 出口平地给斜面底部提供了实体地面, 因此无需 "到达边界" 终止
    # (terrain_edge_reached 在长斜面上截断过早). 保留: 摔倒 (bad_orientation),
    # NaN, 和 "掉入虚空" (trunk 低于最低出口平地) 以防机器人脱离实体地面.
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=base_mdp.bad_orientation,
        params={
            "limit_angle": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    if "out_of_terrain_bounds" in cfg.terminations:
        del cfg.terminations["out_of_terrain_bounds"]
    cfg.terminations["fell_into_void"] = TerminationTermCfg(
        func=microduck_mdp.root_height_below,
        params={
            "min_height": VOID_FLOOR,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # === 观测: 清理 NaN/Inf (对罕见的接触发散的鲁棒性) ===
    # 罕见接触 (~1/25M 步-env) 会使 free-joint 发散为 NaN. 由于子步长偏移,
    # nan_state 终止只能在下一步 (reset) 捕获, 但 NaN 已进入当前步 obs ->
    # rsl_rl 的 check_nan 会终止训练. nan_policy="sanitize" 将返回 obs 中的
    # NaN/Inf 替换为 0 (不崩溃); nan_state 随后 reset 故障 env.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    # === 事件 ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    # 滚动起步 (动量在轮子上, 无打滑). 位于 reset_base 之后.
    cfg.events["reset_rolling_entry"] = EventTermCfg(
        func=microduck_mdp.reset_rolling_entry,
        mode="reset",
        params={"speed_range": ENTRY_VELOCITY_X},
    )

    # === 课程: 缓坡 -> 陡坡 ===
    # 从最缓的坡 (2°) 开始, 当机器人下滑足够远时晋级到更陡的坡 (直至 20°,
    # terrain_levels_slope, 基于行驶距离). 现在可行因为 descent_speed 让它前进
    # (此前它保持静止 -> 从不晋级). 它逐步学习平衡, 而非一开始就面对 20°
    # (那会让它栽跟头).
    for name in list(cfg.curriculum.keys()):
        del cfg.curriculum[name]
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=microduck_mdp.terrain_levels_slope)

    return cfg


MicroduckRollerSlopeRlCfg = RslRlOnPolicyRunnerCfg(
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
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
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
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_slope",
    run_name="roller_slope",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8_000,
)
