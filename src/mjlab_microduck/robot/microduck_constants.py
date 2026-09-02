"""Microduck 机器人实体配置, MJCF 路径, 以及 BAM 执行器配置."""

from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_microduck.actuator import (
    BacklashEncoderBamActuatorCfg,
    FrictionDRBamActuatorCfg,
)

_ROBOT_DIR: Path = Path(__file__).parent / "microduck"

MICRODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
# 全碰撞模型, 被 standup / ground-pick / walk-rollers 任务共用.
MICRODUCK_ALLCOLLISIONS_XML: Path = _ROBOT_DIR / "robot_allcollisions.xml"
# 70mm / 15g 球道具, 用于 BallKick 任务.
MICRODUCK_BALL_XML: Path = _ROBOT_DIR / "ball.xml"
# 轮滑模型: 14 个驱动关节 + 被动轮铰链 (passive_*wheel).
MICRODUCK_ALLCOLLISIONS_ROLLERS_XML: Path = _ROBOT_DIR / "robot_allcollisions_rollers.xml"
# Backlash 模型: 每个 servo 关节串联一个非驱动的 passive_<joint>_backlash
# 铰链 (±1° 间隙, 共 2°). 通过 config_mjcf_{allcollisions,walk}_backlash.json
# 导出 (add_backlash.py 后处理器).
MICRODUCK_ALLCOLLISIONS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_backlash.xml"
MICRODUCK_WALK_BACKLASH_XML: Path = _ROBOT_DIR / "robot_walk_backlash.xml"
MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML: Path = _ROBOT_DIR / "robot_allcollisions_rollers_backlash.xml"

assert MICRODUCK_WALK_XML.exists(), f"XML not found: {MICRODUCK_WALK_XML}"
assert MICRODUCK_ALLCOLLISIONS_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_XML}"
assert MICRODUCK_BALL_XML.exists(), f"XML not found: {MICRODUCK_BALL_XML}"
assert MICRODUCK_ALLCOLLISIONS_ROLLERS_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_ROLLERS_XML}"
assert MICRODUCK_ALLCOLLISIONS_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_ALLCOLLISIONS_BACKLASH_XML}"
assert MICRODUCK_WALK_BACKLASH_XML.exists(), f"XML not found: {MICRODUCK_WALK_BACKLASH_XML}"
assert MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML.exists(), (
    f"XML not found: {MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML}"
)


def get_walk_spec() -> mujoco.MjSpec:
    """将 walk 模型 MJCF 加载为 MjSpec."""
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML))


def get_standup_spec() -> mujoco.MjSpec:
    """将全碰撞 standup 模型 MJCF 加载为 MjSpec."""
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))


def get_ground_pick_spec() -> mujoco.MjSpec:
    """将 ground-pick (全碰撞) 模型 MJCF 加载为 MjSpec."""
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_XML))


def get_walk_rollers_spec() -> mujoco.MjSpec:
    """将轮滑 walk 模型 MJCF 加载为 MjSpec."""
    # 注意: 曾经加载的是 robot_allcollisions.xml (无轮子) — 轮子环境
    # 静默地跑在了无轮的 standup 模型上.
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_ROLLERS_XML))


def get_ball_spec() -> mujoco.MjSpec:
    """将球道具 MJCF 加载为 MjSpec."""
    return mujoco.MjSpec.from_file(str(MICRODUCK_BALL_XML))


def get_backlash_spec() -> mujoco.MjSpec:
    """将全碰撞 backlash 模型 MJCF 加载为 MjSpec."""
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_BACKLASH_XML))


def get_walk_backlash_spec() -> mujoco.MjSpec:
    """将 walk backlash 模型 MJCF 加载为 MjSpec."""
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_BACKLASH_XML))


def get_rollers_backlash_spec() -> mujoco.MjSpec:
    """将轮滑 backlash 模型 MJCF 加载为 MjSpec."""
    return mujoco.MjSpec.from_file(str(MICRODUCK_ALLCOLLISIONS_ROLLERS_BACKLASH_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # 下半身 — STAND2 姿态: 躯干相对双脚前移约 5mm, 使 CoM 落在踝关节
        # 轴线上 (旧 HOME 时 CoM 在踝后约 5mm, 让机器人向后偏置, 使得
        # standup 策略把头前倾作为配重). 腿 pitch 链前倾:
        # hip_pitch 30°→26.24°, ankle 30°→25.95°, knee 0°→0.28°. 与
        # scene.xml / scene_walk.xml 中的 STAND 关键帧一致.
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.0873,
        r".*right_hip_roll.*": 0.0873,
        r".*left_hip_pitch.*": -0.4579,
        r".*right_hip_pitch.*": 0.4579,
        r".*left_knee.*": -0.0049,
        r".*right_knee.*": 0.0049,
        r".*left_ankle.*": 0.4530,
        r".*right_ankle.*": -0.4530,
        # 头部
        r".*neck_pitch.*": 0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# -- 旧执行器 (XML position, MuJoCo 内置 PD + 摩擦) --
# actuators = DelayedActuatorCfg(
# delay_min_lag=0,
# delay_max_lag=3,
# base_cfg=XmlPositionActuatorCfg(joint_names_expr=(r".*",)),
# )

# -- BAM M6 执行器 (完整电压控制 + 负载相关摩擦) --
# 排除 passive_* 关节 (新模型中的颌连杆没有 XML 执行器).
# 电压域随机化 (镜像 mjlab_microban):
#   - vin_range: 启动时逐环境采样的电池电压 (替代固定 vin)
#   - vin_drop_gain_range: 负载相关的电压跌落 V_drop = gain * sum(|tau|)
#   - vin_min: 跌落后有效电压的硬下限
# kp_fw 保持在 200 (microduck 保留的固件刚度; microban 用 125).
_BAM_ACTUATOR_KWARGS = {
    "motor_name": "xl330",
    "model": "m6",
    "target_names_expr": (r"^(?!passive_).*",),
    "kp_fw": 200.0,  # microduck 保留的固件刚度 (microban 用 125)
    # vin_range=(6.9, 7.9),
    "vin_range": (6.5, 8.2),
    "vin_drop_gain_range": (0.0, 0.2),
    "vin_min": 6.0,
    # max_current=1.75,
    "delay_min_lag": 3,
    "delay_max_lag": 6,
}
actuators = FrictionDRBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# 同样的 BAM 执行器, 但固件位置环通过
# passive_<joint>_backlash 铰链读取编码器 (真实编码器位于
# 齿轮间隙的输出侧). 仅用于 backlash 模型; 目标 regex 已经
# 把 passive_* backlash 关节排除在驱动之外.
backlash_actuators = BacklashEncoderBamActuatorCfg(**_BAM_ACTUATOR_KWARGS)

# -- BAM M4 执行器
# actuators = DelayedActuatorCfg(
# delay_min_lag=0,
# delay_max_lag=3,
# base_cfg=make_bam_m4_actuator_cfg(),
# )

# backlash 模型的 HOME 姿态. HOME_FRAME 中未锚定的模式
# (如 r".*left_hip_roll.*") 也会匹配 passive_left_hip_roll_backlash
# 并试图将其初始化为 -0.0873 rad — 超出其 ±1° 范围. 模式
# 匹配在声明顺序中先到先得, 因此放在最前面的锚定 backlash 规则
# 把每个 backlash 关节钉在 0, servo 关节则落入正常 HOME 值.
BACKLASH_HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={r".*_backlash$": 0.0, **HOME_FRAME.joint_pos},
    joint_vel={".*": 0.0},
)

MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)


MICRODUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)


MICRODUCK_GROUND_PICK_ROBOT_CFG = EntityCfg(
    spec_fn=get_ground_pick_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)


# Backlash 机器人: 基础模型 + 每个 servo 一个 ±1° 串联 backlash 铰链.
# 编码器穿过 backlash 读取 (BacklashEncoderBamActuator 反馈 +
# joint_pos/vel_rel_backlash 观测 — 见 tasks/backlash.py).
# allcollisions 变体 → VelStand/StandUp backlash 任务 (镜像
# MICRODUCK_STANDUP_ROBOT_CFG); walk 变体 → Velocity backlash
# 任务 (镜像 MICRODUCK_WALK_ROBOT_CFG, 让 backlash 与基础对比
# 不被碰撞模型混淆).
MICRODUCK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_WALK_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)


# 轮滑 backlash 机器人: 轮子保持自由 (add_backlash.py 不动 passive_*wheel).
# collisions=() 镜像 MICRODUCK_WALK_ROLLERS_ROBOT_CFG —
# 轮子碰撞 geom 没有显式名称; XML 默认值生效.
MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG = EntityCfg(
    spec_fn=get_rollers_backlash_spec,
    init_state=BACKLASH_HOME_FRAME,
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(backlash_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)


# 自由漂浮, 非铰接的球道具, 用于 BallKick 任务. 位置由
# reset_ball_in_front_of_foot 事件在每个 episode 设置; 此处的初始位置
# 仅影响首次 reset 之前的原始状态.
MICRODUCK_BALL_CFG = EntityCfg(
    spec_fn=get_ball_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, 0.035)),
)


# 轮滑机器人: 4 个被动轮关节 (passive_*wheel) 没有 XML
# 执行器; BAM 配置的目标 regex 已经排除它们, 所以动作空间
# 保持 14 维. 使用与所有其他变体相同的标准 BAM 执行器
# (曾经是普通 XmlActuatorCfg PD — 与家族其他成员存在执行器物理
# 不匹配, 且关节摩擦域随机化不可能).
MICRODUCK_WALK_ROLLERS_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_rollers_spec,
    init_state=HOME_FRAME,
    collisions=(),  # 轮子碰撞 geom 没有显式名称; XML 默认值生效
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": MICRODUCK_WALK_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    viewer.launch(scene.compile())
