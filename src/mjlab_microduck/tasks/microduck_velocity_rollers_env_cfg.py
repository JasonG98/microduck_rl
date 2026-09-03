"""Microduck velocity 环境 — 轮滑变体.

已迁移到 mjlab 1.3.0 + 标准 BAM (2026-07), 与 velocity 环境的 sim2real 机制
对齐, 并针对新 roller 模型更新:

- `get_walk_rollers_spec` 现在加载 `robot_groundcontact_rollers.xml`
    (之前静默加载无轮 standup 模型): 14 个驱动关节 + 4 个被动轮
    (passive_{L,R}{F,R}wheel), 每个刀刃两个, 关节顺序中交叉排列 (在每个踝
    之后) — 一切都按 NAME 而非索引解析关节.
  - 腿部运行标准 BAM 执行器, 与其他变体一致 (原为纯 XML PD — 执行器物理
    不匹配, 且无关节摩擦 DR).
  - Obs 迁移到统一 61D 布局 (twist + 零填充 head/body 指令槽位), 使 roller
    policy 通过 runtime 的 --new-cmd-obs 路径加载. 对称性关闭
    (SYMMETRY_CFG 是为旧 51D 布局硬编码的).
  - DR/noise/delays 与 velocity 环境的 FIXED (非累积, 每环境验证) 版本对齐;
    轮轴承 frictionloss DR 保留 (被动轮上的 dr.dof_frictionloss + 现有课程).

任务设计 (不变 — roller 配方):
  cmd_x 语义: 0 = 滑行, >0 = 推加速, <0 = 刹车.
  cmd[2] = 通过 RelativeHeadingVelocityCommand 的航向误差.
  唯一正任务奖励是 wheel_speed — 机器人必须真正转动轮子; 制动/
  skating_air_time/forward_lean/heading_tracking 塑造滑行风格.
"""

import math
from copy import deepcopy

# 对称性 — 关闭: SYMMETRY_CFG 的 obs 置换为旧 51D 布局硬编码, 在 61D obs 上
# 会失效 (与所有其他 v1.5+ 环境情况相同).
ENABLE_SYMMETRY = False

# ── Domain randomisation 开关 (与 velocity 环境对齐) ────────────────────────
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # 每环境 BAM 摩擦预算 (腿部)
ENABLE_ARMATURE_RANDOMIZATION = True  # 仅腿部 — 不含轮轴承
ENABLE_WHEEL_FRICTION_RANDOMIZATION = True  # 被动轮上的轴承 frictionloss
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # obs 层每环境旋转
ENABLE_ENCODER_BIAS = True

# ── 范围 (与 velocity 环境对齐, 除非 roller 专有) ───────────────────────────
COM_RANDOMIZATION_RANGE = 0.003  # ±3mm 初始, 通过课程渐升
HEAD_COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.2, 0.2)  # roller 专有: 比 walk 的 ±0.3 更温和
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE = (-0.015, 0.015)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
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


def make_microduck_velocity_rollers_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """创建 Microduck 轮滑速度跟踪环境配置."""
    # passive_.*: 999.0 → 被动轮关节被匹配但实际被忽略
    std_standing = {
        r".*hip_yaw.*": 0.05,
        r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.05,
        r".*knee.*": 0.05,
        r".*ankle.*": 0.05,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    std_walking = {
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.6,  # 放宽: 滑行需要宽的横向推力
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    std_running = {
        r".*hip_yaw.*": 0.5,
        r".*hip_roll.*": 0.8,  # 放宽: 滑行需要宽的横向推力
        r".*hip_pitch.*": 0.8,
        r".*knee.*": 0.8,
        r".*ankle.*": 0.5,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    # 2026-07 模型: roller_blade body 被并入踝部 (刀刃网格现在是
    # ankle_{l,r}_v1 上的视觉 geom); 轮胎直接挂在踝部下. 每个踝子树唯一的
    # 碰撞 geom 是它的两个轮胎, 所以这保留了旧的每足语义: 2 槽, 左在前.
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

    # 机器人设置
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    # 动作配置
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # 注意: 曾在此尝试环境侧的 action clip 以限制目标, 但部署流水线
    # (infer_policy.py) 不裁剪 → clip 只存在于仿真, 造成训练/部署不匹配.
    # 过度指令的威慑放在 policy 侧 (下方的 action_over_limit 奖励), 烤进
    # 网络中, 随 ONNX 转移.

    # === 奖励 ===
    keep = {"pose", "upright", "body_ang_vel", "angular_momentum", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_running
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].params["running_threshold"] = 0.5
    cfg.rewards["pose"].weight = 2.0

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -1.0

    cfg.rewards["com_height_target"] = RewardTermCfg(
        func=microduck_mdp.com_height_target,
        weight=2.0,
        params={"target_height_min": 0.0935, "target_height_max": 0.1235},
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    # 仅门控到 STANCE 脚 (sensor_name), 这样抬摆动脚不再受惩罚 — 旧的未门控
    # -5.0 通过让两个刀刃都平贴地面 (swizzle) 来最小化, 并主动对抗跨步.
    # 权重也软化了 -5.0 -> -2.0, 留出略微倾斜推力的空间.
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(func=microduck_mdp.neck_action_rate_l2, weight=-0.5)
    cfg.rewards["neck_joint_pos_l2"] = RewardTermCfg(func=microduck_mdp.neck_joint_pos_l2, weight=-0.5)
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(func=microduck_mdp.joint_torques_l2, weight=-1e-3)
    # 威慑 OVER-COMMANDING 关节越过其硬限位 (policy 侧, 通过 ONNX 转移).
    # hip_roll 的 ±0.38 rad 限位对比 ±10 rad ctrlrange, 让低 kp 舵机可被指令
    # 远超限位并以最大力矩撞击 — 一个脆弱的纯仿真把戏. 此项仅惩罚超过
    # (限位 + 0.3 过冲) 的 COMMAND, 所以关节保留全部可达范围 (qpos 惩罚会
    # 偷走该范围并破坏步态), 同时抑制野蛮的过驱.
    cfg.rewards["action_over_limit"] = RewardTermCfg(
        func=microduck_mdp.action_over_limit_penalty,
        weight=-0.5,
        params={"action_name": "joint_pos", "overshoot": 0.3},
    )
    # 将 hip_roll 拉回中立, 使站姿不再靠在 hip_roll 限位上外撇. L1 = 常量
    # 梯度: 它在静止时温和地并拢双腿, 但强跨步奖励 (wheel_speed,
    # single_support, air_time) 在主动推力时轻松压过它 → 闭合姿态而不阻止
    # 横向推力行程. 调参: 仍外撇就调高, 压扁跨步就调低. (物理注意: 如果
    # 软 hip_roll 舵机在体重下无法保持窄站姿, policy 会弯膝/降 CoM 来卸载
    # — 或, 如果不存在稳定窄站姿, 它会保持部分外撇.)
    cfg.rewards["hip_roll_neutral"] = RewardTermCfg(
        func=microduck_mdp.joint_deviation_l1,
        weight=-2.0,  # -1.0 -> -2.0: 更强的对中拉力. 仿真已让 hip_roll
        # 保持窄, 但更强的修正可能帮助真机抵抗让腿外撇的因素
        # (部署/扰动). 压扁推力就调低.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip_roll.*",))},
    )
    # 唯一正任务奖励 — 机器人必须转轮才能获得任何奖励
    # vel_scale 0.5 -> 0.3: tanh 目标速度. 在训练好的 ckpt 上实测, policy
    # 在最大推力下仅达 ~0.33 m/s, 所以 0.5 的目标停在未饱和的 tanh 斜坡上,
    # 不断推它超过能力 (过度延伸 → 起飞不稳). 0.3 在可达速度附近饱和, 所以
    # 它在那里"满足"而非过驱.
    cfg.rewards["wheel_speed"] = RewardTermCfg(
        func=microduck_mdp.wheel_speed_reward,
        weight=10.0,
        params={"command_name": "twist", "vel_scale": 0.3},
    )
    # 制动: cmd_x < 0 时奖励停止. cmd_x >= 0 时静默 (滑行/推力).
    cfg.rewards["braking"] = RewardTermCfg(
        func=microduck_mdp.braking_reward,
        weight=1.0,
        params={"command_name": "twist", "vel_std": 0.3},
    )
    # 推力时的 air time: 支付恢复脚的抬升, 但仅在身体确实向前移动时
    # (vel_gate_ref) — 否则快速原地抖动会刷这项. threshold_min 从 0.15 升到
    # 0.25 以禁止超短摆动 (限制狂躁踢腿节奏); 下方的 glide 奖励慢速相位.
    # air_time 奖励每次摆动 → 驱动摆动频率; glide 奖励保持在单刀刃 → 驱动
    # 承诺. 平衡偏向 glide (3.0) 而非 air_time (2.0), 因为节奏仍太快.
    # air_time 保持足够高 (2.0) 使抬脚仍值得.
    # 平稳步态: 激进的 [0.40, 1.00] 窗口强迫大而长的摆动 → 暴力踢腿让真机
    # 翻倒. 回到温和的 [0.15, 0.45] (允许小摆动, 不强迫长摆动), 权重
    # 2.0 -> 1.5 使摆动激励降低 (更低节奏). glide (下方) 奖励滑行, 所以
    # 它只偶尔推力.
    cfg.rewards["skating_air_time"] = RewardTermCfg(
        func=microduck_mdp.skating_air_time_reward,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "threshold_min": 0.15,
            "threshold_max": 0.45,
            "vel_gate_ref": 0.2,
        },
    )
    # 滑行相位 (要求单支撑, 不同于之前失败的尝试): 奖励在一只刀刃上滑行
    # 且腿部安静, 让 policy 承诺到每次划桨而非狂躁踢腿. 权重从 1.5 升到
    # 3.0 以真正压过 air_time 的摆动频率拉力.
    cfg.rewards["glide"] = RewardTermCfg(
        func=microduck_mdp.glide_reward,
        weight=4.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "vel_ref": 0.2,
        },
    )
    # 注意: 曾试过 recover_pose 奖励 (奖励默认腿姿 + 安静 + 暂停时滑行) 以
    # 实现 "划桨 → 恢复到中立 → 划桨", 但奖励对称默认姿态 + 去掉
    # single_support 的双支撑惩罚重新打开了对称 swizzle → 回退. 正确重试
    # 必须相位门控 (仅在划桨后短暂奖励中立, 而非持续), 并保留双支撑惩罚.
    # 单支撑跨步 vs 双支撑 swizzle. 奖励恰好一刃着地并惩罚推力时双刃着地
    # — 核心反 swizzle 信号. 也门控前向速度, 使不推进的迈步无收益.
    cfg.rewards["single_support"] = RewardTermCfg(
        func=microduck_mdp.single_support_reward,
        weight=3.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "vel_gate_ref": 0.2,
        },
    )
    # 平衡左/右腿使用. 对称增强关闭时, 没什么阻止偏跨步 (主要用一条腿推)
    # 导致偏转和不稳, 尤其在起飞时. 惩罚累积摆动时间不平衡 |L-R|/(L+R);
    # 真实跨步的瞬时单脚摆动不对称是允许的.
    cfg.rewards["gait_symmetry"] = RewardTermCfg(
        func=microduck_mdp.gait_symmetry_penalty,
        weight=-1.0,
        params={"sensor_name": "feet_ground_contact"},
    )
    # 注意: 曾在此试过 contact_frequency 惩罚以放慢节奏, 但它惩罚接触变化
    # — 通过永不抬脚 (swizzle) 来最小化, 所以它推向我们要离开的步态. 回退;
    # 上方加宽的 air_time 窗口是安全的节奏放慢器 (它禁止短摆动而不奖励
    # 不迈步).
    # 鼓励推力时轻微前倾, 以抵消后向力矩.
    cfg.rewards["forward_lean"] = RewardTermCfg(
        func=microduck_mdp.forward_lean_reward,
        weight=1.5,
        params={"command_name": "twist", "target_pitch": 0.262, "std": 0.1},
    )
    # 航向指令禁用 (专注直线), 但我们保持航向不让它漂移: heading_hold
    # 奖励偏航角保持接近出生航向. 修正性 (允许偏航转回) — 不同于偏航率
    # 惩罚, 后者冻结偏航并让漂移更糟 (试过并回退). 跨步稳固后重新加回真实
    # heading_tracking (转弯).
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=1.0,
        params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )

    # === 终止 ===
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # === 事件 ===
    # BAM (mjlab_frictionloss 分支) 每步写入每环境的 dof_frictionloss/dof_damping;
    # 这个 no-op 事件为按世界展开注册这些字段.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    del cfg.events["foot_friction"]  # 轮子滚动; 地面摩擦在 XML 中

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

    # 轮轴承摩擦 DR: 真实轴承有一点阻力; XML 保持 frictionloss=0 以便训练,
    # 课程逐步引入. mjlab 1.3.0 原生 dr 操作 (operation="abs" 直接写值;
    # 非累积).
    if ENABLE_WHEEL_FRICTION_RANDOMIZATION:
        cfg.events["randomize_wheel_friction"] = EventTermCfg(
            func=dr.dof_frictionloss,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
                "operation": "abs",
                "ranges": (0.000, 0.000),  # 由 wheel_friction_curriculum 渐升
            },
        )

    # ── DR 与 velocity 环境的 FIXED 版本对齐 ───────────────────────────────
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
        # 仅腿/头 — 轮轴承的微小 armature 被排除 (其 DR 是上方的
        # frictionloss 事件).
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
    # 1.3.0 基础模板添加基于传感器的 foot_height + height_scan; roller 环境
    # 没有地形高度传感器.
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
    # IMU 延迟 0-1 控制步 (与 velocity 一致: 真实 dxl IMU 路径快)
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # 观测噪声 — 与 velocity 环境对齐
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    # IMU 安装失准 DR (obs 层, 仅 actor — 与 velocity 一致)
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # joint_vel 上 1-ctrl-step 滞后 (Dynamixel present_velocity 滑动平均)
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(cfg.observations["actor"].terms["joint_vel"])
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # 从 joint_pos/vel obs 中排除被动轮关节 (obs 维 14, 匹配动作空间).
    # 按组深拷贝, 使下方 encoder-bias `biased` 标志仅应用于 actor.
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

    # critic 的特权轮速 (新模型 4 个轮).
    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        scale=1.0,
        params={"asset_cfg": wheel_cfg},
    )

    # 指令 obs 与 61D 家族布局对齐: head/body 槽零填充 (roller 任务通过
    # twist 槽驱动航向).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding,
            params={"dim": 6},
        )

    # === 指令 ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False  # RelativeHeadingVelocityCommand 内部处理航向
    command.ranges.heading = None  # heading_command=False 时必须为 None
    # cmd_x 语义: 0=滑行, >0=推加速, <0=刹车停止
    command.ranges.lin_vel_x = (-0.5, 0.6)
    command.ranges.lin_vel_y = (0.0, 0.0)
    # ang_vel_z 范围是 cmd[2] = 航向误差 (rad) 的裁剪限位.
    # 设为 0 → cmd[2] 恒为 0 → 无转弯需求 (专注直线).
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.RelativeHeadingVelocityCommandCfg(**vars(command))

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === 课程 ===
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # action_rate 惩罚提高 (-0.5/-0.8/-1.0 -> -1.0/-1.5/-2.0) 以获得更平稳
    # 步态: 这是主要的 "少动" 杠杆 — 它惩罚快/大动作变化, 所以动作变小,
    # 变平滑且变少 (快速交替 = 大动作变化 = 受惩罚). 呆滞/推力不足时调低.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -1.0},
                {"step": 250 * 24, "weight": -1.5},
                {"step": 500 * 24, "weight": -2.0},
            ],
        },
    )

    if ENABLE_WHEEL_FRICTION_RANDOMIZATION:
        # 延迟 + 软化的渐升: 之前的计划在 iter 750 — wheel_speed 峰值时 —
        # 开始加轴承阻力, 并达到 0.003, 这 (与下方的航向渐升一起) 把 policy
        # 推出滑行进入航向刷分的局部最优. 保持轮自由直到滑行稳健, 然后加
        # 温和、真实的阻力.
        cfg.curriculum["wheel_friction"] = CurriculumTermCfg(
            func=microduck_mdp.wheel_friction_curriculum,
            params={
                "event_name": "randomize_wheel_friction",
                "ranges_stages": [
                    {"step": 0 * 24, "ranges": (0.0000, 0.0000)},
                    {"step": 2000 * 24, "ranges": (0.0005, 0.0005)},
                    {"step": 3500 * 24, "ranges": (0.0010, 0.0010)},
                    {"step": 5000 * 24, "ranges": (0.0015, 0.0015)},
                ],
            },
        )

    # (heading_tracking_weight 课程移除 — 我们专注直线滑行时航向被禁用.
    # 与上方的奖励一起重新加回.)

    # CoM 随机化课程 — velocity 的渐升, 对平衡敏感的滑行任务上限更低
    # (审计教训: ±30 mm 在 walker 上强制紧张步态; 轮滑更不宽容).
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


MicroduckRollersRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # 与家族一致; 归一化器由 export.py 烤进 ONNX
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
        entropy_coef=0.03,  # roller 专有: 比 walk 环境更高的探索
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
    experiment_name="velocity_rollers",
    run_name="velocity_rollers",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
