"""Microduck roller SWIZZLE 环境 — 干净的经典 swizzle.

一个独立的 roller 任务, 产生经典 SWIZZLE: 两个刀刃都贴地, 腿外撇再对称
收回 (沙漏图案), 推动鸭子前进. 比交替跨步 (`Mjlab-Velocity-Flat-MicroDuck-
Rollers`, 对真机转移不佳) 更简单/更稳定的替代. 跨步环境保持不变.

方案 A (见 docs/superpowers/specs/2026-07-23-swizzle-env-design.md): 基础
roller 配方自然收敛到 swizzle, 所以我们整体复用跨步环境 (机器人, 61D obs,
指令, 完整 DR, 课程, sim2real — 用 `--roller` 相同部署), 仅交换奖励配方:
  - 移除反 swizzle / 跨步项.
  - 添加 leg_symmetry (腿镜像) + grounded (双刃着地).
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    MicroduckRollersRlCfg,
    make_microduck_velocity_rollers_env_cfg,
)

# swizzle 任务要丢弃的跨步 / 反 swizzle 奖励.
_ANTI_SWIZZLE = (
    "single_support",
    "glide",
    "skating_air_time",
    "gait_symmetry",
    "hip_roll_neutral",
)


def make_microduck_velocity_swizzle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Roller swizzle 环境: 跨步环境去掉反 swizzle 项, 加上对称和着地奖励.

    其他一切 (机器人, obs, 指令, DR) 完全相同.
    """
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    for name in _ANTI_SWIZZLE:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # 腿互相镜像 (swizzle 的定义性对称).
    cfg.rewards["leg_symmetry"] = RewardTermCfg(
        func=microduck_mdp.leg_symmetry_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 保持双刃着地 (经典 swizzle: 不抬脚).
    cfg.rewards["grounded"] = RewardTermCfg(
        func=microduck_mdp.grounded_reward,
        weight=1.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )

    # --- 后向行走 (方案 A): cmd_x < 0 表示后退 (非刹车) ---
    # wheel_speed 奖励指令方向上的轮转 (正向前, 负向后); 制动奖励被移除
    # (负不再表示 "停止"); 指令范围对称化, 使前后获得相等推力范围. 要停止,
    # 指令 cmd_x ~ 0 (滑行). grounded 使用 |cmd_x|, 所以两个方向都压住刀刃.
    cfg.rewards["wheel_speed"].params["bidirectional"] = True
    if "braking" in cfg.rewards:
        del cfg.rewards["braking"]
    cfg.commands["twist"].ranges.lin_vel_x = (-0.6, 0.6)

    # --- 航向课程: 先直线, 再跟随指令方向 ---
    # 跨步环境禁用了航向 (ang_vel_z=(0,0), heading_hold, 无 heading_tracking).
    # 重新启用航向指令, 使 cmd[2] 携带到采样目标的航向误差, 并添加
    # heading_tracking (从 0 开始). 课程然后切换两者:
    #   阶段 1 (直线): heading_hold 主导, heading_tracking 关闭
    #   阶段 2 (跟随): heading_hold -> 0, heading_tracking -> 升
    # cmd[2] = 航向误差裁剪. 从 ±1.0 降到 ±0.5: 限制观测到的航向误差, 使
    # 转向修正率更温和 (±1.0 训练的 policy 转得太猛 — 必须跑 --max-angular-vel
    # 0.3 来驯服). 它仍可达任何航向 (误差只在 0.5 饱和), 所以它完整但平滑地
    # 转向, 且 heading_tracking 权重保持 3.0 使它仍能跟随方向.
    cfg.commands["twist"].ranges.ang_vel_z = (-0.5, 0.5)

    cfg.rewards["heading_tracking"] = RewardTermCfg(
        func=microduck_mdp.heading_tracking_reward,
        weight=0.0,  # 由下方课程渐升 (必须匹配其 step-0 值)
        params={"command_name": "twist", "std": 0.5},
    )

    cfg.curriculum["heading_hold_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "heading_hold",
            "weight_stages": [
                {"step": 0, "weight": 1.0},  # 必须匹配 heading_hold 初始权重
                {
                    "step": 1000 * 24,
                    "weight": 1.0,
                },  # swizzle 巩固期间保持直线
                {"step": 1750 * 24, "weight": 0.5},
                {"step": 2500 * 24, "weight": 0.0},
            ],
        },
    )
    cfg.curriculum["heading_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "heading_tracking",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1000 * 24, "weight": 0.0},  # 到此仅直线
                {"step": 1750 * 24, "weight": 1.5},
                {"step": 2500 * 24, "weight": 3.0},
            ],
        },
    )

    # --- 头部姿态控制 (Y 按钮): policy 产生头部姿态 ---------
    # 头部姿态指令 (相对 HOME 的 4D 增量: [neck_pitch, head_pitch, head_yaw,
    # head_roll]). 从 velocity 环境移植; 范围从小开始 (由下方课程加宽).
    # 每 2-5 秒重采样.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.05, 0.05),  # neck_pitch
            (-0.05, 0.05),  # head_pitch
            (-0.07, 0.07),  # head_yaw
            (-0.015, 0.015),  # head_roll (更紧 — 机械范围小)
        ),
    )

    # 将真实 head 指令喂入 obs (替换 zero_command_padding), 应用于两组.
    # body_command 保持零填充 (此处无身体姿态控制).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )

    # 奖励头部跟踪其指令. 此处权重 0 — 由课程晚期渐升, 以免在 swizzle
    # 巩固前干扰它.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.0,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # 调和两个会与 head_pose_tracking 冲突的 HOME 拉力:
    #  1) neck_joint_pos_l2 把 neck/head 关节拉向 HOME -> 移除它.
    if "neck_joint_pos_l2" in cfg.rewards:
        del cfg.rewards["neck_joint_pos_l2"]
    #  2) pose 奖励包含 neck/head -> 限定到 LEG 关节.
    # 从 std 字典中移除 neck/head/passive 模式以匹配限定后的 asset_cfg.
    for std_key in ["std_standing", "std_walking", "std_running"]:
        if std_key in cfg.rewards["pose"].params:
            std_dict = cfg.rewards["pose"].params[std_key]
            # 仅保留腿关节模式 (过滤掉 neck, head, passive)
            cfg.rewards["pose"].params[std_key] = {
                k: v for k, v in std_dict.items() if "neck" not in k and "head" not in k and "passive" not in k
            }
    # asset_cfg 限定到 LEG 关节 (排除 neck, head, 被动轮)
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )

    # head_pose_tracking 渐升 0 -> 4.0, 到 ~1500 iter 前保持 0 (swizzle 巩固),
    # 使头部控制建立在稳定 swizzle 之上.
    cfg.curriculum["head_pose_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_tracking",
            "weight_stages": [
                {"step": 0, "weight": 0.0},  # 必须匹配初始权重
                {"step": 1500 * 24, "weight": 0.0},  # swizzle 巩固前头部关闭
                {"step": 2250 * 24, "weight": 2.0},
                {"step": 3000 * 24, "weight": 4.0},
            ],
        },
    )
    # 头部指令范围在相同窗口内加宽 (1500 前微小, 3000 前全范围), 所以指令的
    # 头部早期几乎不动, 在 policy 能处理时达到全范围.
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,               ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
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
                    "step": 1500 * 24,
                    "ranges": (
                        (-0.05, 0.05),
                        (-0.05, 0.05),
                        (-0.07, 0.07),
                        (-0.015, 0.015),
                    ),
                },
                {
                    "step": 2250 * 24,
                    "ranges": (
                        (-0.55, 0.55),
                        (-0.55, 0.55),
                        (-0.70, 0.70),
                        (-0.15, 0.15),
                    ),
                },
                {
                    "step": 3000 * 24,
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

    return cfg


# 与跨步 roller 任务相同的 PPO 超参数, 新的 experiment/run 名.
MicroduckSwizzleRlCfg = dataclasses.replace(
    MicroduckRollersRlCfg,
    experiment_name="velocity_swizzle",
    run_name="velocity_swizzle",
)
