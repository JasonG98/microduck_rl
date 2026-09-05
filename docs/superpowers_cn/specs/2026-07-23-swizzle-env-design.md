# Swizzle roller 环境 — 设计

**日期:** 2026-07-23
**分支:** `new_pre_alpha_rollers`

## 目标

一个**独立的** roller 任务，产生**干净的经典 swizzle** : 两片刀片都贴地，双腿**对称地**张开又收回（沙漏形态），推动鸭子前进。这是交替步态（`Mjlab-Velocity-Flat-MicroDuck-Rollers`）的更简单、更稳定的替代方案，其动机是该步态迁移到真实机器人上效果不佳。stride env 保持不动。

Sim2real 是目标 : 与 stride env 相同的机器人、观察、命令语义、域随机化和 ONNX 导出，因此它**以相同方式部署**（`microduck_runtime ... --roller`，相同的标志）。

## 方法（已选 : A — 移除反 swizzle + 奖励对称性）

基础的 roller 速度配方*天然地*收敛到 swizzle（这正是我们与 stride 斗争所对抗的吸引子）。所以获得干净 swizzle 的最简单方法是**移除反 swizzle 机制**并**奖励 swizzle 的定义性特征**（对称性、脚贴地）。无需相位脚本。

已否决 : B（显式的沙漏步态模式塑形）和 C（相位驱动的脚本化轨迹）— 更复杂，仅在 A 的 swizzle 看起来不干净（节奏/幅度）时才需要。

## 结构

- 新文件 `src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py`，包含 `make_microduck_velocity_swizzle_env_cfg(play=False)` 和 `MicroduckSwizzleRlCfg`。由 `make_velocity_env_cfg()` 加上 roller 机器人的方式构建，镜像 `microduck_velocity_rollers_env_cfg.py` 的结构（obs，DR，命令，curriculum）。
- 在 `tasks/__init__.py` 中注册 `Mjlab-Velocity-Swizzle-MicroDuck`。
- 复用来自 stride env 的所有 sim2real 内容 : robot cfg，61D obs 布局，命令（cmd_x 推动/滑行/刹车，直线 : `ang_vel_z=(0,0)`，`heading_hold`），所有 DR 事件 + curriculum（com，wheel_friction），`action_over_limit`，ONNX 导出路径。

## 奖励配方

**保留**（任务 + 稳定性 + sim2real） :
`wheel_speed`（向前推进，任务本身），`braking`，`upright`，`com_height_target`，`pose`，`forward_lean`，`heading_hold`，`action_over_limit`，`feet_flat`，`self_collisions`，正则化器（`action_rate_l2` + curriculum，`neck_action_rate_l2`，`neck_joint_pos_l2`，`joint_torques_l2`）。

**移除**（stride / 反 swizzle 机制） :
`single_support`，`glide`，`skating_air_time`，`gait_symmetry`，`hip_roll_neutral`（最后一个会与 swizzle 的侧向外展动作冲突）。

**新增**（亲 swizzle） :
- `leg_symmetry` — 奖励左右腿镜像。机器人使用镜像的 L/R 符号约定，因此对称配置满足每对的 `q_left + q_right ≈ 0`。对腿部关节对（hip_yaw，hip_roll，hip_pitch，knee，ankle）返回 `-mean_pairs |q_left + q_right|`（L1，恒定梯度 — 与现有 `bilateral_symmetry_penalty` 形式相同）；以正权重使用，使得不对称被惩罚、对称的 swizzle 被偏好。这是 swizzle 的定义性特征。（实现 : 现有 `bilateral_symmetry_penalty` 接受显式的 L/R 索引列表；添加一个轻量包装器，在运行时按名称解析 L/R 腿部关节对，使其无需硬编码索引即可配置。）
- `grounded` — 推动期间奖励两片刀片都接触（n_contact == 2），使脚保持贴地（经典 swizzle，不抬脚）。较小权重。新的 mdp 函数（`single_support_reward` 的镜像，但奖励双支撑）。像其它项一样以 `cmd_x >= 0` 门控。

让 `hip_roll` 的 pose std 保持宽松（如 stride env 中一样），以便双腿能够张开。

## 新的 mdp 函数（在 `tasks/mdp.py` 中）

1. `leg_symmetry_reward(env, asset_cfg)` — 按名称解析 L/R 腿部关节对，返回 `-mean_pairs |q_left + q_right|`（以正权重使用）。
2. `grounded_reward(env, sensor_name, command_name)` — 奖励恰好两片刀片接触，乘以 `clamp(cmd_x, 0)` 缩放。

## 命令 / sim2real（与 stride 相同）

`cmd_x` 推动/滑行/刹车，`lin_vel_y=0`，`ang_vel_z=(0,0)`（直线），完整的 DR（com，head_com，质量/惯量，关节摩擦，armature，轮子摩擦，速度推撞，IMU 错位，encoder bias，obs 延迟），61D obs，`vel_scale=0.3`。与 stride roller 策略相同的运行时标志部署。

## PPO 配置

复用 `MicroduckRollersRlCfg` 的超参数（相同的 actor/critic 512-256-128 ELU，PPO 设置，`entropy_coef=0.03`），新的 `experiment_name`/`run_name` = `velocity_swizzle`。

## 测试 / 验证

- Smoke test : `uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 16 --agent.max-iterations 2` 无错误运行 ; `leg_symmetry` 和 `grounded` 出现在奖励日志中。
- 在真实运行中观察 : `leg_symmetry` 高（对称），`grounded` 高（两脚贴地），`wheel_speed` 上升（向前移动）。视频 : 对称的沙漏 swizzle，两片刀片贴地。

## 调优旋钮（首次运行后）

- 如果对称度不够 → 提高 `leg_symmetry` 权重。
- 如果它抬脚 → 提高 `grounded` 权重。
- 如果它几乎没有移动 → 对称性/接地权重相对 `wheel_speed` 太高 ; 降低它们。
- 如果 swizzle 看起来不干净（节奏/幅度）→ 升级到方法 B。