# 规范 — RL 任务「射门踢球」通过姿态跟随

**日期** : 2026-07-24
**分支** : `new_pre_alpha_ground_pick`
**Task id** : `Mjlab-Shoot-Flat-MicroDuck`

## 目标

通过**跟随由相位插值的 4 个 keyframe 的关节姿态轨迹**来学习一个**单次射门**动作（踢击球）:

```
STAND → PIED_ARRIÈRE (armement) → PIED_AVANT (frappe) → STAND (repos)
```

- **右腿**踢击，**左腿**支撑。
- **没有模拟球** : 我们通过姿态跟随学习*动作*（如 `ground_pick` / crouch）。如果部署时机器人前方有一个真实球，它会被踢到。
- **统一的 61D obs** 与其它 microduck 策略相同 → 导出的 ONNX 可原样部署到运行时的**按键插槽**中（单次 : 执行动作后交还控制给主策略）。

与该分支的 `ground_pick` 任务相同的范式（`[cos, sin, 0]` 相位编码在 twist 插槽中，按相位跟随姿态，61D obs，从 velocity 继承的 DR sim2real）。

## 非目标（YAGNI）

- 无实体球，无接触/球速奖励。
- 无可配置边（仅右侧 ; 左侧 = 如需可稍后对称化）。
- 无行走 / 摔倒恢复 : 所有步态项都被移除。

## 架构

### 文件与注册
- `src/mjlab_microduck/tasks/microduck_shoot_env_cfg.py`
  - `make_microduck_shoot_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg`
  - `MicroduckShootRlCfg`（RslRlOnPolicyRunnerCfg，`experiment_name="shoot"`）
- 在 `src/mjlab_microduck/tasks/__init__.py` 中注册 :
  `Mjlab-Shoot-Flat-MicroDuck`（可选 `-Rough-` 变体）。
- 基础 : 继承 velocity env（通过 `make_velocity_env_cfg` 如 ground_pick），然后激进地剥离所有步态内容。
- 机器人 : `MICRODUCK_WALK_ROBOT_CFG`（标准行走，14 关节，无 rollers）。
- `action.scale = 1.0`。

### 姿态（占位符 → 通过 `read_pose.py` 读取真实机器人）
Dicts `{关节名: rad}`，**14 个关节**（excluded mouth）。env 文件顶部。
- `STAND_POSE` : 中性站立（~sim 的 HOME）。
- `KICK_BACK_POSE` : 右髋**向后伸展** + 右膝弯曲（举臂） ; 左腿 + 头 ≈ HOME。
- `KICK_FWD_POSE` : 右髋**前屈** + 右膝伸直（射门） ; 左腿 + 头 ≈ HOME。

起始时合理的占位符（可调整），将替换为真实读取值。

### 命令与相位
- 复用 `GroundPickPhaseCommand` : `command = [cos(2π·φ), sin(2π·φ), 0]` 在 twist 插槽中。
- **周期** : `SHOOT_PERIOD ≈ 2.5 s`（可通过 `cfg.period` 配置）。
- **新标志 `randomize_phase`** 在 `GroundPickPhaseCommandCfg` / `GroundPickPhaseCommand` 上 :
  - 默认 `True`（非破坏性 : `ground_pick` 保留当前行为）。
  - Shoot 将其设为 `False` → `reset()` 将 φ=0 重置为 `rand()` 之外的值。
  - 原因 : 每个 episode 都从 STAND 开始（机器人状态 = `default_joint_pos`），φ=0 = STAND 目标 → reset 时状态/目标一致（否则策略被要求从静止状态瞬时进入「射门」姿态）。
  - **一致性不变量** : `STAND_POSE` 必须等于 sim 的关节 reset 姿态（`HOME_FRAME` / `default_joint_pos`，非零 : hip_pitch ±0.4579，ankle ±0.4530，hip_roll ±0.0873，neck/head_pitch 0.3491）。由 `test_stand_pose_matches_home_standing_pose` 校验。最初为零的占位符违反了该不变量（最终审查后已修正）。

### Reset（站立高度，无冲量）
- `reset_base.pose_range.z = (0.12, 0.13)` — **绝对站立高度**（`InitialStateCfg` 的默认根 `pos` 为 (0,0,0)，所以 reset z = 0.12–0.13 m，不是可加偏移 ; 与能行走的 velocity env 值相同）。无坠落。
- **不注入进入速度**（站姿射门，与 crouch-glide 相反）。

### 继承的未列出奖励
上表并不详尽 : env 从 velocity 继承一些通用的低权重、非射门特异的正则化器 — `angular_momentum`（-0.02），`dof_pos_limits` — 保留（稳定性，可忽略）。⚠️ `soft_landing`（行走奖励）**被移除** : 它读取为支持单脚传感器而删除的双脚传感器 `feet_ground_contact`，否则第 1 步就 KeyError，而且对站姿射门它是无效的。

### 传感器重命名陷阱（⚠️）
重命名单脚传感器（`feet_ground_contact` → `left_foot_ground_contact`）会破坏 velocity/ground_pick 继承中所有按该名称引用的内容。需处理 :
- **critic obs** `foot_air_time`/`foot_contact`/`foot_contact_forces` → 改指向左脚传感器（critic 保留支撑信息 ; 否则 env 构建时 KeyError）。
- **reward** `soft_landing` → 移除（见上 ; 否则第 1 步 KeyError）。
始终通过实际构建 + **至少一次 `step()`** 验证（reward manager 只在 step 时运行），而不只是 cfg 构建和单元测试。

### ⚠️ 学习权重迁移（首次训练后修订）
发现 : 以**手持（双足支撑）**机器人测得的 BACK/FWD 姿态在所有阶段都把 CoM **保持在双足之间**（在左脚内侧 ~4-5 cm）。由于 `upright` 被强制，一旦右脚抬起机器人就会倾倒 → 没有任何策略能够维持（是几何问题，不是调参问题）。已在 sim 中验证（CoM vs 足部 sites）。

采纳的修复（RL 学习平衡） :
- `mdp.com_over_support_foot` : 高斯奖励（std 4 cm）将 CoM 投影（`root_com_pos_w`）拉向支撑脚，由 `mdp.kick_engagement` 门控（STAND 静止时 0，踢击期间 1）。权重 3.0。
- **拆分的姿态跟随**（`kick_pose_track`/`_l1` 上的 `joint_names` 参数） :
  动作 = 右腿 + 头/颈部（std 0.35，紧） ; 支撑 = 左腿（std 0.9，权重 1.0，**松**）→ 策略可以内收/偏移骨盆以转移体重，而不会让跟随把骨盆固定居中。
因此上面的「平衡 / 支撑」表被扩展 : 增加 `support_leg_pose`（1.0），`com_over_support`（3.0），而 `kick_pose_track`/`kick_pose_l1` 只承载动作的 9 个关节（右腿+头）。

### 目标 : 按相位插值的姿态跟随
`mdp.py` 中新的**纯**函数 :
```python
kick_pose_target(phase, stand, back, forward, windup_end, kick_end, return_end) -> Tensor
```
根据 4 段在姿态向量之间插值（归一化周期 [0,1)）:
```
[0, windup_end)        STAND   → BACK      (armement,     défaut 0.35)
[windup_end, kick_end) BACK    → FORWARD   (frappe sèche, défaut 0.10 = "snap")
[kick_end, return_end) FORWARD → STAND     (retour,       défaut 0.30)
[return_end, 1.0)      STAND              (repos)
```
「snap」来自短的踢击段 : 关节目标快速移动 → 脚快速摆动。3 个时机制界都是可参数的。

按**名称**解析关节（`asset.find_joints([name])`）— 对顺序鲁棒。

跟随奖励（始终激活，像 crouch 一样对称）:
| Reward | 权重 | 作用 |
|---|---|---|
| `kick_pose_tracking` | 6.0 | 高斯跟随 `exp(-((q-目标)/std)²).mean`，std=0.4 |
| `kick_pose_l1` | 2.0 | L1 引导（早期恒定梯度） |

### 平衡 / 支撑（单腿 = 倾倒风险）
| Reward | 权重 | 作用 |
|---|---|---|
| `upright` | 2.0 | 躯干竖直 |
| `support_foot_grounded`（左脚） | 6.0 | 保持支撑脚踩实（单脚传感器 → `found∈{0,1}`，`/2` 后 reward∈{0,0.5}，所以权重 6.0 ≈ 最大贡献 3.0） |
| `feet_flat`（左） | -1.0 | 左刀片平贴 |
| `self_collisions` | -1.0 | |
| `body_ang_vel` | -0.05 | |

`support_foot_grounded` : 复用 ground_pick 的 `feet_grounded_reward` 机制但限制到**左脚**（`left_foot_collision` 上的接触传感器）。

### 正则化（相对 ground_pick 减轻 — 让 snap 通过）
| Reward | 权重 | 作用 |
|---|---|---|
| `action_rate_l2` | -0.5 | 轻 : 重权重会杀死快速踢击 |
| `neck_action_rate_l2` | -0.5 | 头部稳定 |
| `joint_torques_l2` | -1e-3 | |

**移除的**（行走项） : `track_linear_velocity`，`track_angular_velocity`，`air_time`，`foot_clearance`，`foot_swing_height`，`foot_slip`，`pose`。

### 观察 / 部署（一致性）
- **61D obs 与 ground_pick/roller 相同** : `[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]`，head(4)+body(6) 插槽为**零填充**（`zero_command_padding`）。
- 相同的从 velocity 继承的 DR sim2real（CoM，mass/inertia，BAM 摩擦，armature，IMU mismatch obs 级，encoder-biast，推撞 ±0.3），以 NaN 保护结尾。
- 通过现有导出脚本导出 ONNX（烘焙归一化器）。
- 部署到运行时的一个相位插槽，例如 :
  ```
  --ground-pick shoot.onnx --ground-pick-period 2.5 \
  --ground-pick-kp-ratio 1.0 --ground-pick-action-scale <match>
  ```
  按钮 → 射门 → 自动交还主策略。

## 测试

- `tests/test_shoot.py` — 纯函数 :
  - `kick_pose_target` 在关键点 : φ=0 时 STAND，`windup_end` 时 BACK，`kick_end` 时 FORWARD，保持段内 STAND ; 中段插值 ; 边界（每个分量在姿态的 min/max 之间）。
  - 简单情形下 `kick_pose_tracking` / `kick_pose_l1` 奖励的值。
- `tests/test_shoot_cfg.py` — env 以正确的命令构建（`GroundPickPhaseCommand`，`randomize_phase=False`，周期）并且包含预期奖励 / 不含行走项。
- 运行 : `uv run --with pytest pytest tests/ -q`。

## 训练

```bash
uv run train Mjlab-Shoot-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations <N>
```
观察 `Episode_Reward/kick_pose_tracking`（应上升）。Play : play_latest 脚本。

## 开放问题 / 训练时调整
- **时序**（windup/kick/return）和**周期** : 合理的默认 snap，根据得到的脚速度和稳定性调整。
- **`action_rate` 权重** : snap 与 sim2real 平滑的权衡 ; 从轻（-0.5）开始。
- **可选增强（v1 不采用）** : 门控在踢击段上的小「右脚向前速度」奖励，以在无模拟球情况下推动力度。仅在单独的姿态跟随缺乏冲击力时添加。
- **部署时的转换** : 如果 `STAND_POSE` ≠ 主策略的中性点，触发/返回时有轻微冲击（如 crouch 所注意到的）。