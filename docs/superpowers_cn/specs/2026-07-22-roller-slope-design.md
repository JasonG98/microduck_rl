# 斜坡模式 — `roller_slope`（平衡的被动下坡）

日期 : 2026-07-22
状态 : 设计已验证，可进行实施计划。

## 目标

训练一个专用策略，让 **microduck（在 rollers 上）从平地以向前的轻微冲量起步，滚到一段下坡坡道，并被动滑到坡底，始终保持站立且平衡**。下坡过程中不进行任何操控 : 策略的唯一目标是**不摔倒**。

该策略必须通过难度 curriculum 处理逐渐变陡的坡道（**~2° → ~20°**）。

## 已框定的决策（头脑风暴）

| 主题 | 决策 |
|---|---|
| 行为 | 被动平衡下坡（重力使前进，不强制蹬腿） |
| 操控 | 无 — 纯平衡，`twist` 命令强制为零 |
| 方法 | **A** — 专用、隔离的任务（如 `roller_crouch`） |
| 地形形状 | **简单坡道** : 平地起步 + 下坡坡道（不是金字塔） |
| Episode 场景 | 在平地生成 → 向前冲量速度 → 在坡道上滑行 |
| 陡度 | Curriculum **0/2° → 20°** |
| 部署 | 标志 `--slope <onnx>` + `infer_policy.py` 中的按键 **`Y`**（Y 是空闲的） |

## 架构

### 1. 新任务

文件 : `src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py`，从 `microduck_velocity_rollers_env_cfg.py` 克隆。

- 相同的 roller 机器人（`MICRODUCK_WALK_ROLLERS_ROBOT_CFG`），相同的物理，相同的域随机化 / 噪声 / 延迟。
- **相同的 61D 观察**（twist + head/body 置零补齐）→ 该策略通过运行时的 `--new-cmd-obs` 路径加载，并与其它 roller 策略保持可互换。
- 在 `src/mjlab_microduck/tasks/__init__.py` 中通过 `register_mjlab_task` 注册，使用 PPO 配置 `MicroduckRollerSlopeRlCfg`（`experiment_name`/`run_name` = `roller_slope`）。

### 2. 地形「平地 + 坡道」（自定义）

mjlab 提供的倾斜地形是金字塔；因此我们编写一个专用的 `SubTerrainCfg`（例如 `FlatRampTerrainCfg`），其 `function(difficulty, spec, rng)` 方法构建 :

- 一个**平地的起步区**（长度 ~1–2 m），机器人在这里生成；
- 接着是一段**下坡坡道**，其角度由 `difficulty` 在 `[~2°, ~20°]` 上**插值**。

地形通过 `TerrainEntityCfg(terrain_type="generator", ...)` 安装，配合一个生成多个难度等级（因此多个坡道角度）的 `TerrainGeneratorCfg`。每个环境原点必须落在**平地区域**上，坡道在它前面。

> 要在计划中处理的实现风险 : 生成原点定位在平地上（不在 tiles 中心），以及坡道的方向使得「前方」=「向下」。

### 3. 命令 = 无

`twist` 插槽被中和 : `rel_standing_envs = 1.0`，速度范围置 0，`rel_heading_envs = 0.0`。Head/body 保持置零补齐。该策略不接收任何移动指令。

### 4. Reset 与冲量速度

- `reset_base` : 在平地上以静止状态生成，rollers 标称高度 `z`（~`0.1335–0.1435`，同 roller env）。
- **进入速度**通过 `reset_root_state_uniform` 的 `velocity_range` 注入（自身状态 + 范围），**而不是**通过 `push_by_setting_velocity`（它加到当前状态上，可能使 free-joint 发散 → NaN —— 这是在 `roller_crouch` 上已经学到的教训）:
  `x ≈ (0.2, 0.5) m/s` 向前。
- episode 期间保留轻微随机推撞（鲁棒性），同 roller env。

### 5. 奖励

核心是「保持直立 + 自然姿态」，防偷懒最优化（避免它为了最大化稳定性而趴在地上）:

- `upright`（躯干竖直）— **主要的**
- `alive`（每步存活奖励）
- **标称站立姿态** : 奖励趋向 HOME 姿态（沿用 `roller_crouch` 的姿态插值机制，但目标固定 = 站立），以保持正常的 rollers 站姿而非防御性蹲姿
- `feet_flat`（rollers 平贴地面）
- `body_ang_vel`, `angular_momentum`（不颤抖 / 不打转）
- `action_rate_l2`, `neck_action_rate_l2`, `joint_torques_l2`, `self_collisions`（平滑 + sim2real）

> 没有速度/刹车奖励 : 下坡是被动的。我们不奖励「滑得快」，只奖励「下坡时保持直立」。

### 6. 终止条件

- **摔倒** : `bad_orientation`（躯干过度倾斜）。
- **到达坡底** : `out_of_terrain_bounds`（机器人已到坡底 → reset）。
- `nan_state`，超时。

### 7. 难度 curriculum（陡度）

**平缓 → 陡峭** 的进展 : 从近乎平坦的坡道开始，随成功情况逐步把角度提升到 20°。

> 实现风险 : 标准的 `terrain_levels_vel` curriculum 根据已行进距离与命令速度之比来自动提升。这里命令为零，因此**需要一个自定义的升级标准** : 如果机器人存活 / 在没有摔倒的情况下到达坡底则升级，如果过早摔倒则降级。

### 8. 部署 — 按键 `Y`

在 `scripts/infer_policy.py` 中 :

- 新标志 `--slope <onnx>` 加载斜坡策略作为额外会话（与 `--walking` / `--standing` / `--ground-pick` 相同的模式）；
- `GLFW_KEY_Y = 89`（如今**空闲** — 头部在 `H` 上），它**切换**活动会话到/离开斜坡策略；
- 添加一行键盘帮助。

没有破坏任何现有控制（与共享头部控制的按键 `H` 不同）。

## 范围外（YAGNI）

- 没有左右操控，也没有下坡刹车。
- 没有上坡或横向穿越。
- 没有金字塔或多方向地形。
- 不从现有 roller 权重微调（从零开始训练）。

## 交付物

1. `microduck_roller_slope_env_cfg.py`（env + `FlatRampTerrainCfg` + PPO cfg）。
2. 在 `tasks/__init__.py` 中注册任务。
3. `tasks/mdp.py` 中需要的自定义奖励/curriculum（站姿，等级升级）。
4. 在 `scripts/infer_policy.py` 中接线 `--slope` + 按键 `Y`。
5. 纯函数的单元测试（按难度计算的坡道角度，可能的升级标准）。