# 设计 — `roller_standup` : 在 rollers 上站起来

**目标** : 一个专用策略，在摔倒后（趴着或仰躺着）把 microduck **扶起站到 rollers 上**，并且随后能**保持**轮上站立。

把 `standup`（行走鸭）配方移植到 rollers 模型。不修改任何现有 env。

---

## 已决决策

| 决策 | 选择 | 排除的替代方案 |
|---|---|---|
| 形式 | **专用 episodic 策略** | 把站起来嫁接进 roller env（`velstand` 配方）→ 有真实风险破坏已学成的步态 |
| 起始姿态 | **趴 + 仰躺 + 站立** | `坐`（只存在于从 `sit` 策略交接，无 roller 对应物）；侧躺（覆盖最全但收敛难得多）；无`站立`（策略站起来后又会摔倒） |
| 自由轮子 | **反向滚动摩擦 curriculum** | 起手就是真摩擦（引导太硬）；用奖励强加滑冰技术（仓库历史 : 过于指令性的风格奖励会产生寄生最优点 — swizzle、crouch 的偷懒最优点） |
| 目标姿态 | **HOME + 实测高度** | roller-crouch 的 `STAND_POSE`（被标记为开放问题 : ≠ roller 中性点 → 返回时有冲击）；从真实机器人读姿态（阻塞开发） |
| 命令 | **twist 中和**（≈ 0） | 相位 / 按钮插槽命令（见「部署」）；可控制的头部 |

---

## 架构

**新文件** : `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- `make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`
- `MicroduckRollerStandUpRlCfg`（`experiment_name="roller_standup"`）
- Task id : `Mjlab-RollerStandUp-Flat-MicroDuck`（仅 flat，无 rough 变体）

**派生** : `cfg = make_microduck_velocity_rollers_env_cfg()`。

这是 `roller_slope` 的模式（246 行），而非 `roller_crouch` 的模式（479 行，从 `make_velocity_env_cfg()` 重新出发并复制所有 DR 块）。因此我们无风险地继承，不会漂移 :

- 机器人 `MICRODUCK_WALK_ROLLERS_ROBOT_CFG`（14 个活动关节 + 4 个被动轮子，BAM m6，kp_fw 200）；
- 传感器 `feet_ground_contact`（在 `ankle_{l,r}_v1` 上 subtree 模式）和 `self_collision` ;
- 整个 DR : 躯干 CoM + 头部，质量/惯量（pseudo_inertia），BAM 摩擦，armature，encoder bias，obs 级 IMU mismatch，滚动摩擦 ;
- **统一的 61D 观察** `[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]` — 运行时可互换的硬条件 ;
- 终止条件 `nan_state`（扩展保护 : 关节 + free-joint + 轮子）。

rollers 模型**物理上允许**躺平 : `robot_allcollisions_rollers.xml` 除了 4 个轮胎外，还在躯干（`np_f970`）、髋部、腿、头壳和下颌上带有碰撞 geoms。已验证。

---

## 实测常量

通过精确运动学测得（碰撞 geoms 网格顶点的最小值，`STAND` keyframe 姿态，躯干回到接触），在 `scene_rollers.xml` 与 `scene.xml` 上对比 :

| 姿态 | 脚模型 | rollers 模型 |
|---|---|---|
| 站立（`STAND` = HOME） | 0.1172 | **0.1407** |
| 趴着（静止） | 0.0752 | 0.0752 |
| 仰躺（静止） | 0.0476 | 0.0475 |

一致性校验 : `standup` 使用**负载下**测得的 `STAND_Z = 0.115`，相对运动学 0.1172 → ~2 mm 下陷。我们应用同样的修正，结果恰好落在 roller env 已经使用的 `reset_base z = 0.1335–0.1435` 内。

```python
ROLLER_STAND_Z   = 0.138   # tronc debout sur roues, sous charge (+23 mm vs pieds)
ROLLER_PRONE_Z   = 0.075   # hauteur de repos à plat ventre
EPISODE_LENGTH_S = 6.0
```

两个模型的贴地静止高度**相同**（接触的是躯干外壳，不是脚）。这并不意味着 `standup` 的 `prone_z` 范围能原样复用 : 见「Reset」下方的注释 — `prone_z_min` 有分歧（此处 0.076，不是 0.05），因为单一范围服务于两个姿态（趴、仰躺），它们的 reset 接触高度不同。

实测的量正是奖励读取的量 : `height_target_gaussian` 和 `height_l1_penalty` 使用 `root_link_pos_w[:, 2]`，它恰好等于 `xpos[trunk_base].z`（free-joint 在 `trunk_base` 上）— 已在数值上验证。

## 关节索引

被动轮子**交错**在关节顺序中。MuJoCo 中验证过的实际顺序（`m.jnt_qposadr`，rollers 模型，free-joint 之后 18 个关节）:

```
0-4   left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
5-6   passive_LF_wheel, passive_LR_wheel
7-10  neck_pitch, head_pitch, head_yaw, head_roll
11-15 right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
16-17 passive_RF_wheel, passive_RR_wheel
```

```python
_LEG_JOINTS   = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]   # standup : [0-4, 9-13]
_NECK_JOINTS  = [7, 8, 9, 10]                          # standup : [5-8]
_WHEEL_JOINTS = [5, 6, 16, 17]
```

只有 `_LEG_JOINTS` 真正被消费（被姿态奖励）。`_NECK_JOINTS` 和 `_WHEEL_JOINTS` 是为文档和索引测试而声明 : 颈部按**名称**解析（`neck_joint_pos_l2` 每一步都调用 `find_joints(r".*(neck|head).*")`，正是为了对轮子造成的偏移鲁棒），轮子用正则 `^passive_.*`。

交接文档明确标注了这一脆弱点。它由一个测试锁定，该测试构建 env 并验证这些索引处的关节名称（见「测试」）。

---

## 奖励

### 从 roller 继承中移除的

| 移除 | 原因 |
|---|---|
| `wheel_speed`，`braking`，`skating_air_time`，`glide`，`single_support`，`gait_symmetry`，`forward_lean`，`heading_hold` | 步态奖励 : 躺在地上时无意义 |
| `feet_flat` | 起身过程中刀片不都平贴 → 这个惩罚会对抗动作 |
| `hip_roll_neutral` | 站起来需要分开双腿 |
| `pose`，`com_height_target` | 被下面的姿态/高度目标取代 |
| `upright`（基础高斯） | 被 `upright_linear` + `upright_sharp` 取代 |

### 从 roller 继承中保留的

| Reward | 权重 | 作用 |
|---|---|---|
| `action_over_limit` | −0.5 | sim2real 保护（超出限位过驱动），与任务无关 |
| `self_collisions` | −1.0 | |
| `body_ang_vel` | **−0.05** | 刻意**轻** : `standup` 记录到在 −0.15 时它会冻结起身（动作阻隔器） |
| `angular_momentum` | −0.02 | |
| `action_rate_l2` | curriculum −0.4 → −0.8 → −1.0 | roller env 将其拉平到 −1.0 ; 我们沿用 `standup` 的斜坡（早期平缓 → 帮助大翻身动作引导） |
| `neck_action_rate_l2` | −0.5 | 头部稳定 |
| `neck_joint_pos_l2` | −0.5 | 保持头部竖直（`roller_slope` 的选择）— **取代** `standup` 的 `head_pose` 命令 |
| `joint_torques_l2` | −1e-3 | |

### 新增

| Reward | 权重 | 作用 |
|---|---|---|
| `joint_torque_rate_l2` | −2e-3 | 抗抖动 : `standup` 将其确定为唯一不阻隔翻身的阻尼器（它惩罚扭矩的*变化*，而不是幅度或躯干旋转） |

### 起身奖励（从 `standup` 移植，重新映射）

十项**连同它们已经调好的权重**一起复制，这些权重来自 `microduck_standup_env_cfg.py` 中记录的迭代。只改变关节索引和两个高度。所有 mdp 函数都已存在 — **`mdp.py` 无需编写任何内容**。

| Reward | mdp 函数 | 权重 | roller 参数 | 作用 |
|---|---|---|---|---|
| `pose_stand_legs` | `pose_target_match` | +8.0 | `std=0.5`, `joint_indices=_LEG_JOINTS`, `target_overrides=None` (HOME) | 关节姿态目标 |
| `pose_stand_l1` | `pose_l1_penalty` | +5.0 | `joint_indices=_LEG_JOINTS`, `target_overrides=None` | L1 引导 : 即使远离 HOME 也为恒定梯度 |
| `height_stand` | `height_target_gaussian` | +4.0 | `std=0.04`, `target_height=0.138` | 宽高斯 → 从地面拉起 |
| `height_stand_sharp` | `height_target_gaussian` | +4.0 | `std=0.015`, `target_height=0.138` | 窄高斯 → 迫使最后几厘米 |
| `height_stand_l1` | `height_l1_penalty` | +30.0 | `target_height=0.138` | 使「趴着」明显为负（否则偷懒最优点） |
| `com_upward_velocity` | `com_upward_velocity` | +3.0 | `max_height=0.148` | 为*上升*运动付费（目标上方 +10 mm 余量，如 `standup` 中 0.125 vs 0.115） |
| `gentle_rise` | `trunk_vertical_accel_penalty` | −0.02 | | 惩罚 `\|a_z\|` → 匀速平滑上升 |
| `upright_linear` | `body_upright_linear` | +6.0 | | `cos(tilt)` : 躺着时强梯度 |
| `upright_sharp` | `upright_gaussian_at_height` | +6.0 | `std=0.3`, `height_low=0.075`, `height_high=0.138` | 在高度门控上的紧高斯 → 消灭后仰 |
| `standing_composite` | `standing_composite_score` | +15.0 | `height_std=0.04`, `upright_std=0.40`, `pose_std=0.40`, `target_height=0.138`, `joint_indices=_LEG_JOINTS` | 高度 × 竖直 × 姿态的乘法得分 |

在 `standup` 使用 `asset_cfg=SceneEntityCfg("robot", body_names=("trunk_base",))` 的所有位置，这些项都接受该配置。

**此 v1 无冲击惩罚**（躯干/头部）: `standup` 没有，只有 `velstand` 有。保持动作最小化。

---

## 观察与命令

**观察** : 完整地从 roller env 继承（61D）。无修改 — 这是派生自该 env 的原因。

我们在 actor 和 critic 组上添加 `nan_policy = "sanitize"`，如 `roller_slope` : 一个罕见的接触使 free-joint 发散为 NaN，obs 被净化（→ 0）以免杀死训练，有问题的 env 在下一步 reset。

**命令** : `twist` 插槽被中和，与 `standup` 完全一样 :

```python
command = cfg.commands["twist"]
command.rel_standing_envs = 0.0
command.rel_heading_envs  = 0.0
command.heading_command   = False
command.ranges.heading    = None
command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
command.debug_vis = False
command.ranges.lin_vel_x = (-0.01, 0.01)
command.ranges.lin_vel_y = (-0.01, 0.01)
command.ranges.ang_vel_z = (-0.05, 0.05)
cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))
```

`head_pose`（4）和 `body_pose`（6）插槽保持**零填充** — roller 家族（`roller`，`roller_crouch`，`roller_slope`）的约定。这是相对行走 `standup` 的一个明确差异，后者通过真实的 4D `head_pose` 命令驱动头部（见「风险」）。

twist 中和的理由 : 在 `scripts/infer_policy.py` 中，行走的 `standup` 策略被加载为 `--standing`，与 `--walking` 并列，切换是**基于速度命令量值的自动切换**（`infer_policy.py:262`，阈值 0.05）; 当 `standing` 激活时，twist 插槽被置零（`infer_policy.py:239`）。相位插槽（`ground_pick`，`fold`）用于按钮触发的单次 tricks，不用于起身。

---

## Reset

添加 `set_ground_state` 事件（`reset` 模式），插入在继承的 `reset_base` 和 `reset_robot_joints` **之后**（事件顺序跟随 dict 中的插入顺序）:

```python
cfg.events["set_ground_state"] = EventTermCfg(
    func=microduck_mdp.set_random_ground_state,
    mode="reset",
    params={
        "face_down_prob":  0.50,   # ventre — piloté par le curriculum ci-dessous
        "face_up_prob":    0.00,   # dos — introduit tard (le plus dur)
        "sitting_prob":    0.00,   # pas de bucket assis → aucun override de joint à remapper
        "standing_prob":   0.50,
        "prone_z_min":     0.076,  # cf. note ci-dessous — pas un simple héritage du standup
        "prone_z_max":     0.09,
        "standing_z_min":  0.134,  # roller (contre 0.11–0.12 pour les pieds)
        "standing_z_max":  0.144,
        "sitting_tilt_max": math.radians(10),  # ± bruit de pitch/roll ; s'applique AUSSI au bucket debout
    },
)
```

注意 : 在 `set_random_ground_state` 中，`standing` bucket 复用 `sitting` bucket 的四元数 — 因此 `sitting_tilt_max` 也会给站立起点加噪声，这是有意的。

**关于 `prone_z_min` = 0.076（而不是 0.05，错误地从 `standup` 沿用）** : 趴和仰躺共享单一 z 范围，但它们的实测接触高度不同 — 趴 0.0752，仰躺 0.0475 — 因此单一范围无法对两者都理想。`standup` 的注释用**重力下稳定后**实测 ~0.044 的静止来解释其 `0.05` 下界 ; 但在 reset 时刻起作用的，是 HOME 姿态下的接触高度，而不是落下后的静止高度。在 0.05 时，趴着生成时躯干外壳会**陷入地面 25 mm**，策略随后通过 `gentle_rise` / `joint_torque_rate_l2` 为这个 pushout 付费。`prone_z_min = 0.076` 消除了这种相互穿透，代价是仰躺高过其静止点 28–42 mm — 这是一个比接触 pushout 温和得多的伪影。

**`mdp.py` 无修改** : 基础 `reset_robot_joints` 使用 `joint_names=(".*",)` 且 `velocity_range=(0.0, 0.0)`，`default_joint_vel`（HOME_FRAME `joint_vel={".*": 0.0}`）→ 4 个被动轮子每次 reset 都已归零。已验证。

**Curriculum `ground_state_mix`**（`event_param_curriculum`），与 `standup` 相同的 easy → hard 逻辑 : 仰躺后期引入并在最后得到最多训练。

| iter | 站 | 趴 | 仰 |
|---|---|---|---|
| 0 | 0.50 | 0.50 | 0.00 |
| 600 | 0.35 | 0.45 | 0.20 |
| 1500 | 0.25 | 0.40 | 0.35 |
| 2500 | 0.20 | 0.40 | 0.40 |

（Steps 以 `common_step_counter` 为单位 = `iter × 24`。）

**推撞** : `push_robot` 从 roller env 继承（±0.2 m/s，间隔 3–6 s）。我们添加 `standup` 的上升 curriculum，以免干扰引导 : 0 → ±0.08（iter 500）→ ±0.2（iter 1000）。

**终止条件** : 移除 `fell_over`（机器人**以摔倒状态开始** — 基于倾角的终止在此无意义）。`nan_state` 被继承并保留。

**地形** : `plane`。此 v1 无 rough 变体 — 与 roller env 一致，它没有 `rough` 参数。

---

## 滚动摩擦 curriculum，反向

这是设计中唯一真正新颖的部分，也是任务所提问题的核心 : **轮子滚动，没有纵向附着力来推地面。**

该机制已存在并被继承（`randomize_wheel_friction` 通过 `dr.dof_frictionloss` 作用于 `^passive_.*` + `wheel_friction_curriculum`）。在 roller env 中它是**上升** 0 → 0.0015。这里我们让它**下降** :

| iter | frictionloss | 效果 |
|---|---|---|
| 0 | 0.05 | 轮子近乎锁死 → 像有脚一样站起来 |
| 1000 | 0.02 | |
| 2000 | 0.008 | |
| 3000 | 0.003 | |
| 4000 | 0.0015 | 真实滚动值（roller env 的值） |

`wheel_friction_curriculum` 简单地应用最后一个通过的档位（`if env.common_step_counter > stage["step"]`）— 它上升和下降都同样工作。**零代码编写。**

**这个 curriculum 告诉我们** : 如果 `Episode_Reward/standing_composite` 在摩擦下降时崩掉，我们就得到明确的答案，即「粘住的地面」动作无法迁移到自由轮子，需要引导滑冰技术（中间膝关节支撑，一次用一个冰刀）。这是一个可利用的结果，不是失败。

---

## 网络与 PPO

与 `standup` 相同 : actor 和 critic `(512, 256, 128)` elu，`obs_normalization=True`（归一化器由 `export.py` 烘焙进 ONNX），PPO `lr=1e-3` adaptive schedule，`desired_kl=0.01`，`entropy_coef=0.01`，`gamma=0.99`，`lam=0.95`，`num_steps_per_env=24`，`save_interval=250`，`max_iterations=15_000`。**对称 OFF**（`SYMMETRY_CFG` 是为旧 51D 布局接线的，会在 61D 上崩溃 — 与所有 v1.5+ env 相同）。

---

## 测试

`tests/test_roller_standup_cfg.py` :

1. env 能构建（`play=False` 和 `play=True`）；
2. **`_LEG_JOINTS` / `_NECK_JOINTS` / `_WHEEL_JOINTS` 索引处的关节名称是正确的**（防止交错轮子的脆弱点）；
3. 预期的起身奖励存在，滑冰奖励不存在（`wheel_speed`，`glide`，`single_support`，`feet_flat`，…）；
4. `fell_over` 不存在，`nan_state` 存在 ;
5. `wheel_friction` curriculum 确实是**递减**并以 0.0015 结束 ;
6. `ground_state_mix` curriculum : 最后档位的概率之和为 1，且 `face_up_prob` 单调增长 ;
7. **obs 一致性** : actor/critic 项的名称和维度与 `make_microduck_velocity_rollers_env_cfg()` 的相同（否则 ONNX 无法加载进插槽）。

运行 : `uv run --with pytest pytest tests/ -q`。

---

## 训练与部署

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
```

观察 `Episode_Reward/standing_composite`（应上升），尤其注意其在**滚动摩擦档位**（iter 1000/2000/3000/4000）的表现。

Play : `uv run scripts/play_latest.py`。Export : `uv run scripts/export_latest.py`。

目标部署 : 该策略作为 `--standing`，面对作为 `--walking` 的 roller 策略，基于命令量值自动切换。**保留** : `infer_policy.py` 是本地 sim/键盘脚本 ; 机器人运行时是 Rust 二进制 `microduck_runtime`，不在本仓库 — 此处的未验证其是否暴露具有相同切换逻辑的等价 `--standing`。交接文档只列出 `--model`，`--ground-pick`，`--fold-policy`。待确认。这不改变训练 : 如果运行时没有该插槽，该策略仍可用于按钮插槽（那里的命令将是相位而非零 — 那将是唯一要重新审视的点）。

---

## 风险与注意事项

1. **在自由轮子上站起可能没有专用技术就无法实现。** 这是主要风险。摩擦 curriculum 的设计就是让它以可读的方式裁决这个问题，而不是绕开它。
2. **「仰躺」bucket 最难。** `standup` 记录到它在这种姿态下停在「什么都不做」，原因是*动作阻隔器*（高 `body_ang_vel`，过强的 `action_rate`）。这里沿用的值来自「从任何地方爬起来」的版本 — 不要无缘无故地加强它们。
3. **头部零填充 vs `head_pose` 命令。** 如果策略以 `--standing` 部署并且有人操作头部按键，`infer_policy` 写入 `cmd[3:7] = head_offset`，策略看到分布外的内容。为保持 roller 约定而做的明确选择 ; 如果起身期间需要头部控制则重新审视。
4. **Frictionloss 0.05 远离真实值。** 0 → 2000 迭代的档位产生的策略无法迁移 ; 只有最后档位后的 checkpoint（iter 4000+）才能作为部署候选。

## 范围外

- 把起身整合进滚动策略（`velstand` 配方）— 在验证可行性后推迟决定。
- 侧躺起始 buckets。
- rough 变体 / 崎岖地形。
- 躯干/头部冲击惩罚。
- 对 `roller`、`roller_crouch`、`roller_slope`、`standup`、`velstand` env 或 `mdp.py` 的任何修改。