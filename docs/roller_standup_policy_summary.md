# Policy `roller_standup` — 穿着轮滑站起来

**目标**: microduck (穿轮滑) 从地面出发 — 俯卧或仰卧 — 然后**站到轮子上**, 接着**保持**站立.

- **任务**: `Mjlab-RollerStandUp-Flat-MicroDuck`
- **文件**: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- **基础**: 派生自 roller env (`velocity_rollers`) → 同一机器人, 同一物理/DR, **同一 61D 观测** (运行时可互换, 通过 `--new-cmd-obs` 加载).
- **Spec**: `docs/superpowers/specs/2026-08-04-roller-standup-design.md`
- **盲策略**: 无地形扫描; 本体感知 + `projected_gravity`.

## 高度 (实测, 非猜测)

| 姿态 | 脚模型 | 轮滑模型 |
|---|---|---|
| 站立 | 0.1172 → 负载下 `STAND_Z=0.115` | 0.1407 → **`ROLLER_STAND_Z=0.138`** |
| 俯卧 (静止) | 0.075 | 0.075 |
| 仰卧 (静止) | 0.048 | 0.048 |

两种模型在地面上的静止高度相同: 触地的是躯干外壳, 不是脚.

## ⚠️ 关节索引 — 轮子是交错的

```
0-4   左腿           5-6   左轮
7-10  颈/头          11-15  右腿           16-17  右轮
```
`_LEG_JOINTS = [0-4, 11-15]`. `standup` 的索引 (`[0-4, 9-13]`) 只适用于**无轮**模型, 在这里会指向轮子. 由 `tests/test_roller_standup_cfg.py::test_joint_indices_match_actual_roller_model` 锁定.

## Reset — 地面起步

`set_random_ground_state`: 俯卧 (`prone_z` 0.076–0.09, 抬高了下限因为俯卧只有在 0.0752 才离地) / 仰卧 / **已经站立** (`standing_z` 0.134–0.144), ± 10° 的 pitch/roll 噪声. 没有 "坐" 桶. "站立" 桶是必需的: 没有它策略能站起来但稳不住.

**课程 `ground_state_mix`** (易 → 难, 仰卧最后):

| iter | 站立 | 俯卧 | 仰卧 |
|---|---|---|---|
| 0 | 0.50 | 0.50 | 0.00 |
| 600 | 0.35 | 0.45 | 0.20 |
| 1500 | 0.25 | 0.40 | 0.35 |
| 2500 | 0.20 | 0.40 | 0.40 |

## 奖励

从 `standup` 沿用十个项, 权重已调好: `pose_stand_legs` (+8), `pose_stand_l1` (+5), `height_stand` (+4, std 0.04), `height_stand_sharp` (+4, std 0.015), `height_stand_l1` (+30), `com_upward_velocity` (+3), `gentle_rise` (−0.02), `upright_linear` (+6), `upright_sharp` (+6), `standing_composite` (+15). 加上 `joint_torque_rate_l2` (−2e-3), 即不阻碍翻身的抗抖动项.

继承的正则项: `body_ang_vel` **−0.05** (动作阻断项, 保持轻量), `angular_momentum` −0.02, `action_rate_l2` (斜坡 −0.4 → −1.0, **不是** roller 的 −2.0), `neck_action_rate_l2` −0.5, `neck_joint_pos_l2` −0.5 (头部朝正), `joint_torques_l2` −1e-3, `action_over_limit` −0.5, `self_collisions` −1.0.

移除项: 所有滑行奖励, 以及 `feet_flat` (起身过程中刀片并不平贴) 和 `hip_roll_neutral` (起身需要分开双腿).

## ⚠️ 难点: 轮子会滚

没有任何纵向抓地力来蹬地. **滚动摩擦课程是反向的** (roller env 让它上升, 这里让它下降):

| iter | frictionloss | |
|---|---|---|
| 0 | 0.05 | 轮子几乎锁死 → 像有脚一样起身 |
| 1000 | 0.02 | |
| 2000 | 0.008 | |
| 3000 | 0.003 | |
| 4000 | 0.0015 | 真实的滚动摩擦值 |

**在各阶段边界关注 `Episode_Reward/standing_composite`.** 如果它崩塌, 说明 "有脚抓地" 的动作无法迁移到自由轮上 → 需要引导一种滑冰者技巧 (中间用膝盖支撑, 一次一只滑冰). 这是一个结果, 不是失败.

**还要在 play 时关注机器人的水平漂移**, 在每个摩擦阶段. `standing_composite` 既看不到 `root_link_pos_w[:2]` 也看不到水平速度: 一个滑到远处才起身的策略和一个起身即停的策略得分完全一样. 只要这个漂移没有被肉眼测量过, 摩擦课程的结果 (这正是这个 env 要回答的问题) 就不可靠.

**Sim2real**: 只有 iter 4000 之后的 checkpoint 才是部署候选. 在此之前策略依赖的是真实机器人上不存在的摩擦.

## 命令

`twist` 槽中性化: `lin_vel_x`/`lin_vel_y` ± 0.01, `ang_vel_z` **± 0.05** (5× 更宽 — 与 `standup` 同一选择). `head_pose` / `body_pose` 槽**零填充** (roller 约定). 目标部署: 以 `--standing` 面对 roller 策略的 `--walking`, 通过命令幅值自动切换 (`infer_policy.py:262`, 阈值 0.05); twist 槽在那里保持为零 (`infer_policy.py:239`).

**保留意见**: `infer_policy.py` 是本地 sim/键盘脚本. 机器人运行时是 Rust 二进制 `microduck_runtime`, 不在本仓库中 — 未经验证它是否暴露等价的 `--standing`. crouch 交接文档只列出 `--model`, `--ground-pick`, `--fold-policy`. 待确认.

## 终止条件

`fell_over` **已移除** (机器人从摔倒状态起步). 继承 `nan_state`. actor/critic obs 上 `nan_policy="sanitize"`.

## 网络 / PPO

Actor 和 critic `(512, 256, 128)` elu, `obs_normalization=True`. PPO `lr=1e-3` 自适应, `desired_kl=0.01`, `gamma=0.99`, `lam=0.95`, `num_steps_per_env=24`, episode 6 s, `max_iterations=15000`. **对称性 OFF** (`SYMMETRY_CFG` 是为 51D 布局写的).

## 命令

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
uv run scripts/play_latest.py        # alias md-play
uv run scripts/export_latest.py      # alias md-export
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```

### ⚠️ 在 play 时查看仰卧起步

play 默认**从不**显示仰卧起步: play env 是全新构建的, 所以
`common_step_counter` 从 0 开始, 课程应用其阶段 0, 此时
`face_up_prob = 0`. 无论加载的 checkpoint 多成熟, 看到的都是 50% 俯卧 / 50% 站立.
而仰卧恰恰是最难的情况, 正是我们要检查的.

`STANDUP_PLAY_FACE_UP` 强制混合 (与 `roller_slope` 中的
`SLOPE_PLAY_DIFFICULTY` 同一模式), **仅在 `play=True` 路径上生效** —
训练及其易 → 难课程不受影响:

```bash
STANDUP_PLAY_FACE_UP=1.0 md-play    # 100% 仰卧起步
STANDUP_PLAY_FACE_UP=0.4 md-play    # 课程最后阶段的混合比例
STANDUP_PLAY_FACE_UP=none md-play   # 默认 (阶段 0, 无仰卧)
```

剩余 (`1 - face_up`) 按最后阶段 2:1 的俯卧:站立比例分配,
因此 `0.4` 精确复现训练末期的混合 (0.40 / 0.20 / 0.40).

## 🔧 抗暴力修正 (首次上机后)

**症状** (4000+ checkpoint 上): 动作非常猛烈, 头部撞地,
机器人上无法从仰卧起身. **仿真中也存在** → 因此既不是 sim2real 问题,
也不是 checkpoint 太年轻, 而是奖励设计问题.

**根因: `gentle_rise` 在奖励暴力.** `trunk_vertical_accel_penalty`
已经返回 `-|a_z|` (`mdp.py:2171`); 乘上从 `standup` 继承的 **−0.02** 权重,
变成双重负号, 即 `+0.02·|a_z|` — **躯干加速越猛烈, 策略越被奖励**.
日志确认: run `vweolw91` 上 `Episode_Reward/gentle_rise = +0.0118`,
是唯一一个 logged 为正的惩罚项.

`mdp.py` 混用了两种符号约定, 这正是陷阱:

| 项 | 函数返回 | 正确权重 |
|---|---|---|
| `height_stand_l1`, `pose_stand_l1`, `gentle_rise` | `-abs(...)`, 已经为负 | **正** |
| `joint_torques_l2`, `joint_torque_rate_l2`, `action_rate_l2`, `body_impact_cost` | 正的幅值 | **负** |

由 `test_already_negative_penalties_use_positive_weights` 锁定.

⚠️ **行走器的 `standup` 有完全相同的 bug** (同一函数, 同一 −0.02 权重).
这解释了其注释中记录的一系列失败的阻尼尝试 ("*violent / shaky / overshoot-tip-repeat on the real robot*"): 它们对抗的是一个朝反方向主动推动的项. **此处未修** —
那是另一个 env, 需单独处理.

**关联的结构性问题.** 收敛时任务奖励合计 **≈ +41.6** 饱和到 95–99%,
而所有阻尼项合计只有 **≈ −1.2** — 其中 `joint_torque_rate_l2` 为
**−0.0002/步**, `joint_torques_l2` 为 **−0.0001/步**, 即几乎为零.
比例约 35:1: 没有理由保持温柔.

**当前修正状态:**

| | 之前 | 现在 | 原因 |
|---|---|---|---|
| `gentle_rise` | −0.02 (奖励) | **+0.02** (惩罚) | 符号修正; 幅值有意保持小 — 翻身过程中 `\|a_z\|` 必然很高, 大权重会变成动作阻断项 |
| `joint_torque_rate_l2` | −2e-3 | **−0.2** | 安全的杠杆: 惩罚力矩变化, 而非动作本身 |
| `head_impact_penalty` | 缺失 | **仍然缺失** | 试过 −1.0, 冻结了策略 — 见下 |

### ⚠️ 头部撞击惩罚冻结了策略 — 不要原样恢复

用 `velstand` 的值尝试 (`body_impact_cost`, `neck` 子树, −1.0,
阈值 2.0): **策略收敛到躺着不动.** 实测 (run `d8rnko6p`):

| 项 | 之前 (暴力) | 加 head_impact (冻结) |
|---|---|---|
| `standing_composite` | +14.32 | **+3.26** |
| `upright_sharp` | +5.76 | +1.06 |
| `head_impact_penalty` | — | **−1.01** ← 最大负项 |
| `joint_torque_rate_l2` | −0.0002 | −0.255 (因此**不是**元凶) |

推理错误: 以为 "定向" 惩罚不会压制动作.
**在这里是错的 — 从仰卧起身时, 这个机器人以头和肩为支点旋转.** 头部
是翻身的着力点, 不是附带损伤; 惩罚它会封锁唯一可用的机制,
而仰卧本来就是会失败的情况.

**使这种冻结成为可能的懒惰最优**: `pose_stand_legs` 在机器人
躺着时仍保持 **+7.72 / 8** — 腿在俯卧位置就在 HOME, 因此这项奖励几乎免费拿到.
是 `height_stand_l1` (权重 +30) 必须让 "留在地面" 明显为负; 不要削弱它.

**正在验证的假设**: 撞头是暴力的*症状* (符号 bug 在奖励猛烈, 而猛烈起身
最终会落到头上), 而非独立缺陷.
如果符号修正后撞头复发, 恢复方式应是一个**高度门控**的惩罚
(类似 `upright_sharp`), 避开地面翻身阶段.

**方法论教训**: 三项修正是同时施加的, 因此无法确定性地归因冻结 —
只能指出最可疑的. 今后一次只改一项.

**如果仍然暴力的再校准**: `|Δτ|²` 收敛时约 0.1, 因此
`joint_torque_rate_l2` 贡献 ≈ `0.1 × |权重|`. 提**这一项**, **不要**
提 `body_ang_vel` (−0.05) 或 `action_rate_l2` (斜坡 → −1.0): 那些是动作阻断项,
`standup` 文档记录在 −0.15 和 −1.2 时它们会**冻结**仰卧起身. 反之如果仰卧
不再工作, **首先降低** `joint_torque_rate_l2`.

## 范围外

将起身集成进滑行策略 (`velstand` 方案); 侧躺起步桶; rough 变体; 躯干/头部撞击惩罚.

没有任何奖励惩罚躯干水平速度 (`root_link_lin_vel_w[:, :2]`): "起身同时滑到远处" 是一个不受惩罚且拿满分的结果. 有意为之 (不是遗漏): 一个不按高度门控的静止奖励也会惩罚从地面起身所必需的平移 — 即 `standup` 文档记录的 "动作阻断项" 失败模式. 若问题确认可考虑: 按高度门控的静止 (仅在接近 `ROLLER_STAND_Z` 时生效).
