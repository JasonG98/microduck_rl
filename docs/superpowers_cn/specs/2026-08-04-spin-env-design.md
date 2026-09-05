# 规范 — 「Spin」Env（在 rollers 上原地快速旋转）

日期 : 2026-08-04。分支 : `new_pre_alpha_rollers`。

> **修订（首次 run 后）** : 首次校准 run（500 it.）表明机器人系统地在大约 1.16 s 时摔倒，远早于刹车。作为回应，目标减半 — `SPIN_RATE_MAX` 6.0 → **3.0 rad/s**，即**每周期 1 圈而非 2 圈** — 并把 `spin_stay_in_place` 加强到 **−3.0**，**速度上不做 curriculum**。证据和当前生效配置见「初始验证结果」。

## 目标

一个新的 RL 任务，教 rollers 上的 microduck **旋转** : 原地逆时针约 2 圈，转速约 6 rad/s（360°/s）*（初始目标 ; 降为 3 rad/s，见修订）*，然后干净停下并站立。**由相位驱动的循环动作**，部署在运行时的**单次按钮插槽**中，与现有任务 `roller_crouch` 一样。

## 已框定决策

| 问题 | 决策 |
|---|---|
| 支撑 | 在 rollers 上（`MICRODUCK_WALK_ROLLERS_ROBOT_CFG`，4 个被动轮子） |
| 操控 | 单次按钮插槽，命令 = 相位 `[cos(2πφ), sin(2πφ), 0]` |
| 目标 | ~6 rad/s，2 圈，然后刹车到停（初始目标 ; 降为 3 rad/s，见修订） |
| 进入状态 | 静止**或**慢速滚动（0 → 0.3 m/s） |
| 方向 | 仅向左（正偏航，逆时针） |
| 方法 | 「结果」目标（跟随 ω_z）+ 递减的反称引导 |

**运行时约束** : 插槽只发送 `[cos, sin, 0]` — 没有自由通道用于旋转方向。因此策略**总是向左转**。镜像策略之后可放入另一个插槽（按钮 B，`--fold-policy`）。

## 目标物理机制

在 4 个被动轮子上，原地「干净」旋转通过**差速滚动**完成 : 左冰刀向后移动，右冰刀向前（轮子**滚动**，不打滑）。这是一个*反称 swizzle* : 双腿互为反相，而不是经典 swizzle 的镜像。

逆时针旋转的符号验证（参照系 : x 向前，y 向左，z 向上 ; ω_z > 0） : 左侧（+y）的点速度为 `ω ẑ × y ŷ = −ω y x̂`，因此**向后**。4 个轮子前进时都正转（由 `test_wheel_direction.py` 验证），所以对于逆时针 spin :
`ω_左轮 < 0`，`ω_右轮 > 0`，即 **`ω_D − ω_G > 0`**。

## 采用的方法（C）及原因

考虑了三种方法 :

- **A — 纯「结果」目标** : 奖励偏航速度，让 PPO 找到动作。本仓库记录的风险 : 偷懒最优点 / 跳跃抖动而非干净滚动。
- **B — 按姿态「指令式」目标** : 两个剪刀姿态由相位插值，如 `roller_crouch`。*如果*姿态好则收敛快 ; 但对 crouch 它们是**从真实机器人读取的**，而这里动作未知。必须手动组合 : 昂贵且有风险（无有效力矩的姿态产生不了任何东西）。
- **C — A + 递减的反称引导** ← **采用**。A 的结构，加上两个微弱的 shaping 项，注入唯一确定的物理知识（差速滚动），其权重通过 curriculum 递减，让策略细化自己的动作。**泵送频率保持自由**。

## 架构

**文件** : `src/mjlab_microduck/tasks/microduck_spin_env_cfg.py`
- factory `make_microduck_spin_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`
- PPO 配置 `MicroduckSpinRlCfg`
- task id `Mjlab-Spin-Flat-MicroDuck`，在 `tasks/__init__.py` 中注册

克隆 `microduck_roller_crouch_env_cfg.py` 的结构 : roller 机器人，统一 61D obs，完整 DR，`action.scale = 1.0`，平地。

**`ENABLE_SYMMETRY = False`** — 强制 : 左右对称增强会把左转变成右转并毁掉学习。

**命令** : `GroundPickPhaseCommandCfg(period=4.0, randomize_phase=False)`。`period=4.0` 是 `--ground-pick-period` 的默认值 → 无需向运行时传任何内容。`randomize_phase=False` → 每个 episode 从 φ=0（站立）开始，如同部署。

## 相位包络

相位驱动一个**目标偏航速度** ω\*(φ)，在 4 段上呈梯形（周期 4 s，`SPIN_RATE_MAX = 6.0` rad/s — 初始目标 ; 降为 3 rad/s，见修订 ; 段和周期未变）:

```
ACCEL_END = 0.125   [0,     0.125)  0.5 s  ω* : 0 → 6 rad/s   (lancement, rampe linéaire)
HOLD_END  = 0.525   [0.125, 0.525)  1.6 s  ω* = 6 rad/s        (régime)
BRAKE_END = 0.650   [0.525, 0.650)  0.5 s  ω* : 6 → 0          (freinage, rampe linéaire)
            1.0     [0.650, 1.0)    1.4 s  ω* = 0              (repos debout)
```

*（上表 ω\* = 6 rad/s 对应 `SPIN_RATE_MAX` = 6.0，即初始目标 ; 生效值见修订。）*

单周期积分 : `0.5·3 + 1.6·6 + 0.5·3 = 12.6 rad ≈ 2.0 圈`。✅
*(在 `SPIN_RATE_MAX = 6.0`，初始目标。)* 一般形式 : 无论 `rate_max` 为何，积分都等于 `2.1 × SPIN_RATE_MAX`（0.25 + 1.6 + 0.25 = 2.1）。按生效目标（3.0 rad/s）: 每周期 `2.1 × 3.0 = 6.3 rad ≈ 1 圈` — 见修订。

Episode = 20 s = **5 个周期** : 机器人每 episode 重复启动 → 稳态 → 刹车 → 保持五次。每 episode 更多数据，且「保持」段也训练干净退出 trick。**（run 后注）** : 这在几何上仍成立（20 s / 4 s），但校准 run 中没有 episode 存活超过 ~1.16 s，仅覆盖第一个周期的一小部分 — 见「初始验证结果」。

**纯函数** `spin_rate_by_phase(phase, rate_max, accel_end, hold_end, brake_end)`
在 `mdp.py` 中，紧挨 `crouch_pose_blend`。无需模拟器即可测试。

**shaping 门控** : `gate(φ) = spin_rate_by_phase(φ) / rate_max ∈ [0, 1]`。在保持段为 0 → 此时没有引导推动剪刀动作，因此机器人回到中性站立。这正是把干净退出 trick 交给 roller 策略的东西。

## 奖励

### 在 mjlab 中验证过的陷阱（需显式处理）

- `body_ang_vel`（`body_angular_velocity_penalty`）只惩罚 **x/y**（`ang_vel_xy`，注释「Don't penalize z-angular velocity」）→ **保留**（权重 −0.05）: 抑制横滚/俯仰摇摆而不妨碍旋转。
- `angular_momentum`（`angular_momentum_penalty`）惩罚角动量的**3D 范数** → 会直接对抗旋转。**移除。**

### 新奖励（需在 `mdp.py` 中编写）

| Reward | 权重 | 定义 |
|---|---|---|
| `spin_rate_track` | 6.0 | `exp(−((ω_z − ω*(φ))/std)²)`，`std = 1.5` rad/s。ω_z = 躯干在机身坐标系中的偏航（IMU 所见）。主要目标。 |
| `spin_rate_l1` | 0.5 | `−|ω_z − ω*(φ)|` : 高斯远离目标饱和时的恒定梯度引导（与 `crouch_glide_pose_l1` 相同的技巧） |
| `spin_stay_in_place` | −3.0（最初 −1.0，见修订） | 躯干的 `‖v_xy‖²` → 「原地」，并消灭进入冲量。无参考状态，因此对每 episode 的 5 个周期鲁棒 |
| `spin_wheel_differential` | 1.0 | `gate(φ) · tanh(clamp(ω_D − ω_G, min=0) / omega_scale)`，`ω_G = (LF+LR)/2`，`ω_D = (RF+RR)/2` : 奖励刀片以与逆时针一致的相反方向滚动 → 通过**滚动**而不是打滑旋转。轮子按名解析（`passive_LF_?wheel`，…）。生效 `omega_scale = 17.0` rad/s（见下方校准段） |
| `leg_antisymmetry` | 1.0 → 0.25 | `gate(φ) · (−mean|q_G − q_D|)` 作用于 `hip_pitch` 和 `knee`。⚠️ 镜像约定 : *对称*姿态给出 `q_G + q_D ≈ 0`，因此**剪刀**是 `q_G ≈ q_D`。通过 curriculum 递减 |
| `spin_grounded` | 0.5 | `gate(φ) · 1[n_contact ≥ 2]` : 两片刀片贴地，防止「跳起来在空中转」。swizzle 的 `grounded_reward` 不能原样复用（它乘以 `cmd_x`，而这里 `cmd_x` = `cos(2πφ)`） |

**`omega_scale` 校准**（tanh 饱和标度） : 在目标稳态，每个冰刀行进 `v = ω_z · 半轮距`，因此每个轮子以 `v / r` 旋转，`r = 0.0175` m，差分为 `2 · ω_z · 半轮距 / r`。腿根在 rollers 模型中位于 `y = ±0.0175` m，但刀片更分开（踝偏移） : 真实半轮距需在首次 run 中在 sim 的 `left_foot` / `right_foot` sites 处**测量**。以估计半轮距 ~0.03 m 和 `ω_z = 6` rad/s，预期差分约为 ~20 rad/s — 因此初始默认 `omega_scale = 20.0`。**已完成测量（Task 3）** : 真实半轮距 = 0.0499 m，预期差分 = 34.2 rad/s，比估计高 71 % — 超过计划设定的 30 % 阈值。因此 `SPIN_WHEEL_OMEGA_SCALE` 被修正为 **34.0**（中间值，在目标仍为 6 rad/s 时生效 ; 之后重新校准为 **17.0**，见下方「更新」段）。半轮距测量详情见下方「初始验证结果」部分。

**更新（review 后修复波）** : `SPIN_RATE_MAX` 已从 6.0 降到 **3.0 rad/s**（人为决定，无 curriculum — 见下文）。这直接机械地影响 `omega_scale`，不是独立选择 : 稳态预期差分回归 `2 · 3.0 · 0.0499 / 0.0175` = **17.1 rad/s**。保持 `omega_scale = 34.0` 会把该项封顶到其自身最大值的 `tanh(17.1/34) = 0.47`，从而削弱的恰是我们想加强的 shaping。因此 `SPIN_WHEEL_OMEGA_SCALE` 被重新修正为 **17.0**，使用相同的实测半轮距（0.0499 m）作为参考。

### 从 `roller_crouch` 沿用的奖励（稳定性 / sim2real）

| Reward | 权重 |
|---|---|
| `upright`（躯干竖直） | 2.0 |
| `feet_flat`（刀片平贴） | −2.0 |
| `self_collisions` | −1.0 |
| `body_ang_vel`（仅 xy） | −0.05 |
| `action_rate_l2` | −1.0（curriculum −0.5 → −1.0） |
| `neck_action_rate_l2` | −0.5 |
| `joint_torques_l2` | −1e-3 |
| `neck_joint_pos_l2` **排除 `head_yaw`** | −0.2 |

**头部** : 颈部俯仰/横滚保持接近中性（sim2real），但 `head_yaw` **从该项排除** → 自由作为飞轮来启动旋转。实现 : `neck_joint_pos_l2` 硬编码通过正则 `.*(neck|head).*` 解析其关节 ; 因此要么给该函数加一个正则参数，要么写一个 `neck_joint_pos_l2_no_yaw` 变体。选择 : **给 `neck_joint_pos_l2` 添加一个 `pattern` 参数**（默认不变）以避免重复。

## Reset / 进入状态

```python
cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)
cfg.events["reset_base"].params["velocity_range"] = {"x": (0.0, 0.3)}
```

通过 `reset_root_state_uniform` 注入。**绝不**在 `mode="reset"` 中用 `push_by_setting_velocity` : 正是这个在 crouch 上产生了 NaN（对一个可能发散的根速度做 `root_vel +=` → 基础 free-joint 爆炸）。

## 域随机化

与 `roller_crouch` 相同，不偏离（仓库验证过的 sim2real 配方） : 躯干 COM + 头部，质量/惯量，BAM 关节摩擦，armature，轮子摩擦，每 3–6 s 推撞 0.2 m/s，IMU 错位 6°，encoder bias ±0.015 rad。

## 观察

与 roller / ground_pick / crouch **完全相同的 61D 布局** — ONNX 是否能加载进插槽的条件 :
`[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]`
其中 `command = [twist(3), head_pose(4), body_pose(6)]`，head/body 零填充。

因此 : 从 actor 移除 `base_lin_vel`（critic 侧保留），移除 `height_scan` 和 `foot_height`，critic 侧移除 `wheel_vel`，被动关节从 `joint_pos`/`joint_vel` 项中排除，延迟和噪声与 crouch 相同。

陀螺仪在 obs 中 → 策略**观察**自己的 ω_z : 该任务可观测。

## 终止条件

`time_out`，`fell_over`，`out_of_terrain_bounds`（继承）+ `nan_state`（`microduck_mdp.robot_state_is_nan`），如 crouch。

## Curriculum

| 项 | 阶段 |
|---|---|
| `action_rate_weight` | −0.5（0）→ −0.8（250 it.）→ −1.0（500 it.） |
| `leg_antisym_weight` | 1.0（0）→ 0.5（1500 it.）→ 0.25（3000 it.） |
| `com_range` | 0.003 → 0.005（500 it.）→ 0.01（1000 it.） |
| `head_com_range` | 0.003 → 0.005（500 it.）→ 0.01（1000 it.） |

（iterations × 24 步 /env，同其它 env）

**目标速度上无 curriculum** : 起手就是 6 rad/s *（初始目标 ; 降为 3 rad/s，仍无 curriculum，见修订）*。见「Plan B」。

## PPO

`MicroduckSpinRlCfg` = 复制 `MicroduckRollerCrouchRlCfg` : actor/critic（512, 256, 128）elu，obs normalization，自适应 lr 1e-3 PPO，`desired_kl=0.01`，`num_steps_per_env=24`，`symmetry_cfg=None`，`experiment_name="spin"`，`run_name="spin"`，`max_iterations=8000`。

## 测试

`tests/test_spin.py` — 纯函数，无需模拟器 :
- `spin_rate_by_phase` : 4 段边界处的值（0，rate_max，rate_max，0，0）
- 启动斜坡上单调增，刹车段单调减
- **单周期积分 ≈ 4π** 在 `rate_max = 6.0`（保证梯形的**形状**，每周期 `2.1 × rate_max` rad）— 自修订以来不再保护生效目标，见下一条。包络精确值 : 12.6 rad 对 4π = 12.566 → 容差 1 %
- **实际发出的目标**（`mdp.SPIN_RATE_MAX`）无论 `rate_max` 为何都确实积分到每周期 `2.1 × SPIN_RATE_MAX` rad — 在 7d916aa 添加，正是这个测试在目标改变而未考虑圈数时会失败。按生效值（3.0 rad/s）: 6.3 rad ≈ 1 圈
- `gate(φ) = 0` 在整个保持段，其它地方 `∈ [0,1]`

`tests/test_spin_cfg.py` — env 能构建 :
- 命令 = `GroundPickPhaseCommand`，`period == 4.0`，`randomize_phase is False`
- `"angular_momentum" not in cfg.rewards`（rewards 部分的陷阱）
- `symmetry_cfg is None`
- actor obs 维度 == 61
- **观察项顺序与 `roller_crouch` 精确一致**（actor + critic），逐组 — 在 7d916aa 添加，是导出的 ONNX 能否加载进运行时插槽的严格条件

运行 : `uv run --with pytest pytest tests/ -q`

## 训练 / 部署

```bash
uv run train Mjlab-Spin-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 8000
# surveiller Episode_Reward/spin_rate_track (doit monter)
uv run scripts/play_latest.py     # alias md-play
uv run scripts/export_latest.py   # ONNX, normaliseur d'obs baké
```

```bash
microduck_runtime --variant pre-alpha --new-cmd-obs --roller \
  --model output.onnx --new-dxl-imu --kp 200 --action-scale 0.8 \
  --ground-pick spin.onnx \
  --ground-pick-period 4.0 \      # = SPIN_PERIOD
  --ground-pick-kp-ratio 1.0 \    # défaut 0.6 -> forcer 1.0 (entraîné kp 200)
  --ground-pick-action-scale 0.8  # matcher action_scale runtime
```

按钮 **A** → spin，然后自动回到 roller 策略。

## 成功标准

在 play 中 : ~2.6 s 内逆时针约 2 圈，躯干漂移 < ~10 cm，机器人全程站立，下一周期前保持段内中立站稳定。*（该标准是针对 6 rad/s / 2 圈的初始目标设定的 ; 在 3 rad/s 时，见修订，将是稳态时长内的 ~1 圈 — 标准未修订，因为机器人还撑不到那。）*

## 若训练停滞的 Plan B

按序 :
1. **速度 curriculum** : `SPIN_RATE_MAX` 3 → 6 rad/s（需要让 `rate_max` 通过 reward 参数上的 `CurriculumTermCfg` 可控制）。**部分跟进** : 校准 run 之后，目标确实降到 3 rad/s（见修订），但**无 curriculum** — 3 rad/s 目前是固定目标，不是向 6 爬升的起点。人类选择先看看机器人在这速度下能做到什么，再考虑逐步提高。
2. 提高 `spin_wheel_differential` 并推迟 `leg_antisymmetry` 的递减。
3. 放宽 `spin_rate_track` 的 `std`（1.5 → 2.5）以获得更远的有效梯度。
4. 最后手段，切换到方法 B（在姿态编辑器中手动组合剪刀姿态）来引导动作，然后释放。

## 范围外

- 右转（镜像策略放进另一个插槽）— 之后。
- 步行变体（无 rollers）。
- 连续速度命令的 spin（需要运行时命令通道）。

## 初始验证结果

### 实测半轮距与 `omega_scale`

半轮距在 rollers 模型的 `left_foot` / `right_foot` sites 处测得 : **0.0499 m**，相对 spec 的估计 0.03 m。稳态（6 rad/s）预期轮差 : `2 · 6.0 · 0.0499 / 0.0175` = **34.2 rad/s**，比默认 20.0 高 71 % — 超过计划设定的 30 % 阈值。因此 `SPIN_WHEEL_OMEGA_SCALE` 从 20.0 改为 **34.0**。测试仍显式传 `omega_scale=20.0`，以保持独立于该常量。

### Smoke run（Step 2 : 5 次迭代，64 envs，NaN 保护）

无异常完成。`Episode_Termination/nan_state` 全程保持 0.0000，`/tmp/mjlab/nan_dumps/` 从未被创建。六个 spin 奖励确实出现在记录的 `Episode_Reward/` 键中 : `spin_rate_track`，`spin_rate_l1`，`spin_stay_in_place`，`spin_wheel_differential`，`spin_grounded`，`leg_antisymmetry`。

观察一致性（Step 1） : spin env 的 actor obs 项列表**与 `roller_crouch` 相同** — 8 项，同顺序 : `base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, command, head_command, body_command`。这是导出的 ONNX 能加载进运行时插槽的条件。

**要记住的用法说明** : 计划中示例命令用裸标志 `--enable-nan-guard` 会被本仓库的 CLI 拒绝 — 必须传 `--enable-nan-guard True`。

### 500 次迭代校准 run（Step 3）

4096 envs，500 次迭代，~2,32 s/迭代，退出码 0，wandb logger（因此 `scripts/play_latest.py` / `md-play` 能找到该 run）。

**真正确立了什么** : `Mean episode length` = **57.83 步**，在 1000 步的 episode（20 s @ 50 Hz）上，即 **~1,16 s**。`Episode_Termination/fell_over` ≈ **70**，`time_out = 0.0000`，`nan_state = 0`。机器人**每个 episode 都摔**，在相位 φ ≈ 0,29 — 正好在稳态段的中间。它从未到达刹车（φ ≥ 0,525）或保持（φ ≥ 0,650） : **周期的 71 % 从未被训练**。

episode 长度在 run 期间从 23,98 升到 57,83 步 : 因此 `Episode_Reward/spin_rate_track` 的上升（0,0291 → 0,3168）主要反映**存活的延长**，而非跟随的改善。该步骤在计划中陈述的成功标准（「曲线应上升」）**不是**该项的有效信号 : 一个完全静止的机器人已经在该项上拿到 `6.0 × 0.405 = 2.43` — 保持段为一动不动地站着支付全额费用，所以任何存活更久的策略都会机械地捕获更多该段，与跟随质量无关。

### 派生的诊断 — 估计，不是直接测量

以下值来自最后一个日志块中奖励项之间的比值，这抵消了 logger 施加的未知归一化因子。当作估计来看，且可复现自同一方法 :

**成立的部分** : 在它保持站立的 ~1,2 s 内，机器人相当接近地跟随目标。比值 `spin_rate_l1 / spin_rate_track`（−0,0097 / 0,3168，权重 0,5 和 6,0，`std = 1.5`），解 `e = 0.3674 · exp(−(e/1.5)²)` : 偏航速度跟随的平均绝对误差 ≈ **0,35 rad/s**，由两条独立路径确认 — 该比值 `spin_rate_l1 / spin_rate_track`，以及从 reward manager 归一化的逆推。它**能启动** spin ; 它**无法在旋转中保持站立**。

**不成立的部分** : shaping 块（`spin_wheel_differential` 1,0，`spin_grounded` 0,5，`spin_stay_in_place` −1,0）合计权重约 ~1,0，相对主要目标的 6,0 — 约为**13 %**，是打滑策略通过忽略该块所放弃的。而且 `spin_wheel_differential` 对**瞬时旋转中心不变** : 以 6 rad/s 居中的 spin 和以左冰刀为枢轴、6 rad/s 的 pivote 都产生 34,2 的差分 — 因此该项**不**编码居中滚动，只有 `spin_stay_in_place` 编码。`spin_stay_in_place` ≈ −0,0069 意味着 `‖v_xy‖ ≈ 0,35 m/s` : 机器人仍在平移，一致于偏心枢轴（以冰刀为枢轴）而不是围绕身体中心旋转。

### 基于此诊断决定的配置更改

目标减半 — `SPIN_RATE_MAX` 6.0 → **3.0 rad/s** — 并把 `spin_stay_in_place` 加强 −1.0 → **−3.0**（见 rewards 表和上方重新校准到 17.0 的 `SPIN_WHEEL_OMEGA_SCALE`）。**刻意在目标速度上不做 curriculum** : 这是一个首次试验，先看看机器人在速度减半时能做到什么，再视需要考虑逐步提高。

**降低启动期间漂移成本。** 把 `spin_stay_in_place` 加强到 −3.0 使 review 指出的一个缺陷更尖锐 : 该项是 spin 中唯一不受相位调制的项，因此它以全价收取启动斜坡期间过渡性平移 — 恰是机器人必须向地面推撞以注入角动量的时刻，也是进入冲量（最高 0.3 m/s）必须被**转换**为旋转的时刻。该成本现在在 `[0, ACCEL_END)` 上乘以 `SPIN_LAUNCH_DRIFT_SCALE = 0.2`，之后为全价。它**刻意**不像引导项那样在保持段关闭 : 那里静止才是真正的标准。

Step 4（观察动作）仍待完成，留给人类。

⚠️ 这四个测试（三个新的关于衰减、一个修改）**尚未**运行 — 机器人在该 commit 时被分配到别的事。任何长 run 之前先运行 : `uv run --with pytest pytest tests/test_spin.py tests/test_spin_cfg.py -q`。