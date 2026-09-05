# 设计 — Roller 下蹲滑行（按按钮「边滑边蹲下」）

**日期 :** 2026-07-17
**状态 :** 设计已验证，可用于实施计划

## 背景

microduck 机器人会滑冰（roller 策略，任务 `Mjlab-Velocity-Flat-MicroDuck-Rollers`）。
我们想要一个新的动作：在按下按钮时，它**下蹲并继续滑行**（像滑冰者处于低位姿态），保持约 1 秒，然后**自己站起来**并恢复滑冰。

用户的硬约束：**不要修改 Rust 运行时**（`apirrone/microduck_runtime`，以二进制方式安装）。因此该动作必须复用运行时中已有的机制。

**关键发现：** 运行时已经有一个「按按钮触发的单次行为」插槽：`--ground-pick`。它由**按钮 A**（上升沿）触发，在固定时长内执行一个由**相位**驱动的 ONNX 策略，然后自动回到主策略。关键是，它使用**与 roller 策略完全相同的 61D 观察布局** —— 两者在运行时是可互换的。这就是理想的载体，无需改动一行 Rust。

已接受的权衡：该动作是**单次**的（固定时长，没有「持续按住开关」）。下蹲的时长由该插槽的周期决定。

## 采用的方法（方法 B）

创建一个**新的 mjlab 任务**，在 rollers 机器人上训练，执行 下落 → 下蹲滑行 → 回升，由 ground-pick 插槽的相位驱动。导出为 ONNX 并通过 `--ground-pick` 加载。无需修改 Rust。

### 涉及的文件

| 文件 | 操作 |
|---|---|
| `src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py` | **新建。** 该 env，roller + ground-pick 的混合。 |
| `src/mjlab_microduck/tasks/mdp.py` | **新增** reward `crouch_glide_height_by_phase`。 |
| `src/mjlab_microduck/tasks/__init__.py` | **新增** : 注册 `Mjlab-RollerCrouch-Flat-MicroDuck`。 |

### 复用（不要重新发明）

- **物理 / roller 机器人** ← `microduck_velocity_rollers_env_cfg.py` :
  `MICRODUCK_WALK_ROLLERS_ROBOT_CFG`（14 个活动关节 + 4 个被动轮子），
  `roller_blade` 上的接触传感器，滚珠轴承摩擦的 DR（`randomize_wheel_friction` + curriculum），14 维 obs（通过 `SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))` 排除轮子），`action.scale=1.0`，`kp_fw=200`。
- **阶段 / 单次机制** ← `microduck_ground_pick_env_cfg.py` :
  命令 `microduck_mdp.GroundPickPhaseCommand` **原位复用**（生成运行时将发送到 twist 插槽的 `[cos(2πφ), sin(2πφ), 0]`），head/body 补零（`zero_command_padding`），终止条件 `robot_state_is_nan`，`reset_action_history`。
- **DR sim2real** ← 不作修改地从 roller env 沿用（obs 级 IMU mismatch，encoder bias，质量/惯量，BAM 摩擦，armature，轻柔推撞 ±0.2）。

## 核心：由相位驱动的「梯形」高度目标

唯一真正的新颖之处。不再让嘴巴下降（ground-pick），而是根据相位驱动**躯干高度**（`com_height` of `trunk_base`），并带有一个低位平台：

```
hauteur
 haute ┐                    ┌──   debout (rend la main à la policy roller)
       │ \                 /
  basse│  \_______________/       accroupi + glisse (palier 1 s)
       └───────────────────────► phase
       0   0.375      0.625   1
```

- φ ∈ [0, 0.375] : 向蹲姿高度下降
- φ ∈ [0.375, 0.625] : **保持下蹲**（= 4 秒周期中的 1 秒）→ 滑行
- φ ∈ [0.625, 1.0] : 向 roller 站立姿态回升

**新的 reward `crouch_glide_height_by_phase(env, command_name, height_low,
height_high, hold_lo=0.375, hold_hi=0.625, std=...)`** 在 `mdp.py` 中 :
从命令读取相位，计算目标高度（高→低→高 插值，在平台上保持恒定），
奖励 `exp(-((h_测量 - h_目标)/std)²)`。
参考现有的 `com_height_target`（mdp.py:694）以及已存在的 `interpolated/multistage height target`。

初始值 : `height_high ≈ 0.11` m（roller 站立高度，参考 roller 的 `com_height_target` 带 0.0935–0.1235），`height_low ≈ 0.075` m（蹲姿；需在 play 中微调）。相位从命令的 `atan2(sin, cos)` 重建。

## 奖励

| Reward | 作用 | 来源 |
|---|---|---|
| `crouch_glide_height_by_phase` | 主要目标（高→低→高） | **新** |
| `wheel_speed`（权重降低 ~2–3） | 保持冲量，下蹲时不要刹车 | roller env（`wheel_speed_reward`） |
| `upright`（≈2），`body_ang_vel`（−0.05），`angular_momentum`（−0.02） | 平衡 / 稳定 | roller env |
| `return_pose`（阶段末尾） | 收敛到 roller 站立姿态，干净地交还控制 | 改编自 `ground_pick_return_pose` |
| `feet_flat`（−2） | 刀片平贴 → 稳定滑行 | roller env |
| `action_rate_l2`, `neck_action_rate_l2`, `joint_torques_l2`, `self_collisions` | 平滑 / sim2real 迁移 | 两个 env |

**明确不包含 :** `braking`（我们不想停下），`mouth_ground_proximity` / `mouth_perpendicular_to_ground`（不触地），`skating_air_time` / `single_support` / `glide`（做动作期间没有步伐 —— 我们被动滑行）。

## 训练

- `MicroduckRollerCrouchRlCfg` = 复制 `MicroduckRollersRlCfg`
  （MLP 512/256/128，ELU，obs_normalization，PPO，`experiment_name="roller_crouch"`）。
- 在 `tasks/__init__.py` 中注册 :
  `register_mjlab_task(task_id="Mjlab-RollerCrouch-Flat-MicroDuck", ...)`。
- 启动 :
  ```bash
  uv run train Mjlab-RollerCrouch-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 8000
  ```
- 事件以**现实的进入速度**开始（机器人滚着到来），否则它在蹲下时就没有冲量可保持了。通过一个 reset event（非零初始速度）或 episode 开头的 push 来接线。

## 导出 + 部署（精确的运行时参数）

导出 ONNX（归一化器已由 `export.py` 烘焙），然后 :

```bash
microduck_runtime --variant pre-alpha --new-cmd-obs --roller \
  --model output.onnx \
  --new-dxl-imu --kp 200 --action-scale 0.8 \
  --max-linear-vel 0.6 --max-linear-vel-backward 0.5 --max-angular-vel 0.0 \
  --ground-pick roller_crouch.onnx \
  --ground-pick-period 5.0 \
  --ground-pick-kp-ratio 1.0 \
  --ground-pick-action-scale 0.8
```

按钮 **A** → 下蹲滑行，然后自动回到 roller 策略。

**训练/部署一致性陷阱（对 sim2real 重要） :**
- `--ground-pick-kp-ratio 1.0` : 默认是 **0.6**（做动作期间把 kp 降到 120）。我们在 kp=200 下训练 → 必须强制为 **1.0** 才能匹配。
- `--ground-pick-action-scale` 必须匹配训练的 `action_scale`（上面是 0.8）。
- `--ground-pick-period 5.0` 必须匹配训练的运动周期/时长（默认 4.0，我们保留它）。

## 风险与验证

- **单次，固定时长 :** 蹲姿持续 `ground-pick-period` 然后自动回升。没有自由保持 —— 方法 B 的既定限制。
- **做动作期间的冲量 :** 相位替换了速度命令 → 蹲下期间**没有主动推力**。如果进入冲量太弱，它会减速。因此要用现实的进入速度训练。
- **验证 :**
  1. 在模拟中（`play`）: 它下沉，保持轮子在平台期间转动，不摔倒地站起来，最终姿态干净地与 roller 站立姿态衔接。
  2. 在真实机器人上 : 以低速滑行，按下 A，观察。
  3. 确认 roller 策略在返回后能干净地接管。

## 待实现时确认的开放问题

- `height_low`（蹲姿）的确切值 —— 需在 play 中调整。
- 注入进入速度到 episode 的最佳方式（event reset 还是初始 push）。
- `wheel_speed` 与 `crouch_glide_height_by_phase` 的相对权重（保持冲量而不妨碍蹲下）。