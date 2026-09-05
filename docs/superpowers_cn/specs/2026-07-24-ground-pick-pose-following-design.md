# Ground-pick 通过按相位插值的姿态跟随

**日期** : 2026-07-24
**分支** : `new_pre_alpha_ground_pick`
**目标文件** : `src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py`（原位重写）
**Task id** : `Mjlab-GroundPick-Flat-MicroDuck`（不变）

## 1. 目标

将当前 ground_pick 的*任务空间*目标（奖励嘴巴下探到地面，然后分别奖励起身回站立）替换为**姿态跟随的目标** : 定义两个关节姿态目标 — STAND 和 DOWN — 并奖励跟随**按相位插值的姿态**（STAND→DOWN→STAND）。

动机（沿用 roller_crouch 的方法，已验证） : 按插值姿态的目标**在构造上是对称的** — 「起身」（目标 → STAND）与「下蹲」（目标 → DOWN）得到完全相同的奖励，这解决了一个偷懒最优化问题，即策略能下蹲却起来得不好。信号在每个相位都是**稠密的**（目标连续移动），与固定的由 `sin` 加权的目标不同，后者在过渡阶段不提供任何信号。

该动作仍通过运行时的 `--ground-pick` 插槽在**按钮 A** 上触发（单次，自动回到主策略）。统一的 61D obs 不变 → 策略在插槽中可互换。

## 2. 目标姿态

按**名称**解析关节（`asset.find_joints([name])`）— 鲁棒，与 roller 方法一致。14 个关节（excluded mouth）。

- **STAND_POSE** = HOME（模型的 `default_joint_pos`）。混合来源 ; 不要硬编码重新定义 — 使用模型默认作为来源（blend=0）。部署时，主策略从 HOME 恢复 → 干净返回。

- **DOWN_POSE** = 来自 `scene_walk.xml` 的 **FOLD keyframe** 的初始值（深度前折，头低垂 → 嘴巴朝地）。文件头部的按名 dict，**注释为可由 `read_pose.py` 对真实机器人嘴巴朝地放置的读取替换**。初始值 :

  ```python
  DOWN_POSE = {
      "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 1.57,
      "left_knee": 1.57, "left_ankle": 0.0,
      "neck_pitch": 1.0, "head_pitch": 1.0, "head_yaw": 0.0, "head_roll": 0.0,
      "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -1.57,
      "right_knee": -1.57, "right_ankle": 0.0,
  }
  ```

## 3. 相位轮廓（4 段）

命令 `GroundPickPhaseCommand` : `[cos(2πφ), sin(2πφ), 0]`，周期 **4.0 s**（运行时插槽默认 → 部署时无需修改周期标志）。

```
DESCENT_END=0.15  HOLD_END=0.50  RISE_END=0.65   (période 4 s)
[0, 0.15)     descente  STAND->DOWN   ~0.6 s   blend 0->1
[0.15, 0.50)  bas       DOWN          ~1.4 s   blend 1
[0.50, 0.65)  remontée  DOWN->STAND   ~0.6 s   blend 1->0
[0.65, 1.0)   haut      STAND (repos) ~1.4 s   blend 0
```

`blend ∈ [0,1]` : 0 = STAND（HOME），1 = DOWN。目标 = `stand + blend·(down - stand)`。边界可调（文件头部的常量）。

**`randomize_phase=False`** : 每个 episode 从 φ=0（= 站立）开始，如同部署时的按钮 A 触发。由于 episodes 在不同时刻重置，envs 在相位上自然地相互去相关（无需随机化）。需要在 `GroundPickPhaseCommandCfg` 上添加一个 `randomize_phase` 标志（默认 `True` → 其它 sit/stand 任务不变），在 `reset()` 中遵循。

## 4. 新的 mdp 函数（从 roller 移植，适配，按名）

位于 `src/mjlab_microduck/tasks/mdp.py`。名称与现有的 `phase_pose_match`（它是固定目标乘 sin 加权的变体）区分开，避免混淆。

- **`phase_pose_blend(phase, descent_end, hold_end, rise_end) -> Tensor`** — 纯函数，4 段 0..1 的 blend（可隔离测试）。
- **`_phase_pose_error(env, asset_cfg, command_name, target_pose, descent_end, hold_end, rise_end, source_pose=None) -> (cur, target)`** — 按名解析关节 ; `source_pose` = HOME（`default_joint_pos`）如果 `None` ; 计算 `phase = atan2(sin,cos)/2π % 1`，`blend`，然后 `target = source + blend·(target_pose - source)`。
- **`phase_pose_track(env, command_name, target_pose, source_pose=None, std=0.3, descent_end, hold_end, rise_end, asset_cfg) -> Tensor`** — 高斯 `exp(-((cur-target)/std)²).mean(-1)`。
- **`phase_pose_track_l1(env, ...同样args无std...) -> Tensor`** — 引导 `-(cur-target).abs().mean(-1)`（当高斯饱和时恒定梯度）。

`target_pose` = `DOWN_POSE`（按名 dict）。`source_pose=None` → HOME。

## 5. 奖励

相对当前进行最小重写 — 我们替换姿态返回机制，保留稳定性/正则化/sim2real。

| Reward | 权重 | 状态 | 作用 |
|---|---|---|---|
| `phase_pose_track`（std 0.3） | **6.0** | **新** | 按相位插值姿态 STAND↔DOWN 跟随 |
| `phase_pose_track_l1` | **2.0** | **新** | L1 引导 |
| `mouth_ground_proximity`（std 0.10） | **1.0** | 重新调优（原为 2.0） | 安全网 : 如果 DOWN 不完美则确保嘴巴着地 ; 接近过程门控（+sin） |
| `upright` | 0.2 | 保留 | 躯干 ~竖直（低，机器人会倾斜） |
| `feet_grounded` | 3.0 | 保留 | 整个动作期间两脚踩地 |
| `self_collisions` | -1.0 | 保留 | |
| `head_impact_penalty`（阈值 2 N） | -0.5 | 保留 | 头部不重击（DOWN 使头部压低） |
| `action_rate_l2` | -0.8→-2.0（curric） | 保留 | 平滑 |
| `neck_action_rate_l2` | -1.0 | 保留 | |
| `joint_torques_l2` | -5e-3 | 保留 | |
| `body_ang_vel` | -0.05 | 保留 | |
| `angular_momentum` | -0.02 | 保留 | |
| `soft_landing` | -1e-5 | 保留 | |

**移除的** : `mouth_perpendicular_to_ground`，`ground_pick_return_pose_legs`，`ground_pick_return_pose_neck`（被姿态跟随取代）。

其余**不变** : DR 块（CoM/head-CoM/mass-inertia/friction/armature/IMU-misalign/encoder-bias/pushes），61D obs + head/body 置零补齐，终止条件（`nan_state`），curricula（`action_rate_weight`，`com_range`，`head_com_range`），RlCfg（`experiment_name="ground_pick"`）。

## 6. 部署（sim2real 一致性）

```bash
microduck_runtime ... \
  --ground-pick ground_pick.onnx \
  --ground-pick-period 4.0 \       # = période env (défaut, rien à changer)
  --ground-pick-kp-ratio 1.0 \     # entraîné kp 200 → forcer 1.0 (défaut 0.6 baisse à 120)
  --ground-pick-action-scale 1.0   # = action.scale env
```

## 7. 测试

`tests/`（运行 `uv run --with pytest pytest tests/ -q`） :

- **纯函数** : `phase_pose_blend` 在关键点（φ=0→0，φ=0.075→0.5，φ=0.3→1，φ=0.575→0.5，φ=0.8→0，逐段单调） ; `phase_pose_track`/`_l1` : max 值和符号（cur==target）。
- **env 构建** : `make_microduck_ground_pick_env_cfg()` 能构建 ; 命令 = 带 `randomize_phase=False`，`period=4.0` 的 `GroundPickPhaseCommand` ; rewards `phase_pose_track`/`phase_pose_track_l1` 存在 ; `mouth_perpendicular_to_ground`/`ground_pick_return_pose_*` 不存在 ; `mouth_ground_proximity` 存在且权重 1.0。

## 8. 训练 / play / 导出

```bash
uv run train Mjlab-GroundPick-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 20000
uv run scripts/play_latest.py     # md-play
uv run scripts/export_latest.py   # normaliseur baké dans l'ONNX
```
观察 `Episode_Reward/phase_pose_track`（应上升）。

## 9. 范围外 / 备注

- **`pose_target_match` 重复**（mdp.py 1577 和 1914） : 潜在问题，此处不处理。
- **DOWN_POSE 调整** : 如果使用 FOLD 值嘴巴无法充分触地，调整 dict（理想情况下用 `read_pose.py` 读取真实机器人嘴巴朝地的放置），而不是加大 `mouth_ground_proximity`。
- **部署时的转换** : STAND=HOME = 主策略的中性点 → 返回时无冲击（与 roller 上注意到的 STAND≠HOME 问题相反）。