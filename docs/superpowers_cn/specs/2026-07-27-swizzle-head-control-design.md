# Swizzle 头部控制（Y 按钮）— 设计

**日期:** 2026-07-27
**分支:** `new_pre_alpha_rollers`
**触及任务:** `Mjlab-Velocity-Swizzle-MicroDuck`（`microduck_velocity_swizzle_env_cfg.py`）

## 目标

让操作员在 rollers 滑行时移动鸭子的**头部**到不同姿态（Y 按钮），**且头部移动时 swizzle 不会散架**。头部姿态是物理的头部/颈部关节（看上下左右）— 与 `heading_tracking`（身体的行进方向，保持不变）无关。

当前的 swizzle 策略在头部作为外部偏移移动时会倾倒（它不补偿 CoM 偏移），因此头部必须是**策略管理的** : 策略同时产生头部姿态并保持平衡。这与行走策略在 `--new-cmd-obs` 模式下的做法一致 — 头部是一个注入到观察中的**命令**，策略产生姿态（没有外部「双重叠加」）。

## 方法（已选 : A — 通过 obs 命令管理头部）

将 `microduck_velocity_env_cfg.py` 中已有的头部命令机制移植到 swizzle env : 把真实的头部姿态命令喂入当前置零填补的 `head_command` obs 插槽，奖励头部跟随该命令，并通过一个 curriculum 在**后期**将其逐渐引入，以免干扰 swizzle。

没有外部偏移选项（此前已否决 : 如果头部在策略不补偿的情况下移动，swizzle 无法保持直立）。

## 更改（全部在 `make_microduck_velocity_swizzle_env_cfg` 内）

1. **头部姿态命令项。** 添加 `cfg.commands["head_pose"] = UniformPoseCommandCfg(...)`，从 velocity env 复制 : 4D `[neck_pitch, head_pitch, head_yaw, head_roll]` 相对默认的增量，`resampling_time_range = (2.0, 5.0)`，逐关节范围（head_roll 更紧，匹配小型机械行程范围）。
2. **真实的 `head_command` obs。** 用真实的命令 obs `func=<head command obs>, params={"command_name": "head_pose"}` 替换当前 `zero_command_padding(dim=4)` 的头部插槽，actor 和 critic 都用。（保持 61D 布局 ; body_command 保持零填充 — 这里无身体姿态控制。）
3. **`head_pose_tracking` reward。** 添加 `cfg.rewards["head_pose_tracking"]`（`microduck_mdp.head_pose_tracking`，`command_name="head_pose"`，`std=0.5`），初始权重 0（curriculum 逐渐引入）。
4. **后期 curriculum。** 一个 `reward_weight` curriculum 将 `head_pose_tracking` 从 0 → **4.0** 逐渐引入，在 **~1500 iterations** 前保持 0（swizzle 已稳固），然后在下 ~1000 iterations 内攀升，镜像 velstand 的身体姿态介入。外加一个头部姿态命令范围 curriculum : 从紧的范围（头部小幅增量）开始，在同一窗口内放宽，使头部早期几乎不动，一旦策略能处理就达到全范围。这就是让头部控制「不难管理」的原因 — 它是在已稳定的 swizzle 之上添加的。（这些值是起点，可调。）
5. **协调颈部惩罚（必需）。** env 目前有 `neck_joint_pos_l2`，它把颈部/头部关节拉向 HOME — 它会与 `head_pose_tracking`（把它们拉向命令）**冲突**，因此头部永远不会动。将头部姿态关节从 `neck_joint_pos_l2` 中排除（或删除它），镜像 velocity env 的处理方式（其注释 : 把两者都保留会「在 head_pose_tracking 把关节拉向命令的同时又把它们拉向 HOME」）。保留 `neck_action_rate_l2`（平滑，无冲突）。

其它一切（swizzle，后退移动，heading curriculum，DR，obs 布局，command）不变。需要**重新训练** swizzle 任务。

## 运行时

无运行时代码更改。`microduck_runtime` 的 **Y 按钮**已经驱动 `head_command` obs 插槽（new-cmd-obs 模式将头部偏移作为命令注入，"不要双重叠加"）。一旦 swizzle 策略带头部控制重新训练，它就响应 Y。部署标志不变（`--roller --new-cmd-obs ...`）。

## 测试 / 验证

- Smoke test : `uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 16 --agent.max-iterations 2` 运行 ; `head_pose_tracking` 出现在奖励日志中 ; `head_pose` 命令和真实的 `head_command` obs 构建无错误。
- 真实运行 : `head_pose_tracking` 在 curriculum 介入后上升 ; swizzle 保持稳定（头部 curriculum 开启时摔倒率不飙升）。在 viewer / 机器人上 : 移动头部命令会移动头部，而 roller 继续滑行。

## 调优旋钮

- 头部介入时干扰 swizzle → 把 curriculum 介入推得更晚，或更慢地放宽头部范围。
- 头部跟随不佳 → 提高 `head_pose_tracking` 目标权重，或检查颈部惩罚是否仍在与之冲突。
- 头部过于抖动 → 保持/提高 `neck_action_rate_l2`。