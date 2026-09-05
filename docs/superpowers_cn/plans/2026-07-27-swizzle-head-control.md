# Swizzle 头部控制实现计划

> **面向 agentic 工作者的说明:** 必选子技能: 使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐步实现本计划。步骤使用复选框（`- [ ]`）语法以跟踪进度。

**目标:** 为 swizzle roller 任务添加操作员头部姿态控制（Y 按钮）, 使策略在保持平衡的同时, 将头部移动到指令的姿态。

**架构:** 通过观测命令由策略管理头部（与行走的 `--new-cmd-obs` 路径一致）。swizzle env 目前将 `head_command` obs 槽零填充; 我们向其中喂入真实的 `head_pose` 命令, 奖励 `head_pose_tracking`, 移除两个会把颈/头拉回 HOME 的奖励项（它们会与命令对抗）, 并通过 curriculum 在后面让头部**晚期**加入, 以免干扰已经正常的 swizzle。仅修改一个文件的配置; 需要重新训练。

**技术栈:** mjlab / mjlab_microduck 任务配置（Python）, rsl_rl PPO。复用 `microduck_velocity_env_cfg.py` 中已有的机制（`UniformPoseCommandCfg`, `head_pose_tracking`, `pose_command_range_curriculum`, `reward_weight`）。

## 全局约束

- 仅修改 swizzle 任务: `src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py`。stride、velocity、standup、roller-slope/crouch 任务以及 `mdp.py` **都不修改**。
- 保持 61D obs 布局 `[twist(3), head(4), body(6)]`: 替换 `head_command` 槽的内容（零填充 → 真实命令）, 但保持 `body_command` 零填充（此处不做 body-pose 控制）。
- 无新增 mdp 函数 — 所有 reward/command/curriculum 函数均已存在于 `microduck_mdp` 中。
- Runtime 不变: `microduck_runtime` 的 Y 按钮已经驱动 `head_command` obs 槽。

---

### 任务 1: 将头部姿态控制接入 swizzle env

**文件:**
- 修改: `src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py`
- 测试: `tests/test_swizzle_head_cfg.py`（新建）

**接口:**
- 消费（已存在, 请勿重新定义）:
  - `microduck_mdp.UniformPoseCommandCfg(resampling_time_range, ranges)` — 头部姿态命令项。
  - `mdp.generated_commands`（来自 `mjlab.tasks.velocity`）— 按名字读取命令的 obs 函数; 用作 `params={"command_name": "head_pose"}`。
  - `microduck_mdp.head_pose_tracking` — 奖励 `func`, params `{"command_name": "head_pose", "std": 0.5}`。
  - `microduck_mdp.reward_weight` — curriculum 函数, params `{"reward_name", "weight_stages": [{"step","weight"}, ...]}`。
  - `microduck_mdp.pose_command_range_curriculum` — curriculum 函数, params `{"command_name", "range_stages": [{"step","ranges"}, ...]}`。
- 产出: 具备 `head_pose` 命令、真实 `head_command` obs、`head_pose_tracking` 奖励、移除 `neck_joint_pos_l2`、将 `pose` 奖励限定到腿部关节、以及两个头部 curriculum 的 swizzle env cfg。

- [ ] **步骤 1: 写出会失败的测试**

创建 `tests/test_swizzle_head_cfg.py`:

```python
from mjlab.tasks.velocity import mdp
from mjlab_microduck.tasks.microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
)


def test_swizzle_head_control_wired():
    cfg = make_microduck_velocity_swizzle_env_cfg()

    # Head-pose command term exists.
    assert "head_pose" in cfg.commands

    # head_command obs is the REAL command (not zero-padded) on both groups.
    for group in ("actor", "critic"):
        term = cfg.observations[group].terms["head_command"]
        assert term.func is mdp.generated_commands
        assert term.params["command_name"] == "head_pose"

    # head_pose_tracking reward exists.
    assert "head_pose_tracking" in cfg.rewards

    # The two HOME-pullers that would fight the head command are handled:
    #  - neck_joint_pos_l2 removed
    assert "neck_joint_pos_l2" not in cfg.rewards
    #  - pose reward scoped to leg joints via a negative-lookahead regex that
    #    excludes neck/head (and passive wheels)
    pose_joints = cfg.rewards["pose"].params["asset_cfg"].joint_names
    assert any(
        "(?!" in j and "neck" in j and "head" in j for j in pose_joints
    ), f"pose reward not scoped away from neck/head: {pose_joints}"

    # Late head curricula exist.
    assert "head_pose_tracking_weight" in cfg.curriculum
    assert "head_pose_range" in cfg.curriculum
```

- [ ] **步骤 2: 运行测试以确认其失败**

运行: `MUJOCO_GL=egl uv run pytest tests/test_swizzle_head_cfg.py -v`
预期: FAIL（`head_pose` 命令 / `head_pose_tracking` 奖励不存在; `head_command` obs 仍是 `zero_command_padding`）。

- [ ] **步骤 3: 向 swizzle env cfg 添加导入**

在 `microduck_velocity_swizzle_env_cfg.py` 中, 扩展导入（当前为 `from mjlab.managers import CurriculumTermCfg, RewardTermCfg`）, 添加 `ObservationTermCfg`, 并为 `generated_commands` 导入 velocity mdp:

```python
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
```

- [ ] **步骤 4: 添加 head_pose 命令 + 真实 head_command obs + head_pose_tracking 奖励 + neck 调和**

在 `make_microduck_velocity_swizzle_env_cfg` 内部、现有 reward/heading 设置之后、`return cfg` 之前, 添加:

```python
    # --- Head-pose control (Y button): the policy produces the head pose ---------
    # Head-pose command (4D deltas from HOME: [neck_pitch, head_pitch, head_yaw,
    # head_roll]). Ported from the velocity env; ranges start small (widened by the
    # curriculum below). Resample every 2-5 s.
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll (tighter — small mechanical range)
        ),
    )

    # Feed the REAL head command into the obs (replaces zero_command_padding) on
    # both groups. body_command stays zero-padded (no body-pose control here).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )

    # Reward the head tracking its command. Weight 0 here — ramped in LATE by the
    # curriculum so it doesn't disturb the swizzle before it's solid.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.0,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # Reconcile the two HOME-pullers that would fight head_pose_tracking:
    #  1) neck_joint_pos_l2 pulls the neck/head joints to HOME -> remove it.
    if "neck_joint_pos_l2" in cfg.rewards:
        del cfg.rewards["neck_joint_pos_l2"]
    #  2) the pose reward includes neck/head -> scope it to LEG joints only.
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
```

- [ ] **步骤 5: 添加晚期的头部 curriculum**

紧接着步骤 4 的块之后（仍在 `return cfg` 之前）:

```python
    # head_pose_tracking ramps 0 -> 4.0, staying 0 until ~1500 it. (swizzle solid),
    # so head control is added on top of a stable swizzle.
    cfg.curriculum["head_pose_tracking_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_tracking",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},   # must match initial weight
                {"step": 1500 * 24,  "weight": 0.0},   # head off while swizzle solidifies
                {"step": 2250 * 24,  "weight": 2.0},
                {"step": 3000 * 24,  "weight": 4.0},
            ],
        },
    )
    # Head-command range widens over the SAME window (tiny until 1500, full by 3000),
    # so the commanded head barely moves early and reaches full range once the policy
    # can handle it.
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,               ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0,          "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 1500 * 24,  "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 2250 * 24,  "ranges": ((-0.55, 0.55), (-0.55, 0.55), (-0.70, 0.70), (-0.15, 0.15))},
                {"step": 3000 * 24,  "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
            ],
        },
    )
```

- [ ] **步骤 6: 运行 cfg 测试以确认其通过**

运行: `MUJOCO_GL=egl uv run pytest tests/test_swizzle_head_cfg.py -v`
预期: PASS。

- [ ] **步骤 7: 对 env 做端到端冒烟测试**

运行: `MUJOCO_GL=egl uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 16 --agent.max-iterations 2`
预期: 无错误; 奖励日志列出 `head_pose_tracking` 而不再列出 `neck_joint_pos_l2`; 出现一行值为 0.0 的 `Curriculum/head_pose_tracking_weight`。

- [ ] **步骤 8: 提交**

```bash
git add src/mjlab_microduck/tasks/microduck_velocity_swizzle_env_cfg.py tests/test_swizzle_head_cfg.py
git commit -m "swizzle: add head-pose control (Y button, policy-managed, late curriculum)"
```

---

## 完整训练运行的说明（不属于本任务）

由于头部 curriculum 要到约 3000 次迭代才结束, 训练时长应超过之前使用的 2500:

```bash
uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3500
```

观察: `head_pose_tracking` 在约 1500 次迭代后上升; swizzle 的摔倒率在它生效时不飙升。若头部干扰了 swizzle → 推迟生效时刻 / 更慢地扩大范围。若头部不跟随 → 提高最终权重。部署保持不变（`--roller --new-cmd-obs`, Y 按钮移动头部）。