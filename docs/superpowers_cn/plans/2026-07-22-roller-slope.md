# `roller_slope` 斜坡模式 — 实现计划

> **面向 agentic worker：** 必备子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现本计划。步骤使用复选框（`- [ ]`）语法跟踪。

**目标：** 训练一个专用策略，让 microduck（rollers）在平地上启动并带一个初始冲量，滚上一条下行斜坡，然后在无任何操控的情况下保持站立滑行到底。

**架构：** 一个从 `velocity_rollers` 克隆出来的独立新任务（相同机器人，相同 61D 观测 → 可在 runtime 互换）。自定义地形「平坦 + 斜坡」，角度按难度插值，自研坡度课程，命令被中和，平衡 + 名义站立姿态奖励。`infer_policy.py` 中增加 `Y` 按钮切换开关。

**技术栈：** Python、mjlab 1.3.x、MuJoCo（MjSpec terrains）、rsl_rl（PPO）、PyTorch、onnxruntime（部署）、pytest。

## 全局约束

- **统一 61D 观测**：twist (3D) + head_command (4D) + body_command (6D) 零填充。绝不改变此布局 — 策略必须通过 `--new-cmd-obs` 加载。
- **按名称解析关节**，绝不用索引（被动轮子交错）。
- **通过 `reset_root_state_uniform`（velocity_range）设置入口速度**，绝不在 reset 模式下使用 `push_by_setting_velocity`（会累积到根状态上 → free-joint 发散 → NaN）。`roller_crouch` 的教训。
- **物理代码中使用弧度**；坡度常数以度为单位表达（`RAMP_DEG_MIN=2.0`、`RAMP_DEG_MAX=20.0`）并转换。
- **简单提交**，遵循仓库风格（不含 `Co-authored-by`）。
- 测试放在 `tests/`，用 `uv run pytest` 运行。

---

## 文件结构

- **创建** `src/mjlab_microduck/tasks/slope_terrain.py` — `ramp_angle_by_difficulty()` + `FlatRampTerrainCfg`（平坦+斜坡地形几何）。单一职责：地形。
- **修改** `src/mjlab_microduck/tasks/mdp.py` — 添加 `slope_move_masks()`（纯函数）+ `terrain_levels_slope()`（坡度课程）。
- **创建** `src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py` — `make_microduck_roller_slope_env_cfg()` + `MicroduckRollerSlopeRlCfg`。
- **修改** `src/mjlab_microduck/tasks/__init__.py` — 注册任务。
- **修改** `scripts/infer_policy.py` — `--slope` 标志 + `Y` 键。
- **创建** `tests/test_slope_terrain.py`、`tests/test_slope_curriculum.py`、`tests/test_roller_slope_cfg.py`。

---

## 任务 1：按难度给出斜坡角度（纯函数）

**文件：**
- 创建：`src/mjlab_microduck/tasks/slope_terrain.py`
- 测试：`tests/test_slope_terrain.py`

**接口：**
- 产出：`ramp_angle_by_difficulty(difficulty: float, deg_min: float = 2.0, deg_max: float = 20.0) -> float`（返回**弧度**）。模块常量 `RAMP_DEG_MIN = 2.0`、`RAMP_DEG_MAX = 20.0`。

- [ ] **第 1 步：编写失败测试**

```python
# tests/test_slope_terrain.py
import math
from mjlab_microduck.tasks.slope_terrain import (
    ramp_angle_by_difficulty,
    RAMP_DEG_MIN,
    RAMP_DEG_MAX,
)


def test_ramp_angle_endpoints():
    assert math.isclose(ramp_angle_by_difficulty(0.0), math.radians(RAMP_DEG_MIN), abs_tol=1e-9)
    assert math.isclose(ramp_angle_by_difficulty(1.0), math.radians(RAMP_DEG_MAX), abs_tol=1e-9)


def test_ramp_angle_midpoint():
    mid_deg = (RAMP_DEG_MIN + RAMP_DEG_MAX) / 2.0
    assert math.isclose(ramp_angle_by_difficulty(0.5), math.radians(mid_deg), abs_tol=1e-9)


def test_ramp_angle_clamps_out_of_range():
    assert math.isclose(ramp_angle_by_difficulty(-1.0), math.radians(RAMP_DEG_MIN), abs_tol=1e-9)
    assert math.isclose(ramp_angle_by_difficulty(2.0), math.radians(RAMP_DEG_MAX), abs_tol=1e-9)
```

- [ ] **第 2 步：运行测试 — 它必须失败**

运行：`uv run pytest tests/test_slope_terrain.py -v`
预期：失败 — `ModuleNotFoundError: No module named 'mjlab_microduck.tasks.slope_terrain'`

- [ ] **第 3 步：最小实现**

```python
# src/mjlab_microduck/tasks/slope_terrain.py
"""Terrain custom « plat + rampe descendante » pour la tâche roller_slope.

Le robot spawne sur une zone plate, reçoit une impulsion vers +x, roule
jusqu'à la rampe et se laisse glisser. L'angle de la rampe est interpolé par
la difficulté (curriculum) sur [RAMP_DEG_MIN, RAMP_DEG_MAX] degrés.
"""

from __future__ import annotations

import math

import numpy as np

RAMP_DEG_MIN = 2.0
RAMP_DEG_MAX = 20.0


def ramp_angle_by_difficulty(
    difficulty: float, deg_min: float = RAMP_DEG_MIN, deg_max: float = RAMP_DEG_MAX
) -> float:
    """Angle de rampe (radians) interpolé linéairement par la difficulté [0,1]."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    return math.radians(deg_min + d * (deg_max - deg_min))
```

- [ ] **第 4 步：运行测试 — 它必须通过**

运行：`uv run pytest tests/test_slope_terrain.py -v`
预期：通过（3 个测试）

- [ ] **第 5 步：提交**

```bash
git add src/mjlab_microduck/tasks/slope_terrain.py tests/test_slope_terrain.py
git commit -m "roller-slope: angle de rampe par difficulte (fonction pure + tests)"
```

---

## 任务 2：自定义地形 `FlatRampTerrainCfg`

**文件：**
- 修改：`src/mjlab_microduck/tasks/slope_terrain.py`
- 测试：`tests/test_slope_terrain.py`

**接口：**
- 消费：`ramp_angle_by_difficulty`（任务 1）、`mjlab.terrains.terrain_generator` 的 `SubTerrainCfg`、`TerrainGeometry`、`TerrainOutput`。
- 产出：`FlatRampTerrainCfg(SubTerrainCfg)`，字段为 `flat_length: float = 2.0`、`ramp_length: float = 5.0`、`deg_min: float = 2.0`、`deg_max: float = 20.0`、`thickness: float = 0.5`；方法 `function(difficulty, spec, rng) -> TerrainOutput`。spawn 原点位于平坦区域上。

**几何笔记（须记住）：** 平坦面位于局部 `z=0`。斜坡是一个围绕 `+y` 转过四元数 `[cos(a/2), 0, sin(a/2), 0]` 的 box — 即围绕 `+y` 旋转 `+a` 会降低 `+x` 边缘（当 `x` 增大时斜坡向下）。平坦/斜坡的精确拼接（无台阶、无缝隙）**必须在 viewer 中验证**（第 6 步），因为斜坡中心的 `z` 很敏感。

- [ ] **第 1 步：编写失败测试**

```python
# tests/test_slope_terrain.py  (ajouter)
import mujoco
import numpy as np
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg


def _empty_terrain_spec():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    return spec


def test_flat_ramp_builds_geoms_and_origin_on_flat():
    cfg = FlatRampTerrainCfg(flat_length=2.0, ramp_length=5.0)
    cfg.size = (8.0, 4.0)  # posé normalement par le générateur
    spec = _empty_terrain_spec()
    out = cfg.function(difficulty=0.5, spec=spec, rng=np.random.default_rng(0))
    # deux géométries : plat + rampe
    assert len(out.geometries) == 2
    # origine sur le plat (x dans [0, flat_length], z ~ 0)
    assert 0.0 <= out.origin[0] <= 2.0
    assert abs(out.origin[2]) < 1e-6


def test_flat_ramp_steeper_at_higher_difficulty():
    # à difficulté plus haute, le bout de rampe descend plus bas
    cfg = FlatRampTerrainCfg()
    cfg.size = (8.0, 4.0)
    easy = cfg.function(0.0, _empty_terrain_spec(), np.random.default_rng(0))
    hard = cfg.function(1.0, _empty_terrain_spec(), np.random.default_rng(0))
    # la rampe (2e géométrie) est plus basse (centre z plus négatif) en difficile
    assert hard.geometries[1].geom.pos[2] < easy.geometries[1].geom.pos[2]
```

- [ ] **第 2 步：运行测试 — 它必须失败**

运行：`uv run pytest tests/test_slope_terrain.py -k flat_ramp -v`
预期：失败 — `ImportError: cannot import name 'FlatRampTerrainCfg'`

- [ ] **第 3 步：最小实现**

```python
# src/mjlab_microduck/tasks/slope_terrain.py  (ajouter en tête)
from dataclasses import dataclass

import mujoco

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)


@dataclass(kw_only=True)
class FlatRampTerrainCfg(SubTerrainCfg):
    """Zone plate de départ suivie d'une rampe descendante (angle par difficulté)."""

    flat_length: float = 2.0   # longueur du plat de départ le long de +x (m)
    ramp_length: float = 5.0   # longueur horizontale de la rampe le long de +x (m)
    deg_min: float = RAMP_DEG_MIN
    deg_max: float = RAMP_DEG_MAX
    thickness: float = 0.5     # épaisseur des box (m)

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng
    ) -> TerrainOutput:
        del rng  # non utilisé
        body = spec.body("terrain")
        angle = ramp_angle_by_difficulty(difficulty, self.deg_min, self.deg_max)
        width = self.size[1]
        t = self.thickness

        # Plat : box dont la surface supérieure est à z=0, x dans [0, flat_length].
        flat = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(self.flat_length / 2.0, width / 2.0, t / 2.0),
            pos=(self.flat_length / 2.0, 0.0, -t / 2.0),
        )

        # Rampe : box tourné de +angle autour de +y (le bord +x descend).
        # Longueur de surface = ramp_length / cos(angle).
        surf_len = self.ramp_length / math.cos(angle)
        ramp_cx = self.flat_length + self.ramp_length / 2.0
        # Centre z : mi-descente de la surface, moins la demi-épaisseur projetée.
        ramp_cz = -(self.ramp_length * math.tan(angle) / 2.0) - (t / 2.0) * math.cos(angle)
        half = angle / 2.0
        ramp = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(surf_len / 2.0, width / 2.0, t / 2.0),
            pos=(ramp_cx, 0.0, ramp_cz),
            quat=(math.cos(half), 0.0, math.sin(half), 0.0),
        )

        origin = np.array([self.flat_length * 0.4, 0.0, 0.0])
        return TerrainOutput(
            origin=origin,
            geometries=[
                TerrainGeometry(geom=flat, color=(0.5, 0.5, 0.5, 1.0)),
                TerrainGeometry(geom=ramp, color=(0.45, 0.55, 0.75, 1.0)),
            ],
        )
```

- [ ] **第 4 步：运行测试 — 它们必须通过**

运行：`uv run pytest tests/test_slope_terrain.py -v`
预期：通过（5 个测试）

- [ ] **第 5 步：提交**

```bash
git add src/mjlab_microduck/tasks/slope_terrain.py tests/test_slope_terrain.py
git commit -m "roller-slope: terrain custom plat+rampe (FlatRampTerrainCfg + tests)"
```

- [ ] **第 6 步：目视验证（人工检查点）**

几何体（尤其是 `ramp_cz` 和四元数的符号）必须用肉眼确认。
在任务 4（env 组装后）之后，启动 play viewer（见任务 4 第 6 步）并检查：
平坦区域与斜坡**无缝衔接、无台阶也无缝隙**，且斜坡在 `+x` 方向（机器人前方）**向下**。
如果出现垂直偏移，调整 `ramp_cz`；
如果斜坡是向上而不是向下，则反转四元数的符号（`-half`）。

---

## 任务 3：坡度课程 `terrain_levels_slope`

**文件：**
- 修改：`src/mjlab_microduck/tasks/mdp.py`
- 测试：`tests/test_slope_curriculum.py`

**接口：**
- 产出：
  - `slope_move_masks(distance: torch.Tensor, size_x: float) -> tuple[torch.Tensor, torch.Tensor]` — 纯辅助函数。`move_up = distance > size_x * 0.5`（已到底部 → 斜坡更陡）；`move_down = (distance < size_x * 0.2) & ~move_up`（过早摔倒/卡住 → 斜坡更缓）。以 `bool` 返回 `(move_up, move_down)`。
  - `terrain_levels_slope(env, env_ids) -> torch.Tensor` — mjlab 课程签名；计算从原点出发在 `x` 方向上的行进距离，应用 `slope_move_masks`，调用 `terrain.update_env_origins`，返回平均难度级别。

- [ ] **第 1 步：编写失败测试**

```python
# tests/test_slope_curriculum.py
import torch
from mjlab_microduck.tasks.mdp import slope_move_masks


def test_move_up_when_reached_bottom():
    # distance > size_x/2 → monte en difficulté
    dist = torch.tensor([5.0, 4.1])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert bool(up[0]) and bool(up[1])
    assert not bool(down[0]) and not bool(down[1])


def test_move_down_when_stuck_early():
    # distance < size_x*0.2 (=1.6) → descend en difficulté
    dist = torch.tensor([0.5, 1.0])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert not bool(up[0]) and not bool(up[1])
    assert bool(down[0]) and bool(down[1])


def test_stay_in_middle_band():
    # entre 1.6 et 4.0 → ni haut ni bas
    dist = torch.tensor([2.5])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert not bool(up[0]) and not bool(down[0])
```

- [ ] **第 2 步：运行测试 — 它必须失败**

运行：`uv run pytest tests/test_slope_curriculum.py -v`
预期：失败 — `ImportError: cannot import name 'slope_move_masks'`

- [ ] **第 3 步：最小实现**

在 `src/mjlab_microduck/tasks/mdp.py` 中添加（靠近其他课程，例如 `com_range_curriculum` 之后）。检查文件头部已导入 `torch`（是的）。

```python
def slope_move_masks(distance: "torch.Tensor", size_x: float):
    """Masques de promotion/rétrogradation du curriculum de pente.

    move_up   : a parcouru plus de la moitié de la tuile → il a dévalé la rampe,
                on la rend plus raide.
    move_down : a à peine avancé (< 20% de la tuile) → chute/blocage précoce,
                on adoucit la rampe.
    """
    move_up = distance > size_x * 0.5
    move_down = (distance < size_x * 0.2) & (~move_up)
    return move_up, move_down


def terrain_levels_slope(env, env_ids):
    """Curriculum de raideur pour roller_slope (pas de vitesse commandée).

    Progression basée sur la distance en x parcourue depuis l'origine de spawn.
    """
    asset = env.scene["robot"]
    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    distance = (
        asset.data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    )
    move_up, move_down = slope_move_masks(distance, terrain_generator.size[0])
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
```

- [ ] **第 4 步：运行测试 — 它必须通过**

运行：`uv run pytest tests/test_slope_curriculum.py -v`
预期：通过（3 个测试）

- [ ] **第 5 步：提交**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_slope_curriculum.py
git commit -m "roller-slope: curriculum de raideur terrain_levels_slope (+ helper pur teste)"
```

---

## 任务 4：`roller_slope` env cfg + 注册

**文件：**
- 创建：`src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py`
- 修改：`src/mjlab_microduck/tasks/__init__.py`
- 测试：`tests/test_roller_slope_cfg.py`

**接口：**
- 消费：`make_microduck_velocity_rollers_env_cfg`（物理/DR/obs 基础）、`FlatRampTerrainCfg`（任务 2）、`terrain_levels_slope`（任务 3）、现有 mdp 函数：`body_upright_gaussian`、`is_alive`、`pose_target_match`、`pose_l1_penalty`、`feet_flat_penalty`、`neck_action_rate_l2`、`joint_torques_l2`、`robot_state_is_nan`、`reset_action_history`、`zero_command_padding`。
- 产出：`make_microduck_roller_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg` 和 `MicroduckRollerSlopeRlCfg`（`RslRlOnPolicyRunnerCfg`、`experiment_name="roller_slope"`）。

> 复用 roller env 的 DR/obs/reset 代码块：我们**从** `make_microduck_velocity_rollers_env_cfg()` **出发**，并且只修改地形、命令、奖励、终止、课程。不要重写 DR。

- [ ] **第 1 步：编写失败测试**

```python
# tests/test_roller_slope_cfg.py
from mjlab_microduck.tasks.microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
)
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg


def test_terrain_is_flat_ramp_generator():
    cfg = make_microduck_roller_slope_env_cfg()
    assert cfg.scene.terrain.terrain_type == "generator"
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None and gen.curriculum is True
    assert any(isinstance(st, FlatRampTerrainCfg) for st in gen.sub_terrains.values())


def test_command_is_neutralised():
    cfg = make_microduck_roller_slope_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.rel_standing_envs == 1.0
    assert cmd.rel_heading_envs == 0.0


def test_entry_velocity_set_on_reset_base():
    cfg = make_microduck_roller_slope_env_cfg()
    vr = cfg.events["reset_base"].params["velocity_range"]
    assert vr["x"][0] > 0.0  # impulsion vers l'avant


def test_has_upright_and_pose_rewards():
    cfg = make_microduck_roller_slope_env_cfg()
    for name in ("upright", "alive", "standing_pose", "feet_flat"):
        assert name in cfg.rewards
```

- [ ] **第 2 步：运行测试 — 它必须失败**

运行：`uv run pytest tests/test_roller_slope_cfg.py -v`
预期：失败 — `ModuleNotFoundError`（env cfg 模块缺失）

- [ ] **第 3 步：实现**

```python
# src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py
"""Microduck roller slope — descente passive équilibrée.

Le robot spawne sur du plat (impulsion vers l'avant), roule sur une rampe
descendante et se laisse glisser en restant debout. Aucun pilotage : la
commande twist est neutralisée (rel_standing_envs=1.0). Terrain custom
plat+rampe (FlatRampTerrainCfg), curriculum de raideur (terrain_levels_slope).
Obs 61D unifié → interchangeable au runtime (--new-cmd-obs).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.tasks.velocity import mdp
from mjlab.envs import mdp as base_mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.slope_terrain import FlatRampTerrainCfg
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

ENTRY_VELOCITY_X = (0.2, 0.5)  # impulsion vers l'avant au reset (m/s)


def make_microduck_roller_slope_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    # === TERRAIN : plat + rampe, curriculum de raideur ===
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 4.0),
            curriculum=True,
            num_rows=10,          # 10 niveaux de raideur
            num_cols=1,
            difficulty_range=(0.0, 1.0),
            sub_terrains={"flat_ramp": FlatRampTerrainCfg(flat_length=2.0, ramp_length=5.0)},
        ),
        max_init_terrain_level=0,  # démarrer sur la rampe la plus douce
    )

    # === COMMANDE neutralisée (équilibre pur) ===
    command = cfg.commands["twist"]
    command.rel_standing_envs = 1.0
    command.rel_heading_envs = 0.0
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    if getattr(command.ranges, "ang_vel_z", None) is not None:
        command.ranges.ang_vel_z = (0.0, 0.0)

    # === RESET : impulsion vers l'avant sur le plat ===
    cfg.events["reset_base"].params["velocity_range"] = {"x": ENTRY_VELOCITY_X}

    # === RÉCOMPENSES : équilibre + posture debout nominale ===
    keep = {"action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"] = RewardTermCfg(
        func=microduck_mdp.body_upright_gaussian,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "std": 0.2},
    )
    cfg.rewards["alive"] = RewardTermCfg(func=microduck_mdp.is_alive, weight=1.0)
    # posture debout nominale (cible fixe = default_joint_pos, aucun override)
    cfg.rewards["standing_pose"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match, weight=3.0, params={"std": 0.4},
    )
    cfg.rewards["standing_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty, weight=1.0,
    )
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5,
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3,
    )
    cfg.rewards["action_rate_l2"].weight = -1.0

    # === TERMINATIONS : chute + bas atteint ===
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=base_mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.terminations["out_of_bounds"] = TerminationTermCfg(func=mdp.out_of_terrain_bounds)
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan, time_out=False,
    )

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset",
    )

    # === CURRICULUM : raideur de la rampe ===
    for name in list(cfg.curriculum.keys()):
        del cfg.curriculum[name]
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=microduck_mdp.terrain_levels_slope)

    return cfg


MicroduckRollerSlopeRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_slope",
    run_name="roller_slope",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8_000,
)
```

然后在 `src/mjlab_microduck/tasks/__init__.py` 中注册，**严格遵循**已有的 `roller_crouch` 注册模式（导入 `make_...` + `Microduck...RlCfg`，然后用 `"Microduck-Roller-Slope"` 风格的 id 调用 `register_mjlab_task(...)`）。复制 `roller_crouch` 代码块并把 `crouch`→`slope` 替换。

- [ ] **第 4 步：运行测试 — 它们必须通过**

运行：`uv run pytest tests/test_roller_slope_cfg.py -v`
预期：通过（4 个测试）

- [ ] **第 5 步：确认任务注册 + 完整构建**

运行：
```bash
uv run python -c "import gymnasium as gym; import mjlab_microduck.tasks; print([e for e in gym.registry if 'Slope' in e])"
```
预期：列表包含 id `Microduck-Roller-Slope`（或已注册的变体）。

- [ ] **第 6 步：地形 + 下落的目视验证（人工检查点 — 关闭任务 2 第 6 步）**

运行一段短训练，然后 play（或根据仓库用法运行 `scripts/play_latest.py`）并观察：
1. 平坦 + 斜坡无台阶/缝隙地拼接；斜坡在机器人前方**向下**。
2. 机器人在平坦上 spawn，向前出发，抵达斜坡。
如果几何体有误，修正 `slope_terrain.py`（见任务 2 第 6 步）并重新提交。

- [ ] **第 7 步：提交**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_slope_env_cfg.py src/mjlab_microduck/tasks/__init__.py tests/test_roller_slope_cfg.py
git commit -m "roller-slope: env descente passive (terrain plat+rampe, cmd nulle, rewards equilibre) + enregistrement"
```

---

## 任务 5：部署 — `--slope` 标志 + `Y` 键

**文件：**
- 修改：`scripts/infer_policy.py`

**接口：**
- 消费：从 `roller_slope` 策略导出的 `.onnx`。
- 产出：CLI 参数 `--slope <path>`；属性 `self.slope_session` + 标志 `self.slope_mode`；方法 `toggle_slope_mode()`；接线 `GLFW_KEY_Y = 89` 键。

> 斜坡策略以零 twist 命令运行（如同 standing 模式）。在 slope 模式下，必须中和 walking/standing 的自动切换。

- [ ] **第 1 步：添加 CLI 参数并加载会话**

在 `main()` 中（靠近其他 `add_argument`，约第 471 行）：
```python
    parser.add_argument("--slope", type=str, default=None, help="Path to slope policy ONNX file (press Y to toggle)")
```
把 `slope_onnx_path=args.slope` 传给控制器的构造函数（给 `__init__` 添加参数 `slope_onnx_path=None`，约第 51-57 行，并像其他参数一样加载）：
```python
        self.slope_session = None
        self.slope_mode = False
        if slope_onnx_path:
            print(f"\nLoading slope policy from: {slope_onnx_path}")
            self.slope_session = ort.InferenceSession(slope_onnx_path)
```

- [ ] **第 2 步：添加 `toggle_slope_mode` 并中和自动切换**

在 `toggle_body_pose_mode` 之后（约第 285 行）：
```python
    def toggle_slope_mode(self):
        """Bascule vers/depuis la politique pente (descente passive)."""
        if self.slope_session is None:
            print("Slope unavailable: no --slope policy loaded")
            return
        self.slope_mode = not self.slope_mode
        if self.slope_mode:
            self.ort_session = self.slope_session
            self.current_policy = "slope"
            self.set_vel_cmd(0.0, 0.0, 0.0)  # descente passive : commande nulle
            print("Slope mode: ON (descente passive)")
        else:
            self.ort_session = self.walking_session or self.standing_session
            self.current_policy = "walking" if self.walking_session else "standing"
            print("Slope mode: OFF")
```
在 `_update_policy_session`（约第 250 行）中，在开头添加防护（在 `ground_pick_mode` 防护之后）：
```python
        if self.slope_mode:
            return  # Ne pas basculer pendant le mode pente
```

- [ ] **第 3 步：接线 `Y` 键**

在其他键盘常量附近添加键码（约第 680 行）：
```python
    GLFW_KEY_Y = 89
```
在 `key_callback` 中添加分支（例如在 `GLFW_KEY_B` 分支之后）：
```python
            elif key == GLFW_KEY_Y:
                policy.toggle_slope_mode()
```
添加键盘帮助行（靠近 `print`，约第 821 行）：
```python
    print("  Y:                toggle slope mode (requires --slope, descente passive)")
```

- [ ] **第 4 步：确认脚本可无错误加载**

运行：`uv run python scripts/infer_policy.py --help`
预期：帮助信息显示并列出 `--slope`。

- [ ] **第 5 步：提交**

```bash
git add scripts/infer_policy.py
git commit -m "roller-slope: deploiement --slope + touche Y (bascule mode pente)"
```

---

## 自检（由计划作者完成）

- **规格覆盖**：专用任务（任务 4）✓；自定义平坦+斜坡地形（任务 2）✓；平坦起步 + 冲量（任务 4 重置 velocity_range）✓；零命令（任务 4）✓；平衡 + 站立姿态 + 防摔奖励（任务 4）✓；摔倒/到底/nan 终止（任务 4）✓；0→20° 课程（任务 1 角度 + 任务 3 晋升）✓；61D 观测可互换（继承自 roller env，未修改）✓；`Y` 键（任务 5）✓。
- **占位符**：无「TBD/TODO」；两个人检查点（viewer 几何体）是明确的检查项，不是实现漏洞。
- **类型一致性**：`ramp_angle_by_difficulty`（任务 1）被 `FlatRampTerrainCfg`（任务 2）复用；`slope_move_masks`（任务 3）被 `terrain_levels_slope`（任务 3）消费；任务 4 中测试的奖励名（`upright`、`alive`、`standing_pose`、`feet_flat`）与实现对齐。
- **已声明的风险**：斜坡几何体（`ramp_cz`、四元数符号）须在 viewer 中确认；mjlab 的确切 API 名（`terrain.terrain_levels`、`TerrainEntityCfg`、注册 id）须在实现时对照现有 `roller_crouch` 模式验证。
