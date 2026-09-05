# Roller Crouch-Glide 实现计划

> **面向 agentic worker：** 必备子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现本计划。步骤使用复选框（`- [ ]`）语法跟踪。

**目标：** 增加一个「蹲下滑行然后站起身来」的动作，由 A 按钮触发，不修改 Rust runtime，通过训练一个加载到 `--ground-pick` 槽位的 mjlab policy 实现。

**架构：** 一个新的 mjlab 任务，在轮式（rollers）机器人上训练，由相位命令 `GroundPickPhaseCommand`（即 runtime 的 ground-pick 槽位所发送的命令）驱动。一个新的 reward 沿相位跟踪「梯形」的目标躯干高度（高 → 低 → 保持 1 秒 → 高）。与 roller policy 相同的 61D 观测布局 → 可在 runtime 互换。导出 ONNX，通过 `--ground-pick` 加载。

**技术栈：** Python、PyTorch、mjlab 1.3.0、MuJoCo、uv、ONNX。目标 runtime：`apirrone/microduck_runtime`（Rust，二进制 — 不修改）。

## 全局约束

- **不修改 Rust runtime。** 本动作复用现有的 `--ground-pick` 槽位（A 按钮，one-shot）。
- **必须统一使用 61D 观测布局**（`--new-cmd-obs`）：`[twist(3), head(4), body(6)]`，head/body 零填充。任何新 policy 必须保持此布局。
- **14 个主动关节**（通过 `SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))` 排除被动轮子），`action.scale = 1.0`，`kp_fw = 200`。
- **训练/部署一致性（sim2real）：** 部署时强制 `--ground-pick-kp-ratio 1.0`（默认 0.6）、`--ground-pick-action-scale` = 运行时 action_scale、`--ground-pick-period 5.0`。
- **相位编码（由 runtime 强制）：** `command = [cos(2π·φ), sin(2π·φ), 0]`，周期 4 秒。滑行保持 = 1 秒 → `hold_lo=0.375`，`hold_hi=0.625`。
- **简单提交**（不含 `Co-Authored-By`）。
- 通过 `uv run --with pytest pytest` 运行测试（不向项目添加 pytest 依赖）。
- 参考规格：`docs/superpowers/specs/2026-07-17-roller-crouch-glide-design.md`。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/mjlab_microduck/tasks/mdp.py` | **修改。** 添加 3 个函数：`crouch_height_target`（纯函数）、`crouch_glide_reward_from_values`（纯函数）、`crouch_glide_height_by_phase`（env 包装器）以及 `forward_speed_reward`。 |
| `tests/test_crouch_glide.py` | **创建。** 纯函数的单元测试。 |
| `src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py` | **创建。** 该 env（roller + 相位混合体）+ `MicroduckRollerCrouchRlCfg`。 |
| `src/mjlab_microduck/tasks/__init__.py` | **修改。** 导入并注册 `Mjlab-RollerCrouch-Flat-MicroDuck`。 |
| `tests/test_roller_crouch_cfg.py` | **创建。** 冒烟测试：env 以正确的命令/rewards 构建。 |

---

## 任务 1：「梯形」高度目标（纯函数）

**文件：**
- 修改：`src/mjlab_microduck/tasks/mdp.py`（添加函数，位于 `com_height_target` 之后，约第 737 行）
- 测试：`tests/test_crouch_glide.py`

**接口：**
- 产出：`crouch_height_target(phase: torch.Tensor, height_low: float, height_high: float, hold_lo: float = 0.375, hold_hi: float = 0.625) -> torch.Tensor` — 接收相位 (B,) ∈ [0,1)，返回目标高度 (B,)。

- [ ] **第 1 步：编写失败测试**

创建 `tests/test_crouch_glide.py`：

```python
import math
import torch
from mjlab_microduck.tasks import mdp


def test_crouch_height_target_endpoints_are_high():
    # phase 0 (début) et phase ~1 (fin) → hauteur haute (debout)
    phase = torch.tensor([0.0, 0.999])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([0.11, 0.11]), atol=2e-3)


def test_crouch_height_target_plateau_is_low():
    # tout le palier [0.375, 0.625] → hauteur basse constante
    phase = torch.tensor([0.375, 0.5, 0.624])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.full((3,), 0.075), atol=1e-6)


def test_crouch_height_target_descent_midpoint():
    # milieu de la descente (phase = hold_lo/2 = 0.1875) → milieu des deux hauteurs
    phase = torch.tensor([0.1875])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([(0.11 + 0.075) / 2]), atol=1e-6)


def test_crouch_height_target_rise_midpoint():
    # milieu de la remontée (phase = 0.8125) → milieu des deux hauteurs
    phase = torch.tensor([0.8125])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([(0.11 + 0.075) / 2]), atol=1e-6)
```

- [ ] **第 2 步：运行测试以确认其失败**

运行：`uv run --with pytest pytest tests/test_crouch_glide.py -v`
预期：失败 — `AttributeError: module ... has no attribute 'crouch_height_target'`

- [ ] **第 3 步：实现该函数**

在 `src/mjlab_microduck/tasks/mdp.py` 中，紧接 `com_height_target` 之后（第 737 行之后）：

```python
def crouch_height_target(
    phase: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
) -> torch.Tensor:
    """Cible de hauteur du tronc « en trapèze » le long de la phase [0,1).

    phase ∈ [0, hold_lo)      : descente   height_high -> height_low
    phase ∈ [hold_lo, hold_hi): palier      height_low   (la glisse accroupie)
    phase ∈ [hold_hi, 1.0)    : remontée    height_low  -> height_high

    Args:
        phase: (B,) phase par env, dans [0, 1).
        height_low: hauteur du tronc accroupi (m).
        height_high: hauteur du tronc debout (m).
        hold_lo, hold_hi: bornes du palier bas en fraction de phase.
    Returns:
        (B,) hauteur-cible en mètres.
    """
    descend = phase < hold_lo
    hold = (phase >= hold_lo) & (phase < hold_hi)

    frac_d = phase / hold_lo
    t_descend = height_high + (height_low - height_high) * frac_d

    t_hold = torch.full_like(phase, height_low)

    frac_r = (phase - hold_hi) / (1.0 - hold_hi)
    t_rise = height_low + (height_high - height_low) * frac_r

    return torch.where(descend, t_descend, torch.where(hold, t_hold, t_rise))
```

- [ ] **第 4 步：运行测试以确认其通过**

运行：`uv run --with pytest pytest tests/test_crouch_glide.py -v`
预期：通过（4 个测试）

- [ ] **第 5 步：提交**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_crouch_glide.py
git commit -m "roller-crouch: cible de hauteur en trapezoide (fonction pure + tests)"
```

---

## 任务 2：crouch-glide 和 forward-speed rewards

**文件：**
- 修改：`src/mjlab_microduck/tasks/mdp.py`
- 测试：`tests/test_crouch_glide.py`（补充）

**接口：**
- 消费：`crouch_height_target`（任务 1）。
- 产出：
  - `crouch_glide_reward_from_values(com_height, cmd_cos, cmd_sin, height_low, height_high, hold_lo=0.375, hold_hi=0.625, std=0.02) -> torch.Tensor`（纯函数）。
  - `crouch_glide_height_by_phase(env, command_name="twist", height_low=0.075, height_high=0.11, hold_lo=0.375, hold_hi=0.625, std=0.02, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor`（env 包装器）。
  - `forward_speed_reward(env, vel_ref=0.2, asset_cfg=_DEFAULT_ASSET_CFG) -> torch.Tensor` — 奖励前进速度（保持动能），独立于命令。

- [ ] **第 1 步：编写失败测试**

向 `tests/test_crouch_glide.py` 补充：

```python
def test_reward_is_one_when_height_matches_target():
    # phase 0.5 (plein palier) → cible = height_low ; si com_height == height_low → reward 1
    cmd_cos = torch.tensor([math.cos(2 * math.pi * 0.5)])  # -1
    cmd_sin = torch.tensor([math.sin(2 * math.pi * 0.5)])  # ~0
    com_height = torch.tensor([0.075])
    r = mdp.crouch_glide_reward_from_values(
        com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02
    )
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-3)


def test_reward_decays_when_off_by_one_std():
    # à height_low + std de la cible → exp(-1) ≈ 0.368
    cmd_cos = torch.tensor([math.cos(2 * math.pi * 0.5)])
    cmd_sin = torch.tensor([math.sin(2 * math.pi * 0.5)])
    com_height = torch.tensor([0.075 + 0.02])
    r = mdp.crouch_glide_reward_from_values(
        com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02
    )
    assert torch.allclose(r, torch.tensor([math.exp(-1.0)]), atol=1e-3)


def test_reward_at_phase_zero_expects_high_stance():
    # phase 0 → cible = height_high ; rester debout est récompensé, être accroupi non
    cmd_cos = torch.tensor([1.0, 1.0])   # cos(0)
    cmd_sin = torch.tensor([0.0, 0.0])   # sin(0)
    com_height = torch.tensor([0.11, 0.075])  # debout vs accroupi
    r = mdp.crouch_glide_reward_from_values(
        com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02
    )
    assert r[0] > 0.99          # debout à phase 0 → ~1
    assert r[1] < 0.2           # accroupi à phase 0 → faible
```

- [ ] **第 2 步：确认失败**

运行：`uv run --with pytest pytest tests/test_crouch_glide.py -v`
预期：失败 — `crouch_glide_reward_from_values` 不存在。

- [ ] **第 3 步：实现这三个函数**

在 `src/mjlab_microduck/tasks/mdp.py` 中，`crouch_height_target` 之后：

```python
def crouch_glide_reward_from_values(
    com_height: torch.Tensor,
    cmd_cos: torch.Tensor,
    cmd_sin: torch.Tensor,
    height_low: float,
    height_high: float,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
) -> torch.Tensor:
    """Récompense gaussienne du suivi de la cible de hauteur (fonction pure).

    Décode la phase depuis [cos, sin] puis compare la hauteur mesurée à la
    cible-trapèze. Retourne exp(-((h - cible)/std)^2) ∈ (0, 1].
    """
    phase = (torch.atan2(cmd_sin, cmd_cos) / (2 * torch.pi)) % 1.0
    target = crouch_height_target(phase, height_low, height_high, hold_lo, hold_hi)
    return torch.exp(-((com_height - target) / std) ** 2)


def crouch_glide_height_by_phase(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    height_low: float = 0.075,
    height_high: float = 0.11,
    hold_lo: float = 0.375,
    hold_hi: float = 0.625,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward principale : suit la cible de hauteur du tronc le long de la phase.

    La hauteur du CoM est calculée comme dans `com_height_target` (world z moins
    l'origine du terrain, nan->0). La phase provient de la commande GroundPick.
    """
    asset: Entity = env.scene[asset_cfg.name]
    com_height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0
    )
    cmd = env.command_manager.get_command(command_name)
    return crouch_glide_reward_from_values(
        com_height, cmd[:, 0], cmd[:, 1],
        height_low, height_high, hold_lo, hold_hi, std,
    )


def forward_speed_reward(
    env: ManagerBasedRlEnv,
    vel_ref: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Récompense la vitesse avant du tronc (conserver l'élan / ne pas freiner).

    Indépendante de la commande (la commande porte la phase, pas la vitesse).
    tanh(clamp(vx, 0)/vel_ref) → sature à ~1, ne récompense jamais reculer.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vx = asset.data.root_link_lin_vel_b[:, 0]
    return torch.tanh(torch.clamp(vx, min=0.0) / vel_ref)
```

- [ ] **第 4 步：确认通过**

运行：`uv run --with pytest pytest tests/test_crouch_glide.py -v`
预期：通过（共 7 个测试）

- [ ] **第 5 步：提交**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_crouch_glide.py
git commit -m "roller-crouch: rewards crouch-glide-height et forward-speed"
```

---

## 任务 3：环境 + 任务注册

**文件：**
- 创建：`src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py`
- 修改：`src/mjlab_microduck/tasks/__init__.py`
- 测试：`tests/test_roller_crouch_cfg.py`

**接口：**
- 消费：`crouch_glide_height_by_phase`、`forward_speed_reward`、`ground_pick_return_pose`（任务 2 + 已有）、`GroundPickPhaseCommandCfg`、`GroundPickPhaseCommand`、`MICRODUCK_WALK_ROLLERS_ROBOT_CFG`。
- 产出：`make_microduck_roller_crouch_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`、`MicroduckRollerCrouchRlCfg`、任务 `Mjlab-RollerCrouch-Flat-MicroDuck`。

- [ ] **第 1 步：编写失败冒烟测试**

创建 `tests/test_roller_crouch_cfg.py`：

```python
from mjlab_microduck.tasks.microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


def test_cfg_uses_phase_command():
    cfg = make_microduck_roller_crouch_env_cfg()
    assert isinstance(
        cfg.commands["twist"], microduck_mdp.GroundPickPhaseCommandCfg
    )
    assert cfg.commands["twist"].period == 4.0


def test_cfg_has_crouch_and_forward_rewards():
    cfg = make_microduck_roller_crouch_env_cfg()
    assert "crouch_glide_height" in cfg.rewards
    assert "forward_speed" in cfg.rewards
    # rewards de patinage actif retirées (pas de stride pendant le trick)
    for gone in ("braking", "skating_air_time", "single_support", "glide", "wheel_speed"):
        assert gone not in cfg.rewards


def test_cfg_has_entry_velocity_event():
    cfg = make_microduck_roller_crouch_env_cfg()
    assert "entry_velocity" in cfg.events
```

- [ ] **第 2 步：确认失败**

运行：`uv run --with pytest pytest tests/test_roller_crouch_cfg.py -v`
预期：失败 — `ModuleNotFoundError: ...microduck_roller_crouch_env_cfg`

- [ ] **第 3 步：创建环境文件**

创建 `src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py`：

```python
"""Microduck roller crouch-glide task.

Geste one-shot déclenché au bouton A via le slot --ground-pick du runtime :
le robot s'accroupit et glisse sur son élan (palier ~1 s), puis se relève et
rend la main à la policy roller.

Hybride :
  - physique / robot roller  ← microduck_velocity_rollers_env_cfg.py
  - machinerie phase one-shot ← microduck_ground_pick_env_cfg.py
    (commande GroundPickPhaseCommand : [cos(2πφ), sin(2πφ), 0], période 4 s)

Cible de hauteur « en trapèze » (haut→bas→palier 1 s→haut) via
crouch_glide_height_by_phase. Obs 61D unifié → interchangeable au runtime.
"""

import math
from copy import deepcopy

ENABLE_SYMMETRY = False

# DR — repris du roller env
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_WHEEL_FRICTION_RANDOMIZATION  = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE          = 0.003
HEAD_COM_RANDOMIZATION_RANGE     = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE     = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S         = (3.0, 6.0)
VELOCITY_PUSH_RANGE              = (-0.2, 0.2)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE               = (-0.015, 0.015)

# Geste : hauteurs cibles (m) et vitesse d'entrée (élan)
CROUCH_HEIGHT_HIGH = 0.11    # tronc debout
CROUCH_HEIGHT_LOW  = 0.075   # tronc accroupi (à affiner en play)
CROUCH_STD         = 0.02
ENTRY_VELOCITY_X   = (0.2, 0.5)  # m/s : le robot arrive en roulant

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_roller_crouch_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Env crouch-glide sur rollers, piloté par la phase du slot ground-pick."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(roller_blade|roller_blade_2)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === REWARDS ===
    keep = {"upright", "body_ang_vel", "angular_momentum", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -1.0

    # Reward principale : cible de hauteur trapèze le long de la phase
    cfg.rewards["crouch_glide_height"] = RewardTermCfg(
        func=microduck_mdp.crouch_glide_height_by_phase,
        weight=4.0,
        params={
            "command_name": "twist",
            "height_low": CROUCH_HEIGHT_LOW,
            "height_high": CROUCH_HEIGHT_HIGH,
            "hold_lo": 0.375,
            "hold_hi": 0.625,
            "std": CROUCH_STD,
        },
    )
    # Conserver l'élan (ne pas freiner) — indépendant de la commande
    cfg.rewards["forward_speed"] = RewardTermCfg(
        func=microduck_mdp.forward_speed_reward,
        weight=2.0,
        params={"vel_ref": 0.2},
    )
    # Fin de phase : converger vers la pose roller debout pour rendre la main proprement
    _LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    _NECK_JOINTS = [5, 6, 7, 8]
    cfg.rewards["return_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose,
        weight=3.0,
        params={"std": 0.3, "command_name": "twist", "joint_indices": _LEG_JOINTS},
    )
    cfg.rewards["return_pose_neck"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose,
        weight=3.0,
        params={"std": 0.15, "command_name": "twist", "joint_indices": _NECK_JOINTS},
    )
    # Stabilité de glisse
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-0.5
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )

    # === TERMINATIONS ===
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan, time_out=False,
    )

    # === EVENTS ===
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset",
    )
    del cfg.events["foot_friction"]

    # Vitesse d'entrée : le robot démarre en roulant vers l'avant (élan à conserver)
    cfg.events["entry_velocity"] = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="reset",
        params={
            "velocity_range": {"x": ENTRY_VELOCITY_X, "y": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)

    if ENABLE_WHEEL_FRICTION_RANDOMIZATION:
        cfg.events["randomize_wheel_friction"] = EventTermCfg(
            func=dr.dof_frictionloss,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*",)),
                "operation": "abs",
                "ranges": (0.000, 0.000),
            },
        )
    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia, mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # === OBSERVATIONS (unified 61D layout) ===
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel, scale=1.0, params={"asset_cfg": wheel_cfg},
    )

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # === COMMAND: phase (comme ground_pick) ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{**vars(command), "class_type": microduck_mdp.GroundPickPhaseCommand, "period": 4.0}
    )

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === CURRICULUM ===
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.5},
                {"step": 250 * 24, "weight": -0.8},
                {"step": 500 * 24, "weight": -1.0},
            ],
        },
    )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * 24, "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    return cfg


MicroduckRollerCrouchRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_crouch",
    run_name="roller_crouch",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8_000,
)
```

- [ ] **第 4 步：注册任务**

在 `src/mjlab_microduck/tasks/__init__.py` 中，在 rollers 代码块之后添加导入（第 54 行之后）：

```python
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
```

并在 rollers 代码块之后添加注册（第 175 行之后）：

```python
register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ RollerCrouch task registered: Mjlab-RollerCrouch-Flat-MicroDuck")
```

- [ ] **第 5 步：确认冒烟测试通过**

运行：`uv run --with pytest pytest tests/test_roller_crouch_cfg.py -v`
预期：通过（3 个测试）。（该测试会构建 env — 它会编译 MuJoCo spec，因此较慢；这很正常。）

- [ ] **第 6 步：确认任务已正确注册**

运行：`uv run python -c "import mjlab_microduck.tasks"`
预期：无错误地显示 `✓ RollerCrouch task registered: Mjlab-RollerCrouch-Flat-MicroDuck`。

- [ ] **第 7 步：提交**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py \
        src/mjlab_microduck/tasks/__init__.py tests/test_roller_crouch_cfg.py
git commit -m "roller-crouch: env crouch-glide + enregistrement de la tache"
```

---

## 任务 4：训练冒烟运行（runtime 验证）

**文件：** 无（观察性验证）。

**接口：**
- 消费：任务 `Mjlab-RollerCrouch-Flat-MicroDuck`（任务 3）。

- [ ] **第 1 步：运行一个极短的训练**

运行：
```bash
uv run train Mjlab-RollerCrouch-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max_iterations 5
```
预期：训练启动，记录 rewards（包括 `crouch_glide_height`、`forward_speed`），5 次迭代无崩溃，写出一个 checkpoint。

- [ ] **第 2 步：确认无观测形状错误**

检查启动日志：actor 观测必须是 **61D**（与该策略家族的其他 policy 一致）。如果维度不同，说明 head/body 填充或轮子排除接线有误 — 在继续前修复。

- [ ] **第 3 步：提交（如有配置文件中需要调整的内容）**

```bash
git add -A && git commit -m "roller-crouch: ajustement post smoke-run"
```
（如果没有需要提交的内容，跳过此步。）

---

## 任务 5：完整训练 + play 验证

**文件：** 可能对 `microduck_roller_crouch_env_cfg.py` 迭代（reward 权重、`CROUCH_HEIGHT_LOW`）。

- [ ] **第 1 步：运行完整训练**

运行：
```bash
uv run train Mjlab-RollerCrouch-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max_iterations 8000
```

- [ ] **第 2 步：在 play 中观察**

运行：`uv run scripts/play_latest.py`（或该项目中此任务的 play 入口）。
观察循环：机器人**下降**，在轮子持续转动期间**滑行约 1 秒**（它不刹车），然后**站起身来**，最终姿态回归 roller 站立姿态。它不应摔倒。

- [ ] **第 3 步：必要时迭代**

典型调整（在 `microduck_roller_crouch_env_cfg.py` 中）：
- 它下降得不够 → 降低 `CROUCH_HEIGHT_LOW`（例如 0.07）和/或调高 `crouch_glide_height` 的权重。
- 它在蹲下时刹车 → 调高 `forward_speed` 的权重。
- 它在低位摔倒 → 调高 `upright`，降低入口速度 `ENTRY_VELOCITY_X`，或缩短保持期（使 `hold_lo`/`hold_hi` 更接近）。
- 起身过于猛烈 → 调高 `return_pose_*` 和/或 `action_rate_l2`。

每次更改后，重新运行训练并再次观察。提交每个被采用的调整：
```bash
git add src/mjlab_microduck/tasks/microduck_roller_crouch_env_cfg.py
git commit -m "roller-crouch: reglage <ce qui a change>"
```

---

## 任务 6：ONNX 导出 + 部署到机器人

**文件：** 无（手动/硬件）。

- [ ] **第 1 步：将 policy 导出为 ONNX**

运行：`uv run scripts/export_latest.py`（观测归一化器由 `scripts/export.py` 烘焙进图中）。
获取 `.onnx` 文件，重命名为 `roller_crouch.onnx`，复制到机器人上（例如 `~/microduck/policies/roller_crouch.onnx`）。

- [ ] **第 2 步：使用 ground-pick 槽位运行 runtime**

在机器人上：
```bash
microduck_runtime --variant pre-alpha --new-cmd-obs --roller \
  --model output.onnx \
  --new-dxl-imu --kp 200 --action-scale 0.8 \
  --max-linear-vel 0.6 --max-linear-vel-backward 0.5 --max-angular-vel 0.0 \
  --ground-pick ~/microduck/policies/roller_crouch.onnx \
  --ground-pick-period 5.0 \
  --ground-pick-kp-ratio 1.0 \
  --ground-pick-action-scale 0.8
```

**关键参数（sim2real 一致性）：**
- `--ground-pick-kp-ratio 1.0` — 默认值 0.6 会把 kp 降到 120，而我们在 200 上训练。
- `--ground-pick-action-scale 0.8` — 必须匹配训练时的 `action_scale`。
- `--ground-pick-period 5.0` — 必须匹配训练时的周期。

- [ ] **第 3 步：测试该动作**

让机器人以小速度向前，按下 **A**。验证：它蹲下，滑行约 1 秒，站起身来，然后 roller policy 干净地接管。如果不够稳定，回到任务 5（迭代权重/高度/入口速度）。

---

## 验证笔记（自检）

- **规格覆盖：** 1 秒梯形目标（任务 1）；crouch + 防刹车 + return-pose rewards（任务 2/3）；roller 机器人 + 相位 + 61D 观测 + DR（任务 3）；入口速度（任务 3，`entry_velocity` 事件）；部署标志包括 `kp-ratio` 陷阱（任务 6）。✅
- **相位 vs 速度陷阱：** roller env 的 `wheel_speed_reward`/`braking`/`coasting_reward` 把 `command[:,0]` 用作*速度* — 在这里无效，因为 `command[:,0]=cos(2πφ)`。因此它们**被移除**，并由 `forward_speed_reward`（独立于命令）取代。由 `test_cfg_has_crouch_and_forward_rewards` 测试。
- **命名一致性：** `crouch_glide_height`（reward 键）与 `crouch_glide_height_by_phase`（函数）— 这是有意的：键是 term 名，函数是 `func=` 指向的函数。
```
