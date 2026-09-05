# Roller StandUp — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 一个专用的 policy `Mjlab-RollerStandUp-Flat-MicroDuck`，让 microduck 摔倒（背朝下或脸朝下）之后能在自己的 rollers 上重新站起，并且能保持站在轮子上。

**架构：** 只有一个新的 env 文件，从 `make_microduck_velocity_rollers_env_cfg()` 派生 —— 这样它就自动继承了 rollers 机器人、传感器、全部 domain randomization 和 61D 观测（这是运行时可互换性的硬条件）。移除滑行（patinage）奖励，移植 `standup` 的十个站起奖励（重映射到 rollers 模型的关节索引，轮子是被动关节、交错排列在其中），把 reset 替换为地面出发，并反转滚动摩擦的 curriculum（轮子从制动 → 自由），以便在施加真实轮子物理之前先引导出该动作。

**技术栈：** Python 3.12, mjlab 1.3.0, MuJoCo / mujoco-warp, rsl_rl (PPO), uv, pytest.

参考 Spec：`docs/superpowers/specs/2026-08-04-roller-standup-design.md`

## 全局约束

- **不改动** `src/mjlab_microduck/tasks/mdp.py`，也不改 `roller`、`roller_crouch`、`roller_slope`、`standup`、`velstand` 这些 env 文件。所有需要的 mdp 函数都已存在。
- **必须与** `make_microduck_velocity_rollers_env_cfg()` **保持 61D 观测对齐**：`[gyro(3), projected_gravity(3), joint_pos(14), joint_vel(14), last_action(14), command(13)]`。`head_pose`（4）和 `body_pose`（6）这些槽位保持 **zero-padding**。没有这种对齐，ONNX 就无法加载进 runtime 的某个槽位。
- **rollers 模型的关节索引**（被动轮子交错排列；已在 MuJoCo 中验证）：
  `_LEG_JOINTS = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]`, `_NECK_JOINTS = [7, 8, 9, 10]`, `_WHEEL_JOINTS = [5, 6, 16, 17]`.
  切不可**复用** `standup` 的索引（`[0-4, 9-13]` / `[5-8]`），那些只适用于无轮的模型。
- **实测高度**：`ROLLER_STAND_Z = 0.138`, `ROLLER_PRONE_Z = 0.075`。不要用 `standup` 的值（0.115 / 0.07）去替换。
- `EPISODE_LENGTH_S = 6.0`, `NUM_STEPS_PER_ENV = 24`。curriculum 的 `step` 以 `iters × NUM_STEPS_PER_ENV` 表示。
- **对称性 OFF**：`symmetry_cfg=None`。`SYMMETRY_CFG` 是按旧的 51D 布局接线的，在 61D 上会崩。
- 仓库风格：rollers 环境里的注释用法语，4 空格缩进，`SceneEntityCfg` **在每个 term 里重建**（绝不要共享对象 —— mjlab 会就地解析并修改这些对象）。
- 简单的提交，不带 `Co-Authored-By`。
- **已存在、不在范围内**：`tests/test_wheel_glide.py` 在本次工作之前有 4 个失败的测试（一个带过期正则 `passive_LF_?wheel` 的错误 asset）。不要修它们，也不要因此惊慌。其余测试通过（46 个）。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`（新建） | env 的全部配置 + `MicroduckRollerStandUpRlCfg`。单文件，和本仓库其他所有 env 一致。 |
| `src/mjlab_microduck/tasks/__init__.py`（修改） | 新任务的 import + `register_mjlab_task`。 |
| `tests/test_roller_standup_cfg.py`（新建） | 构建配置的测试 + 关节索引的锁定。无 sim、无 GPU（类似 `test_roller_slope_cfg.py`）。 |
| `docs/roller_standup_policy_summary.md`（新建，Task 5） | 交接摘要，仿照 `docs/roller_slope_policy_summary.md` 的格式。 |

---

## Task 1 : Env 骨架 —— 派生、命令中和、移除滑行、注册

**Files:**
- Create: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- Modify: `src/mjlab_microduck/tasks/__init__.py`
- Test: `tests/test_roller_standup_cfg.py`

**Interfaces:**
- Consumes: `make_microduck_velocity_rollers_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`（已存在），`microduck_mdp.VelocityCommandCommandOnlyCfg`（已存在）。
- Produces: `make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg`；`MicroduckRollerStandUpRlCfg: RslRlOnPolicyRunnerCfg`；模块常量 `ROLLER_STAND_Z: float`, `ROLLER_PRONE_Z: float`, `EPISODE_LENGTH_S: float`, `NUM_STEPS_PER_ENV: int`, `_LEG_JOINTS: list[int]`, `_NECK_JOINTS: list[int]`, `_WHEEL_JOINTS: list[int]`；已注册的任务 `"Mjlab-RollerStandUp-Flat-MicroDuck"`。

- [ ] **Step 1 : 编写先失败的测试**

创建 `tests/test_roller_standup_cfg.py`：

```python
from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
    EPISODE_LENGTH_S,
    make_microduck_roller_standup_env_cfg,
)
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)

# Récompenses de PATINAGE : elles ne doivent pas survivre dans un env de relevé.
SKATING_REWARDS = (
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "feet_flat",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    "upright",
)


def test_env_builds_train_and_play():
    assert make_microduck_roller_standup_env_cfg() is not None
    assert make_microduck_roller_standup_env_cfg(play=True) is not None


def test_episode_is_short():
    # Épisode court : monter puis stabiliser, comme standup (6 s).
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 6.0


def test_no_skating_rewards_survive():
    cfg = make_microduck_roller_standup_env_cfg()
    for name in SKATING_REWARDS:
        assert name not in cfg.rewards, f"reward de patinage survivante : {name}"


def test_smoothness_regularisers_kept():
    # Gardées de l'héritage roller : le relevé a besoin de douceur sim2real, mais
    # body_ang_vel doit rester LÉGER (standup documente qu'à -0.15 il gelait).
    cfg = make_microduck_roller_standup_env_cfg()
    for name in (
        "action_over_limit",
        "self_collisions",
        "body_ang_vel",
        "angular_momentum",
        "action_rate_l2",
        "neck_action_rate_l2",
        "neck_joint_pos_l2",
        "joint_torques_l2",
    ):
        assert name in cfg.rewards, f"régularisateur perdu : {name}"
    assert cfg.rewards["body_ang_vel"].weight == -0.05


def test_twist_command_is_neutralised():
    # Pas de pilotage : la policy se déploie en --standing, où le runtime laisse
    # le slot twist à zéro (cf. infer_policy.py:239).
    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.ranges.lin_vel_x == (-0.01, 0.01)
    assert cmd.ranges.lin_vel_y == (-0.01, 0.01)
    assert cmd.ranges.ang_vel_z == (-0.05, 0.05)
    assert cmd.heading_command is False
    assert cmd.ranges.heading is None
    assert cmd.rel_standing_envs == 0.0


def test_twist_command_is_not_heading_relative():
    # L'env roller installe un RelativeHeadingVelocityCommandCfg (cmd[2] = erreur
    # de cap, calculée en interne). Ici cmd[2] doit être un vrai zéro bruité.
    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = make_microduck_roller_standup_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.VelocityCommandCommandOnlyCfg)
    assert not isinstance(cmd, microduck_mdp.RelativeHeadingVelocityCommandCfg)


def test_obs_nan_policy_sanitize():
    # Un contact rare fait diverger le free-joint en NaN : on assainit l'obs
    # plutôt que de tuer l'entraînement (même choix que roller_slope).
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.observations["actor"].nan_policy == "sanitize"
    assert cfg.observations["critic"].nan_policy == "sanitize"


def test_obs_parity_with_roller_env():
    # Parité 61D obligatoire : sinon l'ONNX ne se charge pas dans un slot runtime.
    standup = make_microduck_roller_standup_env_cfg()
    roller = make_microduck_velocity_rollers_env_cfg()
    for grp in ("actor", "critic"):
        assert list(standup.observations[grp].terms.keys()) == list(
            roller.observations[grp].terms.keys()
        ), f"layout d'observation divergent sur le groupe {grp}"


def test_terrain_is_plain_plane():
    # Hérité de l'env roller : sol plat, pas de générateur. Pas de variante rough
    # pour cette v1.
    cfg = make_microduck_roller_standup_env_cfg()
    assert cfg.scene.terrain.terrain_type == "plane"
    assert cfg.scene.terrain.terrain_generator is None


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks

    import mjlab_microduck.tasks  # noqa: F401  (l'import déclenche l'enregistrement)

    assert "Mjlab-RollerStandUp-Flat-MicroDuck" in list_tasks()
```

- [ ] **Step 2 : 运行测试确认它们会失败**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
预期：collect 报错，`ModuleNotFoundError: No module named 'mjlab_microduck.tasks.microduck_roller_standup_env_cfg'`。

- [ ] **Step 3 : 创建 env 文件**

创建 `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`：

```python
"""Microduck roller standup — se relever sur rollers.

Policy DÉDIÉE épisodique : le robot démarre au sol (à plat ventre, à plat dos) ou
déjà debout, et doit se remettre debout sur ses rollers puis TENIR la station.
Portage de la recette `standup` (canard marcheur) vers le modèle rollers.

Dérive de l'env roller (`make_microduck_velocity_rollers_env_cfg`) → hérite tel
quel le robot rollers, les capteurs, toute la DR et l'observation 61D, donc
interchangeable au runtime (--new-cmd-obs). C'est le pattern de roller_slope.

Deux différences structurelles avec `standup` :
  - les roues passives sont INTERCALÉES dans l'ordre des joints → indices
    remappés (_LEG_JOINTS ci-dessous), verrouillés par
    tests/test_roller_standup_cfg.py ;
  - pas de commande head_pose : les slots head/body restent zero-paddés
    (convention de la famille roller) et la tête est tenue droite par
    neck_joint_pos_l2, qui résout par NOM.

La pièce nouvelle est le curriculum de friction de roulement, INVERSÉ (roues
freinées → libres) : les roues roulent, donc il n'y a aucune adhérence pour
pousser sur le sol. On bootstrappe avec des roues quasi bloquées puis on rampe
vers la vraie valeur. Si `standing_composite` s'écroule à un palier, le geste
« pieds adhérents » ne transfère pas et il faudra guider une technique de
patineur (appui genou, un patin à la fois).

Déploiement visé : en `--standing` face à la policy roller en `--walking`, avec
la bascule automatique sur la magnitude de la commande de vitesse
(infer_policy.py:262, seuil 0.05) ; le slot twist y est laissé à zéro
(infer_policy.py:239).
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# ── Hauteurs de tronc (m) ─────────────────────────────────────────────────────
# Mesurées par cinématique exacte (minimum des sommets de maillage des géoms
# collidantes, pose STAND, tronc ramené au contact) sur scene_rollers.xml :
# debout 0.1407, repos à plat ventre 0.0752, repos à plat dos 0.0475.
# Contrôle : le modèle SANS roues donne 0.1172 en cinématique contre STAND_Z=0.115
# mesuré sous charge par standup → ~2 mm d'affaissement, appliqué ici aussi.
# 0.138 tombe dans le reset_base z (0.1335–0.1435) déjà utilisé par l'env roller.
ROLLER_STAND_Z = 0.138
ROLLER_PRONE_Z = 0.075

EPISODE_LENGTH_S  = 6.0   # monter + stabiliser, comme standup
NUM_STEPS_PER_ENV = 24

# ── Indices de joints — les roues passives sont INTERCALÉES ───────────────────
# Ordre réel du modèle rollers (18 joints après le free-joint), vérifié dans
# MuJoCo via get_walk_rollers_spec().compile() :
#   0-4   left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle
#   5-6   passive_LF_wheel, passive_LR_wheel
#   7-10  neck_pitch, head_pitch, head_yaw, head_roll
#   11-15 right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
#   16-17 passive_RF_wheel, passive_RR_wheel
# Le standup utilise [0-4, 9-13] / [5-8] : ce sont les indices du modèle SANS
# roues, ils ne valent PAS ici. Verrouillé par tests/test_roller_standup_cfg.py.
#
# Seul _LEG_JOINTS est consommé (par les récompenses de pose). _NECK_JOINTS et
# _WHEEL_JOINTS servent à la documentation et au test d'indices : le cou est
# résolu par NOM (neck_joint_pos_l2 appelle find_joints(r".*(neck|head).*") à
# chaque pas) et les roues par la regex ^passive_.*.
_LEG_JOINTS   = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
_NECK_JOINTS  = [7, 8, 9, 10]
_WHEEL_JOINTS = [5, 6, 16, 17]

# Récompenses de PATINAGE de l'env roller : aucun sens quand on est par terre.
# feet_flat : les lames ne sont PAS à plat pendant la montée → combattrait le geste.
# hip_roll_neutral : se relever demande d'écarter les jambes.
# pose / com_height_target : remplacés par les cibles pose/hauteur du relevé.
# upright (gaussienne de base) : remplacée par upright_linear + upright_sharp.
_SKATING_REWARDS = (
    "wheel_speed",
    "braking",
    "skating_air_time",
    "glide",
    "single_support",
    "gait_symmetry",
    "forward_lean",
    "heading_hold",
    "feet_flat",
    "hip_roll_neutral",
    "pose",
    "com_height_target",
    "upright",
)


def make_microduck_roller_standup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Env « se relever sur rollers » : départ au sol, cible = debout sur roues."""
    cfg = make_microduck_velocity_rollers_env_cfg(play=play)

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Récompenses de patinage retirées ─────────────────────────────────────
    for name in _SKATING_REWARDS:
        cfg.rewards.pop(name, None)

    # ── Commande : slot twist neutralisé (≈ 0) ───────────────────────────────
    # L'env roller installe un RelativeHeadingVelocityCommandCfg (cmd[2] = erreur
    # de cap calculée en interne). Ici on ne pilote rien : on repasse au
    # command-only neutralisé, comme standup. Les slots head_pose (4) et
    # body_pose (6) restent zero-paddés → parité d'obs 61D préservée.
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

    # ── Robustesse numérique (même choix que roller_slope) ───────────────────
    # Un contact rare (~1/25M pas) fait diverger le free-joint en NaN : on
    # assainit l'obs (→ 0) pour ne pas tuer l'entraînement, l'env fautif se reset
    # au pas suivant.
    for grp in ("actor", "critic"):
        cfg.observations[grp].nan_policy = "sanitize"

    return cfg


# ── Config du runner RL — identique à standup ─────────────────────────────────
MicroduckRollerStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # le normaliseur DOIT être baké dans l'ONNX par export.py
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
        # Symétrie OFF : SYMMETRY_CFG est câblé pour l'ancien layout 51D et casse
        # sur le 61D (même situation que tous les envs v1.5+).
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_standup",
    run_name="roller_standup",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=15_000,
)
```

- [ ] **Step 4 : 注册任务**

在 `src/mjlab_microduck/tasks/__init__.py` 中，把 import 加在 `microduck_roller_slope_env_cfg` 的 import 块 **之后**：

```python
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
```

然后在文件最末尾（在 `Mjlab-RollerSlope-Flat-MicroDuck` 的注册之后）：

```python
# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ RollerStandUp task registered: Mjlab-RollerStandUp-Flat-MicroDuck")
```

- [ ] **Step 5 : 运行测试确认它们通过**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
预期：10 passed。
如果 `test_obs_parity_with_roller_env` 失败，说明有什么地方动了观测 —— 在继续之前修好它，这是项目的硬约束。

- [ ] **Step 6 : 确认没有其他测试回归**

```bash
uv run --with pytest pytest tests/ -q
```
预期：`4 failed, 56 passed` —— 那 4 个失败是 `tests/test_wheel_glide.py` 里本就存在的（本次工作之前该套件是 `4 failed, 46 passed`）。其他都没有失败。

- [ ] **Step 7 : Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py \
        src/mjlab_microduck/tasks/__init__.py \
        tests/test_roller_standup_cfg.py
git commit -m "roller-standup: squelette de l'env (dérivé roller, twist neutralisé)"
```

---

## Task 2 : 站起奖励 + 关节索引锁定

**Files:**
- Modify: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- Test: `tests/test_roller_standup_cfg.py`

**Interfaces:**
- Consumes: 来自 Task 1 —— `make_microduck_roller_standup_env_cfg`, `ROLLER_STAND_Z`, `ROLLER_PRONE_Z`, `_LEG_JOINTS`, `_NECK_JOINTS`, `_WHEEL_JOINTS`。来自 `mdp.py`（已存在、未修改）：`pose_target_match(target_overrides, asset_cfg, std, joint_indices)`、`pose_l1_penalty(target_overrides, asset_cfg, joint_indices)`、`height_target_gaussian(target_height, asset_cfg, std)`、`height_l1_penalty(target_height, asset_cfg)`、`com_upward_velocity(asset_cfg, max_height)`、`trunk_vertical_accel_penalty(asset_cfg)`、`body_upright_linear(asset_cfg)`、`upright_gaussian_at_height(std, height_low, height_high, asset_cfg)`、`standing_composite_score(target_height, height_std, upright_std, pose_std, joint_indices, target_overrides, asset_cfg)`、`joint_torque_rate_l2()`。
- Produces: 这些奖励项 `pose_stand_legs`、`pose_stand_l1`、`height_stand`、`height_stand_sharp`、`height_stand_l1`、`com_upward_velocity`、`gentle_rise`、`upright_linear`、`upright_sharp`、`standing_composite`、`joint_torque_rate_l2` 位于 `cfg.rewards` 中。

- [ ] **Step 1 : 编写先失败的测试**

在 `tests/test_roller_standup_cfg.py` 末尾追加：

```python
def test_joint_indices_match_actual_roller_model():
    """Verrou : les roues passives sont intercalées dans l'ordre des joints.

    Réutiliser les indices du standup ([0-4, 9-13]) donnerait des récompenses
    qui pointent sur des roues. Ce test compile le vrai MjSpec du robot rollers
    et vérifie les noms aux indices utilisés. Pur CPU, pas de sim.
    """
    import mujoco

    from mjlab_microduck.robot.microduck_constants import get_walk_rollers_spec
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
        _LEG_JOINTS,
        _NECK_JOINTS,
        _WHEEL_JOINTS,
    )

    model = get_walk_rollers_spec().compile()
    articulated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]

    assert [articulated[i] for i in _LEG_JOINTS] == [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    assert [articulated[i] for i in _NECK_JOINTS] == [
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    ]
    assert [articulated[i] for i in _WHEEL_JOINTS] == [
        "passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel",
    ]
    # Aucun recouvrement, et les trois listes couvrent tous les joints.
    assert len(set(_LEG_JOINTS) | set(_NECK_JOINTS) | set(_WHEEL_JOINTS)) == len(articulated)


def test_recovery_rewards_present_with_expected_weights():
    cfg = make_microduck_roller_standup_env_cfg()
    expected = {
        "pose_stand_legs":      8.0,
        "pose_stand_l1":        5.0,
        "height_stand":         4.0,
        "height_stand_sharp":   4.0,
        "height_stand_l1":     30.0,
        "com_upward_velocity":  3.0,
        "gentle_rise":         -0.02,
        "upright_linear":       6.0,
        "upright_sharp":        6.0,
        "standing_composite":  15.0,
        "joint_torque_rate_l2": -2e-3,
    }
    for name, weight in expected.items():
        assert name in cfg.rewards, f"récompense de relevé manquante : {name}"
        assert cfg.rewards[name].weight == weight, f"poids inattendu sur {name}"


def test_recovery_rewards_use_roller_heights_not_walker_heights():
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import (
        ROLLER_PRONE_Z,
        ROLLER_STAND_Z,
    )

    cfg = make_microduck_roller_standup_env_cfg()
    assert ROLLER_STAND_Z == 0.138  # PAS le 0.115 du modèle sans roues
    for name in ("height_stand", "height_stand_sharp", "height_stand_l1"):
        assert cfg.rewards[name].params["target_height"] == ROLLER_STAND_Z
    assert cfg.rewards["standing_composite"].params["target_height"] == ROLLER_STAND_Z
    # com_upward_velocity se coupe juste AU-DESSUS de la cible (10 mm de marge),
    # sinon la policy se gare à l'altitude de coupure sans finir la montée.
    assert cfg.rewards["com_upward_velocity"].params["max_height"] == ROLLER_STAND_Z + 0.010
    # upright_sharp est gatée entre le repos au sol et la station debout.
    assert cfg.rewards["upright_sharp"].params["height_low"] == ROLLER_PRONE_Z
    assert cfg.rewards["upright_sharp"].params["height_high"] == ROLLER_STAND_Z


def test_pose_rewards_target_legs_only_at_roller_indices():
    from mjlab_microduck.tasks.microduck_roller_standup_env_cfg import _LEG_JOINTS

    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("pose_stand_legs", "pose_stand_l1", "standing_composite"):
        assert cfg.rewards[name].params["joint_indices"] == _LEG_JOINTS
        # target_overrides=None → la cible est HOME (default_joint_pos).
        assert cfg.rewards[name].params["target_overrides"] is None


def test_trunk_asset_cfgs_are_distinct_objects():
    """mjlab résout et MUTE les SceneEntityCfg en place : un objet partagé entre
    plusieurs termes provoque des indices périmés. Chaque terme doit avoir le sien.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    names = (
        "height_stand", "height_stand_sharp", "height_stand_l1",
        "com_upward_velocity", "gentle_rise", "upright_linear",
        "upright_sharp", "standing_composite",
    )
    seen = [id(cfg.rewards[n].params["asset_cfg"]) for n in names]
    assert len(set(seen)) == len(seen), "asset_cfg partagé entre plusieurs termes"
```

- [ ] **Step 2 : 运行测试确认它们会失败**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
预期：`test_joint_indices_match_actual_roller_model` **通过**（Task 1 的常量已经正确 —— 这是一个回归锁，不是红灯测试）；另外 4 个以 `KeyError: 'pose_stand_legs'` 或 `assert 'pose_stand_legs' in cfg.rewards` 失败。

- [ ] **Step 3 : 添加站起奖励**

在 `microduck_roller_standup_env_cfg.py` 中，把这个代码块插到「Robustesse numérique」块 **之后**、`return cfg` **之前**：

```python
    # ── Récompenses de relevé — transplant du standup, remappé ───────────────
    # Les poids viennent des itérations documentées dans
    # microduck_standup_env_cfg.py : ne les retoucher qu'avec une raison. Seuls
    # les indices de joints et les deux hauteurs changent ici.
    # NB : un SceneEntityCfg NEUF par terme — mjlab les résout et les mute en
    # place, un objet partagé donne des indices périmés.

    # Pose cible = HOME (target_overrides=None), JAMBES seulement : le cou et la
    # tête sont tenus par neck_joint_pos_l2 (hérité), qui résout par NOM.
    cfg.rewards["pose_stand_legs"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=8.0,
        params={
            "std": 0.5,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )
    # Bootstrap L1 : gradient constant même loin de HOME (la gaussienne sature).
    cfg.rewards["pose_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=5.0,
        params={
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )

    # Hauteur en trois couches : gaussienne large (tire depuis le sol),
    # gaussienne étroite (force les derniers cm, là où la large est saturée),
    # et L1 fort qui rend « rester par terre » net NÉGATIF — sans lui, la policy
    # se contente de l'optimum paresseux « immobile au sol ».
    cfg.rewards["height_stand"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std": 0.04,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={
            "std": 0.015,
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_stand_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=30.0,
        params={
            "target_height": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Paye le MOUVEMENT de montée, pas seulement la destination : sans ça,
    # « rester assis en collectant la pose partielle » domine. La coupure est
    # 10 mm AU-DESSUS de la cible, sinon la policy se gare à l'altitude de
    # coupure et ne finit pas la montée.
    cfg.rewards["com_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.com_upward_velocity,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "max_height": ROLLER_STAND_Z + 0.010,
        },
    )
    # Montée douce : pénalise |a_z|. Compatible avec com_upward_velocity — une
    # vitesse verticale constante collecte l'une ET a a_z = 0 → les deux
    # pressions sélectionnent ensemble une montée lisse à vitesse constante.
    cfg.rewards["gentle_rise"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Tronc vertical en deux couches : cos(tilt) a un fort gradient quand on est
    # couché mais s'essouffle près de la verticale ; la gaussienne serrée gatée
    # en hauteur prend le relais et tue le penché-arrière (mode d'échec du
    # standup : basculer en arrière en tendant les jambes).
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear,
        weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["upright_sharp"] = RewardTermCfg(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=6.0,
        params={
            "std": 0.3,
            "height_low": ROLLER_PRONE_Z,
            "height_high": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Score MULTIPLICATIF hauteur × verticalité × pose : comme les facteurs se
    # multiplient, être bon sur 2 critères sur 3 ne rapporte rien → casse les
    # compromis « penché à la bonne hauteur » que les récompenses additives
    # laissent passer. Stds volontairement LARGES pour rester visible pendant la
    # montée (des stds serrées donnaient un score ~5e-5, donc zéro gradient).
    cfg.rewards["standing_composite"] = RewardTermCfg(
        func=microduck_mdp.standing_composite_score,
        weight=15.0,
        params={
            "target_height": ROLLER_STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Anti-jitter : pénalise la VARIATION de couple, pas son amplitude ni la
    # rotation du tronc → amortit la tremblote sans bloquer le retournement.
    # Le standup l'a identifié comme le seul amortisseur qui ne tue pas le
    # relevé depuis le dos.
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=-2e-3,
    )
```

- [ ] **Step 4 : 运行测试确认它们通过**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
预期：15 passed。

- [ ] **Step 5 : Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py \
        tests/test_roller_standup_cfg.py
git commit -m "roller-standup: recompenses de relevé + verrou des indices de joints"
```

---

## Task 3 : 地面出发 —— reset、移除 `fell_over`、姿态 curriculum

**Files:**
- Modify: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- Test: `tests/test_roller_standup_cfg.py`

**Interfaces:**
- Consumes: `microduck_mdp.set_random_ground_state(env, env_ids, asset_cfg, face_down_prob, face_up_prob, sitting_prob, standing_prob, prone_z_min, prone_z_max, sitting_z_min, sitting_z_max, standing_z_min, standing_z_max, sitting_joint_overrides, sitting_joint_noise_std, sitting_tilt_max)` 和 `microduck_mdp.event_param_curriculum(env, env_ids, event_name, param_stages)` —— 均已存在、未修改。
- Produces: 事件 `cfg.events["set_ground_state"]` 和 curriculum `cfg.curriculum["ground_state_mix"]`；不含 `fell_over` 的 `cfg.terminations`。

- [ ] **Step 1 : 编写先失败的测试**

在 `tests/test_roller_standup_cfg.py` 末尾追加：

```python
def test_starts_from_ground_states():
    # Ventre + dos + debout. Pas de bucket "assis" : il n'existait dans standup
    # que pour le hand-off depuis la policy sit, dont il n'y a pas d'équivalent
    # roller — et ses sitting_joint_overrides sont des indices du modèle SANS roues.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "set_ground_state" in cfg.events
    params = cfg.events["set_ground_state"].params
    assert params["sitting_prob"] == 0.0
    assert params["sitting_joint_overrides"] is None
    assert params["face_down_prob"] > 0.0
    assert params["standing_prob"] > 0.0
    # face_up (le dos) démarre à 0 : introduit tard par le curriculum.
    assert params["face_up_prob"] == 0.0


def test_ground_state_heights_are_roller_specific():
    cfg = make_microduck_roller_standup_env_cfg()
    params = cfg.events["set_ground_state"].params
    # Repos au sol : géométrie identique aux deux modèles (c'est la coque du
    # tronc qui touche, pas les pieds) → plages du standup réutilisées.
    assert (params["prone_z_min"], params["prone_z_max"]) == (0.05, 0.09)
    # [Corrigé à 0.076 après la revue finale — voir docs/superpowers/specs/2026-08-04-roller-standup-design.md]
    # Debout : hauteur ROLLER (+23 mm vs le modèle sans roues, qui est à 0.11–0.12).
    assert params["standing_z_min"] == 0.134
    assert params["standing_z_max"] == 0.144
    assert params["standing_z_min"] < 0.138 < params["standing_z_max"]


def test_ground_state_event_runs_after_base_reset():
    # set_ground_state écrase la pose posée par reset_base / reset_robot_joints :
    # l'ordre des événements suit l'ordre d'insertion, il doit donc venir APRÈS.
    cfg = make_microduck_roller_standup_env_cfg()
    order = list(cfg.events.keys())
    assert order.index("set_ground_state") > order.index("reset_base")
    assert order.index("set_ground_state") > order.index("reset_robot_joints")


def test_no_fall_termination():
    # Le robot DÉMARRE tombé : une terminaison sur inclinaison tuerait l'épisode
    # au premier pas. nan_state (hérité) reste, lui.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_ground_state_curriculum_ramps_easy_to_hard():
    cfg = make_microduck_roller_standup_env_cfg()
    assert "ground_state_mix" in cfg.curriculum
    stages = cfg.curriculum["ground_state_mix"].params["param_stages"]
    assert cfg.curriculum["ground_state_mix"].params["event_name"] == "set_ground_state"
    # Les steps sont croissants et démarrent à 0.
    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)
    # Le dos (face_up) est introduit tard puis croît de façon monotone.
    face_up = [s["params"]["face_up_prob"] for s in stages]
    assert face_up[0] == 0.0
    assert face_up == sorted(face_up)
    assert face_up[-1] >= 0.35
    # Chaque palier est une distribution valide, et le "déjà debout" ne disparaît
    # jamais (sinon la policy se relève puis retombe faute d'apprendre à tenir).
    for stage in stages:
        p = stage["params"]
        total = (
            p["standing_prob"] + p["sitting_prob"]
            + p["face_down_prob"] + p["face_up_prob"]
        )
        assert abs(total - 1.0) < 1e-9
        assert p["sitting_prob"] == 0.0
        assert p["standing_prob"] > 0.0
```

- [ ] **Step 2 : 运行测试确认它们会失败**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
预期：新增的 5 个失败——`assert 'set_ground_state' in cfg.events`（KeyError / AssertionError）、`assert 'fell_over' not in cfg.terminations`、`assert 'ground_state_mix' in cfg.curriculum`。

- [ ] **Step 3 : 添加地面 reset、移除 `fell_over` 并加 curriculum**

在 `microduck_roller_standup_env_cfg.py` 中，把这个代码块插到站起奖励 **之后**、`return cfg` **之前**：

```python
    # ── Départ AU SOL : à plat ventre / à plat dos / déjà debout ─────────────
    # Ajouté en DERNIER dans cfg.events : l'ordre d'exécution suit l'ordre
    # d'insertion, et ce terme doit écraser la pose posée par reset_base /
    # reset_robot_joints.
    # Le bucket « déjà debout » n'est pas décoratif : sans lui la policy apprend
    # à monter mais pas à TENIR, et elle retombe juste après s'être relevée.
    # Pas de bucket « assis » → aucun sitting_joint_overrides à remapper (ceux du
    # standup sont des indices du modèle SANS roues).
    # Les probabilités ci-dessous = palier 0 du curriculum ground_state_mix.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.50,   # ventre (+90° de pitch)
            "face_up_prob":   0.00,   # dos — le plus dur, introduit tard
            "sitting_prob":   0.00,
            "standing_prob":  0.50,
            "sitting_joint_overrides": None,
            # Repos au sol : mesuré à 0.075 (ventre) / 0.048 (dos), identique aux
            # deux modèles — c'est la coque du tronc qui touche, pas les pieds.
            "prone_z_min":    0.05,
            # [Corrigé à 0.076 après la revue finale — voir docs/superpowers/specs/2026-08-04-roller-standup-design.md]
            "prone_z_max":    0.09,
            # Debout sur roues : ROLLER_STAND_Z = 0.138 (contre 0.11–0.12 sans roues).
            "standing_z_min": 0.134,
            "standing_z_max": 0.144,
            # Bruit de pitch/roll au départ. Attention : dans
            # set_random_ground_state le bucket « debout » réutilise le quaternion
            # du bucket « assis », donc ce bruit s'applique AUSSI aux départs
            # debout — c'est voulu (pas de sur-apprentissage du parfaitement droit).
            "sitting_tilt_max": math.radians(10),
        },
    )

    # Le robot DÉMARRE tombé → la terminaison sur inclinaison n'a aucun sens ici
    # (elle tuerait l'épisode au premier pas). nan_state, hérité, reste.
    cfg.terminations.pop("fell_over", None)

    # Curriculum des poses de départ, easy → hard. Avec un mélange plat dès le
    # départ, la policy optimise la majorité facile et laisse le dos sous-entraîné
    # (leçon du standup : il gelait en « ne rien faire » sur cette pose). On
    # introduit donc debout+ventre d'abord, le dos tard, et on biaise vers les
    # poses dures à la fin pour qu'elles reçoivent le plus d'entraînement.
    cfg.curriculum["ground_state_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                {"step": 0, "params": {
                    "standing_prob": 0.50, "sitting_prob": 0.00,
                    "face_down_prob": 0.50, "face_up_prob": 0.00}},
                {"step": 600 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.35, "sitting_prob": 0.00,
                    "face_down_prob": 0.45, "face_up_prob": 0.20}},
                {"step": 1500 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.25, "sitting_prob": 0.00,
                    "face_down_prob": 0.40, "face_up_prob": 0.35}},
                {"step": 2500 * NUM_STEPS_PER_ENV, "params": {
                    "standing_prob": 0.20, "sitting_prob": 0.00,
                    "face_down_prob": 0.40, "face_up_prob": 0.40}},
            ],
        },
    )
```

- [ ] **Step 4 : 运行测试确认它们通过**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
预期：20 passed。

- [ ] **Step 5 : Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py \
        tests/test_roller_standup_cfg.py
git commit -m "roller-standup: depart au sol (ventre/dos/debout) + curriculum des poses"
```

---

## Task 4 : Curricula —— 反转的滚动摩擦、推搡、action_rate

**Files:**
- Modify: `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- Test: `tests/test_roller_standup_cfg.py`

**Interfaces:**
- Consumes: `microduck_mdp.wheel_friction_curriculum(env, env_ids, event_name, ranges_stages)`、`microduck_mdp.push_curriculum(env, env_ids, event_name, push_stages)`、`microduck_mdp.reward_weight(env, env_ids, reward_name, weight_stages)` —— 均已存在、未修改。从 roller env 继承的事件：`randomize_wheel_friction`、`push_robot`。
- Produces: `cfg.curriculum["wheel_friction"]`（递减）、`cfg.curriculum["push_magnitude"]`、`cfg.curriculum["action_rate_weight"]`（被替换）。

- [ ] **Step 1 : 编写先失败的测试**

在 `tests/test_roller_standup_cfg.py` 末尾追加：

```python
def test_wheel_friction_curriculum_is_decreasing():
    """La pièce nouvelle : roues FREINÉES → LIBRES.

    Les roues roulent, donc il n'y a aucune adhérence longitudinale pour pousser
    sur le sol. On bootstrappe avec des roulements quasi bloqués (le relevé se
    fait comme avec des pieds) puis on rampe vers la vraie valeur. L'env roller,
    lui, fait MONTER cette friction (0 → 0.0015) : le sens est bien inversé ici.
    """
    cfg = make_microduck_roller_standup_env_cfg()
    stages = cfg.curriculum["wheel_friction"].params["ranges_stages"]
    assert cfg.curriculum["wheel_friction"].params["event_name"] == "randomize_wheel_friction"

    steps = [s["step"] for s in stages]
    assert steps[0] == 0 and steps == sorted(steps) and len(set(steps)) == len(steps)

    lows = [s["ranges"][0] for s in stages]
    assert lows == sorted(lows, reverse=True), "la friction doit DÉCROÎTRE"
    assert lows[0] >= 0.02, "départ franchement freiné pour bootstrapper le geste"
    # Arrivée sur la vraie valeur du roulement (celle de l'env roller).
    assert stages[-1]["ranges"] == (0.0015, 0.0015)
    for stage in stages:
        assert stage["ranges"][0] == stage["ranges"][1]


def test_wheel_friction_event_starts_at_stage_zero():
    # Le curriculum n'est évalué qu'à partir du premier pas : sans ça les tout
    # premiers resets utiliseraient la valeur (0, 0) héritée de l'env roller,
    # soit des roues LIBRES pendant le bootstrap — exactement l'inverse du but.
    cfg = make_microduck_roller_standup_env_cfg()
    stage0 = cfg.curriculum["wheel_friction"].params["ranges_stages"][0]["ranges"]
    assert cfg.events["randomize_wheel_friction"].params["ranges"] == stage0


def test_action_rate_ramp_is_the_standup_one_not_the_roller_one():
    # L'env roller monte à -2.0 (gait calme) : c'est un bloqueur de mouvement,
    # il ralentit l'action rapide dont le relevé depuis le dos a besoin. On
    # reprend la rampe du standup, qui plafonne à -1.0.
    cfg = make_microduck_roller_standup_env_cfg()
    weights = [
        s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]
    ]
    assert weights == [-0.4, -0.8, -1.0]
    assert cfg.rewards["action_rate_l2"].weight == -0.6


def test_push_curriculum_ramps_from_zero():
    # Poussées héritées (±0.2 m/s), mais rampées : une bourrade dès le pas 0
    # parasite le bootstrap du relevé.
    cfg = make_microduck_roller_standup_env_cfg()
    assert "push_robot" in cfg.events
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert cfg.curriculum["push_magnitude"].params["event_name"] == "push_robot"
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)
    assert stages[-1]["velocity_range"]["x"] == (-0.2, 0.2)
    highs = [s["velocity_range"]["x"][1] for s in stages]
    assert highs == sorted(highs), "la poussée doit CROÎTRE"


def test_inherited_dr_curricula_survive():
    # La DR héritée de l'env roller ne doit pas avoir été perdue en chemin.
    cfg = make_microduck_roller_standup_env_cfg()
    for name in ("com_range", "head_com_range"):
        assert name in cfg.curriculum, f"curriculum de DR perdu : {name}"
    for name in (
        "randomize_com",
        "randomize_head_com",
        "randomize_armature",
        "randomize_joint_friction",
        "randomize_mass_inertia",
        "randomize_wheel_friction",
        "encoder_bias",
    ):
        assert name in cfg.events, f"événement de DR perdu : {name}"
```

- [ ] **Step 2 : 运行测试确认它们会失败**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
预期：`test_wheel_friction_curriculum_is_decreasing` 在 `assert lows == sorted(lows, reverse=True)` 处失败（roller env 把 friction 从 0 升到 0.0015）、`test_wheel_friction_event_starts_at_stage_zero` 失败、`test_action_rate_ramp_is_the_standup_one_not_the_roller_one` 在 `[-1.0, -1.5, -2.0] != [-0.4, -0.8, -1.0]` 处失败、`test_push_curriculum_ramps_from_zero` 在 `KeyError: 'push_magnitude'` 处失败。`test_inherited_dr_curricula_survive` 已经通过（非回归检查）。

- [ ] **Step 3 : 替换 curricula**

在 `microduck_roller_standup_env_cfg.py` 中，把这个代码块插到 `ground_state_mix` curriculum **之后**、`return cfg` **之前**：

```python
    # ── Friction de roulement INVERSÉE : freinées → libres ───────────────────
    # C'est la seule pièce vraiment nouvelle de cet env, et le cœur de la
    # difficulté : les roues roulent, donc il n'y a AUCUNE adhérence
    # longitudinale pour pousser sur le sol. L'env roller fait MONTER cette
    # friction (0 → 0.0015) ; ici on la fait DESCENDRE, pour bootstrapper le
    # geste sur un problème facile (roues quasi bloquées ≈ des pieds) avant
    # d'imposer la physique réelle du roulement.
    #
    # DIAGNOSTIC à surveiller : si Episode_Reward/standing_composite s'écroule à
    # un palier, le geste « pieds adhérents » ne transfère pas aux roues libres
    # → il faudra guider une technique de patineur (appui genou intermédiaire,
    # un patin à la fois). C'est un résultat exploitable, pas un échec.
    #
    # ATTENTION sim2real : seuls les checkpoints d'APRÈS le dernier palier
    # (iter 4000+) sont candidats au déploiement. Avant, la policy s'appuie sur
    # une friction de roulement qui n'existe pas sur le vrai robot.
    _WHEEL_FRICTION_STAGE0 = (0.0500, 0.0500)
    cfg.curriculum["wheel_friction"] = CurriculumTermCfg(
        func=microduck_mdp.wheel_friction_curriculum,
        params={
            "event_name": "randomize_wheel_friction",
            "ranges_stages": [
                {"step": 0,                        "ranges": _WHEEL_FRICTION_STAGE0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "ranges": (0.0200, 0.0200)},
                {"step": 2000 * NUM_STEPS_PER_ENV, "ranges": (0.0080, 0.0080)},
                {"step": 3000 * NUM_STEPS_PER_ENV, "ranges": (0.0030, 0.0030)},
                {"step": 4000 * NUM_STEPS_PER_ENV, "ranges": (0.0015, 0.0015)},
            ],
        },
    )
    # La valeur de DÉPART de l'événement doit matcher le palier 0 : le curriculum
    # n'est évalué qu'à partir du premier pas, sinon les tout premiers resets
    # utiliseraient le (0, 0) hérité de l'env roller — des roues LIBRES pendant
    # le bootstrap, soit exactement l'inverse du but.
    cfg.events["randomize_wheel_friction"].params["ranges"] = _WHEEL_FRICTION_STAGE0

    # ── action_rate : la rampe du standup, pas celle du roller ───────────────
    # L'env roller monte à -2.0 pour un gait calme. C'est un bloqueur de
    # mouvement : il ralentit l'action rapide dont le relevé depuis le dos a
    # besoin (le standup documente qu'un action_rate trop fort tuait cette
    # récupération). La douceur est portée ici par joint_torque_rate_l2.
    cfg.rewards["action_rate_l2"].weight = -0.6
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,                       "weight": -0.4},
                {"step": 250 * NUM_STEPS_PER_ENV, "weight": -0.8},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -1.0},
            ],
        },
    )

    # ── Poussées rampées ────────────────────────────────────────────────────
    # push_robot est hérité de l'env roller (±0.2 m/s, toutes les 3–6 s) mais
    # sans curriculum. Une bourrade dès le pas 0 parasite le bootstrap du
    # relevé : on la fait monter comme le standup.
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0, "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 500 * NUM_STEPS_PER_ENV, "velocity_range": {
                    "x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                {"step": 1000 * NUM_STEPS_PER_ENV, "velocity_range": {
                    "x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
            ],
        },
    )
```

- [ ] **Step 4 : 运行测试确认它们通过**

```bash
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```
预期：25 passed。

- [ ] **Step 5 : 确认没有其他测试回归**

```bash
uv run --with pytest pytest tests/ -q
```
预期：`4 failed, 71 passed` —— 仅 `tests/test_wheel_glide.py` 里那 4 个本就存在的失败。

- [ ] **Step 6 : Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py \
        tests/test_roller_standup_cfg.py
git commit -m "roller-standup: curriculum de friction de roulement inverse + pousses rampees"
```

---

## Task 5 : GPU 上端到端验证 + 交接文档

Tasks 1–4 的测试都是**静态**的：它们检查配置，不检查执行。它们抓不到越界的 `joint_indices`、传给 mdp 函数的错误参数名，或缺失的传感器。这个任务是唯一真正运行 env 的地方。

**Files:**
- Create: `docs/roller_standup_policy_summary.md`
- （如果一切正常，预计无需改动代码）

**Interfaces:**
- Consumes: 已注册的任务 `Mjlab-RollerStandUp-Flat-MicroDuck`（Task 1）和完整的 env（Tasks 2–4）。
- Produces: 无程序化产出—— 一个交接文档 + 确认 env 能运行。

- [ ] **Step 1 : 运行一个极短训练**

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 3 \
  --agent.logger tensorboard
```

`--agent.logger tensorboard` 避免用一个一次性 run 污染 wandb。

预期：`✓ RollerStandUp task registered: Mjlab-RollerStandUp-Flat-MicroDuck`，然后 3 次迭代不抛异常地跑完，奖励表里显示 `pose_stand_legs`、`height_stand`、`standing_composite` 等项。

可能出现的错误及其原因：
- `joint_pos[:, joint_indices]` 上的 `IndexError` → `_LEG_JOINTS` 的索引超出了关节数量；重读 Task 2。
- `TypeError: ... unexpected keyword argument` → 某个参数名与 mdp 函数的签名不匹配；与 Task 2 的 **Interfaces** 块对照检查。
- 某个传感器名上的 `KeyError` → 某个被移除的奖励是唯二使用某个传感器的地方，或者某个保留的奖励索要一个不存在的传感器。

- [ ] **Step 2 : 确认站起奖励不为全零**

在上一步的输出里，检查 `Episode_Reward/standing_composite` 和 `Episode_Reward/height_stand` 是否**非零**。三次迭代里恰好都是 0.0 意味着某个奖励从未触发（`asset_cfg` 错误、目标高度错误）。

- [ ] **Step 3 : 目测检查地面出发**

```bash
uv run play Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 16
```

预期：机器人出现在**地面**（脸朝下）或**站在轮子上**，绝不在空中、也不穿过地面。此阶段没有任何机器人脸朝上——这是正常的，`face_up_prob = 0` 在 curriculum 的第 0 档，而 play 模式下 curriculum 不运行。

如果有的机器人从高处坠落，说明 `prone_z` 区间没调好；如果机器人穿过地面，说明出发姿态让它 spawn 到了平面下方。

- [ ] **Step 4 : 撰写交接文档**

创建 `docs/roller_standup_policy_summary.md`，仿照 `docs/roller_slope_policy_summary.md`：

```markdown
# Policy `roller_standup` — se relever sur rollers

**But** : le microduck (sur rollers) part du sol — à plat ventre ou à plat dos — et se remet **debout sur ses roues**, puis **tient** la station.

- **Tâche** : `Mjlab-RollerStandUp-Flat-MicroDuck`
- **Fichier** : `src/mjlab_microduck/tasks/microduck_roller_standup_env_cfg.py`
- **Base** : dérivée de l'env roller (`velocity_rollers`) → même robot, même physique/DR, **même observation 61D** (interchangeable au runtime, chargeable via `--new-cmd-obs`).
- **Spec** : `docs/superpowers/specs/2026-08-04-roller-standup-design.md`
- **Politique aveugle** : pas de scan de terrain ; proprioception + `projected_gravity`.

## Hauteurs (mesurées, pas devinées)

| pose | modèle pieds | modèle rollers |
|---|---|---|
| debout | 0.1172 → `STAND_Z=0.115` sous charge | 0.1407 → **`ROLLER_STAND_Z=0.138`** |
| à plat ventre (repos) | 0.075 | 0.075 |
| à plat dos (repos) | 0.048 | 0.048 |

Les hauteurs de repos au sol sont identiques aux deux modèles : c'est la coque du tronc qui touche, pas les pieds.

## ⚠️ Indices de joints — les roues sont INTERCALÉES

```
0-4   jambe gauche      5-6   roues gauches
7-10  cou / tête       11-15  jambe droite      16-17  roues droites
```
`_LEG_JOINTS = [0-4, 11-15]`. Les indices du `standup` (`[0-4, 9-13]`) valent pour le modèle **sans** roues et pointeraient sur des roues ici. Verrouillé par `tests/test_roller_standup_cfg.py::test_joint_indices_match_actual_roller_model`.

## Reset — départ au sol

`set_random_ground_state` : ventre (`prone_z` 0.05–0.09) / dos / **déjà debout** (`standing_z` 0.134–0.144), ± 10° de bruit en pitch/roll. Pas de bucket « assis ». Le bucket « debout » est nécessaire : sans lui la policy monte mais ne tient pas.

**Curriculum `ground_state_mix`** (easy → hard, le dos en dernier) :

| iter | debout | ventre | dos |
|---|---|---|---|
| 0 | 0.50 | 0.50 | 0.00 |
| 600 | 0.35 | 0.45 | 0.20 |
| 1500 | 0.25 | 0.40 | 0.35 |
| 2500 | 0.20 | 0.40 | 0.40 |

## Récompenses

Dix termes repris du `standup` avec leurs poids déjà réglés : `pose_stand_legs` (+8), `pose_stand_l1` (+5), `height_stand` (+4, std 0.04), `height_stand_sharp` (+4, std 0.015), `height_stand_l1` (+30), `com_upward_velocity` (+3), `gentle_rise` (−0.02), `upright_linear` (+6), `upright_sharp` (+6), `standing_composite` (+15). Plus `joint_torque_rate_l2` (−2e-3), l'anti-jitter qui n'empêche pas le retournement.

Régularisateurs hérités : `body_ang_vel` **−0.05** (bloqueur de mouvement, à garder LÉGER), `angular_momentum` −0.02, `action_rate_l2` (rampe −0.4 → −1.0, **pas** le −2.0 du roller), `neck_action_rate_l2` −0.5, `neck_joint_pos_l2` −0.5 (tête droite), `joint_torques_l2` −1e-3, `action_over_limit` −0.5, `self_collisions` −1.0.

Retirées : toutes les récompenses de patinage, plus `feet_flat` (les lames ne sont pas à plat pendant la montée) et `hip_roll_neutral` (se relever demande d'écarter les jambes).

## ⚠️ Le point dur : les roues roulent

Aucune adhérence longitudinale pour pousser sur le sol. Le **curriculum de friction de roulement est INVERSÉ** (l'env roller la fait monter, ici elle descend) :

| iter | frictionloss | |
|---|---|---|
| 0 | 0.05 | roues quasi bloquées → se relève comme avec des pieds |
| 1000 | 0.02 | |
| 2000 | 0.008 | |
| 3000 | 0.003 | |
| 4000 | 0.0015 | la vraie valeur du roulement |

**Surveiller `Episode_Reward/standing_composite` aux paliers.** S'il s'écroule, le geste « pieds adhérents » ne transfère pas aux roues libres → il faudra guider une technique de patineur (appui genou intermédiaire, un patin à la fois). C'est un résultat, pas un échec.

**Sim2real** : seuls les checkpoints d'après iter 4000 sont candidats au déploiement. Avant, la policy s'appuie sur une friction qui n'existe pas sur le vrai robot.

## Commande

Slot `twist` neutralisé (± 0.01), slots `head_pose` / `body_pose` **zero-paddés** (convention roller). Déploiement visé : en `--standing` face à la policy roller en `--walking`, avec la bascule automatique sur la magnitude de la commande (`infer_policy.py:262`, seuil 0.05) ; le slot twist y est laissé à zéro (`infer_policy.py:239`).

**Réserve** : `infer_policy.py` est le script de sim/clavier local. Le runtime robot est le binaire Rust `microduck_runtime`, absent du repo — il n'est pas vérifié qu'il expose un équivalent `--standing`. Le doc de passation du crouch ne liste que `--model`, `--ground-pick`, `--fold-policy`. À confirmer.

## Terminaisons

`fell_over` **supprimée** (le robot démarre tombé). `nan_state` héritée. `nan_policy="sanitize"` sur les obs actor/critic.

## Réseau / PPO

Actor et critic `(512, 256, 128)` elu, `obs_normalization=True`. PPO `lr=1e-3` adaptive, `desired_kl=0.01`, `gamma=0.99`, `lam=0.95`, `num_steps_per_env=24`, épisode 6 s, `max_iterations=15000`. **Symétrie OFF** (`SYMMETRY_CFG` est câblé pour le layout 51D).

## Commandes

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
uv run scripts/play_latest.py        # alias md-play
uv run scripts/export_latest.py      # alias md-export
uv run --with pytest pytest tests/test_roller_standup_cfg.py -q
```

## Hors périmètre

Intégrer le relevé dans la policy de roulage (recette `velstand`) ; buckets de départ sur le côté ; variante rough ; pénalités d'impact tronc/tête.
```

- [ ] **Step 5 : Commit**

```bash
git add docs/roller_standup_policy_summary.md
git commit -m "roller-standup: doc de passation"
```

---

## 计划之后

启动一次真正的训练：

```bash
uv run train Mjlab-RollerStandUp-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 15000
```

**要读的信号**：`Episode_Reward/standing_composite` 应该上升，尤其是**它在 iters 1000 / 2000 / 3000 / 4000（滚动摩擦的档位）的表现**，它回答了这个设计最初要回答的问题——在自由轮子上用「脚面粘连式」动作能否站起来，还是必须教授一种溜冰者技巧？