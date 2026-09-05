# Ground-pick 姿态跟随（pose-following）— 实现计划

> **面向 agentic 工作者的说明:** 必选子技能: 使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐步实现本计划。步骤使用复选框（`- [ ]`）语法以跟踪进度。

**目标:** 重写任务 `Mjlab-GroundPick-Flat-MicroDuck`，使其动作由按相位插值的关节姿态跟踪（STAND→DOWN→STAND）来驱动，取代当前的任务空间目标（口部-地面接近距离 + 姿态返回）。

**架构:** 新增三个 mdp 纯函数/准纯函数（`phase_pose_blend`、`phase_pose_track`、`phase_pose_track_l1`），它们根据一个 4 段相位轮廓在 HOME（STAND）与一个 `DOWN_POSE` 字典之间计算插值后的关节目标，并**按名字**解析。为现有相位命令新增一个 `randomize_phase` 标志。随后重写 `microduck_ground_pick_env_cfg.py` 的奖励块，其余部分（DR、61D obs、curricula、RlCfg）保持不变。

**技术栈:** Python, PyTorch, mjlab 1.3.0, MuJoCo, uv, pytest（通过 `uv run --with pytest`）。

## 全局约束

- 关节解析**按名字**（`asset.find_joints([name])[0][0]`），绝不在代码中硬编码索引。
- 统一的 61D obs **保持不变**（head/body 零填充）→ 策略可在 runtime 槽位中互换。
- task id 不变: `Mjlab-GroundPick-Flat-MicroDuck`（+ `-Rough-` 变体）。
- 相位的周期 = **4.0 s**（`--ground-pick-period` 槽位的默认值）。
- 相位轮廓（分数）: `DESCENT_END=0.15`, `HOLD_END=0.50`, `RISE_END=0.65`。
- ground_pick 任务 `randomize_phase=False`（保证按钮 A 在 φ=0 时部署的一致性）; cfg 默认 `True` 以免破坏 sit/stand。
- STAND = HOME（`asset.data.default_joint_pos`，无需重新定义）。DOWN = 按名字的 `DOWN_POSE` 字典。
- 14 个活跃关节（不含 mouth）。机器人 `MICRODUCK_GROUND_PICK_ROBOT_CFG`（无轮子 → 索引 0-4 左腿、5-8 颈/头、9-13 右腿，但仍按名字解析）。
- mdp 文件: 相关导入已存在（`torch`, `Optional`, `Entity`, `SceneEntityCfg`, `ManagerBasedRlEnv`, `_DEFAULT_ASSET_CFG`）。

---

### 任务 1: 函数 `phase_pose_blend`（4 段 blend, 纯函数）

**文件:**
- 修改: `src/mjlab_microduck/tasks/mdp.py`（新增一个函数; 插入到 `phase_pose_match` ~第 2041 行之前）
- 测试: `tests/test_ground_pick_pose.py`（新建）

**接口:**
- 产出: `phase_pose_blend(phase: torch.Tensor, descent_end: float, hold_end: float, rise_end: float) -> torch.Tensor` — 返回与 `phase` 相同 shape 的 blend ∈ [0,1]（0 = STAND, 1 = DOWN）。

- [ ] **步骤 1: 写出会失败的测试**

创建 `tests/test_ground_pick_pose.py`:

```python
import torch
from mjlab_microduck.tasks.mdp import phase_pose_blend

DESCENT_END, HOLD_END, RISE_END = 0.15, 0.50, 0.65


def test_phase_pose_blend_keypoints():
    phase = torch.tensor([0.0, 0.075, 0.15, 0.30, 0.50, 0.575, 0.65, 0.80])
    b = phase_pose_blend(phase, DESCENT_END, HOLD_END, RISE_END)
    expected = torch.tensor([0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0])
    assert torch.allclose(b, expected, atol=1e-6), b


def test_phase_pose_blend_range():
    phase = torch.linspace(0.0, 1.0, 101)
    b = phase_pose_blend(phase, DESCENT_END, HOLD_END, RISE_END)
    assert b.min() >= 0.0 and b.max() <= 1.0
```

- [ ] **步骤 2: 运行测试以确认其失败**

运行: `uv run --with pytest pytest tests/test_ground_pick_pose.py -q`
预期: FAIL — `ImportError: cannot import name 'phase_pose_blend'`

- [ ] **步骤 3: 编写最小实现**

在 `src/mjlab_microduck/tasks/mdp.py` 中、`def phase_pose_match(` 之前（~第 2041 行）:

```python
def phase_pose_blend(
    phase: torch.Tensor,
    descent_end: float,
    hold_end: float,
    rise_end: float,
) -> torch.Tensor:
    """Blend 0..1 le long de la phase [0,1) — 0 = pose STAND, 1 = pose DOWN.

    [0, descent_end)       : 0 -> 1  (se baisser)
    [descent_end, hold_end): 1       (bas)
    [hold_end, rise_end)   : 1 -> 0  (se lever)
    [rise_end, 1.0)        : 0       (haut / repos)
    """
    b = torch.zeros_like(phase)
    descend = phase < descent_end
    b = torch.where(descend, phase / descent_end, b)
    low = (phase >= descent_end) & (phase < hold_end)
    b = torch.where(low, torch.ones_like(phase), b)
    rise = (phase >= hold_end) & (phase < rise_end)
    b = torch.where(rise, 1.0 - (phase - hold_end) / (rise_end - hold_end), b)
    return b
```

- [ ] **步骤 4: 运行测试以确认其通过**

运行: `uv run --with pytest pytest tests/test_ground_pick_pose.py -q`
预期: PASS（2 passed）

- [ ] **步骤 5: 提交**

```bash
git add tests/test_ground_pick_pose.py src/mjlab_microduck/tasks/mdp.py
git commit -m "feat(mdp): phase_pose_blend — blend 4 segments STAND<->DOWN par la phase"
```

---

### 任务 2: 奖励 `phase_pose_track` / `phase_pose_track_l1`（+ 辅助函数 `_phase_pose_error`）

**文件:**
- 修改: `src/mjlab_microduck/tasks/mdp.py`（紧接着 `phase_pose_blend` 之后新增）
- 测试: `tests/test_ground_pick_pose.py`（追加）

**接口:**
- 消费: `phase_pose_blend`（任务 1）。
- 产出:
  - `_phase_pose_error(env, asset_cfg, command_name, target_pose: dict, descent_end, hold_end, rise_end, source_pose: dict | None = None) -> (cur: Tensor, target: Tensor)` — 按名字解析的 `(B, k)` 张量。
  - `phase_pose_track(env, command_name="twist", target_pose: dict | None = None, source_pose: dict | None = None, std=0.3, descent_end=0.15, hold_end=0.50, rise_end=0.65, asset_cfg=_DEFAULT_ASSET_CFG) -> Tensor` — 高斯 `exp(-((cur-target)/std)²).mean(-1)`。
  - `phase_pose_track_l1(env, command_name="twist", target_pose=None, source_pose=None, descent_end=0.15, hold_end=0.50, rise_end=0.65, asset_cfg=_DEFAULT_ASSET_CFG) -> Tensor` — `-(cur-target).abs().mean(-1)`。

- [ ] **步骤 1: 写出会失败的测试**

向 `tests/test_ground_pick_pose.py` 添加一个轻量假 env + 相关断言:

```python
from mjlab_microduck.tasks.mdp import phase_pose_track, phase_pose_track_l1


class _FakeData:
    def __init__(self, joint_pos, default_pos):
        self.joint_pos = joint_pos
        self.default_joint_pos = default_pos


class _FakeAsset:
    def __init__(self, names, joint_pos, default_pos):
        self._ids = {n: i for i, n in enumerate(names)}
        self.data = _FakeData(joint_pos, default_pos)

    def find_joints(self, query):
        # mjlab renvoie (ids, names) ; on ne gère que la requête [name]
        (name,) = query
        return ([self._ids[name]], [name])


class _FakeCmdMgr:
    def __init__(self, cmd):
        self._cmd = cmd

    def get_command(self, _name):
        return self._cmd


class _FakeEnv:
    def __init__(self, names, joint_pos, default_pos, phase):
        import math
        self.device = "cpu"
        self.scene = {"robot": _FakeAsset(names, joint_pos, default_pos)}
        ang = 2 * math.pi * phase
        cmd = torch.tensor([[math.cos(ang), math.sin(ang), 0.0]])
        self.command_manager = _FakeCmdMgr(cmd)


NAMES = ["j0", "j1"]
DOWN = {"j0": 1.0, "j1": -1.0}
# HOME (STAND source) = 0 pour les deux joints
HOME = torch.tensor([[0.0, 0.0]])


def _env(cur, phase):
    return _FakeEnv(NAMES, torch.tensor([cur]), HOME.clone(), phase)


def test_phase_pose_track_perfect_at_down():
    # phase 0.30 -> blend 1 -> cible = DOWN ; cur == DOWN -> gaussienne 1, l1 0
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    cfg = SceneEntityCfg("robot")
    env = _env([1.0, -1.0], phase=0.30)
    r = phase_pose_track(env, target_pose=DOWN, asset_cfg=cfg)
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-6), r
    env2 = _env([1.0, -1.0], phase=0.30)
    l1 = phase_pose_track_l1(env2, target_pose=DOWN, asset_cfg=cfg)
    assert torch.allclose(l1, torch.tensor([0.0]), atol=1e-6), l1


def test_phase_pose_track_l1_at_home_when_down_target():
    # phase 0.30 -> cible DOWN=[1,-1] ; cur=HOME=[0,0] -> l1 = -mean(|1|,|1|) = -1
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    cfg = SceneEntityCfg("robot")
    env = _env([0.0, 0.0], phase=0.30)
    l1 = phase_pose_track_l1(env, target_pose=DOWN, asset_cfg=cfg)
    assert torch.allclose(l1, torch.tensor([-1.0]), atol=1e-6), l1


def test_phase_pose_track_returns_to_stand():
    # phase 0.80 -> blend 0 -> cible = HOME ; cur=HOME -> gaussienne 1
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    cfg = SceneEntityCfg("robot")
    env = _env([0.0, 0.0], phase=0.80)
    r = phase_pose_track(env, target_pose=DOWN, asset_cfg=cfg)
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-6), r
```

- [ ] **步骤 2: 运行测试以确认其失败**

运行: `uv run --with pytest pytest tests/test_ground_pick_pose.py -q`
预期: FAIL — `ImportError: cannot import name 'phase_pose_track'`

- [ ] **步骤 3: 编写最小实现**

在 `src/mjlab_microduck/tasks/mdp.py` 中、`phase_pose_blend` 之后:

```python
def _phase_pose_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    target_pose: dict,
    descent_end: float,
    hold_end: float,
    rise_end: float,
    source_pose: Optional[dict] = None,
):
    """(cur, target) pour la pose interpolée par la phase, résolue PAR NOM.

    Cible = source + blend(phase)·(target_pose - source), source = STAND
    (`source_pose` si fourni, sinon le DEFAULT/HOME du modèle). blend ∈ [0,1]
    (0 = STAND, 1 = target_pose) via `phase_pose_blend`.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    phase = (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0  # (B,)
    blend = phase_pose_blend(phase, descent_end, hold_end, rise_end)     # (B,)

    names = list(target_pose.keys())
    ids = [int(asset.find_joints([n])[0][0]) for n in names]
    default = asset.data.default_joint_pos[:, ids]                       # (B,k)

    source = default.clone()
    if source_pose:
        for j, n in enumerate(names):
            if n in source_pose:
                source[:, j] = source_pose[n]
    target_vec = torch.tensor(
        [target_pose[n] for n in names], device=env.device, dtype=default.dtype
    ).unsqueeze(0)                                                       # (1,k)

    target = source + blend.unsqueeze(-1) * (target_vec - source)        # (B,k)
    cur = asset.data.joint_pos[:, ids]                                   # (B,k)
    return cur, target


def phase_pose_track(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    std: float = 0.3,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Gaussienne sur la pose articulaire vs cible interpolée STAND<->DOWN.

    Reward directif : indique la config articulaire exacte à chaque phase. Se
    relever (cible → STAND) est récompensé exactement comme se baisser (cible →
    DOWN) — symétrique par construction. Résolution PAR NOM.
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return torch.exp(-((cur - target) / std) ** 2).mean(dim=-1)


def phase_pose_track_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    target_pose: Optional[dict] = None,
    source_pose: Optional[dict] = None,
    descent_end: float = 0.15,
    hold_end: float = 0.50,
    rise_end: float = 0.65,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Bootstrap L1 vers la cible interpolée (pénalité négative).

    Gradient constant partout — donne une direction vers la cible même quand la
    gaussienne ci-dessus a saturé à ~0 loin de la cible.
    """
    cur, target = _phase_pose_error(
        env, asset_cfg, command_name, target_pose or {},
        descent_end, hold_end, rise_end, source_pose,
    )
    return -(cur - target).abs().mean(dim=-1)
```

- [ ] **步骤 4: 运行测试以确认其通过**

运行: `uv run --with pytest pytest tests/test_ground_pick_pose.py -q`
预期: PASS（5 passed）

- [ ] **步骤 5: 提交**

```bash
git add tests/test_ground_pick_pose.py src/mjlab_microduck/tasks/mdp.py
git commit -m "feat(mdp): phase_pose_track/_l1 — suivi de pose interpolée par la phase (par nom)"
```

---

### 任务 3: `GroundPickPhaseCommandCfg` 上的 `randomize_phase` 标志

**文件:**
- 修改: `src/mjlab_microduck/tasks/mdp.py`（`GroundPickPhaseCommand` 类 ~3611/3626, cfg ~3644）
- 测试: `tests/test_ground_pick_pose.py`（追加）

**接口:**
- 产出: `GroundPickPhaseCommandCfg.randomize_phase: bool = True`; `GroundPickPhaseCommand.reset()` 在 `randomize_phase=False` 时把相位设为 0, 否则用 `torch.rand`。

- [ ] **步骤 1: 写出会失败的测试**

向 `tests/test_ground_pick_pose.py` 添加:

```python
def test_ground_pick_cmd_cfg_has_randomize_phase_default_true():
    from mjlab_microduck.tasks.mdp import GroundPickPhaseCommandCfg
    from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
    # construit une cfg minimale en copiant une cfg velocity par défaut
    base = UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0),
        ),
    )
    cfg = GroundPickPhaseCommandCfg(**{**vars(base)})
    assert cfg.randomize_phase is True
    assert cfg.period == 4.0
```

说明: 若本地 `UniformVelocityCommandCfg.Ranges` 的签名不同, 请调整相关字段 — 关键断言是 `cfg.randomize_phase is True`。

- [ ] **步骤 2: 运行测试以确认其失败**

运行: `uv run --with pytest pytest tests/test_ground_pick_pose.py::test_ground_pick_cmd_cfg_has_randomize_phase_default_true -q`
预期: FAIL — `AttributeError: 'GroundPickPhaseCommandCfg' object has no attribute 'randomize_phase'`

- [ ] **步骤 3: 编写最小实现**

在 `src/mjlab_microduck/tasks/mdp.py` 的 `GroundPickPhaseCommand` 类中, 修改 `__init__` 与 `reset`:

替换（在 `__init__` 中, ~第 3614 行）:
```python
        self._period = float(getattr(cfg, "period", self.PERIOD))
```
为:
```python
        self._period = float(getattr(cfg, "period", self.PERIOD))
        self._randomize_phase = bool(getattr(cfg, "randomize_phase", True))
```

替换 `reset` 方法（~第 3626 行）:
```python
    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        return {}
```
为:
```python
    def reset(self, env_ids: torch.Tensor | None) -> dict:
        if env_ids is not None and len(env_ids) > 0:
            if self._randomize_phase:
                self._gp_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
            else:
                self._gp_phase[env_ids] = 0.0
        return {}
```

在 cfg `GroundPickPhaseCommandCfg`（~第 3644 行）中, 在 `period` 之后添加字段:
```python
@_dataclass(kw_only=True)
class GroundPickPhaseCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GroundPickPhaseCommand
    period: float = 4.0  # cycle length in seconds; sitstand uses 8.0
    randomize_phase: bool = True  # False = chaque épisode démarre à φ=0 (parité slot bouton A)

    def build(self, env: ManagerBasedRlEnv) -> "GroundPickPhaseCommand":
        return GroundPickPhaseCommand(self, env)
```

- [ ] **步骤 4: 运行测试以确认其通过**

运行: `uv run --with pytest pytest tests/test_ground_pick_pose.py::test_ground_pick_cmd_cfg_has_randomize_phase_default_true -q`
预期: PASS。若因本地 API 差异导致 `UniformVelocityCommandCfg` 构造失败, 请调整测试中 `base` 的字段（实现本身是正确的）。

- [ ] **步骤 5: 提交**

```bash
git add tests/test_ground_pick_pose.py src/mjlab_microduck/tasks/mdp.py
git commit -m "feat(mdp): flag randomize_phase sur GroundPickPhaseCommandCfg (défaut True)"
```

---

### 任务 4: 重写 env cfg 中的奖励块与姿态

**文件:**
- 修改: `src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py`
- 测试: `tests/test_ground_pick_cfg.py`（新建）

**接口:**
- 消费: `phase_pose_track`, `phase_pose_track_l1`（任务 2）; `randomize_phase`（任务 3）。
- 产出: `make_microduck_ground_pick_env_cfg(play=False, rough=False)` 返回的 cfg: 命令 `GroundPickPhaseCommand` 带 `randomize_phase=False`, `period=4.0`; 奖励包含 `phase_pose_track`（6.0）与 `phase_pose_track_l1`（2.0）、`mouth_ground_proximity`（1.0）; 不再包含 `mouth_perpendicular_to_ground`、`ground_pick_return_pose_legs`、`ground_pick_return_pose_neck`。

- [ ] **步骤 1: 写出会失败的测试**

创建 `tests/test_ground_pick_cfg.py`:

```python
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
)
from mjlab_microduck.tasks.mdp import GroundPickPhaseCommand


def test_ground_pick_cfg_builds_with_pose_rewards():
    cfg = make_microduck_ground_pick_env_cfg()
    rewards = cfg.rewards
    assert "phase_pose_track" in rewards
    assert "phase_pose_track_l1" in rewards
    assert rewards["phase_pose_track"].weight == 6.0
    assert rewards["phase_pose_track_l1"].weight == 2.0
    # filet bouche-sol conservé mais allégé
    assert "mouth_ground_proximity" in rewards
    assert rewards["mouth_ground_proximity"].weight == 1.0
    # anciennes mécaniques retirées
    assert "mouth_perpendicular_to_ground" not in rewards
    assert "ground_pick_return_pose_legs" not in rewards
    assert "ground_pick_return_pose_neck" not in rewards


def test_ground_pick_cfg_command_is_phase_no_randomize():
    cfg = make_microduck_ground_pick_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.class_type is GroundPickPhaseCommand
    assert cmd.period == 4.0
    assert cmd.randomize_phase is False
```

- [ ] **步骤 2: 运行测试以确认其失败**

运行: `uv run --with pytest pytest tests/test_ground_pick_cfg.py -q`
预期: FAIL — `assert 'phase_pose_track' in rewards`（KeyError/False）。

- [ ] **步骤 3: 编写最小实现**

在 `src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py` 中:

(a) 在 `def make_microduck_ground_pick_env_cfg(` 之前添加姿态/相位常量:

```python
# ── Poses cibles du geste (rad, par NOM) ──────────────────────────────────────
# STAND = HOME (default_joint_pos du modèle) — ne pas redéfinir ici : source du
# blend. DOWN = pli avant profond (bouche vers le sol), valeurs initiales tirées
# du keyframe FOLD de scene_walk.xml. ⚠️ REMPLAÇABLE par une lecture read_pose.py
# du vrai robot posé bouche-au-sol quand disponible.
DOWN_POSE = {
    "left_hip_yaw": 0.0, "left_hip_roll": 0.0, "left_hip_pitch": 1.57,
    "left_knee": 1.57, "left_ankle": 0.0,
    "neck_pitch": 1.0, "head_pitch": 1.0, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": 0.0, "right_hip_roll": 0.0, "right_hip_pitch": -1.57,
    "right_knee": -1.57, "right_ankle": 0.0,
}

# Timing du cycle (fractions de phase), période 4 s :
#   descente [0, DESCENT_END) ~0.6s / bas [DESCENT_END, HOLD_END) ~1.4s /
#   remontée [HOLD_END, RISE_END) ~0.6s / repos [RISE_END, 1) ~1.4s
GP_PERIOD    = 4.0
DESCENT_END  = 0.15
HOLD_END     = 0.50
RISE_END     = 0.65
POSE_STD     = 0.3
```

(b) 在奖励删除循环（~第 145-155 行）中, 替换动作相关内容。**移除** `mouth_perpendicular_to_ground` 两个块（~176-183）与 `ground_pick_return_pose_*` 两个块（~189-212）, 并将 `mouth_ground_proximity` 重调为 `weight=1.0`（~163-172, 把 `weight=2.0` 改为 `weight=1.0`）。

具体操作:
- 编辑 `cfg.rewards["mouth_ground_proximity"]` 块: `weight=2.0` → `weight=1.0`。
- 完整删除 `cfg.rewards["mouth_perpendicular_to_ground"] = RewardTermCfg(...)` 块。
- 删除 `_LEG_JOINTS = [...]` / `cfg.rewards["ground_pick_return_pose_legs"]` 以及 `_NECK_JOINTS = [...]` / `cfg.rewards["ground_pick_return_pose_neck"]` 块。
- 若奖励删除列表中含有 `"pose"`, 保留（不变）— 但同时也**移除**已过时的注释行 `# replaced by phase-conditioned ground_pick_return_pose`（可选）。

(c) 添加两个新的姿态跟踪奖励（放在被移除块的位置, 位于 “main ground pick objectives” 段落）:

```python
    # Suivi de pose interpolée par la phase (STAND<->DOWN<->STAND). Directif et
    # symétrique : le retour debout est récompensé exactement comme la descente.
    cfg.rewards["phase_pose_track"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track,
        weight=6.0,
        params={
            "command_name": "twist",
            "target_pose": DOWN_POSE,
            "std": POSE_STD,
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["phase_pose_track_l1"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track_l1,
        weight=2.0,
        params={
            "command_name": "twist",
            "target_pose": DOWN_POSE,
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
```

(d) 在 “Command” 块（~第 368 行）中, 传入周期并关闭相位随机化:

替换:
```python
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{**vars(command), "class_type": microduck_mdp.GroundPickPhaseCommand}
    )
```
为:
```python
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": GP_PERIOD,
            "randomize_phase": False,
        }
    )
```

- [ ] **步骤 4: 运行测试以确认其通过**

运行: `uv run --with pytest pytest tests/test_ground_pick_cfg.py -q`
预期: PASS（2 passed）。

随后确认整套测试通过:
运行: `uv run --with pytest pytest tests/ -q`
预期: PASS（全部）。

- [ ] **步骤 5: 提交**

```bash
git add tests/test_ground_pick_cfg.py src/mjlab_microduck/tasks/microduck_ground_pick_env_cfg.py
git commit -m "feat(ground_pick): suivi de pose interpolée par la phase (STAND->DOWN->STAND)"
```

---

### 任务 5: 端到端验证（任务的运行时构建）

**文件:**
- 测试: `tests/test_ground_pick_cfg.py`（追加）

**接口:**
- 消费: 此前所有内容。

- [ ] **步骤 1: 写出会失败/未被覆盖的测试**

向 `tests/test_ground_pick_cfg.py` 添加:

```python
def test_ground_pick_rough_variant_builds():
    cfg = make_microduck_ground_pick_env_cfg(rough=True)
    assert "phase_pose_track" in cfg.rewards


def test_ground_pick_play_variant_builds():
    cfg = make_microduck_ground_pick_env_cfg(play=True)
    assert cfg.commands["twist"].randomize_phase is False
```

- [ ] **步骤 2: 运行验证**

运行: `uv run --with pytest pytest tests/test_ground_pick_cfg.py -q`
预期: PASS。

- [ ] **步骤 3: 验证任务注册（导入包）**

运行: `uv run python -c "import mjlab_microduck.tasks; print('ok')"`
预期: 输出包含 `GroundPick` 的行 `✓ ... registered`, 随后是 `ok`, 无异常。

- [ ] **步骤 4: 提交**

```bash
git add tests/test_ground_pick_cfg.py
git commit -m "test(ground_pick): variantes rough/play + import du package"
```

---

## 自审

**1. Spec 覆盖:**
- §1 按姿态的指导性目标 → 任务 1、2、4。✓
- §2 姿态（STAND=HOME 源, DOWN=FOLD 按名字）→ 任务 4（a）, 任务 2（`source_pose=None`→default）。✓
- §3 4 段轮廓、周期 4 s + `randomize_phase=False` → 任务 1、任务 3、任务 4（a,d）。✓
- §4 mdp 函数 `phase_pose_blend/track/_l1` 按名字 → 任务 1、2。✓
- §5 奖励（新增 + 移除 + mouth 重调为 1.0）→ 任务 4（b,c）, 任务 4 测试。✓
- §6 部署（周期 4, kp-ratio 1.0）→ 在 spec 中说明; 任务 4 测试验证 period=4。✓
- §7 测试（纯函数 + env 构建）→ 任务 1、2、4、5。✓
- §9 重复的 `pose_target_match` 超出范围 → 未修改（符合）。✓

**2. 占位符扫描:** 无 TODO/TBD; 所有代码均已给出。✓

**3. 类型一致性:** `phase_pose_track(target_pose=..., std=..., asset_cfg=...)` 与 `phase_pose_track_l1(target_pose=..., asset_cfg=...)` 在任务 2（定义）、任务 4（调用）与测试之间保持一致。`randomize_phase` 在任务 3（定义）与任务 4/测试（使用）之间一致。`GroundPickPhaseCommand`/`GroundPickPhaseCommandCfg` 名称未变。✓