"""策略 manifest (schema 2) 以及已发布策略必须通过的校验.

一套词汇对应两种形态 -- 单策略仓库 (顶层字段) 和官方策略集 (``policies`` 下逐条相同的字段).
本模块负责写第一种; 守护进程 (`pollen-robotics/microduck`, ``updater/src/policy.rs`` 和
``robotd-params``) 两者都会读取. 契约在那里是 `docs/policy-manifest.md`; 下面这些数值正是
守护进程在 ``duck_ipc_proto`` 中发布并据此拒绝与其不符的策略的.

刻意不依赖 mjlab / torch, 这样测试能在笔记本上毫秒级跑完, CLI 也无需 GPU 就能校验 ONNX 文件.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 2
# `duck_ipc_proto`: 守护进程会拒绝与这些不一致的策略, 也会在加载时拒绝图与之不符的网络.
# 61 = 48 本体感觉 + 13 命令; 14 = 舵机的数量.
MODEL_API = 1
OBS_LEN = 61
ACTION_LEN = 14
ROBOT: dict[str, Any] = {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50}

# 仓库携带的唯一一个 `.onnx`. 守护进程取仓库里唯一那个 `.onnx`, 有多个则拒绝.
POLICY_FILE = "policy.onnx"

Kind = Literal["episodic", "perpetual"]
KINDS: tuple[str, ...] = ("episodic", "perpetual")

ZERO_TWIST: tuple[float, float, float] = (0.0, 0.0, 0.0)

# 守护进程的策略槽位, 用于步态的 `slot` 提示 (仅展示: `robotctl policy load <slot>`).
SLOTS: tuple[str, ...] = ("walk", "stand", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade")


class ManifestError(ValueError):
    """守护进程会拒绝的 manifest, 或其即便能加载也会运行出错的 manifest."""


@dataclass(frozen=True)
class Provenance:
    """权重来源. 对守护进程仅展示; 这部分是大家手动填写时容易略过的部分."""

    task_id: str | None = None
    repo: str = "pollen-robotics/microduck_rl"
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    run: str | None = None
    checkpoint: int | None = None
    source_file: str | None = None
    exported: str = field(default_factory=lambda: _now_utc())

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """导出所从的 checkout 的 `commit`, `branch`, `dirty`, 不在 git 内则返回 `{}`."""
    root = str(repo_root or Path(__file__).resolve().parents[3])

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, check=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = git("rev-parse", "--short=9", "HEAD")
    if commit is None:
        return {}
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "branch": branch, "dirty": bool(status)}


def build_manifest(
    *,
    name: str,
    kind: str,
    description: str,
    duration_s: float | None = None,
    chain: bool = False,
    unwind_s: float | None = None,
    idle: tuple[float, float, float] = ZERO_TWIST,
    action_scale: float | None = None,
    entry_pose: str = "standing",
    slot: str | None = None,
    command_help: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """守护进程能无意外加载的单策略 manifest.

    从这里只能发布常量命令一族 -- 技能网络的输入是固定的 twist. phase 和姿态 flag 的编码是官方
    策略集自己的领地, 社区策略不属于这种形态.
    """
    if kind not in KINDS:
        raise ManifestError(f"kind 必须是 {KINDS} 之一, 而不是 {kind!r}")
    if not name or "/" in name or name != name.strip():
        raise ManifestError(f"name 必须是一个客户端可请求的裸词, 而不是 {name!r}")
    if kind == "episodic":
        if duration_s is None or duration_s <= 0:
            raise ManifestError("episodic 策略会自我终止: 用 duration_s > 0 说明它运行多久")
        if unwind_s:
            raise ManifestError("episodic 策略在 duration_s 到时就已回来; unwind_s 是给 perpetual 用的")
    else:
        # 有两种 perpetual: 一种是步态, 放在某个槽位里 (`policy load walk <repo>`) 在这里
        # 不需要额外字段; 另一种是保持某个姿态 (如 flamingo), 属主以 `policy add --hold`
        # 的一次性技能方式运行, 此时需要 `unwind_s`, 这样机器人不会被单脚悬空放开.
        # `unwind_s` 就是用来区分这两者的.
        if duration_s is not None:
            raise ManifestError(
                "perpetual 策略没有自己的时长; 让 duration_s 留空 "
                "(步态会一直运行直到另行指示; 保持姿态作为技能添加时给 --hold)"
            )
        if unwind_s is not None and unwind_s <= 0:
            raise ManifestError("给出 unwind_s 时必须 > 0")
        if chain:
            raise ManifestError("chain 是给通过按住按键重复的一次性 episodic 技能用的")
    if slot is not None and slot not in SLOTS:
        raise ManifestError(f"slot 必须是 {SLOTS} 之一, 而不是 {slot!r}")
    if action_scale is not None and not 0 < action_scale <= 2.0:
        raise ManifestError(f"action_scale {action_scale} 超出范围 (0, 2]")
    if len(idle) != 3:
        raise ManifestError("idle 是一个三维 twist 向量")

    command: dict[str, Any] = {
        "encoding": "constant",
        "idle": [float(v) for v in idle],
        "twist": "unused (zeros)",
        "head": "unused (zeros)",
        "body": "unused (zeros)",
    }
    if command_help:
        command.update(command_help)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_api": MODEL_API,
        "obs_len": OBS_LEN,
        "action_len": ACTION_LEN,
        "robot": dict(ROBOT),
        "name": name,
        "kind": kind,
        "entry_pose": entry_pose,
        "description": description,
        "command": command,
    }
    if kind == "episodic":
        manifest["duration_s"] = float(duration_s)  # type: ignore[arg-type]
        manifest["chain"] = bool(chain)
    else:
        manifest["duration_s"] = None
        if unwind_s is not None:
            manifest["unwind_s"] = float(unwind_s)
    if slot is not None:
        manifest["slot"] = slot
    if action_scale is not None:
        manifest["action_scale"] = float(action_scale)
    if training:
        manifest["training"] = training
    if eval:
        manifest["eval"] = eval
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    """拒绝守护进程会拒绝的内容, 再加它会加载并运行出错的那类错误.

    两种形态和任意 schema 版本都接受, 因为字段缺失不算证据 -- 仓库没有义务携带任何字段.
    只有写出来且写错的说法才会失败.
    """
    if "policies" in manifest:
        for entry in manifest["policies"]:
            if "file" not in entry:
                raise ManifestError("策略集中的每一条都需要一个 `file`")
            validate_manifest({k: v for k, v in entry.items() if k != "file"})
        return
    if (obs := manifest.get("obs_len")) is not None and obs != OBS_LEN:
        raise ManifestError(f"obs_len {obs}: 这个机器人构建的是 {OBS_LEN}")
    if (act := manifest.get("action_len")) is not None and act != ACTION_LEN:
        raise ManifestError(f"action_len {act}: 这个机器人有 {ACTION_LEN}")
    if (api := manifest.get("model_api")) is not None and api > MODEL_API:
        raise ManifestError(f"model_api {api}: 本仓库面向的是 {MODEL_API}")
    model = (manifest.get("robot") or {}).get("model")
    if model is not None and model.lower() != ROBOT["model"]:
        raise ManifestError(f"robot.model {model!r}: 这是 {ROBOT['model']} 策略仓库")
    kind = manifest.get("kind")
    if kind is not None and kind not in (*KINDS, "scripted"):
        raise ManifestError(f"kind {kind!r} 不是 episodic, perpetual, scripted 之一")
    encoding = (manifest.get("command") or {}).get("encoding")
    if encoding is not None and encoding not in ("constant", "phase", "posture_flag"):
        raise ManifestError(f"command.encoding {encoding!r} 不是守护进程能驱动的方式")
    if kind == "episodic" and encoding in (None, "constant"):
        duration = manifest.get("duration_s")
        if duration is None or duration <= 0:
            raise ManifestError("episodic 常量命令策略需要 duration_s > 0")
    idle = (manifest.get("command") or {}).get("idle")
    if idle is not None and len(idle) != 3:
        raise ManifestError("command.idle 是一个三维 twist 向量")


# ---------------------------------------------------------------------------------------------
# ONNX 文件: 守护进程在加载时施加的形状门槛, 在上传前先施加一遍.


@dataclass(frozen=True)
class OnnxShape:
    input_name: str
    output_name: str
    obs_len: int
    action_len: int


def inspect_onnx(path: Path) -> OnnxShape:
    """图的单输入和单输出宽度, 正如守护进程在加载时核对的那样."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise ManifestError(
            f"{path.name}: 期望一个输入和一个输出, 但找到 "
            f"{[i.name for i in inputs]} -> {[o.name for o in graph.output]}"
        )

    def last_dim(value) -> int:
        dims = value.type.tensor_type.shape.dim
        if not dims:
            raise ManifestError(f"{path.name}: {value.name} 没有形状")
        last = dims[-1]
        if not last.HasField("dim_value"):
            raise ManifestError(f"{path.name}: {value.name} 的最后一维是符号化的")
        return int(last.dim_value)

    return OnnxShape(
        input_name=inputs[0].name,
        output_name=graph.output[0].name,
        obs_len=last_dim(inputs[0]),
        action_len=last_dim(graph.output[0]),
    )


def check_onnx(path: Path) -> OnnxShape:
    """拒绝守护进程在加载时会拒绝的文件: 宽度错误, 或不是 61 -> 14 的文件."""
    if not path.exists():
        raise ManifestError(f"{path}: 不存在这样的文件")
    shape = inspect_onnx(path)
    if shape.obs_len != OBS_LEN:
        raise ManifestError(
            f"{path.name}: 观测宽度是 {shape.obs_len}, 这个机器人构建的是 {OBS_LEN} "
            "(51 维策略是旧的 3 值命令一族, 守护进程会拒绝)"
        )
    if shape.action_len != ACTION_LEN:
        raise ManifestError(f"{path.name}: {shape.action_len} 个动作, 这个机器人有 {ACTION_LEN}")
    return shape


def smoke_run_onnx(path: Path, steps: int = 50, seed: int = 0) -> None:
    """用合理输入运行网络, 拒绝 NaN/inf 或饱和的输出.

    不是物理排练 -- 那由 `scripts/infer_policy.py` 承担 -- 但能在任何东西上传之前抓住
    导出损坏 (未内嵌的 normalizer 在原始观测上产生 NaN, 或图无法执行).
    """
    import numpy as np
    import onnxruntime as ort

    shape = inspect_onnx(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    obs = np.zeros((1, shape.obs_len), dtype=np.float32)
    outputs = []
    for _ in range(steps):
        (out,) = session.run([shape.output_name], {shape.input_name: obs})
        if not np.all(np.isfinite(out)):
            raise ManifestError(f"{path.name}: 网络产生了非有限的动作")
        outputs.append(out)
        # 把动作回填到最后动作槽位, 其余做抖动, 模拟观测在机器人上演变的方式;
        # 足够脱离零点了.
        obs = rng.normal(0.0, 0.05, size=obs.shape).astype(np.float32)
        obs[0, -ACTION_LEN - 13 : -13] = np.clip(out[0], -1, 1)
    spread = float(np.std(np.stack(outputs)))
    if spread == 0.0:
        raise ManifestError(f"{path.name}: 网络输出从未改变; 它是真正的策略吗?")


# ---------------------------------------------------------------------------------------------
# 仓库里还要放什么.


def install_commands(manifest: dict[str, Any], repo_id: str) -> str:
    """把这个策略装上机器人的 `robotctl` 命令 -- 每种形态一个用法.

    Episodic: 一个技能, 时长取自 manifest. 带 `unwind_s` 的 perpetual: 属主以 `--hold` 技能方式
    运行的保持姿态. 不带各自的 perpetual: 步态, 载入槽位.
    """
    name = manifest["name"]
    if manifest["kind"] == "episodic":
        return f"sudo robotctl policy add {name} {repo_id}\nrobotctl robot do {name}"
    if manifest.get("unwind_s") is not None:
        return f"sudo robotctl policy add {name} {repo_id} --hold <seconds>\nrobotctl robot do {name}"
    slot = manifest.get("slot", "<slot>")
    return f"sudo robotctl policy load {slot} {repo_id}"


def render_readme(manifest: dict[str, Any], repo_id: str) -> str:
    """说明如何在机器人上运行该策略的模型卡片, 自动生成以免过期."""
    kind = manifest["kind"]
    name = manifest["name"]
    description = manifest.get("description", "")
    training = manifest.get("training", {})
    run = install_commands(manifest, repo_id)
    if kind == "episodic":
        timing = f"Runs {manifest['duration_s']} s and returns itself to a standing pose."
        if manifest.get("chain"):
            timing += " Holding the button chains another run."
    elif manifest.get("unwind_s") is not None:
        timing = (
            f"Holds until told otherwise; the daemon drives `command.idle` for "
            f"{manifest['unwind_s']} s before handing back to the gait."
        )
    else:
        slot = manifest.get("slot")
        timing = "Runs until told otherwise" + (
            f" — a gait for the `{slot}` slot." if slot else " — a gait, loaded into a policy slot."
        )
    lines = [
        "---",
        "tags:",
        "- microduck",
        "- robotics",
        "- reinforcement-learning",
        "- onnx",
        "library_name: onnx",
        "---",
        "",
        f"# {name}",
        "",
        description,
        "",
        f"A **{kind}** policy for the [microduck](https://github.com/pollen-robotics/microduck) "
        f"({OBS_LEN}-D observation, {ACTION_LEN} actions, {ROBOT['control_hz']} Hz). {timing}",
        "",
        "## Run it on a robot",
        "",
        "```bash",
        run,
        "```",
        "",
        "The observation normalizer is baked into `policy.onnx`; feed raw observations.",
        "`manifest.json` follows schema 2 of the microduck policy manifest "
        "(`docs/policy-manifest.md` in the daemon repo).",
    ]
    if training:
        lines += ["", "## Training", ""]
        for key in ("task_id", "repo", "branch", "commit", "run", "checkpoint", "exported"):
            if key in training:
                lines.append(f"- **{key}**: `{training[key]}`")
        if training.get("dirty"):
            lines.append("- exported from a checkout with uncommitted changes")
    return "\n".join(lines) + "\n"


def dump_manifest(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2) + "\n"
