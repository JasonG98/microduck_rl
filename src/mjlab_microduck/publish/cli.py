r"""`uv run publish` -- 以 microduck 守护进程能够加载的形式, 把策略放到 Hub 上.

    # 从 wandb 运行导出 (内嵌 normalizer, -- 从 checkpoint 出发唯一的可靠路径)
    uv run publish --task Mjlab-PoliteBow-Flat-MicroDuck --wandb-run-path ent/proj/run --checkpoint 3000 \\
        --repo <user>/microduck-polite-bow --kind episodic --duration-s 4.0

    # 从你已导出的 ONNX 文件出发
    uv run publish --onnx out.onnx --repo <user>/microduck-flamingo --kind perpetual --unwind-s 1.5

    # 针对某个槽位的步态 (无 hold, 无 unwind: 会一直运行, 直到另行指示)
    uv run publish --onnx walk.onnx --repo <user>/microduck-my-walk --kind perpetual --slot walk

无论哪种路径, 仓库都会获得 `policy.onnx`, schema-2 的 `manifest.json` 和一份 README; 文件在
上传前会先校验 61 -> 14 形状并进行冒烟运行, 且已有的 `policy.onnx` 在未指定 `--force` 时
不会被覆盖. `--dry-run` 会把仓库内容写到本地目录然后结束.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

import tyro

from mjlab_microduck.publish import manifest as m


@dataclass(frozen=True)
class PublishConfig:
    # -- 去处
    repo: str
    """Hub 仓库 id, `<user-or-org>/microduck-<name>`. 若不存在则创建 (私有)."""
    kind: Literal["episodic", "perpetual"]
    """episodic: 运行 `duration_s` 后自行回来. perpetual: 一直保持, 直到另行指示."""

    # -- 权重来源: 必须是 (--task + checkpoint) 或 --onnx 之一
    task: str | None = None
    """要导出的任务 id, 例如 Mjlab-PoliteBow-Flat-MicroDuck. 需要一个 checkpoint."""
    wandb_run_path: str | None = None
    """`entity/project/run_id`. 与 --task 一起使用时, 是 checkpoint 的来源."""
    checkpoint: int | None = None
    """Checkpoint 迭代数 (model_<N>.pt). 默认: 该次运行的最新一个."""
    checkpoint_file: str | None = None
    """本地的 model_<N>.pt, 用来代替 wandb."""
    onnx: str | None = None
    """已导出的 ONNX. 只会校验, 不会重新导出."""

    # -- manifest 会写明的内容
    name: str | None = None
    """客户端请求时用的名称 (`robotctl robot do <name>`). 默认: 仓库 stem 去掉 `microduck-` 前缀."""
    description: str | None = None
    """单行描述. 默认: 任务 id."""
    duration_s: float | None = None
    """仅供 episodic: 运行的秒数."""
    chain: bool = False
    """仅供 episodic: 按住按键可以衔接下一次运行 (roulade 会, kick 不会)."""
    unwind_s: float | None = None
    """perpetual 保持姿态 (flamingo): 守护进程在交还控制前驱动 `idle` 的秒数. 步态则留空."""
    slot: Literal["walk", "stand", "sitstand", "ground_pick", "kick_left", "kick_right", "roulade"] | None = None
    """perpetual 步态: 属于哪个槽位 (walk, stand, ...). 仅展示用; 决定安装提示."""
    idle: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """表示 '停止当前动作' 的 twist. 目前发布的所有一次性策略都用零向量."""
    action_scale: float | None = None
    """策略自身的输出缩放, 如果它需要的话. 默认: 该步态的值."""
    entry_pose: str = "standing"
    """策略预期开始时所处的姿态."""
    twist_help: str | None = None
    """当槽位有特定含义时, `command.twist` 的文字说明 (flamingo: '[flag, side, 0]')."""

    # -- 如何发布
    private: bool = True
    """创建私有仓库 (要公开用 --no-private). 已存在仓库保持其可见性."""
    force: bool = False
    """覆盖仓库中已有的 policy.onnx."""
    tag: str | None = None
    """给本次修订打标签, 例如 v1."""
    smoke: bool = True
    """上传前用合理输入运行网络, 拒绝 NaN."""
    dry_run: bool = False
    """把 policy.onnx, manifest.json 和 README.md 写到 ./publish-<name>/ 然后结束."""
    device: str | None = None
    """导出设备. 默认: 有则用 cuda:0, 否则用 cpu."""


def _fail(msg: str) -> NoReturn:
    print(f"[publish] error: {msg}", file=sys.stderr)
    sys.exit(2)


def _resolve_weights(cfg: PublishConfig, workdir: Path) -> tuple[Path, dict]:
    """要发布的 ONNX 及它携带的来源信息. 给的是 checkpoint 时执行导出."""
    from_checkpoint = cfg.task is not None or cfg.checkpoint_file is not None
    if (cfg.onnx is None) == (not from_checkpoint):
        _fail("必须提供且只能提供一个来源: --onnx <file>, 或 --task <id> 配合 --wandb-run-path/--checkpoint-file")

    training: dict = {"repo": "pollen-robotics/microduck_rl", **m.git_provenance()}
    if cfg.onnx is not None:
        src = Path(cfg.onnx)
        training["source_file"] = src.name
        if cfg.task:
            training["task_id"] = cfg.task
        return src, training

    if cfg.task is None:
        _fail("--checkpoint-file 需要 --task <id>, 才能构建它训练时所用的环境")
    if cfg.wandb_run_path is None and cfg.checkpoint_file is None:
        _fail("--task 需要 --wandb-run-path (可选加 --checkpoint) 或 --checkpoint-file")

    # 只有这条路径才做重型导入: ONNX 路径必须在没有 GPU 或 mjlab 注册表的情况下也能工作.
    import mjlab.tasks  # noqa: F401  (填充注册表)

    from mjlab_microduck.export import ExportConfig, run_export

    out = workdir / m.POLICY_FILE
    result = run_export(
        cfg.task,
        ExportConfig(
            onnx_file=str(out),
            wandb_run_path=cfg.wandb_run_path,
            checkpoint=cfg.checkpoint,
            checkpoint_file=cfg.checkpoint_file,
            num_envs=1,
            device=cfg.device,
        ),
    )
    training["task_id"] = cfg.task
    if result.wandb_run_path:
        training["run"] = result.wandb_run_path
    if result.checkpoint_iteration is not None:
        training["checkpoint"] = result.checkpoint_iteration
    return result.onnx_path, training


def _default_name(repo: str) -> str:
    stem = repo.rsplit("/", 1)[-1]
    return stem.removeprefix("microduck-").removeprefix("microduck_") or stem


def run(cfg: PublishConfig) -> int:
    if "/" not in cfg.repo:
        _fail("--repo 必须是 `<user-or-org>/<name>`")
    name = cfg.name or _default_name(cfg.repo)

    workdir = Path(tempfile.mkdtemp(prefix="microduck-publish-"))
    try:
        onnx_path, training = _resolve_weights(cfg, workdir)
        shape = m.check_onnx(onnx_path)
        print(f"[publish] {onnx_path.name}: {shape.obs_len} -> {shape.action_len}, ok")
        if cfg.smoke:
            m.smoke_run_onnx(onnx_path)
            print("[publish] 冒烟运行: 输出有限且非常量")

        command_help = {"twist": cfg.twist_help} if cfg.twist_help else None
        manifest = m.build_manifest(
            name=name,
            kind=cfg.kind,
            description=cfg.description or training.get("task_id") or name,
            duration_s=cfg.duration_s,
            chain=cfg.chain,
            unwind_s=cfg.unwind_s,
            idle=cfg.idle,
            action_scale=cfg.action_scale,
            entry_pose=cfg.entry_pose,
            slot=cfg.slot,
            command_help=command_help,
            training=training,
        )
        m.validate_manifest(manifest)

        staged = workdir / "repo"
        staged.mkdir()
        shutil.copyfile(onnx_path, staged / m.POLICY_FILE)
        (staged / "manifest.json").write_text(m.dump_manifest(manifest))
        (staged / "README.md").write_text(m.render_readme(manifest, cfg.repo))

        if cfg.dry_run:
            dest = Path.cwd() / f"publish-{name}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(staged, dest)
            print(f"[publish] 干运行: 已写入 {dest}/ (policy.onnx, manifest.json, README.md)")
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(cfg.repo, repo_type="model", private=cfg.private, exist_ok=True)
        existing = set(api.list_repo_files(cfg.repo))
        onnx_files = {f for f in existing if f.endswith(".onnx")}
        if onnx_files and not cfg.force:
            _fail(
                f"{cfg.repo} 已包含 {sorted(onnx_files)}; 用 --force 覆盖. "
                "一个仓库只携带一个 .onnx, 所以换个名字就是新建一个仓库."
            )
        stale = onnx_files - {m.POLICY_FILE}
        commit = api.upload_folder(
            repo_id=cfg.repo,
            folder_path=str(staged),
            commit_message=f"publish {name}: {cfg.kind}, {training.get('task_id', onnx_path.name)}",
            delete_patterns=sorted(stale) or None,
        )
        url = getattr(commit, "commit_url", None) or f"https://huggingface.co/{cfg.repo}"
        print(f"[publish] 已上传: {url}")
        if cfg.tag:
            api.create_tag(cfg.repo, tag=cfg.tag, tag_message=f"{name} {cfg.tag}")
            print(f"[publish] 已打标签 {cfg.tag}")
        first = m.install_commands(manifest, cfg.repo).splitlines()[0]
        print(f"[publish] 在机器人上的命令: {first}")
        return 0
    except m.ManifestError as e:
        _fail(str(e))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


def main() -> int:
    cfg = tyro.cli(PublishConfig)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
