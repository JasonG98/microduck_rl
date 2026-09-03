"""把训练好的检查点导出到 ONNX, 并内嵌观测归一化器.

这是从检查点到可部署 `.onnx` 的唯一路径: `runner.export_policy_to_onnx`
产出 `actor(normalizer(obs))`, 因此机器上运行的正是训练所见的. 仿真中的 `play`
会自己施加归一化器, 从而掩盖一个忘了内嵌它的手误转换检查点 -- 切勿手工转换.

`scripts/export.py` 是命令行封装; `mjlab_microduck.publish` 会直接调用
:func:`run_export`, 因此已发布的策略不可能跳过这一步.
"""

import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class ExportConfig:
    onnx_file: str = "output.onnx"
    agent: Literal["zero", "random", "trained"] = "trained"
    registry_name: str | None = None
    wandb_run_path: str | None = None
    checkpoint: int | None = None      # 按迭代编号选择检查点 (例如 3000)
    checkpoint_file: str | None = None
    motion_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"

    # 供 demo 脚本使用的内部标志.
    _demo_mode: tyro.conf.Suppress[bool] = False


@dataclass(frozen=True)
class ExportResult:
    """导出产出了什么、它来自哪里, 用于发布者的来源 (provenance) 信息块."""

    onnx_path: Path
    checkpoint_path: Path | None
    wandb_run_path: str | None
    checkpoint_iteration: int | None


def _iteration_of(checkpoint_path: Path | None) -> int | None:
    if checkpoint_path is None:
        return None
    match = re.search(r"model_(\d+)\.pt$", checkpoint_path.name)
    return int(match.group(1)) if match else None


def run_export(task_id: str, cfg: ExportConfig) -> ExportResult:
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    # 检查这是否是运动跟踪任务.
    is_motion_tracking = (
        env_cfg.commands is not None
        and "motion" in env_cfg.commands
        and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    )
    is_tracking_task = is_motion_tracking

    if is_tracking_task and cfg._demo_mode:
        # Demo 模式: num_envs > 1 时使用均匀采样以看到更多多样性.
        assert env_cfg.commands is not None
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.sampling_mode = "uniform"

    if is_tracking_task:
        assert env_cfg.commands is not None
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)

        # 检查 motion file 是否已设置且存在
        motion_file_already_set = (
            hasattr(motion_cmd, 'motion_file')
            and motion_cmd.motion_file is not None
            and Path(motion_cmd.motion_file).exists()
        )

        if DUMMY_MODE:
            if not cfg.registry_name:
                raise ValueError(
                    "使用 dummy agents 时, 跟踪任务需要 `registry_name`."
                )
            # 检查 registry name 是否包含 alias, 若不包含则追加 ":latest".
            registry_name = cfg.registry_name
            if ":" not in registry_name:
                registry_name = registry_name + ":latest"
            import wandb

            api = wandb.Api()
            artifact = api.artifact(registry_name)
            motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
        else:
            if cfg.motion_file is not None:
                print(f"[INFO]: 从 CLI 使用 motion file: {cfg.motion_file}")
                motion_cmd.motion_file = cfg.motion_file
            elif motion_file_already_set:
                print(f"[INFO]: 从 env config 使用 motion file: {motion_cmd.motion_file}")
            else:
                # 尝试从 wandb artifacts 下载
                import wandb

                api = wandb.Api()
                if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
                    raise ValueError(
                        "使用 `checkpoint_file` 时, 跟踪任务需要 `motion_file`, "
                        "或提供 `wandb_run_path` 以便解析 motion artifact."
                    )
                if cfg.wandb_run_path is not None:
                    wandb_run = api.run(str(cfg.wandb_run_path))
                    art = next(
                        (a for a in wandb_run.used_artifacts() if a.type == "motions"),
                        None,
                    )
                    if art is None:
                        raise RuntimeError("该 run 中未找到 motion artifact.")
                    motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

    log_dir: Path | None = None
    resume_path: Path | None = None
    if TRAINED_MODE:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"未找到 checkpoint 文件: {resume_path}")
            print(f"[INFO]: 正在加载 checkpoint: {resume_path.name}")
        elif cfg.checkpoint is not None:
            # 从 wandb 或本地选择指定的 checkpoint 迭代.
            checkpoint_filename = f"model_{cfg.checkpoint}.pt"
            if cfg.wandb_run_path is not None:
                import wandb
                api = wandb.Api()
                wandb_run = api.run(str(cfg.wandb_run_path))
                run_id = cfg.wandb_run_path.split("/")[-1]
                download_dir = log_root_path / "wandb_checkpoints" / run_id
                resume_path = download_dir / checkpoint_filename
                if resume_path.exists():
                    print(f"[INFO]: 正在加载 checkpoint: {checkpoint_filename} (run: {run_id}, 已缓存)")
                else:
                    available = [f.name for f in wandb_run.files() if "model" in f.name]
                    if checkpoint_filename not in available:
                        raise FileNotFoundError(
                            f"在 wandb run 中未找到 checkpoint '{checkpoint_filename}'. "
                            f"可用: {sorted(available)}"
                        )
                    wandb_run.file(checkpoint_filename).download(str(download_dir), replace=True)
                    print(f"[INFO]: 正在加载 checkpoint: {checkpoint_filename} (run: {run_id}, 已下载)")
            else:
                resume_path = get_checkpoint_path(
                    log_root_path, checkpoint=re.escape(checkpoint_filename)
                )
                print(f"[INFO]: 正在加载 checkpoint: {resume_path.name}")
        else:
            if cfg.wandb_run_path is None:
                raise ValueError(
                    "未提供 `checkpoint_file` 时需要 `wandb_run_path`."
                )
            resume_path, was_cached = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path)
            )
            # 从路径提取 run_id 和 checkpoint 名称用于展示.
            run_id = resume_path.parent.name
            checkpoint_name = resume_path.name
            cached_str = "已缓存" if was_cached else "已下载"
            print(
                f"[INFO]: 正在加载 checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
            )
        log_dir = resume_path.parent

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print(
            "[WARN] 使用 dummy agents 时视频录制被禁用 (无 checkpoint/log_dir)."
        )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if TRAINED_MODE and cfg.video:
        print("[INFO] play 期间正在录制视频")
        assert log_dir is not None  # log_dir 在 TRAINED_MODE 块中设置
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "play",
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
        if cfg.agent == "zero":

            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)

            policy = PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

            policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(task_id) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(resume_path), map_location=device)
        policy = runner.get_inference_policy(device=device)

    # mjlab 1.3.0: ONNX 导出 + metadata 已迁移到 mjlab.rl.exporter_utils, 使用
    # runner 内置的 export_policy_to_onnx. 观测归一化会自动烧录进导出的图 -
    # EmpiricalNormalization 是策略 MLPModel 的子模块 (RslRlModelCfg 中
    # obs_normalization=True), 因此 export_policy_to_onnx 输出的是
    # actor(normalizer(obs)). 无需手动处理归一化 (旧的
    # export_velocity_policy_as_onnx 路径已移除).
    from mjlab.rl.exporter_utils import get_base_metadata, attach_metadata_to_onnx

    onnx_path = os.path.abspath(cfg.onnx_file)
    path = os.path.dirname(onnx_path)
    filename = os.path.basename(onnx_path)

    runner.export_policy_to_onnx(path, filename)

    metadata = get_base_metadata(runner.env.unwrapped, run_path=cfg.checkpoint_file)
    attach_metadata_to_onnx(onnx_path, metadata)

    print(f"已写入 {onnx_path}")

    env.close()
    return ExportResult(
        onnx_path=Path(onnx_path),
        checkpoint_path=resume_path,
        wandb_run_path=cfg.wandb_run_path,
        checkpoint_iteration=_iteration_of(resume_path),
    )


def main():
    # 解析第一个参数以选择任务.
    # 导入 tasks 以填充 registry.
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    # 解析其余参数 + 允许覆盖 env_cfg 和 agent_cfg.
    agent_cfg = load_rl_cfg(chosen_task)

    args = tyro.cli(
        ExportConfig,
        args=remaining_args,
        default=ExportConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )
    del remaining_args, agent_cfg

    run_export(chosen_task, args)


if __name__ == "__main__":
    main()
