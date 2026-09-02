"""导出 RL 智能体为 ONNX 的脚本."""

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from rsl_rl.runners import OnPolicyRunner


@dataclass(frozen=True)
class ExportConfig:
    """ONNX 策略导出配置."""

    onnx_file: str = "output.onnx"
    agent: Literal["zero", "random", "trained"] = "trained"
    registry_name: str | None = None
    wandb_run_path: str | None = None
    checkpoint: int | None = None  # 按迭代次数选择 checkpoint (例如 3000)
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

    # 内部标志, 由 demo 脚本使用.
    _demo_mode: tyro.conf.Suppress[bool] = False


def run_export(task_id: str, cfg: ExportConfig):
    """将指定任务的训练策略导出为 ONNX 格式."""
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    # 检查这是否是一个 motion tracking 任务.
    is_motion_tracking = (
        env_cfg.commands is not None
        and "motion" in env_cfg.commands
        and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    )
    is_tracking_task = is_motion_tracking

    if is_tracking_task and cfg._demo_mode:
        # Demo 模式: 使用均匀采样以便在 num_envs > 1 时看到更多多样性.
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
            hasattr(motion_cmd, "motion_file")
            and motion_cmd.motion_file is not None
            and Path(motion_cmd.motion_file).exists()
        )

        if DUMMY_MODE:
            if not cfg.registry_name:
                raise ValueError("Tracking tasks require `registry_name` when using dummy agents.")
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
                print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
                motion_cmd.motion_file = cfg.motion_file
            elif motion_file_already_set:
                print(f"[INFO]: Using motion file from env config: {motion_cmd.motion_file}")
            else:
                # 尝试从 wandb artifacts 下载
                import wandb

                api = wandb.Api()
                if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
                    raise ValueError(
                        "Tracking tasks require `motion_file` when using `checkpoint_file`, "
                        "or provide `wandb_run_path` so the motion artifact can be resolved."
                    )
                if cfg.wandb_run_path is not None:
                    wandb_run = api.run(str(cfg.wandb_run_path))
                    art = next(
                        (a for a in wandb_run.used_artifacts() if a.type == "motions"),
                        None,
                    )
                    if art is None:
                        raise RuntimeError("No motion artifact found in the run.")
                    motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

    log_dir: Path | None = None
    resume_path: Path | None = None
    if TRAINED_MODE:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
            print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        elif cfg.checkpoint is not None:
            # 从 wandb 或本地选择一个特定迭代的 checkpoint.
            checkpoint_filename = f"model_{cfg.checkpoint}.pt"
            if cfg.wandb_run_path is not None:
                import wandb

                api = wandb.Api()
                wandb_run = api.run(str(cfg.wandb_run_path))
                run_id = cfg.wandb_run_path.split("/")[-1]
                download_dir = log_root_path / "wandb_checkpoints" / run_id
                resume_path = download_dir / checkpoint_filename
                if resume_path.exists():
                    print(f"[INFO]: Loading checkpoint: {checkpoint_filename} (run: {run_id}, cached)")
                else:
                    available = [f.name for f in wandb_run.files() if "model" in f.name]
                    if checkpoint_filename not in available:
                        raise FileNotFoundError(
                            f"Checkpoint '{checkpoint_filename}' not found in wandb run. Available: {sorted(available)}"
                        )
                    wandb_run.file(checkpoint_filename).download(str(download_dir), replace=True)
                    print(f"[INFO]: Loading checkpoint: {checkpoint_filename} (run: {run_id}, downloaded)")
            else:
                resume_path = get_checkpoint_path(log_root_path, checkpoint=re.escape(checkpoint_filename))
                print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        else:
            if cfg.wandb_run_path is None:
                raise ValueError("`wandb_run_path` is required when `checkpoint_file` is not provided.")
            resume_path, was_cached = get_wandb_checkpoint_path(log_root_path, Path(cfg.wandb_run_path))
            # 从路径中提取 run_id 和 checkpoint 名称用于显示.
            run_id = resume_path.parent.name
            checkpoint_name = resume_path.name
            cached_str = "cached" if was_cached else "downloaded"
            print(f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})")
        log_dir = resume_path.parent

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print("[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir).")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if TRAINED_MODE and cfg.video:
        print("[INFO] Recording videos during play")
        assert log_dir is not None  # log_dir 在 TRAINED_MODE 块中已设置
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

            PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

            PolicyRandom()
    else:
        runner_cls = load_runner_cls(task_id) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(resume_path), map_location=device)
        runner.get_inference_policy(device=device)

    # mjlab 1.3.0: ONNX 导出 + metadata 移到了 mjlab.rl.exporter_utils 和
    # runner 内置的 export_policy_to_onnx. 观测归一化会自动烘焙到导出的
    # 计算图中 — EmpiricalNormalization 是 policy 的 MLPModel 的子模块
    # (RslRlModelCfg 中 obs_normalization=True), 因此 export_policy_to_onnx
    # 输出的是 actor(normalizer(obs)). 不需要手动处理归一化器
    # (旧的 export_velocity_policy_as_onnx 路径已移除).
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    onnx_path = Path(cfg.onnx_file).resolve()
    path = str(onnx_path.parent)
    filename = onnx_path.name

    runner.export_policy_to_onnx(path, filename)

    metadata = get_base_metadata(runner.env.unwrapped, run_path=cfg.checkpoint_file)
    attach_metadata_to_onnx(str(onnx_path), metadata)

    print(f"Written {onnx_path}")

    env.close()


def main():
    """CLI 入口: 将训练好的策略导出为 ONNX 格式."""
    # 解析第一个参数以选择任务.
    # 导入 tasks 以填充注册表.
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
