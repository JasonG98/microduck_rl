"""查找指定用户的最新 wandb run 并启动 `uv run play`.

默认: 最新一个 run (所有任务混合). 加上类型 flag, 只取该类型的最新 run:
md-play --crouch      # 最新 Mjlab-RollerCrouch-...     md-play --roller      # 最新 Mjlab-...-Rollers     md-play
--swizzle     # 最新 Mjlab-...-Swizzle-...     md-play --slope       # 最新 Mjlab-RollerSlope-... 未识别的参数会
原样转发给 `uv run play` (例如 md-play --crouch --action-scale 0.8).
"""

import argparse

from wandb_utils import resolve_run, run_command

# flag -> 在 task_id (metadata args[0]) 中搜索的子串
TYPE_SUBSTR = {
    "crouch": "Crouch",  # Mjlab-RollerCrouch-Flat-MicroDuck
    "roller": "MicroDuck-Rollers",  # Mjlab-Velocity-Flat-MicroDuck-Rollers (≠ RollerSlope/RollerCrouch)
    "swizzle": "Swizzle",  # Mjlab-Velocity-Swizzle-MicroDuck
    "slope": "Slope",  # Mjlab-RollerSlope-Flat-MicroDuck
}


def main():
    """查找并播放匹配用户/类型过滤条件的最新 wandb run."""
    parser = argparse.ArgumentParser(description="播放指定用户的最新 wandb run")
    parser.add_argument(
        "--user",
        default="coralie",
        help="按用户过滤 (匹配 email, 默认: coralie)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印 play 命令, 不执行",
    )
    for t in TYPE_SUBSTR:
        parser.add_argument(
            f"--{t}",
            dest="type",
            action="store_const",
            const=t,
            help=f"只取最新的 '{t}' run",
        )
    # 未识别的参数 (例如 --action-scale 0.8) 会转发给 `uv run play`
    args, extra = parser.parse_known_args()

    task_substr = TYPE_SUBSTR[args.type] if getattr(args, "type", None) else None
    _, info = resolve_run(args.user, task_substr)

    cmd = [
        "uv",
        "run",
        "play",
        info["env_name"],
        "--wandb-run-path",
        info["run_path"],
        *extra,
    ]
    run_command(cmd, args.dry_run)


if __name__ == "__main__":
    main()
