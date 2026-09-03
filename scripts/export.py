"""将训练好的 checkpoint 导出为 ONNX - 对 `mjlab_microduck.export` 的轻薄封装.

    uv run scripts/export.py <TASK_ID> --wandb-run-path <entity/project/run> [--checkpoint 3000]

导出逻辑放在包内, 这样 `uv run publish` 也能复用它; 详见该模块的 docstring.
"""

from mjlab_microduck.export import main

if __name__ == "__main__":
    main()