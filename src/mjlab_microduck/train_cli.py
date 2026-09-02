"""`train` 控制台脚本 — 刻意与 mjlab 自身的保持一致.

本项目必须继续声明一个 `train` 脚本, 尽管它已不再需要: `--hf-jobs` 从
`mjlab.tasks` 插件入口点 (train_hook.py) 拦截, 走的是 mjlab 自身的导入路径,
所以这个包装器已无事可做.

为什么仍然保留: 同名控制台脚本遵循后写者覆盖规则, 安装后两份 dist-info RECORD 都声明了
`.venv/bin/train`. 因此删除这个声明并不会把名字还给 mjlab — 它会让下一次 `uv
sync` 卸载该文件 (我们的, 按照我们的 RECORD), 而没有任何东西重新安装 mjlab 的版本.
于是 `train` 从 venv 中彻底消失, `uv run train` 落到 PATH 上的某个 `train`:
在一台机器上是 liblinear 的, 它回应 `can't open input file Mjlab-Velocity-Flat-MicroDuck`
(2026-08-31).

所以冲突保留, 但被改造为无害: 无论哪个 shim 胜出, `train` 行为一致,
该 flag 也只在唯一一处处理.
"""

from __future__ import annotations

import sys


def main() -> int | None:
    """委托给 mjlab 的 train 入口点 (处理 ``--hf-jobs`` 拦截)."""
    # 这次导入会运行 mjlab 的插件加载器, 后者导入
    # mjlab_microduck.tasks -> train_hook.maybe_submit_to_hf_jobs().
    # 一次 `--hf-jobs` 调用会在下方导入内提交并退出;
    # 永远不会回到这里.
    from mjlab.scripts.train import main as mjlab_train_main

    return mjlab_train_main()


if __name__ == "__main__":
    sys.exit(main())
