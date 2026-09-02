"""保持 `train <task> ...

--hf-jobs` 可用, 无论谁拥有 `train` 脚本.
该 flag 过去存在于我们自己的 `train` 控制台脚本中, 在
`[project.scripts]` 中声明并文档化为 "覆盖" mjlab 的. 它什么也没覆盖:
mjlab 1.3.0 也声明了 `train`, 两个发行版声明同一个脚本名在安装时
是后写者覆盖, mjlab 胜了 — `uv sync` 在 `.venv/bin/train` 留下了
`mjlab.scripts.train:main`, 所以我们的包装器从未被调用,
`uv run train ... --hf-jobs` 死在 tyro 的
`Unrecognized options: --hf-jobs` (2026-08-31). 没有任何警告:
安装成功, flag 静默消失.

所以该 flag 不再在任何控制台脚本中实现. 它在此处从
`mjlab.tasks` 插件入口点拦截: mjlab 自身的 `mjlab/__init__.py` 在模块作用域
调用 `_import_registered_packages()`, 这会导入 `mjlab_microduck.tasks` —
而 mjlab 的 `train` 在执行 `from mjlab.scripts.train import main` 时就到达了这里,
即在它两阶段的 tyro 解析看到 argv 之前. 这条路径属于 mjlab 自身,
任何安装顺序都无法夺走.

`uv run scripts/hf/train_hf.py <task> ...` 直接调用 `submit()`, 如果
此拦截逻辑不再触发, 它仍是后路.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_FLAG = "--hf-jobs"

#: 由 ``hf_jobs.submit`` 设置到任务环境上 — 在任务内部,
#: ``uv run train`` 必须始终意为 "本地训练".
_IN_JOB_ENV = "MICRODUCK_IN_HF_JOB"


def _invoked_as_train() -> bool:
    """当 argv[0] 是 mjlab 的 trainer (控制台脚本或 `-m`) 时返回 True.

    `play --hf-jobs` 不得提交训练任务; 让那个命令自己的解析器去拒绝该 flag.
    """
    prog = Path(sys.argv[0]).name
    return prog.removesuffix(".py").removesuffix("-script") == "train"


def maybe_submit_to_hf_jobs() -> None:
    """消费 `--hf-jobs` 并退出进程; 没有 flag 时是空操作.

    在 `mjlab_microduck.tasks` 导入时调用, 因此它在 mjlab 的插件加载器内执行.
    `SystemExit` 是 `BaseException`, 所以它会穿透加载器的 `except Exception`
    并从 `import mjlab` 传播出去 — 本地 trainer 永远不会启动.
    """
    if _FLAG not in sys.argv[1:]:
        return
    if os.environ.get(_IN_JOB_ENV):
        return
    if not _invoked_as_train():
        return

    from mjlab_microduck.hf_jobs import submit

    sys.exit(submit([a for a in sys.argv[1:] if a != _FLAG]))
