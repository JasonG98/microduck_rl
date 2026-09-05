"""`train <task> ... --hf-jobs` 必须持续提交到 HF Jobs.

这个 flag 本来属于我们自己的 `train` console script, 它本该 shadow
mjlab 的同名 console script.同名 console script 是 last-writer-wins,
mjlab 的 shim 赢了, flag 静默消失: `uv run train ... --hf-jobs` 撞上
mjlab 的 tyro parser 并以 `Unrecognized options: --hf-jobs` 死掉
(2026-08-31).

现在改为从 `mjlab.tasks` plugin entry point 拦截 (train_hook.py),
mjlab 自己会 import 它, 所以两个 shim 都能处理这个 flag.

`train` 的声明要保留 —— 删掉它会让事情更糟而不是更好: 两个 RECORD
都声称 bin/train, 所以 `uv sync` 会卸载文件, 而没人重建 mjlab 的, 于是
`uv run train` 会从 PATH 找到 liblinear 的 `train` ("can't open input
file Mjlab-Velocity-Flat-MicroDuck").

三件事必须同时成立, 任何一件都不会单独响亮地失败:

1. `train` 保持声明, 且与 mjlab 的行为一致.
2. mjlab 必须仍能在解析 argv 之前 reach 到 `mjlab_microduck.tasks` ——
   如果 mjlab 的 plugin loading 重构, 会静默丢掉这个 flag.
3. 两个 shim 必须都拦截, 因为最终落进 bin/ 的那个不是我们能决定的.
"""

import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import distribution
from pathlib import Path

import pytest

from mjlab_microduck import train_hook

_ROOT = Path(__file__).resolve().parents[1]
_TASK = "Mjlab-Velocity-Flat-MicroDuck"


def _our_scripts():
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    return pyproject["project"].get("scripts", {})


def test_train_script_stays_declared():
    """删掉它会卸载 bin/train 且不重建 (见模块 docstring)."""
    assert _our_scripts().get("train") == "mjlab_microduck.train_cli:main", (
        "[project.scripts] 必须继续声明 `train`.本包和 mjlab 都在各自的"
        " RECORD 里声称 bin/train, 所以删掉声明会让 `uv sync` DELETE 该"
        " script 而不是回退到 mjlab 的, 然后 `uv run train` 会跑 PATH 上"
        "另一个无关的 `train`."
    )


def test_our_train_script_only_delegates_to_mjlab():
    """碰撞只有在两个 shim 可以互换时才安全."""
    import mjlab.scripts.train

    from mjlab_microduck import train_cli

    called = []
    original = mjlab.scripts.train.main
    mjlab.scripts.train.main = lambda: called.append(True) or 5
    try:
        assert train_cli.main() == 5, "我们的 `train` 必须返回 mjlab 的 exit code"
    finally:
        mjlab.scripts.train.main = original
    assert called == [True], "我们的 `train` 必须原封不动地调用 mjlab 的 trainer"


def test_train_on_path_is_a_mjlab_trainer():
    """直接捕捉 vanished-script 失败: `train` 必须是我们的或 mjlab 的,
    绝不能是机器上随便安装的别的什么."""
    exe = shutil.which("train")
    assert exe is not None, (
        "PATH 上没有 `train` —— venv script 被卸载且未重建; 运行 `uv sync --reinstall-package mjlab-microduck`."
    )
    head = Path(exe).read_bytes()[:8192]
    assert b"mjlab" in head, (
        f"`train` 解析到 {exe}, 它不是 mjlab trainer.bin/train 被卸载了, 这是 PATH 上另一个无关的二进制."
    )


def test_we_register_the_task_plugin_entry_point():
    """拦截挂在这个 entry point 上; 没有它, 就没有 flag."""
    groups = {ep.group: ep.value for ep in distribution("mjlab-microduck").entry_points}
    assert groups.get("mjlab.tasks") == "mjlab_microduck.tasks", (
        "`mjlab.tasks` entry point 必须保持指向 mjlab_microduck.tasks: "
        "它既是 task 注册的方式, 也是 --hf-jobs 被拦截的地方."
    )


@pytest.fixture
def fake_submit(monkeypatch):
    calls = []
    monkeypatch.setattr("mjlab_microduck.hf_jobs.submit", lambda argv: calls.append(argv) or 0)
    return calls


def test_hook_submits_and_strips_the_flag(monkeypatch, fake_submit):
    monkeypatch.setattr(
        sys,
        "argv",
        ["/x/.venv/bin/train", _TASK, "--env.scene.num-envs", "4096", "--hf-jobs"],
    )
    with pytest.raises(SystemExit) as exc:
        train_hook.maybe_submit_to_hf_jobs()
    assert exc.value.code == 0
    # --hf-jobs 不能传到 job 自己的 `uv run train` (那是 mjlab 的).
    assert fake_submit == [[_TASK, "--env.scene.num-envs", "4096"]]


def test_hook_propagates_the_submission_exit_code(monkeypatch):
    monkeypatch.setattr("mjlab_microduck.hf_jobs.submit", lambda argv: 3)
    monkeypatch.setattr(sys, "argv", ["train", _TASK, "--hf-jobs"])
    with pytest.raises(SystemExit) as exc:
        train_hook.maybe_submit_to_hf_jobs()
    assert exc.value.code == 3


def test_no_flag_trains_locally(monkeypatch, fake_submit):
    monkeypatch.setattr(sys, "argv", ["train", _TASK, "--env.scene.num-envs", "4096"])
    assert train_hook.maybe_submit_to_hf_jobs() is None
    assert fake_submit == []


def test_other_commands_do_not_submit(monkeypatch, fake_submit):
    """`play --hf-jobs` 必须走到 play 的 parser, 不能提交一个训练 job."""
    monkeypatch.setattr(sys, "argv", ["/x/.venv/bin/play", _TASK, "--hf-jobs"])
    assert train_hook.maybe_submit_to_hf_jobs() is None
    assert fake_submit == []


def test_interception_is_disarmed_inside_the_job(monkeypatch, fake_submit):
    """否则一个泄露的 flag 会让 job 再提交一个 job."""
    monkeypatch.setenv("MICRODUCK_IN_HF_JOB", "1")
    monkeypatch.setattr(sys, "argv", ["train", _TASK, "--hf-jobs"])
    assert train_hook.maybe_submit_to_hf_jobs() is None
    assert fake_submit == []


def test_submitted_job_env_disarms_the_interception():
    """上面这个 env 变量只有在 submit() 真的设置它时才有用."""
    src = (_ROOT / "src/mjlab_microduck/hf_jobs.py").read_text()
    assert f'"{train_hook._IN_JOB_ENV}": "1"' in src, f"submit() 必须把 {train_hook._IN_JOB_ENV} 放到 job 的 env 上"


# 关键假设, 通过真实 import 路径来检验: 无论是 `from
# mjlab.scripts.train import main` (mjlab 的 shim) 还是我们自己的 shim, 都
# 必须在 mjlab 解析 argv 之前 reach 到 hook.用 subprocess, 因为它在
# import 内部以 SystemExit 结束.
_PROBE = """
import sys
sys.argv = ["train", "{task}", "--env.scene.num-envs", "4096", "--hf-jobs"]

import mjlab_microduck.hf_jobs as hf_jobs

def fake_submit(argv):
    print("SUBMIT", argv, flush=True)
    return 7

hf_jobs.submit = fake_submit

try:
    {trigger}
except SystemExit as e:
    print("EXIT", e.code, flush=True)
    raise SystemExit(0)

print("NOT-INTERCEPTED", flush=True)
raise SystemExit(1)
"""

_SHIMS = {
    # 正是 mjlab 写出来的 bin/train 的内容
    "mjlab": "from mjlab.scripts.train import main",
    # ... 以及本包写出来的那种
    "ours": "from mjlab_microduck.train_cli import main; main()",
}


@pytest.mark.parametrize("shim", sorted(_SHIMS))
def test_both_train_shims_reach_the_hook_before_parsing_argv(shim):
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(task=_TASK, trigger=_SHIMS[shim])],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=_ROOT,
    )
    assert "NOT-INTERCEPTED" not in proc.stdout, (
        f"{shim} `train` shim 不再 reach 到 --hf-jobs 拦截 "
        "(mjlab 的 plugin loading 变了, 或它现在跑在 argv 解析之后).\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-2000:]}"
    )
    assert proc.returncode == 0, f"probe 失败:\n{proc.stderr[-2000:]}"
    assert f"SUBMIT ['{_TASK}', '--env.scene.num-envs', '4096']" in proc.stdout
    assert "EXIT 7" in proc.stdout
