"""在 linux-aarch64 (DGX Spark / GB10) 上 PyPI 的 torch wheel 是 CPU-only 的:
torch.version.cuda is None -> torch.cuda.device_count() == 0 -> mjlab 的
select_gpus() 索引一个空列表并在第一次训练步之前就崩溃抛出
`IndexError: list index out of range`
(mjlab/utils/gpu.py:70).

修复方法 (在 pyproject.toml 中) 把 torch 路由到 PyTorch 的 CUDA index,
仅在 aarch64 上生效.这有两个静默崩点, 由这些测试锁住 —— 两种情况下
`uv sync` 都成功, 你只会在启动训练时才发现:

1. `torch` 必须保持为 DIRECT dependency: uv 只对 direct dependencies
   应用 [tool.uv.sources], 所以删除 `torch==...` 那行 (看起来多余, 因为
   torch 已经通过 mjlab/rsl_rl 间接进来) 会让 source 绑定变成 no-op 而
   不发出任何警告.
2. x86_64 的解析必须留在 PyPI 上, 否则 HF Jobs 会静默切换 wheel.
"""

import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CUDA_INDEX = "https://download.pytorch.org/whl/cu"


def _packages(name):
    lock = tomllib.loads((_ROOT / "uv.lock").read_text())
    return [p for p in lock["package"] if p["name"] == name]


def _registry(pkg):
    return pkg.get("source", {}).get("registry", "")


def _markers(pkg):
    return " ".join(pkg.get("resolution-markers", []))


def _aarch64_entry(pkgs):
    """其 resolution-markers 选中 linux-aarch64 的条目."""
    hits = [
        p for p in pkgs if "platform_machine == 'aarch64'" in _markers(p) and "sys_platform == 'linux'" in _markers(p)
    ]
    assert len(hits) == 1, f"期望 1 个 aarch64 条目, 找到 {len(hits)}"
    return hits[0]


def test_torch_is_a_direct_dependency():
    """没有这一行, [tool.uv.sources] 对 torch 就是静默 no-op."""
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    assert any(d.split("=")[0].split("[")[0].strip() == "torch" for d in deps), (
        "torch 必须保留在 [project.dependencies] 中: uv 只对 DIRECT "
        "dependencies 应用 [tool.uv.sources].删掉它会把 aarch64 静默"
        "退回到 PyPI 的 CPU-only wheel."
    )


def test_torch_source_is_pinned_to_a_cuda_index_on_aarch64():
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    uv_cfg = pyproject["tool"]["uv"]
    assert "torch" in uv_cfg.get("sources", {}), (
        "[tool.uv.sources] 不再有 torch 条目 -> aarch64 会回退到 PyPI 的"
        " CPU wheel, 然后 `train` 会在 select_gpus() 中死于 IndexError."
    )
    sources = uv_cfg["sources"]["torch"]
    indexes = {p["name"]: p["url"] for p in uv_cfg.get("index", [])}
    for src in sources:
        assert "aarch64" in src["marker"], "torch source 必须保持 aarch64 范围"
        assert indexes[src["index"]].startswith(_CUDA_INDEX), f"index {src['index']} 不是 PyTorch CUDA index"


def test_lockfile_routes_aarch64_torch_to_cuda_wheels():
    torch_pkgs = _packages("torch")
    aarch64 = _aarch64_entry(torch_pkgs)
    assert _registry(aarch64).startswith(_CUDA_INDEX), (
        f"aarch64 上的 torch 来自 {_registry(aarch64)!r} —— 是 CPU wheel."
        "请检查 [tool.uv.sources] 后重新运行 `uv lock`."
    )
    wheels = " ".join(w["url"] for w in aarch64["wheels"])
    assert "aarch64" in wheels, "aarch64 torch 条目里没有 aarch64 wheel"
    assert "%2Bcu" in wheels or "+cu" in wheels, "aarch64 wheel 没有 +cuXXX 本地版本号 -> CPU 构建"


def test_x86_64_resolution_stays_on_pypi():
    """HF Jobs 跑在 x86_64 上: 它们的解析不能动."""
    others = [p for p in _packages("torch") if "platform_machine == 'aarch64'" not in _markers(p)]
    assert others, "没有找到非 aarch64 的 torch 条目"
    for pkg in others:
        assert _registry(pkg) == "https://pypi.org/simple", (
            f"x86_64 torch 移到了 {_registry(pkg)!r} —— HF Jobs 会切换 wheel."
        )
        assert "+cu" not in pkg["version"], "x86_64 torch 不能被 CUDA-pin"


def test_torch_version_identical_across_platforms():
    """修复只改 wheel 的 SOURCE, 不改版本: CUDA index 带的构建比 PyPI pin
    更新, 所以一个 `>=` 会把 torch 2.9.1 -> 2.13.0 拖上来, 而这个升级无人
    验证过."""
    versions = {p["version"].split("+")[0] for p in _packages("torch")}
    assert len(versions) == 1, f"torch 版本跨平台不一致: {versions}"


def _on_spark():
    return (
        sys.platform == "linux"
        and platform.machine() == "aarch64"
        and shutil.which("nvidia-smi") is not None
        and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    )


@pytest.mark.skipif(not _on_spark(), reason="不是带 GPU 的 linux-aarch64 机器")
def test_installed_torch_actually_sees_the_gpu():
    """直接复现崩溃: 这就是 select_gpus() 读到的东西."""
    import torch

    assert torch.cuda.device_count() > 0, (
        f"torch {torch.__version__} (cuda={torch.version.cuda}) 看不到 GPU, "
        "尽管 nvidia-smi 报告了一块 -> select_gpus() 会抛 IndexError."
    )
