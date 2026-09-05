"""将一次 mjlab-microduck 训练任务提交为 Hugging Face Job.

由 --hf-jobs 拦截逻辑调用 (见 train_hook.py):

    uv run train Mjlab-Kick-Flat-MicroDuck \
        --env.scene.num-envs 4096 --agent.max_iterations 4000 --hf-jobs

任何非 --hf-* / 提交相关的参数都会原样转发给任务内的
`uv run train`.

认证: 使用 `hf auth login` 缓存的 HF token 或 HF_TOKEN 环境变量.
提交时会列出账号下的组织, 由你选择运行所在的命名空间
(个人或组织) — 仓库, uv-cache 桶以及任务本身都位于该命名空间下.
传入 --namespace 可跳过提示 (自动化场景).

所有操作都通过 huggingface_hub Python API 完成 (Jobs API, hub >= 1.x)
— 不需要独立的 `hf` CLI.

源码: 已跟踪文件 (已提交或未提交) 的快照会被上传到一个
私有 HF dataset 仓库, 并在任务内以只读方式挂载. 检查点由
训练过程中并行的 watcher (scripts/hf/uploader.py) 推送到
私有 HF model 仓库.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from netrc import netrc
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token

DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DEFAULT_FLAVOR = "l4x1"
DEFAULT_TIMEOUT = "12h"

# 在容器内运行的引导脚本. `$VAR` 由容器 shell 从任务的环境变量展开.
BOOTSTRAP = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -qq -y --no-install-recommends git curl ca-certificates xz-utils >/dev/null
# 锁定 uv 版本: 缓存桶跨任务持久化, 而浮动的 "latest" uv
# 读取旧版 uv 写入的条目会损坏安装 (2026-07-21 出现过:
# 一个 0.9.x 时代任务留下的 bam 内建 wheel 缓存条目导致 0.11.30 失败,
# 报错 "The wheel is invalid: Missing .dist-info directory").
curl -LsSf https://astral.sh/uv/0.11.30/install.sh | sh >/dev/null
export PATH="/root/.local/bin:$PATH"
# HF Jobs 将 uv cache 和 /work 放在不同的文件系统上 -> uv 无法
# hardlink, 其回退策略会损坏构建的 wheel (bam -> "Missing
# .dist-info directory"). copy = 可靠安装 (uv 官方推荐的做法).
export UV_LINK_MODE=copy

mkdir -p /work && cd /work
echo "[bootstrap] 解压源码 $SRC_TARBALL"
tar -xzf "/src/$SRC_TARBALL"

echo "[bootstrap] uv sync"
# 自愈被污染的持久化缓存: 一个坏条目会让 sync 确定性地失败,
# 因此先清空缓存重建一次, 再放弃.
uv sync --no-progress || {
    echo "[bootstrap] uv sync 失败 — 清理 uv 缓存后重试"
    uv cache clean || true
    uv sync --no-progress
}

echo "[bootstrap] 启动检查点上传器"
mkdir -p logs/rsl_rl
nohup uv run python scripts/hf/uploader.py > /tmp/uploader.log 2>&1 &
UPLOADER_PID=$!

echo "[bootstrap] 开始训练: uv run train $TRAIN_ARGS"
set +e
uv run train $TRAIN_ARGS
TRAIN_RC=$?
set -e

echo "[bootstrap] 训练退出码 $TRAIN_RC, 执行最终上传"
# 终止 watcher 循环, 然后同步执行一次最终上传
kill $UPLOADER_PID 2>/dev/null || true
CKPT_ONE_SHOT=1 uv run python scripts/hf/uploader.py || true

# 在环境还热着时自动将最终检查点导出为可部署的 ONNX —
# 单独跑一个导出任务需要再付一次完整的引导开销 (镜像
# 拉取 + apt + uv sync) 仅仅为了执行这一条命令. 尽力而为:
# 导出失败不应将一次成功的训练标记为失败.
if [ "$TRAIN_RC" -eq 0 ] && [ "${AUTO_EXPORT:-1}" = "1" ]; then
    set +e
    TASK_ID=${TRAIN_ARGS%% *}
    CKPT=$(ls -t logs/rsl_rl/*/model_*.pt 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        echo "[bootstrap] 从 $(basename "$CKPT") 自动导出 ONNX"
        uv run python scripts/export.py "$TASK_ID" \
            --checkpoint-file "$(basename "$CKPT")" \
            --num-envs 1 --onnx-file /work/policy.onnx \
        && uv run python - <<'PY'
import os
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj="/work/policy.onnx",
                    path_in_repo="exported/policy.onnx",
                    repo_id=os.environ["CKPT_REPO"], repo_type="model")
print("[bootstrap] 已上传 exported/policy.onnx")
PY
        [ $? -ne 0 ] && echo "[bootstrap] 自动导出失败 (训练仍然成功)"
    else
        echo "[bootstrap] 未找到检查点, 跳过自动导出"
    fi
    set -e
fi

exit $TRAIN_RC
"""


def _wandb_api_key() -> str | None:
    """尽力而为地查找用户的 wandb API key.

    顺序: WANDB_API_KEY 环境变量 -> ~/.netrc (machine api.wandb.ai).
    """
    if k := os.environ.get("WANDB_API_KEY"):
        return k
    try:
        n = netrc(str(Path.home() / ".netrc"))
        auth = n.authenticators("api.wandb.ai")
        if auth and auth[2]:
            return auth[2]
    except (FileNotFoundError, OSError):
        pass
    return None


def _repo_root() -> Path:
    """当前目录的仓库根目录 — 感知 worktree.

    (旧的 scripts/hf/train_hf.py 用脚本文件所在位置; 改为从 cwd 解析意味着
    从 worktree 运行时会快照该 worktree.)
    """
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    return Path(out.decode().strip())


def _build_tarball(repo_root: Path, out_path: Path) -> str:
    """创建 HEAD 加上未提交已跟踪改动的 tarball.

    返回短 SHA.
    """
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root).decode().strip()

    # 使用 `git ls-files` 以包含已跟踪但被修改的文件 (工作树
    # 状态), 同时跳过被忽略的垃圾 (.venv, logs, *.onnx, wandb/ 等).
    files = (
        subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"], cwd=repo_root).decode().splitlines()
    )

    with tarfile.open(out_path, "w:gz") as tar:
        for rel in files:
            p = repo_root / rel
            if p.exists() and p.is_file():
                tar.add(p, arcname=rel)
    return sha


def _pick_namespace(api: HfApi, preset: str | None) -> str:
    """选择任务运行所在的命名空间 (个人账号或组织).

    除非传入 --namespace 或没有可选项, 否则进行交互式提示. 非终端 (脚本, CI) 回退到
    个人账号.
    """
    info = api.whoami()
    user = info.get("name") or info.get("email")
    if not user:
        raise RuntimeError("无法确定 HF 用户名. 请先运行 `hf auth login`.")
    orgs = [o["name"] for o in info.get("orgs", []) if o.get("name")]

    if preset is not None:
        if preset not in (user, *orgs):
            raise RuntimeError(
                f"--namespace {preset!r} 既不是你的账号 ({user}) 也不是你的组织之一 ({', '.join(orgs) or '无'})."
            )
        return preset

    if not orgs:
        return user

    if not sys.stdin.isatty():
        print(f"[hf] 非交互模式, 默认使用个人命名空间: {user}")
        return user

    choices = [user, *orgs]
    print("[hf] 在哪个命名空间下运行?")
    print(f"  1) {user} (个人)")
    for i, org in enumerate(orgs, start=2):
        print(f"  {i}) {org}")
    while True:
        raw = input(f"选择 [1-{len(choices)}, 默认 1]: ").strip()
        if raw == "":
            return user
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        if raw in choices:
            return raw
        print(f"  无效选择: {raw!r}")


def _await_scheduling(api: HfApi, job_id: str, namespace: str, budget_s: float = 1200.0) -> tuple[str, str | None]:
    """轮询直到任务离开 SCHEDULING 阶段 (镜像拉取 / 排队 / 挂载).

    这一阶段合理地需要数分钟 (仅 pytorch 镜像拉取在冷 GPU 节点上就要 ~5 分钟),
    也是卷挂载失败浮现的地方, 大约 7 分钟时出现 ("init container exhausted retries").
    返回第一个非 SCHEDULING 阶段的 (stage, message), 或预算耗尽时返回最后
    观察到的状态 (长时间排队不是错误 — 调用方的流式循环会继续监督).
    """
    deadline = time.monotonic() + budget_s
    last_note = time.monotonic()
    stage, message = "", None
    while time.monotonic() < deadline:
        status = api.inspect_job(job_id=job_id, namespace=namespace).status
        stage, message = status.stage, status.message
        if stage and stage != "SCHEDULING":
            return stage, message
        if time.monotonic() - last_note > 60:
            print("[job] 仍在调度中 (排队 / 镜像拉取 / 卷挂载)...")
            last_note = time.monotonic()
        time.sleep(10)
    return stage, message


def submit(argv: list[str]) -> int:
    """从 ``argv`` 解析提交参数并启动 HF job."""
    ap = argparse.ArgumentParser(
        prog="train --hf-jobs",
        description="将一次 microduck 训练任务提交到 HF Jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("task", help="mjlab 任务 id, 例如 Mjlab-Kick-Flat-MicroDuck")
    ap.add_argument("--flavor", default=DEFAULT_FLAVOR, help="HF Jobs 硬件规格")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="要运行的 Docker 镜像")
    ap.add_argument("--timeout", default=DEFAULT_TIMEOUT, help="任务最大运行时长")
    ap.add_argument(
        "--namespace",
        default=None,
        help="运行所在的 HF 命名空间 (你的用户名或某个组织); 跳过提示.",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="本次运行的简短标签; 默认为 task+时间戳",
    )
    ap.add_argument(
        "--detach",
        action="store_true",
        help="提交后立即返回 (不流式传输日志).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="构建 tarball 并打印任务规格但不提交.",
    )
    ap.add_argument(
        "--src-repo",
        default=None,
        help="用于源码 tarball 的 HF dataset 仓库. 默认为 <namespace>/mjlab-microduck-src",
    )
    ap.add_argument(
        "--ckpt-repo",
        default=None,
        help="用于检查点的 HF model 仓库. 默认为 <namespace>/<run-name>",
    )
    ap.add_argument(
        "--uv-cache-bucket",
        default=None,
        help="用作 UV_CACHE_DIR 以跨运行持久化 wheel 的 HF 桶. "
        "默认为 <namespace>/mjlab-uv-cache. 需配合 --uv-cache 使用.",
    )
    ap.add_argument(
        "--uv-cache",
        action="store_true",
        help="挂载持久化 uv 缓存桶 (默认关闭: FUSE "
        "桶挂载不支持 hardlink, 所以 `uv sync` 会回退到 "
        "通过网络挂载全量拷贝 ~6 GB 解压后的包 — 比直接从 "
        "PyPI 重新下载 wheel 慢得多, 后者 HF 数据中心带宽 "
        "约 1 分钟即可完成; 而且会跨 uv 版本中毒: "
        "0.9.x 时代的条目导致 0.11.30 报 'wheel is invalid: "
        "Missing .dist-info' 而死, 2026-07-21).",
    )
    ap.add_argument(
        "--no-wandb",
        action="store_true",
        help="不转发 wandb API key (如果启用了 wandb 训练会失败).",
    )
    args, train_args = ap.parse_known_args(argv)

    api = HfApi()
    try:
        namespace = _pick_namespace(api, args.namespace)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    print(f"[hf] 命名空间: {namespace}")

    token = get_token()
    if not token:
        print("错误: 没有缓存的 HF token. 请运行 `hf auth login`.", file=sys.stderr)
        return 1

    repo_root = _repo_root()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.task}-{stamp}".lower()
    src_repo = args.src_repo or f"{namespace}/mjlab-microduck-src"
    ckpt_repo = args.ckpt_repo or f"{namespace}/{run_name}"

    env: dict[str, str] = {
        # 在任务内部解除 --hf-jobs 拦截, 使任务自身的
        # `uv run train` 只能本地训练 (见 train_hook.py).
        "MICRODUCK_IN_HF_JOB": "1",
        "CKPT_REPO": ckpt_repo,
        "TRAIN_ARGS": " ".join(shlex.quote(a) for a in [args.task, *train_args]),
    }
    secrets: dict[str, str] = {"HF_TOKEN": token}

    # 转发 wandb 凭证 (环境变量, 然后 ~/.netrc)
    if not args.no_wandb:
        wb_key = _wandb_api_key()
        if not wb_key:
            print(
                "[wandb] ✗ 未找到 API key (已检查 $WANDB_API_KEY 和 ~/.netrc).\n"
                "        请本地运行 `wandb login`, 或传入 --no-wandb 跳过.",
                file=sys.stderr,
            )
            return 1
        secrets["WANDB_API_KEY"] = wb_key
        src = "env" if os.environ.get("WANDB_API_KEY") else "~/.netrc"
        print(f"[wandb] 从 {src} 转发 API key")
        for k in ("WANDB_PROJECT", "WANDB_ENTITY"):
            if os.environ.get(k):
                env[k] = os.environ[k]

    volumes = [Volume(type="dataset", source=src_repo, mount_path="/src", read_only=True)]

    # 持久化 uv 缓存 — 通过 --uv-cache 显式开启 (见该 flag 的帮助文本:
    # 从 FUSE 桶跨文件系统安装比从 PyPI 重新下载更慢, 且
    # 2026-07-21 一个过期条目曾确定性地杀死 uv sync).
    cache_bucket: str | None = None
    if args.uv_cache:
        cache_bucket = args.uv_cache_bucket or f"{namespace}/mjlab-uv-cache"
        volumes.append(Volume(type="bucket", source=cache_bucket, mount_path="/uv-cache"))
        env["UV_CACHE_DIR"] = "/uv-cache"
        print(f"[uv-cache] 使用桶 {cache_bucket}")

    # 1. 构建 tarball
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / f"src-{stamp}.tar.gz"
        print(f"[src] 构建 tarball -> {tar_path.name} (来自 {repo_root})")
        sha = _build_tarball(repo_root, tar_path)
        size_mb = tar_path.stat().st_size / 1e6
        print(f"[src] HEAD={sha}, {size_mb:.1f} MB")
        env["SRC_TARBALL"] = tar_path.name
        env["GIT_SHA"] = sha

        if args.dry_run:
            print("[dry-run] 将提交任务:")
            print(f"  命名空间: {namespace}")
            print(f"  镜像:     {args.image}")
            print(f"  规格:     {args.flavor}, 超时: {args.timeout}")
            print(f"  卷:       {[f'{v.type}:{v.source} -> {v.mount_path}' for v in volumes]}")
            print(f"  环境变量: {dict(env.items())}")
            print(f"  密钥:     {dict.fromkeys(secrets, '***')}")
            print(f"  ckpt 仓库: https://huggingface.co/{ckpt_repo}")
            return 0

        # 2. 上传 tarball + 预创建仓库/桶
        api.create_repo(src_repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"[src] 上传到 dataset {src_repo}")
        api.upload_file(
            path_or_fileobj=str(tar_path),
            path_in_repo=tar_path.name,
            repo_id=src_repo,
            repo_type="dataset",
        )
        api.create_repo(ckpt_repo, repo_type="model", private=True, exist_ok=True)
        if cache_bucket is not None:
            api.create_bucket(cache_bucket, private=True, exist_ok=True)

        # 3. 提交. GPU 节点的卷挂载会瞬态失败 ("init container
        # exhausted retries", 在 SCHEDULING 阶段约 7 分钟时浮现 —
        # 2026-07 观察到, 同时一个完全相同的探测任务挂载正常).
        # 监督调度阶段并在挂载失败时重新提交.
        print(f"[ckpt] 检查点 -> https://huggingface.co/{ckpt_repo}")
        print(f"[job] 提交中 (命名空间={namespace}, 规格={args.flavor}, 超时={args.timeout})")
        job = None
        stage, message = "", None
        for attempt in range(3):
            if attempt:
                print(f"[job] ✗ 卷挂载失败 (节点不稳定) — 重新提交 ({attempt + 1}/3)")
                time.sleep(10)
            try:
                job = api.run_job(
                    image=args.image,
                    command=["bash", "-c", BOOTSTRAP],
                    env=env,
                    secrets=secrets,
                    flavor=args.flavor,
                    timeout=args.timeout,
                    volumes=volumes,
                    namespace=namespace,
                )
            except Exception as e:
                msg = str(e)
                if "402" in msg or "Payment Required" in msg or "credit" in msg.lower():
                    print(
                        "\n[job] ✗ Hugging Face Jobs 计费错误.\n"
                        f"    命名空间 {namespace!r} 的 Jobs 额度不足.\n"
                        "    → 充值额度:   https://huggingface.co/settings/billing\n"
                        "    → 或开通 HF Pro: https://huggingface.co/settings/billing/subscription",
                        file=sys.stderr,
                    )
                    return 2
                if "403" in msg or "Forbidden" in msg or "required permissions" in msg.lower():
                    print(
                        "\n[job] ✗ Hugging Face Jobs 权限错误 (403).\n"
                        "    你的 HF token 认证正常但无权使用 Jobs API\n"
                        f"    操作命名空间 {namespace!r}. 这是 token 范围问题, 不是计费问题.\n"
                        "    → 创建/编辑一个细粒度 token 并启用 Jobs 权限:\n"
                        "        https://huggingface.co/settings/tokens\n"
                        "      (细粒度 → 在你的用户和/或组织下勾选 'Jobs' 权限),\n"
                        "      然后本地重新登录:  hf auth login\n"
                        '    → 验证:  python -c "from huggingface_hub import HfApi; "\n'
                        "               \"print(list(HfApi().list_jobs(namespace='<ns>')))\"  (不得 403)\n"
                        "    (如果 Jobs 已启用但仍被阻止, 该命名空间可能还需要\n"
                        "     包含 Jobs 的 HF 套餐/额度 — 见上方计费链接.)",
                        file=sys.stderr,
                    )
                    return 3
                raise
            print(f"[job] id:  {job.id}")
            if getattr(job, "url", None):
                print(f"[job] url: {job.url}")
            if args.detach:
                print("[job] --detach: 不监督启动 — 查看上方 URL; 瞬态 'Volume mount failed' 错误需要手动重新提交.")
                return 0
            stage, message = _await_scheduling(api, job.id, namespace)
            if stage == "ERROR" and "mount" in (message or "").lower():
                continue  # 节点不稳定 — 重新提交
            break

    assert job is not None
    if stage == "ERROR":
        print(f"[job] ✗ 启动失败: {message}", file=sys.stderr)
        return 1

    # 监督至完成: 流式传输日志, 在流断开时重新挂载
    # (容器仍在启动时返回空), 并报告终态.
    print("[job] 流式传输日志 (Ctrl-C 分离; 任务继续运行)")
    try:
        while True:
            try:
                for line in api.fetch_job_logs(job_id=job.id, namespace=namespace, follow=True):
                    print(line)
            except Exception as e:
                print(f"[job] 日志流断开 ({e}); 重新挂载")
            status = api.inspect_job(job_id=job.id, namespace=namespace).status
            if status.stage == "COMPLETED":
                print("[job] ✓ 已完成")
                return 0
            if status.stage in ("ERROR", "DELETED", "CANCELED"):
                print(f"[job] ✗ {status.stage}: {status.message}", file=sys.stderr)
                return 1
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"\n[job] 已分离. 任务 {job.id} 仍在运行: {getattr(job, 'url', job.id)}")
        return 0
