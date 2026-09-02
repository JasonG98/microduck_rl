# HF Jobs 训练

在 Hugging Face 的托管 GPU 上训练 mjlab-microduck. 认证使用缓存的 HF
token (`hf auth login` 或 `HF_TOKEN`); 一切都通过
`huggingface_hub` Python API 完成 — 不需要独立的 `hf` CLI.

## 一次性设置

```fish
hf auth login    # 或 export HF_TOKEN (任何能缓存 token 的方式都行)
wandb login      # 从 ~/.netrc 自动检测并转发
```

## 提交一个 run

你正常的 train 命令, 加上 `--hf-jobs`:

```fish
uv run train Mjlab-Kick-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 4000 --hf-jobs
```

会询问你在哪个 namespace 下运行 — 你的个人账户或你的某个 org. Repo, uv-cache bucket, 计费和 job 本身都在所选 namespace 中. 传 `--namespace <name>` 可跳过提示
(非交互运行默认用个人账户).

不加 `--hf-jobs` 时命令行为与之前完全一致 (本地训练).
提交 flag 在本地消费; 其余参数都转发给 job 内部的
`uv run train`.

常用 flag:
- `--namespace <name>` — 运行所在的账户/org; 跳过提示
- `--flavor l4x1` (默认) / `a10g-large` / `a100-large`
- `--timeout 12h` (默认) — 超时后 job 被杀掉
- `--detach` — 提交后立即返回 (默认流式输出日志; Ctrl-C 可脱离而不杀 job)
- `--dry-run` — 构建 tarball, 打印 job spec, 不提交
- `--run-name <tag>` — 覆盖自动生成的 `<task>-<timestamp>` 名称
- `--no-uv-cache` — 禁用持久 `uv` cache bucket (每次都冷启动)
- `--no-wandb` — 不转发 wandb key

(`uv run scripts/hf/train_hf.py <task> ...` 仍然可用 — 它是指向
同一份代码的 shim, 实际逻辑在 `src/mjlab_microduck/hf_jobs.py` 中.)

## 底层发生了什么

1. `git ls-files` 对你运行所在的 repo 做快照 (含已跟踪 + 未提交文件, 支持 worktree) → `src-<stamp>.tar.gz`.
2. Tarball 上传到私有 dataset `<namespace>/mjlab-microduck-src`.
3. 创建私有 model repo `<namespace>/<run-name>` 用于存放 checkpoint.
4. 私有 HF bucket `<namespace>/mjlab-uv-cache` 挂载到 `/uv-cache`
   并作为 `UV_CACHE_DIR` 使用, 使 wheel 下载跨 run 持久化 (首次
   冷, 后续快).
5. `HfApi.run_job` 启动一个容器:
   - 安装 `uv`, 解压 tarball, 运行 `uv sync` (热缓存),
   - 在后台启动 `scripts/hf/uploader.py` (监视 `logs/rsl_rl/**/model_*.pt`, 每 60s 推送一次),
   - 运行 `uv run train <task> <args>`,
   - 退出时做一次最终上传.
6. wandb 凭证作为 secret 转发 — run 会实时出现在你的
   wandb 项目中.

## 浏览 checkpoint

提交器在启动时打印 `https://huggingface.co/<namespace>/<run-name>`;
训练过程中新的 `.pt` 文件会出现在那里.

## 管理 job

job id 和 URL 在提交时打印. 从 Python:

```python
from huggingface_hub import HfApi
api = HfApi()
api.list_jobs()                                  # 或 namespace="pollen-robotics"
for l in api.fetch_job_logs(job_id="...", follow=True): print(l)
api.cancel_job(job_id="...")
```

(或使用 `hf jobs ps/logs/cancel` CLI, 如果你装了的话.)
