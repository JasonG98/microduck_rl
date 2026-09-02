"""在 HF Job 内运行的 checkpoint 上传器.

监视 `logs/rsl_rl/**/model_*.pt` 并将新增/更新的文件上传到目标 HF Model repo. 设计为从 job bootstrap
以 `nohup uv run` 方式启动, 认证来自 `hf jobs run` 注入的 HF_TOKEN secret.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def main() -> int:
    """监视 checkpoint 目录并上传新文件到 Hugging Face repo."""
    repo_id = os.environ.get("CKPT_REPO")
    if not repo_id:
        print("[uploader] CKPT_REPO 未设置, 退出", flush=True)
        return 1

    poll_interval = float(os.environ.get("CKPT_POLL_INTERVAL", "60"))
    root = Path(os.environ.get("CKPT_ROOT", "logs/rsl_rl"))

    one_shot = os.environ.get("CKPT_ONE_SHOT") == "1"

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    mode = "one-shot" if one_shot else f"every {poll_interval}s"
    print(f"[uploader] 正在监视 {root} -> {repo_id} ({mode})", flush=True)

    sent: dict[Path, float] = {}
    while True:
        try:
            files = list(root.glob("**/model_*.pt"))
            # 也顺便捡起 dump 出来的配置
            files += list(root.glob("**/params/*.yaml"))
            files += list(root.glob("**/params/*.json"))

            to_upload: list[CommitOperationAdd] = []
            for f in files:
                try:
                    mtime = f.stat().st_mtime
                except FileNotFoundError:
                    continue
                if sent.get(f) == mtime:
                    continue
                # path-in-repo 相对于 logs/rsl_rl, 使 repo 镜像 run 目录
                rel = f.relative_to(root)
                to_upload.append(CommitOperationAdd(path_in_repo=str(rel), path_or_fileobj=str(f)))
                sent[f] = mtime

            if to_upload:
                msg = f"upload {len(to_upload)} file(s)"
                api.create_commit(
                    repo_id=repo_id,
                    repo_type="model",
                    operations=to_upload,
                    commit_message=msg,
                )
                print(f"[uploader] 已推送 {len(to_upload)} 个文件", flush=True)
        except Exception as e:
            print(f"[uploader] 错误: {e}", flush=True)

        if one_shot:
            return 0
        time.sleep(poll_interval)


if __name__ == "__main__":
    sys.exit(main())
