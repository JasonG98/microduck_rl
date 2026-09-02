"""向后兼容的 shim: 提交逻辑已迁移到 mjlab_microduck.hf_jobs.

推荐使用集成的 flag:
    uv run train <task> <train args...> --hf-jobs [--namespace <ns>] [...]

这个脚本保持旧的调用方式可用:
    uv run scripts/hf/train_hf.py <task> [submission flags] <train args...>
"""

import sys

from mjlab_microduck.hf_jobs import submit

if __name__ == "__main__":
    sys.exit(submit(sys.argv[1:]))
