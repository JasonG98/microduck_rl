"""小型 temporal-CNN pick-up 分类器 + pause/resume 状态机.

输入: 63 维特征向量 (features.py) 最近 ``WINDOW`` 个 tick, 最旧的在前, 形状 (B, WINDOW, 63).
输出: p(held) 的 logit. 特征 normalizer 是模块的 buffer, 因此导出的 ONNX 直接接收原始特征.
约 25k 参数, 每次调用约 0.3 MMAC — 在 Radxa 的 A55 核上可忽略.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mjlab_microduck.pickup.features import FEAT_DIM

WINDOW = 50  # 50 Hz 下为 1 s


class PickupNet(nn.Module):
    def __init__(self, feat_dim: int = FEAT_DIM, window: int = WINDOW, ch: int = 32, hidden: int = 32,
                 keep: list[int] | None = None):
        super().__init__()
        # 网络实际读取的通道, 在计算图内部选择, 因此输入契约保持 63 维.
        # 在这里丢弃某个特征 (而不是把它 mask 成 0) 也会省掉它的计算:
        # 第一层卷积占大部分开销, 且随其输入宽度增长.
        keep = list(range(feat_dim)) if keep is None else list(keep)
        self.register_buffer("keep", torch.tensor(keep, dtype=torch.long), persistent=False)  # 来自 net_kwargs
        self.keep_list = keep
        self.register_buffer("mean", torch.zeros(feat_dim))
        self.register_buffer("std", torch.ones(feat_dim))
        # 0 = 忽略该特征 (例如舵机电流, 其 sim2real 存疑);
        # 无论哪种情况输入契约都保持 63 维
        self.register_buffer("mask", torch.ones(feat_dim))
        self.conv = nn.Sequential(
            nn.Conv1d(len(keep), ch, 5, stride=2), nn.ELU(),
            nn.Conv1d(ch, ch, 5, stride=2), nn.ELU(),
            nn.Conv1d(ch, ch, 5, stride=2), nn.ELU(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, len(keep), window)).numel()
        self.head = nn.Sequential(nn.Linear(n, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, W, F) → (B,)
        x = ((x - self.mean) / self.std * self.mask)[..., self.keep]
        x = self.conv(x.transpose(1, 2))
        return self.head(x.flatten(1)).squeeze(-1)


def export_onnx(net: PickupNet, path: str) -> None:
    """导出 features[1, W, 63] → p_held[1] (已应用 sigmoid)."""

    class _Prob(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def forward(self, x):
            return torch.sigmoid(self.n(x))

    net = net.cpu().eval()
    dummy = torch.zeros(1, net.window, FEAT_DIM)
    torch.onnx.export(_Prob(net), dummy, path, input_names=["features"], output_names=["p_held"],
                      opset_version=17, dynamo=False)


@dataclass
class StateMachineCfg:
    p_pause: float = 0.8      # p > p_pause 持续 n_pause 个 tick 时 POLICY → PAUSED
    n_pause: int = 5          # 100 ms
    # p < p_resume 持续 n_resume 个 tick 时 PAUSED → POLICY. 恢复速度决定了
    # 被放下的机器人能否站住 (暂停姿态在 ~0.2-0.5 s 内就会倾倒):
    # 闭环 sim, 0.2×10 ticks → 恢复 p50 0.30 s, 11.7 % 倾倒 >40°;
    # 0.35×4 ticks → 0.18 s, 2.9 % 倾倒 (搬运中误恢复 0 → 1.5 / h).
    p_resume: float = 0.35
    n_resume: int = 4         # 80 ms
    min_pause_s: float = 0.3  # 不允许瞬间来回翻转 (被放下的机器人 ~1 s 内就会倾倒: 保持短)


class PauseStateMachine:
    """对 p(held) 的批量迟滞. ``paused`` 即 runtime 的暂停标志."""

    def __init__(self, num_envs: int, device, cfg: StateMachineCfg | None = None, dt: float = 0.02):
        self.cfg = cfg or StateMachineCfg()
        self.dt = dt
        self.paused = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.cnt = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.t_paused = torch.zeros(num_envs, device=device)

    def reset(self, env_ids):
        self.paused[env_ids] = False
        self.cnt[env_ids] = 0
        self.t_paused[env_ids] = 0.0

    def update(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (just_paused, just_resumed) 掩码."""
        c = self.cfg
        self.t_paused = torch.where(self.paused, self.t_paused + self.dt, torch.zeros_like(self.t_paused))
        hit = torch.where(self.paused, (p < c.p_resume) & (self.t_paused >= c.min_pause_s), p > c.p_pause)
        self.cnt = torch.where(hit, self.cnt + 1, torch.zeros_like(self.cnt))
        need = torch.where(self.paused, torch.full_like(self.cnt, c.n_resume), torch.full_like(self.cnt, c.n_pause))
        flip = self.cnt >= need
        just_paused = flip & ~self.paused
        just_resumed = flip & self.paused
        self.paused = self.paused ^ flip
        self.cnt[flip] = 0
        self.t_paused[flip] = 0.0
        return just_paused, just_resumed
