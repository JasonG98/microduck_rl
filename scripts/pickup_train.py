#!/usr/bin/env python3
"""在 pickup_datagen.py 分片上训练拾取分类器.

  uv run scripts/pickup_train.py --train data/pickup/train_*.npz --val data/pickup/val.npz --out logs/pickup/full
  uv run scripts/pickup_train.py ... --no-current --out logs/pickup/nocur   # 忽略 servo current

写出 <out>/model.pt (state dict + cfg, 可被 pickup_datagen --detector 与
pickup_demo 加载) 以及 <out>/pickup_detector.onnx (features[1,50,63] → p_held[1]).
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from mjlab_microduck.pickup.features import CURRENT_SLICE, FEAT_DIM, PAUSED_IDX
from mjlab_microduck.pickup.model import WINDOW, PickupNet, StateMachineCfg, export_onnx


def load(paths, dev):
    feats, meta = [], {k: [] for k in ("held", "since_reset", "paused", "fallen", "cmd")}
    for p in paths:
        d = np.load(p)
        feats.append(torch.as_tensor(d["feat"]).to(dev))
        for k in meta:
            meta[k].append(torch.as_tensor(d[k]).to(dev))
        print(f"已加载 {p}: {d['feat'].shape}")
    return feats, meta


def valid_index(meta, i):
    sr = meta["since_reset"][i]
    t, n = torch.nonzero(sr >= WINDOW - 1, as_tuple=True)
    return t, n


def gather(feat, t, n):
    tt = t.unsqueeze(1) + torch.arange(-WINDOW + 1, 1, device=t.device)
    return feat[tt, n.unsqueeze(1)].float()  # (B, W, F)


@torch.no_grad()
def evaluate(net, feats, meta, dev, max_per_shard=400_000, thr=0.5):
    net.eval()
    rows = {}
    for i, feat in enumerate(feats):
        t, n = valid_index(meta, i)
        if len(t) > max_per_shard:
            sel = torch.randperm(len(t), device=dev)[:max_per_shard]
            t, n = t[sel], n[sel]
        p = torch.cat([torch.sigmoid(net(gather(feat, t[j:j + 8192], n[j:j + 8192]))) for j in range(0, len(t), 8192)])
        held = meta["held"][i][t, n]; paused = meta["paused"][i][t, n]
        fallen = meta["fallen"][i][t, n]; cmd = meta["cmd"][i][t, n].float()
        cats = {
            "walking (policy, floor, cmd>0.05)": ~held & ~paused & (cmd > 0.05) & ~fallen,
            "standing (policy, floor, cmd≈0)": ~held & ~paused & (cmd <= 0.05) & ~fallen,
            "fallen on floor": ~held & fallen,
            "floor, paused": ~held & paused & ~fallen,
            "held, policy running": held & ~paused,
            "held, paused": held & paused,
        }
        for k, m in cats.items():
            pos = k.startswith("held")
            ok = (p[m] > thr) if pos else (p[m] <= thr)
            rows.setdefault(k, []).append((ok.float().sum().item(), m.sum().item()))
        # 类似 ROC: 统计全部样本的 p
        rows.setdefault("_bce", []).append((Fn.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), held.float(), reduction="sum").item(), len(p)))
    net.train()
    out = {k: sum(a for a, _ in v) / max(1, sum(b for _, b in v)) for k, v in rows.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-current", action="store_true")
    ap.add_argument("--drop-current", action="store_true", help="从计算图中移除 current 通道 (比 --no-current 的 mask 更省)")
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--init", default=None, help="作为起点的 model.pt (结构须相同)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--neg-weight", type=float, default=2.0, help="策略在地面运行窗口的 loss 权重 (误暂停是代价最高的错误)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    feats, meta = load(args.train, dev)
    vfeats, vmeta = load(args.val, dev)
    idx = [valid_index(meta, i) for i in range(len(feats))]
    sizes = torch.tensor([len(t) for t, _ in idx], dtype=torch.float)

    keep = [i for i in range(FEAT_DIM) if not (args.drop_current and CURRENT_SLICE.start <= i < CURRENT_SLICE.stop)]
    net_kwargs = {"ch": args.ch, "keep": keep}
    net = PickupNet(**net_kwargs).to(dev)
    # 从随机 tick 采样得到 normalizer
    smp = torch.cat([f[torch.randint(0, f.shape[0], (50_000,), device=dev), torch.randint(0, f.shape[1], (50_000,), device=dev)].float() for f in feats])
    net.mean.copy_(smp.mean(0)); net.std.copy_(smp.std(0).clamp_min(1e-3))
    net.mean[PAUSED_IDX] = 0.0; net.std[PAUSED_IDX] = 1.0
    if args.no_current or args.drop_current:
        net.mask[CURRENT_SLICE] = 0.0
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev)["state_dict"])
        print("已从以下模型初始化:", args.init)
    print(f"参数量 {sum(p.numel() for p in net.parameters())}")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        shard = torch.multinomial(sizes, args.batch, replacement=True).to(dev)
        xs, ys, ws = [], [], []
        for i in range(len(feats)):
            k = int((shard == i).sum())
            if k == 0:
                continue
            ti, ni = idx[i]
            j = torch.randint(0, len(ti), (k,), device=dev)
            t, n = ti[j], ni[j]
            xs.append(gather(feats[i], t, n))
            y = meta["held"][i][t, n].float(); ys.append(y)
            floor_policy = (~meta["held"][i][t, n]) & (~meta["paused"][i][t, n])
            ws.append(torch.where(floor_policy, torch.full_like(y, args.neg_weight), torch.ones_like(y)))
        x, y, w = torch.cat(xs), torch.cat(ys), torch.cat(ws)
        loss = (Fn.binary_cross_entropy_with_logits(net(x), y, reduction="none") * w).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(net, vfeats, vmeta, dev, max_per_shard=200_000)
            print(f"step {step} loss {loss.item():.4f} val_bce {ev.pop('_bce'):.4f} 用时 {time.time() - t0:.0f}s")
            for k, v in ev.items():
                print(f"    {k:38s} {'检出' if k.startswith('held') else '正确 (无 pause)'} {100 * v:6.2f}%")

    os.makedirs(args.out, exist_ok=True)
    ck = {"state_dict": net.state_dict(), "net_kwargs": net_kwargs, "sm_cfg": vars(StateMachineCfg()),
          "no_current": args.no_current, "train": args.train}
    torch.save(ck, os.path.join(args.out, "model.pt"))
    export_onnx(net, os.path.join(args.out, "pickup_detector.onnx"))
    print("已保存", args.out)


if __name__ == "__main__":
    main()
