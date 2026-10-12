#!/usr/bin/env python3
"""对 `pickup_datagen.py --detector ...` rollout 的闭环指标.

  uv run scripts/pickup_eval.py data/pickup/eval_nocur.npz [more.npz ...]

事件级, 逐 env, 只统计 reset 之后至少 1 s 的 tick:
  false pause      pause 起始, 且之前 0.5 s 内没有 hand-held tick
                   (按 context 折算为每小时的策略在地面运行时间)
  pick-up          held episode ≥ 0.5 s: latency 为 held 起始 → pause; 持握期间从未 pause 记为 missed
  put-down         held → not held, 在地面停留 ≥ 1.5 s: latency → resume
                   (1.5 s 内未 resume 记为 stuck); tipped = resume 前 tilt > 40°
  false resume     搬运途中出现 resume 起始 (在搬运, 之前 0.5 s 双脚离地);
                   放下过程中的 resume - 双脚已承重, 柔顺手正在松开
                   - 属于标签抖动, 单独统计
Put-down 分为 set-down (双脚承重) 与 drop (从高处释放, 落地冲击大, 其中的
tipping 是 drop 造成, 不是检测器).
"""

import sys

import numpy as np

DT = 0.02


def onsets(x):
    return np.nonzero(x[1:] & ~x[:-1])[0] + 1


def analyze(path):
    d = np.load(path)
    held, paused, sr = d["held"], d["paused"], d["since_reset"]
    grav = d["feat"][..., 3:6].astype(np.float32)
    tilt = np.degrees(np.arccos(np.clip(-grav[..., 2], -1, 1)))
    cmd = d["cmd"].astype(np.float32); fallen = d["fallen"]
    phase, ff = d["phase"], d["feet_frac"].astype(np.float32)
    T, N = held.shape
    r = dict(fp=[], fp_ctx={"walking": 0, "standing": 0, "fallen": 0}, floor_time={"walking": 0.0, "standing": 0.0, "fallen": 0.0},
             pick_lat=[], pick_miss=0, put_lat=[], put_stuck=0, put_tipped=0, false_resume=0, setdown_flicker=0, held_time=0.0,
             drops=0, drop_tipped=0)
    for n in range(N):
        h, p, s = held[:, n], paused[:, n], sr[:, n]
        ok = s >= 50
        run_floor = ok & ~h & ~p
        for ctx, m in (("walking", (cmd[:, n] > 0.05) & ~fallen[:, n]), ("standing", (cmd[:, n] <= 0.05) & ~fallen[:, n]), ("fallen", fallen[:, n])):
            r["floor_time"][ctx] += (run_floor & m).sum() * DT
        r["held_time"] += (ok & h).sum() * DT
        for t in onsets(p):
            if not ok[t]:
                continue
            if not h[max(0, t - 25):t + 1].any():
                ctx = "fallen" if fallen[t, n] else ("walking" if cmd[t, n] > 0.05 else "standing")
                r["fp_ctx"][ctx] += 1
        for t in onsets(~p):  # resume 起始点
            if ok[t] and h[t]:
                if phase[t, n] == 1 and ff[max(0, t - 25):t + 1, n].max() < 0.25:
                    r["false_resume"] += 1
                else:
                    r["setdown_flicker"] += 1
        # held episode
        on, off = onsets(h), onsets(~h)
        for t in on:
            if not ok[t] or p[t - 1]:
                continue
            e = off[off > t]; e = e[0] if len(e) else T
            if (e - t) * DT < 0.5:
                continue
            pp = np.nonzero(p[t:e])[0]
            if len(pp):
                r["pick_lat"].append(pp[0] * DT)
            else:
                r["pick_miss"] += 1
        for t in off:
            if not ok[t] or not p[t - 1]:
                continue  # 只统计已 pause 机器人的 put-down
            if t + 75 > T or h[t:t + 75].any() or (s[t:t + 75] < s[t]).any():
                continue
            rr = np.nonzero(~p[t:t + 75])[0]
            if ff[t, n] < 0.25:  # 从高处释放: 属于 drop
                r["drops"] += 1
                r["drop_tipped"] += int((tilt[t:t + (rr[0] + 1 if len(rr) else 75), n] > 40).any())
                continue
            if len(rr):
                r["put_lat"].append(rr[0] * DT)
                if (tilt[t:t + rr[0] + 1, n] > 40).any():
                    r["put_tipped"] += 1
            else:
                r["put_stuck"] += 1
                if (tilt[t:t + 75, n] > 40).any():
                    r["put_tipped"] += 1
    return r


def merge(rs):
    out = rs[0]
    for r in rs[1:]:
        for k, v in r.items():
            if isinstance(v, list):
                out[k] += v
            elif isinstance(v, dict):
                for kk in v:
                    out[k][kk] += v[kk]
            else:
                out[k] += v
    return out


def report(r):
    print("误暂停 (策略在运行, 无人接触鸭子):")
    for ctx in ("walking", "standing", "fallen"):
        hrs = r["floor_time"][ctx] / 3600
        print(f"  {ctx:9s} {r['fp_ctx'][ctx]:5d} 共 {hrs:6.2f} h  →  {r['fp_ctx'][ctx] / max(hrs, 1e-9):7.2f} / 小时")
    pl = np.array(r["pick_lat"]); n_pick = len(pl) + r["pick_miss"]
    print(f"拾取: {n_pick}  检出 {100 * len(pl) / max(1, n_pick):.1f}%  latency p50/p90/p99 "
          f"{np.percentile(pl, 50):.2f}/{np.percentile(pl, 90):.2f}/{np.percentile(pl, 99):.2f} s")
    ql = np.array(r["put_lat"]); n_put = len(ql) + r["put_stuck"]
    print(f"放下: {n_put}  1.5 s 内恢复 {100 * len(ql) / max(1, n_put):.1f}%  latency p50/p90/p99 "
          f"{np.percentile(ql, 50):.2f}/{np.percentile(ql, 90):.2f}/{np.percentile(ql, 99):.2f} s  "
          f"恢复前 tipped >40° {100 * r['put_tipped'] / max(1, n_put):.1f}%")
    print(f"掉落: {r['drops']}  恢复前 tipped >40° {100 * r['drop_tipped'] / max(1, r['drops']):.1f}% (下落本身造成)")
    print(f"set-down 标签抖动 (恢复时双脚已承重): {r['setdown_flicker']}")
    print(f"搬运途中误恢复: {r['false_resume']} / 持握 {r['held_time'] / 3600:.2f} h "
          f"→ {r['false_resume'] / max(r['held_time'] / 3600, 1e-9):.1f} / 小时")


if __name__ == "__main__":
    report(merge([analyze(p) for p in sys.argv[1:]]))
