#!/usr/bin/env python3
"""持握场景测试集: 每个场景一次长时间强制拾取中检测器的表现.

每个场景是一次闭环 `pickup_datagen.py --detector ... --force-pickup-at 2` 运行, 其中所有 env 在
t = 2 s 被拾起并一直持握到结束, 并通过 HandCfg 覆写构造对应情形 (抓头部, 把鸭子倒过来, 旋转 180°, ...).
每个场景报告:

  paused%    从抬起后 1 s 起, 被持握时间中机器人处于 paused 的比例
  latency    抬起 → 首次暂停 (p50 / p90)
  resumes    持握期间每个 env 的误恢复次数 (每一次 = 策略在手中运行)

  uv run scripts/pickup_battery.py --detector logs/pickup/nocur/model.pt --tag v1
"""

import argparse
import os
import subprocess
import sys

import numpy as np

DT = 0.02
COMMON = "drop_prob=0.0;shake_prob=0.0"
SCENARIOS = {
    # 机器人上已经可用的情形 - 作为对照
    "trunk, casual tilt": "head_grip_prob=0.0;orient_prob=0.0;yaw_turn_prob=0.0;tilt_buckets=((20.0,1.0),)",
    "HEAD grip, casual tilt": "head_grip_prob=1.0;orient_prob=0.0;yaw_turn_prob=0.0;tilt_buckets=((20.0,1.0),)",
    "trunk, upside down (pitch 180)": "head_grip_prob=0.0;orient_prob=1.0;orient_pitch=(2.9,3.14);orient_roll=(-0.1,0.1);yaw_turn_prob=0.0;tilt_buckets=((10.0,1.0),)",
    "trunk, on its side (roll 90)": "head_grip_prob=0.0;orient_prob=1.0;orient_pitch=(-0.1,0.1);orient_roll=(1.45,1.57);yaw_turn_prob=0.0;tilt_buckets=((10.0,1.0),)",
    "trunk, spun 180 about vertical": "head_grip_prob=0.0;orient_prob=0.0;yaw_turn_prob=1.0;yaw_turn=(2.9,3.14);yaw_rate=(0.0,0.0);tilt_buckets=((10.0,1.0),)",
    "HEAD grip, upside down": "head_grip_prob=1.0;orient_prob=1.0;orient_pitch=(2.9,3.14);orient_roll=(-0.1,0.1);yaw_turn_prob=0.0;tilt_buckets=((10.0,1.0),)",
}


def analyze(path, pick_t=2.0):
    d = np.load(path)
    held, paused, sr = d["held"], d["paused"], d["since_reset"]
    T, N = held.shape
    t0 = int(pick_t / DT)
    cover, lat, res, n = [], [], [], 0
    for e in range(N):
        if (sr[t0:, e] < sr[t0, e]).any():  # 场景中途发生过 episode reset
            continue
        h = held[t0:, e]
        if not h[int(1.0 / DT):].any():
            continue
        n += 1
        p = paused[t0:, e]
        on = np.argmax(h)
        first = np.nonzero(p[on:])[0]
        lat.append(first[0] * DT if len(first) else np.inf)
        win = slice(on + int(1.0 / DT), None)
        cover.append(p[win][h[win]].mean())
        r = np.nonzero(p[on:-1] & ~p[on + 1:] & h[on + 1:])[0]
        res.append(len(r))
    lat = np.array(lat)
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    return n, np.mean(cover), np.percentile(lat, 50), np.percentile(lat, 90), np.mean(res), np.mean(np.isinf(lat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--only", default=None, help="场景名的子串")
    args = ap.parse_args()
    rows = []
    for name, hand in SCENARIOS.items():
        if args.only and args.only not in name:
            continue
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        out = f"data/pickup/battery_{args.tag}_{slug}.npz"
        if not os.path.exists(out):
            subprocess.run([sys.executable, "scripts/pickup_datagen.py", "--detector", args.detector, "--out", out,
                            "--num-envs", str(args.num_envs), "--seconds", str(args.seconds), "--seed", "200",
                            "--force-pickup-at", "2.0", "--hand", f"{COMMON};{hand}"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rows.append((name, *analyze(out)))
    print(f"\n[{args.tag}] {'scenario':34s} {'envs':>5s} {'paused%':>8s} {'lat p50':>8s} {'lat p90':>8s} {'resumes/env':>12s} {'never':>6s}")
    for name, n, cov, l50, l90, res, never in rows:
        print(f"[{args.tag}] {name:34s} {n:5d} {100 * cov:7.1f}% {l50:7.2f}s {l90:7.2f}s {res:12.2f} {100 * never:5.1f}%")


if __name__ == "__main__":
    main()
