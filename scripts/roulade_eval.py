"""Roulade 前滚翻策略的 headless 分批评测 (重建自 2026-09-07 记录的 eval_roulade.py 口径).

按起始状态分桶 rollout 一个 checkpoint, 每桶 N 个并行环境跑若干轮 (每轮一个
固定 250 步的完整 episode), 报告:

- 完成率: 旋转前沿 (env._roulade_max) 在 episode 内达到 >=330 度 (与训练落地
  年金 gate 同口径) 的占比.
- head-latch 率: 真正过头顶 (第一象限窗内头 FLAT TOP 触地) 的占比.
- 完成耗时: 达到 330 度的环境步.
- 结束站立率与站定质量: 最后 50 步窗内 trunk z 与倾角 (仅对完成的 episode).
- 头颈偏差: 最后 50 步窗内 4 个 head 关节相对模型 default (HOME) 的角度偏差.

用法:

    uv run python scripts/roulade_eval.py --checkpoint logs/rsl_rl/microduck_roulade/<run>/model_9999.pt

可选: --envs 4096 --rounds 3 --buckets "stand|50-180|180-260|260-340" --steps 250
      --csv /tmp/roulade_eval.csv (追加一行汇总)

需要 GPU (mjlab warp); 本机无 GPU 时会在加载环境前报错退出.
"""

import argparse
import csv
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path

import torch

_TASK = "Mjlab-Roulade-Flat-MicroDuck"
_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
_COMPLETE_RAD = math.radians(330.0)


def _parse_bucket(spec: str) -> tuple[dict, str]:
    """把桶描述转成 set_roulade_state 的参数覆盖.

    站立桶: 全部 standing spawn; 中途桶: 全部 mid-roll, pitch 均匀落在
    [lo, hi] 度区间 (沿用配置的 tuck/z/omega 默认).
    """
    spec = spec.strip()
    if spec == "stand":
        return {"standing_prob": 1.0, "midroll_prob": 0.0}, spec
    m = re.fullmatch(r"(\d+)-(\d+)", spec)
    if m is None:
        raise SystemExit(f"无法解析桶描述: {spec!r} (应为 stand 或 lo-hi 度区间)")
    lo, hi = int(m.group(1)), int(m.group(2))
    if not (0 <= lo < hi <= 360):
        raise SystemExit(f"桶区间非法: {spec!r}")
    return {
        "standing_prob": 0.0,
        "midroll_prob": 1.0,
        "midroll_pitch_min": math.radians(lo),
        "midroll_pitch_max": math.radians(hi),
    }, spec


def _deg(x: torch.Tensor) -> torch.Tensor:
    return x * 180.0 / math.pi


def _tilt_deg(quat_w: torch.Tensor) -> torch.Tensor:
    """根 link 相对竖直的倾角 (度): cos(tilt)=1-2(qx^2+qy^2), 对前后左右倾倒对称."""
    qw, qx, qy, qz = quat_w.unbind(dim=-1)
    del qw, qz
    cos_t = torch.clamp(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)
    return _deg(torch.acos(cos_t))


def main() -> None:
    ap = argparse.ArgumentParser(description="roulade 策略 headless 分桶评测")
    ap.add_argument("--checkpoint", required=True, help="model_XXXX.pt 路径")
    ap.add_argument("--task", default=_TASK)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--rounds", type=int, default=3, help="每桶独立重置的轮数")
    ap.add_argument("--steps", type=int, default=250, help="每轮 rollout 的环境步数")
    ap.add_argument("--buckets", default="stand|50-180|180-260|260-340",
                    help="竖线分隔的桶: stand 或 lo-hi (度)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--csv", default=None, help="可选 CSV 路径, 追加一行汇总")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        sys.exit("本机无可用 CUDA, 评测需要 mjlab warp (GPU). 请在 GPU 机器上运行.")
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        sys.exit(f"checkpoint 不存在: {ckpt}")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    import mjlab_microduck.tasks.mdp as mmdp  # noqa: F401  (任务插件注册)

    configure_torch_backends()

    # 与 play 相同的加载路径: play cfg + wrapper + runner.load(actor) + 推理策略.
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.envs
    # 关掉 time_out 截断: 每轮我们手动跑固定 steps, 避免 250 步处自动 reset
    # 污染最后窗口的状态读数 (episode 记账属性也会被清零).
    env_cfg.episode_length_s = 1.0e6
    base_env: ManagerBasedRlEnv = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task)
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] 已加载策略: {ckpt.name} (device={device}, envs={args.envs})")

    asset = base_env.scene["robot"]
    servo_ids = mmdp._servo_joint_ids(base_env, asset)
    # head 关节在 servo 视图内的位置 (名称解析, 不硬编码索引).
    head_servo_idx: list[int] = []
    head_entity_idx: list[int] = []
    for nm in _HEAD_JOINT_NAMES:
        res = asset.find_joints("^" + re.escape(nm) + "$")
        ent_id = res[0][0] if isinstance(res, tuple) else res[0]
        ent_id = int(ent_id)
        if ent_id in servo_ids:
            head_servo_idx.append(servo_ids.index(ent_id))
            head_entity_idx.append(ent_id)
    if len(head_servo_idx) != len(_HEAD_JOINT_NAMES):
        sys.exit("模型关节集合与预期 head 关节不一致, 请检查任务模型.")
    head_servo_idx_t = torch.tensor(head_servo_idx, dtype=torch.long, device=device)
    home = mmdp._servo_default_joint_pos(base_env, asset)[:, head_servo_idx_t]  # [N, 4]

    get_root_pos = lambda: asset.data.root_link_pos_w[:, 2]  # noqa: E731
    get_root_quat = lambda: asset.data.root_link_quat_w  # noqa: E731
    get_jpos = lambda: asset.data.joint_pos[:, head_entity_idx]  # noqa: E731

    buckets = [_parse_bucket(b) for b in args.buckets.split("|")]
    term = base_env.event_manager.get_term_cfg("set_roulade_state")
    assert term is not None, "未找到 set_roulade_state reset 事件"

    rows: list[dict] = []
    for overrides, bname in buckets:
        for k, v in overrides.items():
            term.params[k] = v
        agg = {"n": 0, "comp": 0, "latch": 0, "steps": [], "stand": 0, "z": [], "tilt": [], "dev": []}
        for _round in range(args.rounds):
            obs, _ = env.reset()
            completed = torch.zeros(args.envs, dtype=torch.bool, device=device)
            done_at = torch.zeros(args.envs, dtype=torch.float32, device=device)
            z_buf: list[torch.Tensor] = []
            tilt_buf: list[torch.Tensor] = []
            head_buf: list[torch.Tensor] = []
            for step_i in range(args.steps):
                with torch.no_grad():
                    act = policy(obs)
                obs, _, _, _ = env.step(act)
                fr = base_env._roulade_max.clone()
                newly = (~completed) & (fr >= _COMPLETE_RAD)
                completed = completed | newly
                done_at[newly] = float(step_i + 1)
                z_buf.append(get_root_pos().clone())
                tilt_buf.append(_tilt_deg(get_root_quat()).clone())
                head_buf.append(get_jpos().clone())
            latch = base_env._roulade_head_latch.clone()
            # 最后 50 步窗均值 (不足 50 步则取全部).
            win = max(1, min(50, args.steps))
            z_win = torch.stack(z_buf[-win:]).mean(dim=0)
            tilt_win = torch.stack(tilt_buf[-win:]).mean(dim=0)
            head_win = torch.stack(head_buf[-win:]).mean(dim=0)  # [N, 4] rad
            head_dev = _deg((head_win - home).abs())  # [N, 4]
            is_stand = completed & (tilt_win <= 45.0) & (z_win >= 0.09) & (z_win <= 0.14)
            agg["n"] += int(args.envs)
            agg["comp"] += int(completed.sum())
            agg["latch"] += int((completed & latch).sum())
            agg["steps"].extend(done_at[completed].tolist())
            agg["stand"] += int(is_stand.sum())
            agg["z"].extend(z_win[is_stand].tolist())
            agg["tilt"].extend(tilt_win[is_stand].tolist())
            agg["dev"].extend(head_dev[is_stand].tolist())
        n = max(agg["n"], 1)
        comp_n = max(agg["comp"], 1)
        zs = agg["z"] or [float("nan")]
        ts = agg["tilt"] or [float("nan")]

        def _stat(vals, fn):
            try:
                return fn(vals)
            except (ValueError, TypeError):
                return float("nan")

        row = {
            "bucket": bname,
            "n": agg["n"],
            "completion_pct": 100.0 * agg["comp"] / n,
            "head_latch_pct": 100.0 * agg["latch"] / n,
            "mean_steps": _stat(agg["steps"], lambda v: sum(v) / max(len(v), 1)),
            "stand_pct_of_comp": 100.0 * agg["stand"] / comp_n,
            "z_mean_mm": 1000.0 * _stat(zs, lambda v: sum(v) / len(v)),
            "tilt_mean_deg": _stat(ts, lambda v: sum(v) / len(v)),
        }
        devs = agg["dev"]
        if devs:
            dev_t = torch.tensor(devs).mean(dim=0)
            for nm, d in zip(_HEAD_JOINT_NAMES, dev_t.tolist()):
                row[f"dev_{nm}_deg"] = d
        rows.append(row)
        print(f"[RESULT] bucket={bname} n={agg['n']} completion={row['completion_pct']:.1f}% "
              f"latch={row['head_latch_pct']:.1f}% mean_steps={row['mean_steps']:.0f} "
              f"stand_of_comp={row['stand_pct_of_comp']:.1f}% z={row['z_mean_mm']:.1f}mm "
              f"tilt={row['tilt_mean_deg']:.1f}deg "
              + " ".join(f"{nm}={row[f'dev_{nm}_deg']:.1f}deg" for nm in _HEAD_JOINT_NAMES))

    if args.csv:
        fieldnames = list(rows[0].keys())
        with open(args.csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                w.writeheader()
            for r in rows:
                w.writerow({"bucket": f"{ckpt.name}:{r['bucket']}", **r})
        print(f"[INFO] CSV 已追加: {args.csv}")


if __name__ == "__main__":
    main()
