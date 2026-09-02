import math

import torch

from mjlab_microduck.tasks import mdp


def test_crouch_height_target_endpoints_are_high():
    # phase 0 (起始) 和 phase ~1 (结束) -> 高高度 (站立)
    phase = torch.tensor([0.0, 0.999])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([0.11, 0.11]), atol=2e-3)


def test_crouch_height_target_plateau_is_low():
    # 整个平台 [0.375, 0.625] -> 恒定的低高度
    phase = torch.tensor([0.375, 0.5, 0.624])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.full((3,), 0.075), atol=1e-6)


def test_crouch_height_target_descent_midpoint():
    # 下行中点 (phase = hold_lo/2 = 0.1875) -> 两个高度的中点
    phase = torch.tensor([0.1875])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([(0.11 + 0.075) / 2]), atol=1e-6)


def test_crouch_height_target_rise_midpoint():
    # 上行中点 (phase = 0.8125) -> 两个高度的中点
    phase = torch.tensor([0.8125])
    t = mdp.crouch_height_target(phase, height_low=0.075, height_high=0.11)
    assert torch.allclose(t, torch.tensor([(0.11 + 0.075) / 2]), atol=1e-6)


# ── crouch_pose_blend: 4 段 (下行 / 低 / 上行 / 站立) ──────────────────────
# 测试断点: 下行 [0,0.1), 低 [0.1,0.5), 上行 [0.5,0.6), 站立 [0.6,1.0).
_BLEND = {"descent_end": 0.10, "hold_end": 0.50, "rise_end": 0.60}


def test_blend_zero_standing_at_start_and_top_hold():
    phase = torch.tensor([0.0, 0.6, 0.8, 0.999])  # 起始 + 高平台
    b = mdp.crouch_pose_blend(phase, **_BLEND)
    assert torch.allclose(b, torch.zeros(4), atol=1e-6)


def test_blend_one_on_low_hold():
    phase = torch.tensor([0.10, 0.3, 0.499])  # 低平台
    b = mdp.crouch_pose_blend(phase, **_BLEND)
    assert torch.allclose(b, torch.ones(3), atol=1e-6)


def test_blend_descent_and_rise_midpoints():
    # 下行中点 (0.05 在 [0,0.1) 上) -> 0.5; 上行中点 (0.55 在 [0.5,0.6) 上) -> 0.5
    phase = torch.tensor([0.05, 0.55])
    b = mdp.crouch_pose_blend(phase, **_BLEND)
    assert torch.allclose(b, torch.tensor([0.5, 0.5]), atol=1e-6)


def test_reward_is_one_when_height_matches_target():
    # phase 0.5 (整平台) -> 目标 = height_low; 若 com_height == height_low -> reward 1
    cmd_cos = torch.tensor([math.cos(2 * math.pi * 0.5)])  # -1
    cmd_sin = torch.tensor([math.sin(2 * math.pi * 0.5)])  # ~0
    com_height = torch.tensor([0.075])
    r = mdp.crouch_glide_reward_from_values(com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02)
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-3)


def test_reward_decays_when_off_by_one_std():
    # 在 height_low + std 偏离目标 -> exp(-1) ≈ 0.368
    cmd_cos = torch.tensor([math.cos(2 * math.pi * 0.5)])
    cmd_sin = torch.tensor([math.sin(2 * math.pi * 0.5)])
    com_height = torch.tensor([0.075 + 0.02])
    r = mdp.crouch_glide_reward_from_values(com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02)
    assert torch.allclose(r, torch.tensor([math.exp(-1.0)]), atol=1e-3)


def test_reward_at_phase_zero_expects_high_stance():
    # phase 0 -> 目标 = height_high; 站立被奖励, 蹲下不被奖励
    cmd_cos = torch.tensor([1.0, 1.0])  # cos(0)
    cmd_sin = torch.tensor([0.0, 0.0])  # sin(0)
    com_height = torch.tensor([0.11, 0.075])  # 站立 vs 蹲下
    r = mdp.crouch_glide_reward_from_values(com_height, cmd_cos, cmd_sin, height_low=0.075, height_high=0.11, std=0.02)
    assert r[0] > 0.99  # phase 0 站立 -> ~1
    assert r[1] < 0.2  # phase 0 蹲下 -> 低
