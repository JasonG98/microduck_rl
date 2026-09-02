import torch

from mjlab_microduck.tasks.mdp import slope_move_masks


def test_move_up_when_reached_bottom():
    # 距离 > size_x*0.4 (=3.2) -> 升难度
    dist = torch.tensor([5.0, 4.1])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert bool(up[0]) and bool(up[1])
    assert not bool(down[0]) and not bool(down[1])


def test_move_down_when_stuck_early():
    # 距离 < size_x*0.2 (=1.6) -> 降难度
    dist = torch.tensor([0.5, 1.0])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert not bool(up[0]) and not bool(up[1])
    assert bool(down[0]) and bool(down[1])


def test_stay_in_middle_band():
    # 1.6 与 3.2 之间 -> 既不升也不降
    dist = torch.tensor([2.5])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert not bool(up[0]) and not bool(down[0])


def test_move_up_boundary_at_04():
    # 一旦下行 > 0.4*size_x 就晋升 (机器人走过坡道一截才抵达出口平地).
    dist = torch.tensor([3.3])
    up, down = slope_move_masks(dist, size_x=8.0)
    assert bool(up[0])
    assert not bool(down[0])

    # 3.0 留在中间带 (3.0 < 3.2 且 3.0 > 1.6)
    dist_mid = torch.tensor([3.0])
    up_mid, down_mid = slope_move_masks(dist_mid, size_x=8.0)
    assert not bool(up_mid[0]) and not bool(down_mid[0])
