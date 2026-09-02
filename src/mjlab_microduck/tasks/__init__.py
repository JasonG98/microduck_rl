from mjlab_microduck.train_hook import maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` 会提交到 HF Jobs 并在此退出, 早于下方任何
# cfg 导入: 此模块是 mjlab 插件加载器拉入的入口, 也是任何安装顺序都无法
# 剥夺的唯一训练路径 (见 train_hook.py).未带该 flag 时为空操作.
maybe_submit_to_hf_jobs()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    """On-policy runner, 用于从 symmetry_cfg 中剥离不可序列化的 ``_env`` 以便 yaml 转储."""

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        """初始化 runner, 并将 ``symmetry_cfg`` 与活跃的 ``_env`` 引用解耦."""
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config 会就地注入 _env 到 train_cfg["algorithm"]["symmetry_cfg"],
        # 与 self.alg.symmetry 共享同一个 dict 对象.用一个去掉 _env 的副本替换
        # train_cfg 引用, 这样 dump_yaml 就能序列化配置 (MjSpec 不可 pickle),
        # 同时不动 PPO 内部持有的引用.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .backlash import make_backlash_variant
from .microduck_ball_kick_env_cfg import (
    MicroduckBallKickRlCfg,
    make_microduck_ball_kick_env_cfg,
)
from .microduck_ground_pick_env_cfg import (
    MicroduckGroundPickRlCfg,
    make_microduck_ground_pick_env_cfg,
)
from .microduck_roller_crouch_env_cfg import (
    MicroduckRollerCrouchRlCfg,
    make_microduck_roller_crouch_env_cfg,
)
from .microduck_roller_slope_env_cfg import (
    MicroduckRollerSlopeRlCfg,
    make_microduck_roller_slope_env_cfg,
)
from .microduck_roller_standup_env_cfg import (
    MicroduckRollerStandUpRlCfg,
    make_microduck_roller_standup_env_cfg,
)
from .microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)
from .microduck_sitstand_env_cfg import (
    MicroduckSitStandRlCfg,
    make_microduck_sitstand_env_cfg,
)
from .microduck_spin_env_cfg import (
    MicroduckSpinRlCfg,
    make_microduck_spin_env_cfg,
)
from .microduck_standup_env_cfg import (
    MicroduckStandUpRlCfg,
    make_microduck_standup_env_cfg,
)
from .microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)
from .microduck_velocity_rollers_env_cfg import (
    MicroduckRollersRlCfg,
    make_microduck_velocity_rollers_env_cfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    MicroduckSwizzleRlCfg,
    make_microduck_velocity_swizzle_env_cfg,
)
from .microduck_velstand_env_cfg import (
    MicroduckVelStandRlCfg,
    make_microduck_velstand_env_cfg,
)

# 标准 velocity 任务
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — 行走 + 摔倒恢复 + 身体姿态控制三合一策略.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up 任务 — 机器人初始倒置 (仰卧), 需要自行站起
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand 任务 — 单策略命令式坐 ↔ 站, 动作柔和, 头部可被命令控制
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick 任务 — 下蹲, 用嘴尖触地, 再回到站立
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick 任务 — 从站立起跑用右脚将 70mm/15g 的球向前猛踢
# (仅平地 — 粗糙地形上的球属于另一任务).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# 轮滑 velocity 任务 (被动轮模型; 保留历史 task id)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE 任务 — 标准干净的 swizzle (对称, 双脚着地).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — 在轮滑上起身 (专用策略, 从地面起步).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin 任务 — 在轮滑上原地快速旋转 (使用 ground-pick 槽位).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — 越过平头向前滚翻, 落回双脚.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash 变体 — 每个舵机 ±1° 串联齿轮间隙 + 经由 backlash 的编码器
# 反馈与关节观测 (见 tasks/backlash.py).每个任务族保留其基础任务的碰撞
# 模型: Velocity → robot_walk_backlash.xml,
# VelStand/StandUp → robot_allcollisions_backlash.xml.obs/action 维度
# 与基础任务保持一致.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg).task id 在
# 基础 id 中插入 "-Backlash".Walk 模型任务使用 walk backlash 机器人,
# roller 任务使用 wheels+backlash 机器人, 其余使用 allcollisions
# backlash 机器人 — 每种都与对应基础任务使用同一模型.
_BL_ALLCOL = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    (
        "Mjlab-Velocity-Flat-Backlash-MicroDuck",
        make_microduck_velocity_env_cfg,
        {},
        MicroduckRlCfg,
        _BL_WALK,
    ),
    (
        "Mjlab-Velocity-Rough-Backlash-MicroDuck",
        make_microduck_velocity_env_cfg,
        {"rough": True},
        MicroduckRlCfg,
        _BL_WALK,
    ),
    (
        "Mjlab-VelStand-Flat-Backlash-MicroDuck",
        make_microduck_velstand_env_cfg,
        {},
        MicroduckVelStandRlCfg,
        _BL_ALLCOL,
    ),
    (
        "Mjlab-VelStand-Rough-Backlash-MicroDuck",
        make_microduck_velstand_env_cfg,
        {"rough": True},
        MicroduckVelStandRlCfg,
        _BL_ALLCOL,
    ),
    (
        "Mjlab-StandUp-Flat-Backlash-MicroDuck",
        make_microduck_standup_env_cfg,
        {},
        MicroduckStandUpRlCfg,
        _BL_ALLCOL,
    ),
    (
        "Mjlab-StandUp-Rough-Backlash-MicroDuck",
        make_microduck_standup_env_cfg,
        {"rough": True},
        MicroduckStandUpRlCfg,
        _BL_ALLCOL,
    ),
    (
        "Mjlab-SitStand-Flat-Backlash-MicroDuck",
        make_microduck_sitstand_env_cfg,
        {},
        MicroduckSitStandRlCfg,
        _BL_ALLCOL,
    ),
    (
        "Mjlab-SitStand-Rough-Backlash-MicroDuck",
        make_microduck_sitstand_env_cfg,
        {"rough": True},
        MicroduckSitStandRlCfg,
        _BL_ALLCOL,
    ),
    (
        "Mjlab-GroundPick-Flat-Backlash-MicroDuck",
        make_microduck_ground_pick_env_cfg,
        {},
        MicroduckGroundPickRlCfg,
        _BL_ALLCOL,
    ),
    (
        "Mjlab-GroundPick-Rough-Backlash-MicroDuck",
        make_microduck_ground_pick_env_cfg,
        {"rough": True},
        MicroduckGroundPickRlCfg,
        _BL_ALLCOL,
    ),
    (
        "Mjlab-BallKick-Flat-Backlash-MicroDuck",
        make_microduck_ball_kick_env_cfg,
        {},
        MicroduckBallKickRlCfg,
        _BL_ALLCOL,
    ),
    (
        "Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers",
        make_microduck_velocity_rollers_env_cfg,
        {},
        MicroduckRollersRlCfg,
        _BL_ROLLERS,
    ),
    (
        "Mjlab-Velocity-Swizzle-Backlash-MicroDuck",
        make_microduck_velocity_swizzle_env_cfg,
        {},
        MicroduckSwizzleRlCfg,
        _BL_ROLLERS,
    ),
    (
        "Mjlab-RollerCrouch-Flat-Backlash-MicroDuck",
        make_microduck_roller_crouch_env_cfg,
        {},
        MicroduckRollerCrouchRlCfg,
        _BL_ROLLERS,
    ),
    (
        "Mjlab-RollerSlope-Flat-Backlash-MicroDuck",
        make_microduck_roller_slope_env_cfg,
        {},
        MicroduckRollerSlopeRlCfg,
        _BL_ROLLERS,
    ),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
