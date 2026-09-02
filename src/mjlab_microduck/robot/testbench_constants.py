"""用于 sim2real 验证的 XL330 测试台实体配置."""

from pathlib import Path

import mujoco
from bam.mjlab import BamActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_TESTBENCH_DIR: Path = Path(__file__).parent / "xl330_test_bench"
# 使用仅机器人的 XML (无地面 / 无灯光): mjlab 的 TerrainImporterCfg
# 会自行添加地面, 所以 scene.xml 会造成重复地面.
TESTBENCH_XML: Path = _TESTBENCH_DIR / "xl330_test_bench.xml"

assert TESTBENCH_XML.exists(), f"XML not found: {TESTBENCH_XML}"


# 真机负载质量 (120 g)
TESTBENCH_ARM_MASS: float = 0.12


def _set_arm_mass(spec: mujoco.MjSpec, mass: float) -> None:
    for body in spec.bodies:
        if body.name == "arm":
            original = body.mass
            if original > 0:
                scale = mass / original
                body.mass = mass
                body.fullinertia = [x * scale for x in body.fullinertia]
            break


def get_testbench_spec() -> mujoco.MjSpec:
    """加载测试台 MJCF 并应用真实机械臂负载质量."""
    spec = mujoco.MjSpec.from_file(str(TESTBENCH_XML))
    _set_arm_mass(spec, TESTBENCH_ARM_MASS)
    return spec


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={"1": 0.0},
    joint_vel={".*": 0.0},
)


# 使用 BAM M6 执行器模型 (匹配测试台上的真实 XL330).
testbench_actuators = BamActuatorCfg(
    motor_name="xl330",
    model="m6",
    target_names_expr=(r"1",),
    kp_fw=200.0,
    # max_current=1.75,
    delay_min_lag=0,
    delay_max_lag=3,
)


XL330_TESTBENCH_ROBOT_CFG = EntityCfg(
    spec_fn=get_testbench_spec,
    init_state=HOME_FRAME,
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(testbench_actuators,),
        soft_joint_pos_limit_factor=1.0,
    ),
)
