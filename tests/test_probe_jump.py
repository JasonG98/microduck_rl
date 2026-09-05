import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_jump import JumpProbe


class JumpProbeTest(unittest.TestCase):
    def setUp(self):
        self.probe = JumpProbe("src/mjlab_microduck/robot/microduck/scene_walk.xml")

    def test_pose_solver_is_independent_of_previous_fall(self):
        p = self.probe
        q, z = p.solve_pose(1.2)
        p.data.qpos[p.free : p.free + 7] = [2, -1, 0.03, 0.70710678, 0, 0.70710678, 0]
        p.data.qvel[:] = 5
        other, other_z = p.solve_pose(1.2)
        np.testing.assert_allclose(q, other, atol=1e-8)
        self.assertAlmostEqual(z, other_z)
        np.testing.assert_allclose(p.data.qvel, 0)

    def test_static_crouch_is_supported_not_fallen(self):
        _, stable, s = self.probe.prepare(1.5)
        self.assertTrue(stable)
        self.assertTrue(s["both_feet_contact"])
        self.assertFalse(s["nonfoot_contact"])
        self.assertLess(s["tilt_deg"], 10)
        self.assertLess(abs(s["ground_force_n"] - self.probe.model.body_mass.sum() * 9.81), 0.1)

    def test_com_free_flight_matches_gravity(self):
        p = self.probe
        # 仅验证测量器: 人工高空初态不能进入起跳能力的实验结果.
        p.reset(p.home, 0.8)
        p.data.qvel[2] = 1.0
        initial = p.snapshot()
        for _ in range(20):
            p.step(p.home)
        end = p.snapshot()
        t = end["time_s"]
        self.assertFalse(end["ground_contact"])
        self.assertAlmostEqual(end["com_vz_m_s"], 1 - 9.81 * t, places=5)
        self.assertLess(abs(end["com_z_m"] - initial["com_z_m"] - (t - 0.5 * 9.81 * t * t)), 0.003)


if __name__ == "__main__":
    unittest.main()
