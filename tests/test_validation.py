"""输入校验与错误语义测试。

所有期望值均为测试内手工计算的独立参考，不调用被测核心得出。
"""

import math
import unittest

from mis_envlight import (
    MAX_PATCHES,
    DegenerateStrategyError,
    Patch,
    SceneValidationError,
    build_scene,
    estimate,
    exact_integral,
    mis_contributions,
)

NORMAL = (0.0, 0.0, 1.0)
UP = (0.0, 0.0, 1.0)
SIDE = (0.0, 0.6, 0.8)  # 单位向量，cos=0.8


def simple_patches():
    return [
        Patch(UP, 2.0, 0.5),
        Patch(SIDE, 1.0, 0.25),
    ]


class TestNumericRejection(unittest.TestCase):
    def test_nan_radiance_rejected_with_location(self):
        patches = [Patch(UP, 1.0, 0.5), Patch(SIDE, float("nan"), 0.5)]
        with self.assertRaises(SceneValidationError) as ctx:
            build_scene(patches, NORMAL)
        self.assertIn("patches[1].radiance", str(ctx.exception))

    def test_inf_weight_rejected_with_location(self):
        patches = [Patch(UP, 1.0, float("inf"))]
        with self.assertRaises(SceneValidationError) as ctx:
            build_scene(patches, NORMAL)
        self.assertIn("patches[0].weight", str(ctx.exception))

    def test_negative_radiance_rejected(self):
        with self.assertRaises(SceneValidationError) as ctx:
            build_scene([Patch(UP, -1e-9, 0.5)], NORMAL)
        self.assertIn("patches[0].radiance", str(ctx.exception))

    def test_negative_weight_rejected(self):
        with self.assertRaises(SceneValidationError):
            build_scene([Patch(UP, 1.0, -0.5)], NORMAL)

    def test_bool_radiance_rejected(self):
        with self.assertRaises(SceneValidationError):
            build_scene([Patch(UP, True, 0.5)], NORMAL)

    def test_nan_direction_component_rejected(self):
        with self.assertRaises(SceneValidationError) as ctx:
            build_scene([Patch((0.0, float("nan"), 1.0), 1.0, 0.5)], NORMAL)
        self.assertIn("patches[0].direction[1]", str(ctx.exception))

    def test_non_unit_direction_rejected_not_normalized(self):
        with self.assertRaises(SceneValidationError) as ctx:
            build_scene([Patch((0.0, 0.0, 2.0), 1.0, 0.5)], NORMAL)
        self.assertIn("unit vector", str(ctx.exception))

    def test_zero_direction_rejected(self):
        with self.assertRaises(SceneValidationError):
            build_scene([Patch((0.0, 0.0, 0.0), 1.0, 0.5)], NORMAL)

    def test_non_unit_normal_rejected(self):
        with self.assertRaises(SceneValidationError) as ctx:
            build_scene(simple_patches(), (0.0, 0.0, 1.00001))
        self.assertIn("normal", str(ctx.exception))

    def test_direction_wrong_arity_rejected(self):
        with self.assertRaises(SceneValidationError):
            build_scene([Patch((0.0, 1.0), 1.0, 0.5)], NORMAL)

    def test_f_nan_rejected_with_location(self):
        scene = build_scene(simple_patches(), NORMAL)
        with self.assertRaises(SceneValidationError) as ctx:
            exact_integral(scene, [1.0, float("nan")])
        self.assertIn("f_values[1]", str(ctx.exception))

    def test_f_inf_rejected(self):
        scene = build_scene(simple_patches(), NORMAL)
        with self.assertRaises(SceneValidationError):
            exact_integral(scene, [float("-inf"), 1.0])

    def test_f_bool_rejected(self):
        scene = build_scene(simple_patches(), NORMAL)
        with self.assertRaises(SceneValidationError):
            exact_integral(scene, [True, 1.0])

    def test_f_length_mismatch_rejected(self):
        scene = build_scene(simple_patches(), NORMAL)
        with self.assertRaises(SceneValidationError) as ctx:
            exact_integral(scene, [1.0])
        self.assertIn("f_values", str(ctx.exception))


class TestIntegerRejection(unittest.TestCase):
    def setUp(self):
        self.scene = build_scene(simple_patches(), NORMAL)

    def test_bool_sample_count_rejected(self):
        with self.assertRaises(SceneValidationError) as ctx:
            estimate(self.scene, [1.0, 1.0], True, 1, 0)
        self.assertIn("n_radiance", str(ctx.exception))

    def test_float_sample_count_rejected(self):
        with self.assertRaises(SceneValidationError):
            estimate(self.scene, [1.0, 1.0], 1.5, 1, 0)

    def test_negative_sample_count_rejected(self):
        with self.assertRaises(SceneValidationError):
            estimate(self.scene, [1.0, 1.0], -1, 2, 0)

    def test_zero_total_samples_rejected(self):
        with self.assertRaises(SceneValidationError):
            estimate(self.scene, [1.0, 1.0], 0, 0, 0)

    def test_bool_seed_rejected(self):
        with self.assertRaises(SceneValidationError) as ctx:
            estimate(self.scene, [1.0, 1.0], 1, 1, False)
        self.assertIn("seed", str(ctx.exception))

    def test_float_seed_rejected(self):
        with self.assertRaises(SceneValidationError):
            estimate(self.scene, [1.0, 1.0], 1, 1, 3.5)


class TestSceneLimits(unittest.TestCase):
    def test_empty_scene_rejected(self):
        with self.assertRaises(SceneValidationError):
            build_scene([], NORMAL)

    def test_over_limit_scene_rejected(self):
        patches = [Patch(UP, 1.0, 0.1)] * (MAX_PATCHES + 1)
        with self.assertRaises(SceneValidationError) as ctx:
            build_scene(patches, NORMAL)
        self.assertIn(str(MAX_PATCHES), str(ctx.exception))

    def test_at_limit_scene_accepted(self):
        patches = [Patch(UP, 1.0, 0.1)] * MAX_PATCHES
        scene = build_scene(patches, NORMAL)
        self.assertEqual(scene.size, MAX_PATCHES)

    def test_non_patch_element_rejected(self):
        with self.assertRaises(SceneValidationError) as ctx:
            build_scene([Patch(UP, 1.0, 0.5), ("not", "a", "patch")], NORMAL)
        self.assertIn("patches[1]", str(ctx.exception))

    def test_non_scene_rejected(self):
        with self.assertRaises(SceneValidationError):
            exact_integral("not a scene", [1.0])
        with self.assertRaises(SceneValidationError):
            mis_contributions(None, [1.0], 1, 1)


class TestDegenerateStrategies(unittest.TestCase):
    def test_zero_energy_radiance_request_rejected(self):
        scene = build_scene([Patch(UP, 0.0, 0.5), Patch(SIDE, 0.0, 0.5)], NORMAL)
        with self.assertRaises(DegenerateStrategyError) as ctx:
            estimate(scene, [1.0, 1.0], 1, 1, 0)
        self.assertIn("radiance", str(ctx.exception))

    def test_zero_energy_cosine_only_estimate_is_exactly_zero(self):
        scene = build_scene([Patch(UP, 0.0, 0.5), Patch(SIDE, 0.0, 0.5)], NORMAL)
        result = estimate(scene, [3.0, 4.0], 0, 4, 123)
        self.assertEqual(result.estimate, 0.0)
        self.assertEqual(result.variance, 0.0)
        self.assertEqual(exact_integral(scene, [3.0, 4.0]), 0.0)
        codes = [d.code for d in result.diagnostics]
        self.assertIn("strategy-degenerate-radiance", codes)

    def test_all_below_horizon_cosine_request_rejected(self):
        down = (0.0, 0.0, -1.0)
        scene = build_scene([Patch(down, 2.0, 0.5)], NORMAL)
        with self.assertRaises(DegenerateStrategyError) as ctx:
            estimate(scene, [1.0], 0, 1, 0)
        self.assertIn("cosine", str(ctx.exception))

    def test_all_below_horizon_radiance_only_estimate_is_zero(self):
        down = (0.0, 0.0, -1.0)
        scene = build_scene([Patch(down, 2.0, 0.5)], NORMAL)
        result = estimate(scene, [1.0], 2, 0, 7)
        self.assertEqual(result.estimate, 0.0)


class TestFailureLeavesNoPartialResult(unittest.TestCase):
    def test_failed_estimate_returns_nothing(self):
        # 校验失败发生在抽样之前：抛异常即无任何返回值/副作用对象。
        scene = build_scene(simple_patches(), NORMAL)
        with self.assertRaises(SceneValidationError):
            estimate(scene, [1.0, float("nan")], 4, 4, 0)
        # 同参数修正后可正常得到完整结果，说明无残留状态。
        good = estimate(scene, [1.0, 1.0], 4, 4, 0)
        self.assertEqual(len(good.samples), 8)


if __name__ == "__main__":
    unittest.main()
