"""估计器正确性测试：穷举期望、单亮片、零能量、样本数不等、逐样本权重。

独立参考（精确值、提议概率）在测试内按定义手工计算；
被测核心仅提供逐片贡献表与抽样结果，枚举与加权求和由测试完成。
"""

import itertools
import math
import unittest

from mis_envlight import (
    COSINE,
    RADIANCE,
    Patch,
    build_scene,
    diagnose,
    estimate,
    exact_integral,
    mis_contributions,
)

NORMAL = (0.0, 0.0, 1.0)

# 5 片场景：方向均为单位向量，cos 手工算出为 [1, .8, .8, .8, .8]。
DIRS = [
    (0.0, 0.0, 1.0),
    (0.6, 0.0, 0.8),
    (0.0, 0.6, 0.8),
    (-0.6, 0.0, 0.8),
    (0.0, -0.6, 0.8),
]
COS = [1.0, 0.8, 0.8, 0.8, 0.8]
RAD = [2.0, 0.0, 3.5, 0.5, 1.0]
WGT = [0.4, 0.4, 0.4, 0.4, 0.4]
F = [1.0, 0.5, 2.0, 0.25, 1.5]

# 独立参考：按定义直接求和。
EXACT_REF = math.fsum(RAD[i] * F[i] * COS[i] * WGT[i] for i in range(5))
SUM_L = math.fsum(RAD)
SUM_C = math.fsum(COS)
P_L_REF = [l / SUM_L for l in RAD]
P_C_REF = [c / SUM_C for c in COS]


def scene_b():
    return build_scene(
        [Patch(DIRS[i], RAD[i], WGT[i]) for i in range(5)], NORMAL
    )


class TestExactIntegral(unittest.TestCase):
    def test_hand_computed_two_patch(self):
        # t0 = 2*3*1*0.5 = 3；t1 = 5*7*0*0.5 = 0（cos=0）；t2 = 1*10*0.8*0.25 = 2
        patches = [
            Patch((0.0, 0.0, 1.0), 2.0, 0.5),
            Patch((0.0, 1.0, 0.0), 5.0, 0.5),
            Patch((0.0, 0.6, 0.8), 1.0, 0.25),
        ]
        scene = build_scene(patches, NORMAL)
        self.assertEqual(exact_integral(scene, [3.0, 7.0, 10.0]), 5.0)

    def test_scene_b_matches_reference(self):
        self.assertEqual(exact_integral(scene_b(), F), EXACT_REF)


class TestExhaustiveExpectation(unittest.TestCase):
    """穷举所有一次抽样组合：估计量期望必须等于直接求和。

    对 (n_radiance, n_cosine) 的每种组合，枚举全部抽样结果序列，
    用测试内独立算出的提议概率加权，与被测核心的逐片贡献表相乘求和。
    """

    def check_exhaustive(self, n_rad, n_cos):
        table = mis_contributions(scene_b(), F, n_rad, n_cos)
        total = 0.0
        draws = [RADIANCE] * n_rad + [COSINE] * n_cos
        for outcome in itertools.product(range(5), repeat=len(draws)):
            prob = 1.0
            value = 0.0
            for strategy, idx in zip(draws, outcome):
                if strategy == RADIANCE:
                    prob *= P_L_REF[idx]
                    contrib = table.radiance_contribution[idx]
                else:
                    prob *= P_C_REF[idx]
                    contrib = table.cosine_contribution[idx]
                if prob == 0.0:
                    break  # 零概率结果（贡献表对应位置为 None）不进入期望
                self.assertIsNotNone(contrib)
                value += contrib
            else:
                total += prob * value
        self.assertTrue(
            math.isclose(total, EXACT_REF, rel_tol=1e-12, abs_tol=1e-12),
            f"n=({n_rad},{n_cos}): exhaustive expectation {total!r} != {EXACT_REF!r}",
        )

    def test_one_sample_each_strategy(self):
        self.check_exhaustive(1, 1)

    def test_unequal_counts_exhaustive(self):
        self.check_exhaustive(2, 1)
        self.check_exhaustive(1, 2)
        self.check_exhaustive(2, 2)

    def test_single_strategy_degenerates_to_plain_is(self):
        # 只用 radiance 策略时每样本贡献应退化为 t_i / (n * p_L[i])
        # （n 个样本求和即经典形式 (1/n)Σ t/p 的无偏估计）。
        table = mis_contributions(scene_b(), F, 3, 0)
        for i in range(5):
            if P_L_REF[i] > 0:
                t_i = RAD[i] * F[i] * COS[i] * WGT[i]
                self.assertTrue(
                    math.isclose(
                        table.radiance_contribution[i],
                        t_i / (3 * P_L_REF[i]),
                        rel_tol=1e-15,
                    )
                )
            else:
                self.assertIsNone(table.radiance_contribution[i])


class TestSingleBrightPatch(unittest.TestCase):
    def setUp(self):
        # 只有第 0 片有辐亮度；三片 cos 均 > 0。
        self.patches = [
            Patch((0.0, 0.0, 1.0), 4.0, 0.5),
            Patch((0.6, 0.0, 0.8), 0.0, 0.5),
            Patch((0.0, 0.6, 0.8), 0.0, 0.5),
        ]
        self.f = [1.0, 2.0, 3.0]
        self.scene = build_scene(self.patches, NORMAL)
        # 独立参考：t = 4*1*1*0.5 = 2，其余为 0。
        self.exact = 2.0

    def test_radiance_strategy_concentrated(self):
        table = mis_contributions(self.scene, self.f, 1, 1)
        self.assertEqual(table.p_radiance, (1.0, 0.0, 0.0))

    def test_all_radiance_draws_hit_bright_patch(self):
        result = estimate(self.scene, self.f, 4, 4, 99)
        for rec in result.samples:
            if rec.strategy == RADIANCE:
                self.assertEqual(rec.patch_index, 0)

    def test_estimate_unbiased_exhaustive(self):
        # 单样本组合穷举：radiance 必中 0 片，cosine 三片都可能。
        table = mis_contributions(self.scene, self.f, 1, 1)
        p_cos_ref = [1.0 / 2.6, 0.8 / 2.6, 0.8 / 2.6]
        total = 0.0
        for j in range(3):
            total += p_cos_ref[j] * (
                table.radiance_contribution[0] + table.cosine_contribution[j]
            )
        self.assertTrue(math.isclose(total, self.exact, rel_tol=1e-12))
        self.assertEqual(exact_integral(self.scene, self.f), self.exact)


class TestUnequalSampleCounts(unittest.TestCase):
    def test_unequal_counts_deterministic_and_complete(self):
        scene = scene_b()
        r1 = estimate(scene, F, 3, 9, 20260922)
        r2 = estimate(scene, F, 3, 9, 20260922)
        self.assertEqual(r1.estimate, r2.estimate)  # 确定种子 → 确定结果
        self.assertEqual(len(r1.samples), 12)
        self.assertEqual(r1.n_samples, 12)
        # 抽样顺序固定：先 radiance 后 cosine。
        self.assertTrue(all(s.strategy == RADIANCE for s in r1.samples[:3]))
        self.assertTrue(all(s.strategy == COSINE for s in r1.samples[3:]))
        self.assertIsNotNone(r1.variance)
        self.assertGreaterEqual(r1.variance, 0.0)
        self.assertAlmostEqual(r1.std_error, math.sqrt(r1.variance))

    def test_different_seeds_differ(self):
        scene = scene_b()
        a = estimate(scene, F, 2, 2, 1)
        b = estimate(scene, F, 2, 2, 2)
        self.assertNotEqual(a.estimate, b.estimate)

    def test_single_draw_per_strategy_variance_unavailable(self):
        scene = scene_b()
        result = estimate(scene, F, 1, 1, 0)
        self.assertIsNone(result.variance)
        self.assertIsNone(result.std_error)


class TestPerSampleWeightExplanation(unittest.TestCase):
    def test_records_are_self_consistent(self):
        scene = scene_b()
        n_rad, n_cos = 4, 6
        result = estimate(scene, F, n_rad, n_cos, 42)
        for rec in result.samples:
            # balance heuristic 权重在 (0, 1] 内。
            self.assertGreater(rec.mis_weight, 0.0)
            self.assertLessEqual(rec.mis_weight, 1.0 + 1e-12)
            pdf_self = (
                rec.pdf_radiance if rec.strategy == RADIANCE else rec.pdf_cosine
            )
            self.assertGreater(pdf_self, 0.0)
            # contribution == mis_weight * target / (n_strategy * pdf_self)。
            n_self = n_rad if rec.strategy == RADIANCE else n_cos
            expect = rec.mis_weight * rec.target / (n_self * pdf_self)
            self.assertTrue(
                math.isclose(rec.contribution, expect, rel_tol=1e-12, abs_tol=1e-15),
                f"record {rec}: contribution {rec.contribution!r} != {expect!r}",
            )
            # 权重公式与样本数记账一致。
            pdf_other = (
                rec.pdf_cosine if rec.strategy == RADIANCE else rec.pdf_radiance
            )
            n_other = n_cos if rec.strategy == RADIANCE else n_rad
            den = n_self * pdf_self + n_other * pdf_other
            self.assertTrue(
                math.isclose(rec.mis_weight, n_self * pdf_self / den, rel_tol=1e-12)
            )
            text = rec.explain()
            self.assertIn(f"strategy={rec.strategy}", text)
            self.assertIn(f"patch={rec.patch_index}", text)
        # 估计值等于逐样本贡献之和。
        self.assertEqual(
            result.estimate, math.fsum(r.contribution for r in result.samples)
        )


class TestDiagnostics(unittest.TestCase):
    def test_zero_probability_term_reported(self):
        # 场景 B 第 1 片 L=0 但 cos>0：radiance 策略零概率项。
        diags = diagnose(scene_b(), F)
        zero_terms = [d for d in diags if d.code == "zero-probability-term"]
        self.assertEqual([d.patch_index for d in zero_terms], [1])
        self.assertIn("radiance", zero_terms[0].message)

    def test_no_unreachable_terms_in_valid_scene(self):
        diags = diagnose(scene_b(), F)
        self.assertFalse(any(d.code == "term-unreachable" for d in diags))

    def test_diagnostics_attached_to_estimate(self):
        result = estimate(scene_b(), F, 2, 2, 0)
        self.assertEqual(
            [d.code for d in result.diagnostics],
            [d.code for d in diagnose(scene_b(), F)],
        )


if __name__ == "__main__":
    unittest.main()
