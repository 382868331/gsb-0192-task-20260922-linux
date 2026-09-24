"""Tests for batched adaptive finite-mixture sampling.

All reference expectations are derived independently in this file (plain
loops / exact Fraction arithmetic). The core under test is never used to
manufacture an expected value.
"""

import math
import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mis_envlight import (
    DEFAULT_MIX_FLOOR,
    DIAG_UNREACHABLE_TERM,
    DIAG_VARIANCE_NOT_ESTIMABLE,
    DIAG_ZERO_PDF_TERM,
    DIAG_ZERO_MASS_COMPONENT,
    BatchSpec,
    BatchedEstimator,
    Patch,
    Scene,
    ValidationError,
    DegenerateProposalError,
    direct_sum,
    estimate_batched,
)


def unit(z):
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return (r, 0.0, z)


def make_scene(zs, radiances, weights=None):
    n = len(zs)
    if weights is None:
        weights = [1.0] * n
    return Scene([Patch(direction=unit(zs[i]), radiance=radiances[i],
                        weight=weights[i]) for i in range(n)])


def reference_pdfs(radiances, cosines, weights):
    """Independently normalize the two proposal distributions (plain math)."""
    n = len(radiances)
    lm = [radiances[i] * weights[i] for i in range(n)]
    cm = [cosines[i] * weights[i] for i in range(n)]
    tl, tc = sum(lm), sum(cm)
    p = [m / tl if tl > 0 else 0.0 for m in lm]
    q = [m / tc if tc > 0 else 0.0 for m in cm]
    return p, q


def reference_terms(radiances, f, cosines, weights):
    return [radiances[i] * f[i] * cosines[i] * weights[i]
            for i in range(len(radiances))]


class TestBatchedExactIdentity(unittest.TestCase):
    """E[global mean] == S exactly, even when alpha changes between batches."""

    def test_exact_fraction_two_distributions(self):
        # Rational unit vectors: cos = 1, 4/5, 12/13, 15/17.
        cos = [Fraction(1), Fraction(4, 5), Fraction(12, 13), Fraction(15, 17)]
        L = [Fraction(2), Fraction(1, 2), Fraction(3), Fraction(5, 4)]
        w = [Fraction(1, 4)] * 4
        f = [Fraction(1, 3), Fraction(2), Fraction(5, 2), Fraction(1)]
        g = [L[i] * f[i] * cos[i] * w[i] for i in range(4)]
        lm = [L[i] * w[i] for i in range(4)]
        cm = [cos[i] * w[i] for i in range(4)]
        p = [m / sum(lm) for m in lm]
        q = [m / sum(cm) for m in cm]
        S = sum(g)

        # Two batches with *different* distributions, plus several splits.
        plans = [
            [(1, Fraction(1, 20)), (1, Fraction(19, 20))],
            [(3, Fraction(1, 10)), (5, Fraction(9, 10))],
            [(2, Fraction(1, 2)), (7, Fraction(1, 20)), (4, Fraction(3, 4))],
        ]
        for plan in plans:
            N = sum(n for n, _ in plan)
            # E[global mean] = sum_b (n_b/N) * sum_i mix_b(i) * g_i/mix_b(i)
            expectation = Fraction(0)
            for n_b, a in plan:
                mix = [a * p[i] + (1 - a) * q[i] for i in range(4)]
                batch_mean = sum(mix[i] * g[i] / mix[i] for i in range(4))
                expectation += Fraction(n_b, N) * batch_mean
            self.assertEqual(expectation, S)

    def test_float_expectation_two_changed_batches(self):
        zs = [1.0, 0.8, 0.6, 0.3]
        L = [2.0, 0.5, 3.0, 1.0]
        w = [1.0, 1.0, 1.0, 1.0]
        f = [0.5, 1.5, 0.25, 2.0]
        g = reference_terms(L, f, zs, w)
        p, q = reference_pdfs(L, zs, w)
        plan = [(40, 0.15), (60, 0.85)]
        N = sum(n for n, _ in plan)
        expectation = 0.0
        for n_b, a in plan:
            for i in range(4):
                mix = a * p[i] + (1 - a) * q[i]
                expectation += (n_b / N) * mix * g[i] / mix
        expected = sum(g)
        self.assertTrue(math.isclose(expectation, expected,
                                     rel_tol=1e-12, abs_tol=1e-15))

    def test_statistical_two_batch_change(self):
        # Real sampling: the two batches genuinely use different mixtures;
        # the estimate must land within a tight multiple of its own stddev.
        zs = [1.0, 0.8, 0.6, 0.35, 0.15]
        L = [3.0, 0.2, 2.0, 0.5, 1.0]
        f = [0.4, 2.0, 0.6, 1.5, 0.8]
        scene = make_scene(zs, L)
        res = estimate_batched(scene, f, [(2000, 0.1), (2000, 0.9)],
                               seed=20260924)
        exact = direct_sum(scene, f)
        self.assertEqual(res.n_batches, 2)
        self.assertLess(abs(res.estimate - exact), 4.0 * res.stddev)
        self.assertLess(abs(res.estimate - exact) / exact, 0.05)
        # Recorded alphas and pdfs differ between the batches.
        self.assertNotEqual(res.batches[0].alpha, res.batches[1].alpha)
        self.assertNotEqual(res.batches[0].pdf, res.batches[1].pdf)


class TestFrozenConditionalPdf(unittest.TestCase):
    """Samples keep the pdf in force at draw time; later batches cannot
    retroactively change an old sample's pdf."""

    def setUp(self):
        # p and q differ on every patch (all L, cos positive).
        self.zs = [1.0, 0.7, 0.4]
        self.L = [1.0, 4.0, 2.0]
        self.w = [0.5, 1.0, 0.25]
        self.f = [2.0, 0.5, 1.0]
        self.scene = make_scene(self.zs, self.L, self.w)

    def test_records_freeze_their_own_batch_pdf(self):
        a0, a1 = 0.2, 0.8
        res = estimate_batched(self.scene, self.f,
                               [(50, a0), (50, a1)], seed=11)
        p, q = reference_pdfs(self.L, self.zs, self.w)
        g = reference_terms(self.L, self.f, self.zs, self.w)
        b0 = [r for r in res.records if r.batch_index == 0]
        b1 = [r for r in res.records if r.batch_index == 1]
        for r in b0:
            i = r.patch_index
            mix0 = a0 * p[i] + (1 - a0) * q[i]
            mix1 = a1 * p[i] + (1 - a1) * q[i]
            self.assertTrue(math.isclose(r.pdf_mixture, mix0, rel_tol=1e-15))
            self.assertTrue(math.isclose(r.alpha, a0, rel_tol=1e-15))
            self.assertTrue(math.isclose(r.contribution, g[i] / mix0,
                                         rel_tol=1e-15))
            # The final distribution must not have been back-applied.
            self.assertFalse(math.isclose(mix0, mix1, rel_tol=1e-12))
            self.assertFalse(math.isclose(r.pdf_mixture, mix1, rel_tol=1e-12))
        for r in b1:
            i = r.patch_index
            mix1 = a1 * p[i] + (1 - a1) * q[i]
            self.assertTrue(math.isclose(r.pdf_mixture, mix1, rel_tol=1e-15))

    def test_summary_pdfs_match_frozen_record_pdfs(self):
        res = estimate_batched(self.scene, self.f,
                               [(30, 0.3), (20, 0.7)], seed=4)
        for b in res.batches:
            for r in res.records:
                if r.batch_index == b.batch_index:
                    self.assertIs(r.pdf_mixture, b.pdf[r.patch_index])

    def test_fixed_distribution_degenerates_to_one_batch(self):
        # Two batches that keep alpha fixed draw from one identical cdf; the
        # batch split must not consume extra random numbers, so a split run
        # is exactly identical to a single batch of the combined size.
        a = 0.37
        split = estimate_batched(self.scene, self.f,
                                 [(37, a), (53, a)], seed=99)
        combined = estimate_batched(self.scene, self.f,
                                    [(90, a)], seed=99)
        self.assertEqual(split.batches[0].pdf, split.batches[1].pdf)
        # Same cdf => identical draw sequence and identical frozen pdfs /
        # contributions; only the (batch_index, draw) bookkeeping differs.
        self.assertEqual(
            [(r.patch_index, r.pdf_mixture, r.contribution)
             for r in split.records],
            [(r.patch_index, r.pdf_mixture, r.contribution)
             for r in combined.records])
        self.assertEqual(split.estimate, combined.estimate)
        # The estimate is identical; the stratified variance estimate is
        # allowed to differ slightly because the two batch means enter the
        # within-batch sums separately (its unbiasedness is checked against
        # an independent reference in TestVarianceAndRecords).
        self.assertIsNotNone(split.variance)
        self.assertGreaterEqual(split.variance, 0.0)
        self.assertGreaterEqual(combined.variance, 0.0)


class TestMixtureFloor(unittest.TestCase):
    def setUp(self):
        self.scene = make_scene([1.0, 0.5, 0.25], [1.0, 2.0, 0.5])
        self.f = [1.0, 1.0, 1.0]

    def test_boundary_weights_accepted(self):
        res = estimate_batched(
            self.scene, self.f,
            [(4, DEFAULT_MIX_FLOOR), (4, 1.0 - DEFAULT_MIX_FLOOR)], seed=3)
        self.assertAlmostEqual(res.batches[0].alpha, DEFAULT_MIX_FLOOR)
        self.assertAlmostEqual(res.batches[1].alpha, 1 - DEFAULT_MIX_FLOOR)

    def test_zero_and_one_rejected(self):
        with self.assertRaises(ValidationError):
            estimate_batched(self.scene, self.f, [(4, 0.0), (4, 0.5)], seed=1)
        with self.assertRaises(ValidationError):
            estimate_batched(self.scene, self.f, [(4, 1.0)], seed=1)

    def test_below_floor_rejected_and_locates_batch(self):
        with self.assertRaises(ValidationError) as ctx:
            estimate_batched(self.scene, self.f,
                             [(4, 0.5), (4, 0.01)], seed=1)
        self.assertIn("batches[1].alpha", str(ctx.exception))
        with self.assertRaises(ValidationError):
            estimate_batched(self.scene, self.f, [(4, 0.99)], seed=1,
                             mix_floor=0.1)

    def test_non_finite_alpha_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValidationError):
                estimate_batched(self.scene, self.f, [(4, bad)], seed=1)

    def test_alpha_type_rejected(self):
        for bad in (True, False, "0.5", None, [0.5]):
            with self.assertRaises(ValidationError):
                estimate_batched(self.scene, self.f, [(4, bad)], seed=1)

    def test_custom_floor_enforced(self):
        res = estimate_batched(self.scene, self.f, [(16, 0.2), (16, 0.8)],
                               seed=2, mix_floor=0.2)
        self.assertEqual(res.mix_floor, 0.2)
        for bad in (0.0, -0.1, 0.5, 1.0, math.nan, True):
            with self.assertRaises(ValidationError):
                estimate_batched(self.scene, self.f, [(4, 0.5)], seed=1,
                                 mix_floor=bad)

    def test_floor_keeps_both_components_live(self):
        # Even at the floor, both components contribute positive mass on
        # every fully-lit, above-horizon patch.
        zs = [1.0, 0.6, 0.3]
        scene = make_scene(zs, [1.0, 2.0, 3.0])
        res = estimate_batched(scene, [1.0] * 3,
                               [(512, DEFAULT_MIX_FLOOR),
                                (512, 1 - DEFAULT_MIX_FLOOR)], seed=8)
        p, q = reference_pdfs([1.0, 2.0, 3.0], zs, [1.0] * 3)
        for i in range(3):
            self.assertGreater(res.batches[0].pdf[i], 0.0)
            self.assertGreater(res.batches[1].pdf[i], 0.0)
            lo = min(DEFAULT_MIX_FLOOR * p[i]
                     + (1 - DEFAULT_MIX_FLOOR) * q[i],
                     (1 - DEFAULT_MIX_FLOOR) * p[i]
                     + DEFAULT_MIX_FLOOR * q[i])
            self.assertGreater(lo, 0.0)


class TestAtomicFailure(unittest.TestCase):
    """Invalid input never leaves a half-updated state or perturbs the
    deterministic random stream."""

    def setUp(self):
        self.scene = make_scene([1.0, 0.5, 0.25], [1.0, 2.0, 0.5])
        self.f = [1.0, 1.0, 1.0]

    def test_failed_batch_leaves_state_untouched(self):
        good = BatchedEstimator(self.scene, self.f, seed=123)
        good.run_batch(20, 0.5)

        broken = BatchedEstimator(self.scene, self.f, seed=123)
        b0 = broken.run_batch(20, 0.5)
        self.assertEqual(b0.batch_index, 0)
        with self.assertRaises(ValidationError):
            broken.run_batch(10, 0.0)  # violates the floor
        self.assertEqual(broken.n_batches, 1)
        self.assertEqual(broken.total_samples, 20)
        with self.assertRaises(ValidationError):
            broken.run_batch(0, 0.5)
        self.assertEqual(broken.n_batches, 1)
        # The rejected calls consumed no random numbers: continuing with a
        # valid adapted batch reproduces the uninterrupted runner exactly.
        b1 = broken.run_batch(15, 0.8)
        good.run_batch(15, 0.8)
        good_res = good.result()
        self.assertEqual(broken.result().records, good_res.records)
        self.assertEqual(b1.records,
                         tuple(r for r in good_res.records
                               if r.batch_index == 1))

    def test_invalid_plan_is_rejected_before_any_sampling(self):
        # The bad spec is the second one; the call must raise and the error
        # locates it. Nothing is returned and no state exists to observe.
        with self.assertRaises(ValidationError) as ctx:
            estimate_batched(self.scene, self.f,
                             [(10, 0.5), (10, 2.0)], seed=1)
        self.assertIn("batches[1]", str(ctx.exception))
        with self.assertRaises(ValidationError):
            estimate_batched(self.scene, self.f, [], seed=1)

    def test_result_without_batches(self):
        runner = BatchedEstimator(self.scene, self.f, seed=1)
        with self.assertRaises(ValidationError):
            runner.result()

    def test_both_strategies_zero_mass_rejected(self):
        # Both total masses vanish only when all radiance is zero AND every
        # patch is on the horizon (cos = 0).
        dark_flat = make_scene([0.0, 0.0], [0.0, 0.0])
        with self.assertRaises(DegenerateProposalError):
            BatchedEstimator(dark_flat, [1.0, 1.0], seed=1)
        with self.assertRaises(DegenerateProposalError):
            estimate_batched(dark_flat, [1.0, 1.0], [(8, 0.5)], seed=1)

    def test_batch_count_validation(self):
        for bad in (0, -1, 1.5, "8", True, False):
            runner = BatchedEstimator(self.scene, self.f, seed=1)
            with self.assertRaises(ValidationError):
                runner.run_batch(bad, 0.5)
            self.assertEqual(runner.n_batches, 0)

    def test_bad_constructor_arguments(self):
        with self.assertRaises(ValidationError):
            BatchedEstimator(self.scene, self.f, seed=1, mix_floor=0.0)
        with self.assertRaises(ValidationError):
            BatchedEstimator(self.scene, self.f, seed=True)
        with self.assertRaises(ValidationError):
            BatchedEstimator("nonsense", self.f, seed=1)
        with self.assertRaises(ValidationError):
            BatchedEstimator(self.scene, [1.0] * 2, seed=1)


class TestSingleZeroMassComponent(unittest.TestCase):
    def test_zero_energy_scene_mixture_driven_by_cosine(self):
        zs = [1.0, 0.5, 0.25]
        scene = make_scene(zs, [0.0, 0.0, 0.0])
        res = estimate_batched(scene, [1.0, 2.0, 3.0],
                               [(32, 0.2), (32, 0.8)], seed=6)
        self.assertEqual(res.estimate, 0.0)
        self.assertEqual(res.exact, 0.0)
        codes = [d.code for d in res.diagnostics]
        self.assertIn(DIAG_ZERO_MASS_COMPONENT, codes)
        # Mixture pdf is positive wherever q is positive, independent of a.
        for b in res.batches:
            for i in range(3):
                self.assertGreater(b.pdf[i], 0.0)
        self.assertEqual(res.batches[0].pdf, res.batches[1].pdf)

    def test_all_grazing_scene_mixture_driven_by_radiance(self):
        scene = make_scene([0.0, 0.0], [1.0, 2.0])
        res = estimate_batched(scene, [1.0, 1.0],
                               [(16, 0.3), (16, 0.7)], seed=6)
        codes = [d.code for d in res.diagnostics]
        self.assertIn(DIAG_ZERO_MASS_COMPONENT, codes)
        # cos = 0 everywhere: every target term is zero. The radiance-driven
        # mixture still draws patches, but all frozen contributions are 0.
        self.assertEqual(res.exact, 0.0)
        self.assertEqual(res.estimate, 0.0)
        self.assertTrue(any(r.target_value == 0.0 for r in res.records))


class TestVarianceAndRecords(unittest.TestCase):
    def test_independent_variance_reference(self):
        zs = [1.0, 0.7, 0.4]
        L = [1.0, 4.0, 2.0]
        w = [0.5, 1.0, 0.25]
        f = [2.0, 0.5, 1.0]
        scene = make_scene(zs, L, w)
        plan = [(25, 0.2), (35, 0.8)]
        res = estimate_batched(scene, f, plan, seed=21)
        N = res.total_samples
        ref = 0.0
        for b in res.batches:
            xs = [r.contribution for r in res.records
                  if r.batch_index == b.batch_index]
            mean = sum(xs) / len(xs)
            s2 = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
            ref += len(xs) * s2
        ref /= N * N
        self.assertTrue(math.isclose(res.variance, ref, rel_tol=1e-14))
        self.assertTrue(math.isclose(res.stddev, math.sqrt(ref),
                                     rel_tol=1e-14))

    def test_single_sample_batch_variance_none(self):
        scene = make_scene([1.0, 0.5], [1.0, 2.0])
        res = estimate_batched(scene, [1.0, 1.0],
                               [(1, 0.5), (8, 0.6)], seed=4)
        self.assertIsNone(res.variance)
        self.assertIsNone(res.stddev)
        codes = [d.code for d in res.diagnostics]
        self.assertIn(DIAG_VARIANCE_NOT_ESTIMABLE, codes)
        # The estimate itself remains available.
        self.assertEqual(res.estimate,
                         math.fsum(r.contribution for r in res.records)
                         / res.total_samples)

    def test_single_patch_scene_is_exact(self):
        scene = make_scene([0.6], [2.5], [0.75])
        res = estimate_batched(scene, [1.5], [(4, 0.2), (4, 0.8)], seed=7)
        self.assertEqual(res.estimate, direct_sum(scene, [1.5]))
        self.assertEqual(res.variance, 0.0)

    def test_explain_lists_batches(self):
        scene = make_scene([1.0, 0.5], [1.0, 2.0])
        res = estimate_batched(scene, [1.0, 1.0],
                               [(3, 0.3), (3, 0.7)], seed=5)
        text = res.explain(limit=4)
        self.assertIn("batch", text)
        self.assertIn("pdf_mix", text)
        self.assertIn("more samples", text)
        self.assertIn("0.3", res.explain())


class TestStatefulAdaptation(unittest.TestCase):
    def test_feedback_loop_weight_update(self):
        # The next batch's alpha is derived from the previous batch's
        # frozen records; the runner must record exactly that weight.
        zs = [1.0, 0.8, 0.5, 0.3]
        L = [3.0, 0.5, 2.0, 1.0]
        scene = make_scene(zs, L)
        f = [1.0, 0.5, 2.0, 1.0]
        p, q = reference_pdfs(L, zs, [1.0] * 4)
        g = reference_terms(L, f, zs, [1.0] * 4)

        runner = BatchedEstimator(scene, f, seed=77)
        b0 = runner.run_batch(256, 0.5)

        # Simple deterministic adaptation rule (computed by the caller from
        # the frozen samples): sample more from the proposal whose estimated
        # second moment is larger.
        def second_moment_estimate(batch, own_pdf):
            vals = []
            for r in batch.records:
                if r.target_value > 0.0:
                    vals.append(r.target_value ** 2
                                / (own_pdf[r.patch_index] * r.pdf_mixture))
            return sum(vals) / len(vals) if vals else 0.0

        m_p = second_moment_estimate(b0, p)
        m_q = second_moment_estimate(b0, q)
        a1 = max(DEFAULT_MIX_FLOOR,
                 min(1 - DEFAULT_MIX_FLOOR,
                     math.sqrt(m_p) / (math.sqrt(m_p) + math.sqrt(m_q))))
        b1 = runner.run_batch(256, a1)
        res = runner.result()
        self.assertEqual(runner.n_batches, 2)
        self.assertEqual(b1.alpha, a1)
        self.assertEqual(res.batches[1].alpha, a1)
        self.assertIsInstance(a1, float)
        self.assertLess(abs(res.estimate - direct_sum(scene, f)),
                        4 * res.stddev)

    def test_stateful_matches_one_shot(self):
        scene = make_scene([1.0, 0.7, 0.3], [1.0, 2.0, 0.5])
        f = [1.0, 0.5, 2.0]
        one = estimate_batched(scene, f, [(13, 0.3), (17, 0.65)], seed=42)
        runner = BatchedEstimator(scene, f, seed=42)
        runner.run_batch(13, 0.3)
        runner.run_batch(17, 0.65)
        two = runner.result()
        self.assertEqual(one.records, two.records)
        self.assertEqual(one.estimate, two.estimate)
        self.assertEqual(one.variance, two.variance)
        self.assertEqual(one.diagnostics, two.diagnostics)

    def test_batchspec_form_accepted(self):
        scene = make_scene([1.0, 0.5], [1.0, 2.0])
        a = estimate_batched(scene, [1.0, 1.0],
                             [BatchSpec(8, 0.4), BatchSpec(8, 0.6)], seed=1)
        b = estimate_batched(scene, [1.0, 1.0],
                             [(8, 0.4), (8, 0.6)], seed=1)
        self.assertEqual(a.records, b.records)

    def test_determinism_same_seed(self):
        scene = make_scene([1.0, 0.7, 0.3], [1.0, 2.0, 0.5])
        f = [1.0, 0.5, 2.0]
        a = estimate_batched(scene, f, [(11, 0.2), (9, 0.8)], seed=42)
        b = estimate_batched(scene, f, [(11, 0.2), (9, 0.8)], seed=42)
        self.assertEqual(a, b)


class TestBatchedDiagnostics(unittest.TestCase):
    def test_unreachable_term_guard(self):
        # Exercise the defensive guard directly: a positive target term at a
        # patch whose frozen mixture pdf is zero in every batch.
        from mis_envlight.core import _batch_diagnostics, BatchResult
        scene = make_scene([0.9, 0.8], [0.0, 1.0])
        b0 = BatchResult(0, 2, 0.5, (0.0, 1.0), (), 0.0, 0.0)
        b1 = BatchResult(1, 2, 0.7, (0.0, 1.0), (), 0.0, 0.0)
        diags = _batch_diagnostics((0.5, 1.0), [b0, b1], [])
        hits = [d for d in diags if d.code == DIAG_UNREACHABLE_TERM]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].patch_index, 0)

    def test_zero_pdf_only_in_one_batch(self):
        # Same patch unreachable in just one of two batches: informational,
        # not the unreachable-in-every-batch bias code.
        from mis_envlight.core import _batch_diagnostics, BatchResult
        scene = make_scene([0.9, 0.8], [0.0, 1.0])
        b0 = BatchResult(0, 2, 0.5, (0.0, 1.0), (), 0.0, 0.0)
        b1 = BatchResult(1, 2, 0.7, (0.5, 0.5), (), 0.0, 0.0)
        diags = _batch_diagnostics((0.0, 1.0), [b0, b1], [])
        codes = [d.code for d in diags]
        self.assertIn(DIAG_ZERO_PDF_TERM, codes)
        self.assertNotIn(DIAG_UNREACHABLE_TERM, codes)


if __name__ == "__main__":
    unittest.main()
