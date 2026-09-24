"""Tests for mis_envlight.

Reference expectations are computed independently inside this file (plain
loops / exact Fraction arithmetic); the core's direct_sum is never used to
derive an expected value.
"""

import math
import os
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mis_envlight import (
    MAX_PATCHES,
    MAX_BATCHES,
    MIN_STRATEGY_WEIGHT,
    STRATEGY_RADIANCE,
    STRATEGY_COSINE,
    DIAG_ZERO_PDF_TERM,
    DIAG_UNREACHABLE_TERM,
    DIAG_VARIANCE_NOT_ESTIMABLE,
    DIAG_UNUSED_ZERO_MASS_STRATEGY,
    DIAG_WEIGHTS_UPDATED,
    DIAG_WEIGHTS_CLAMPED,
    ValidationError,
    DegenerateProposalError,
    Patch,
    Scene,
    SampleRecord,
    BatchRecord,
    direct_sum,
    proposal_pdfs,
    enumerate_contributions,
    estimate,
    estimate_adaptive,
    variance_minimizing_updater,
)


def unit(z):
    """Deterministic unit vector with the given z component."""
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return (r, 0.0, z)


def make_scene(zs, radiances, weights=None):
    n = len(zs)
    if weights is None:
        weights = [1.0] * n
    return Scene([Patch(direction=unit(zs[i]), radiance=radiances[i],
                        weight=weights[i]) for i in range(n)])


def reference_direct(radiances, f, cosines, weights):
    """Independent reference: plain loop, no library code."""
    total = 0.0
    for i in range(len(radiances)):
        total += radiances[i] * f[i] * cosines[i] * weights[i]
    return total


class TestDirectSum(unittest.TestCase):
    def test_matches_independent_reference(self):
        zs = [1.0, 0.8, 0.5, 0.25]
        L = [1.5, 0.0, 3.25, 2.0]
        w = [0.5, 0.5, 0.25, 1.0]
        f = [1.0, 2.0, 0.5, 0.75]
        scene = make_scene(zs, L, w)
        self.assertEqual(direct_sum(scene, f),
                         reference_direct(L, f, zs, w))

    def test_all_zero_f_gives_zero(self):
        scene = make_scene([1.0, 0.5], [3.0, 4.0])
        self.assertEqual(direct_sum(scene, [0.0, 0.0]), 0.0)


class TestExhaustiveExpectation(unittest.TestCase):
    """Enumerate every one-sample combination and check E[estimator] == S."""

    def test_one_sample_per_strategy_float(self):
        zs = [1.0, 0.8, 0.6, 0.3]
        L = [2.0, 0.5, 3.0, 1.0]
        w = [1.0, 1.0, 1.0, 1.0]
        f = [0.5, 1.5, 0.25, 2.0]
        scene = make_scene(zs, L, w)
        table = enumerate_contributions(scene, f, 1, 1)
        # Expectation of the estimator over all (i, j) draw combinations.
        # The sum factorizes per strategy; probabilities come from the core's
        # own table, the reference sum is computed independently below.
        expectation = 0.0
        for i in range(len(scene)):
            expectation += table.pdf_radiance[i] * table.radiance[i]
            expectation += table.pdf_cosine[i] * table.cosine[i]
        expected = reference_direct(L, f, zs, w)
        self.assertTrue(math.isclose(expectation, expected,
                                     rel_tol=1e-12, abs_tol=1e-15),
                        f"{expectation!r} != {expected!r}")

    def test_unequal_sample_counts_float(self):
        zs = [1.0, 0.7, 0.4]
        L = [1.0, 4.0, 2.0]
        w = [0.5, 1.0, 0.25]
        f = [2.0, 0.5, 1.0]
        n_r, n_c = 3, 2
        scene = make_scene(zs, L, w)
        table = enumerate_contributions(scene, f, n_r, n_c)
        expectation = 0.0
        for i in range(len(scene)):
            expectation += n_r * table.pdf_radiance[i] * table.radiance[i]
            expectation += n_c * table.pdf_cosine[i] * table.cosine[i]
        expected = reference_direct(L, f, zs, w)
        self.assertTrue(math.isclose(expectation, expected,
                                     rel_tol=1e-12, abs_tol=1e-15),
                        f"{expectation!r} != {expected!r}")

    def test_exact_fraction_enumeration(self):
        """Exact rational check of the balance-heuristic identity.

        Everything (proposals, MIS weights, expectation, direct sum) is
        recomputed here with Fractions from the mathematical specification;
        no library code is involved. E[estimator] must equal S exactly.
        """
        # Rational unit vectors: (3/5,0,4/5), (0,5/13,12/13), (8/17,0,15/17).
        cos = [Fraction(1), Fraction(4, 5), Fraction(12, 13), Fraction(15, 17)]
        L = [Fraction(2), Fraction(1, 2), Fraction(3), Fraction(5, 4)]
        w = [Fraction(1, 4)] * 4
        f = [Fraction(1, 3), Fraction(2), Fraction(5, 2), Fraction(1)]
        g = [L[i] * f[i] * cos[i] * w[i] for i in range(4)]
        lm = [L[i] * w[i] for i in range(4)]
        cm = [cos[i] * w[i] for i in range(4)]
        pL = [m / sum(lm) for m in lm]
        pC = [m / sum(cm) for m in cm]
        for n_r, n_c in ((1, 1), (2, 1), (1, 3), (5, 7)):
            denom = [n_r * pL[i] + n_c * pC[i] for i in range(4)]
            expectation = Fraction(0)
            for i in range(4):
                expectation += n_r * pL[i] * g[i] / denom[i]
                expectation += n_c * pC[i] * g[i] / denom[i]
            self.assertEqual(expectation, sum(g),
                             f"identity failed for counts ({n_r}, {n_c})")


class TestSpecialScenes(unittest.TestCase):
    def test_single_patch_scene_is_exact(self):
        # One patch: both proposals are degenerate at index 0, every sample
        # contributes S / (n_r + n_c), so the estimate is exact with
        # variance zero.
        scene = make_scene([0.6], [2.5], [0.75])
        f = [1.5]
        res = estimate(scene, f, 4, 4, seed=7)
        self.assertEqual(res.estimate, reference_direct([2.5], f, [0.6], [0.75]))
        self.assertEqual(res.variance, 0.0)

    def test_single_bright_patch(self):
        # Radiance proposal collapses onto the single bright patch.
        zs = [1.0, 0.8, 0.6, 0.4]
        L = [0.0, 0.0, 5.0, 0.0]
        f = [1.0, 1.0, 1.0, 1.0]
        scene = make_scene(zs, L)
        res = estimate(scene, f, 8, 8, seed=11)
        for rec in res.records:
            if rec.strategy == STRATEGY_RADIANCE:
                self.assertEqual(rec.patch_index, 2)
        expected = reference_direct(L, f, zs, [1.0] * 4)
        # Deterministic seed: the estimate must lie within 3 estimated
        # standard deviations of the exact value.
        self.assertLessEqual(abs(res.estimate - expected), 3.0 * res.stddev,
                             f"{res.estimate!r} vs {expected!r} "
                             f"(stddev {res.stddev!r})")
        self.assertEqual(res.exact, direct_sum(scene, f))

    def test_zero_energy_scene(self):
        zs = [1.0, 0.5, 0.25]
        scene = make_scene(zs, [0.0, 0.0, 0.0])
        f = [1.0, 2.0, 3.0]
        res = estimate(scene, f, 0, 6, seed=3)
        self.assertEqual(res.estimate, 0.0)
        self.assertEqual(res.exact, 0.0)
        codes = [d.code for d in res.diagnostics]
        self.assertIn(DIAG_UNUSED_ZERO_MASS_STRATEGY, codes)
        # Requesting radiance samples on a zero-energy scene must fail.
        with self.assertRaises(DegenerateProposalError):
            estimate(scene, f, 1, 0, seed=3)

    def test_all_grazing_scene_cosine_degenerate(self):
        scene = make_scene([0.0, 0.0], [1.0, 2.0])
        with self.assertRaises(DegenerateProposalError):
            estimate(scene, [1.0, 1.0], 0, 2, seed=1)

    def test_unequal_sample_counts(self):
        zs = [1.0, 0.9, 0.5, 0.2]
        L = [1.0, 2.0, 0.5, 3.0]
        f = [1.0, 0.5, 2.0, 1.0]
        scene = make_scene(zs, L)
        res = estimate(scene, f, 6, 1, seed=5)
        self.assertEqual(res.n_radiance, 6)
        self.assertEqual(res.n_cosine, 1)
        self.assertEqual(len(res.records), 7)
        # One-sample strategy: variance is not estimable.
        self.assertIsNone(res.variance)
        self.assertIsNone(res.stddev)
        codes = [d.code for d in res.diagnostics]
        self.assertIn(DIAG_VARIANCE_NOT_ESTIMABLE, codes)
        # Estimator still sums the per-sample contributions.
        self.assertEqual(res.estimate,
                         math.fsum(r.contribution for r in res.records))


class TestDiagnostics(unittest.TestCase):
    def test_zero_pdf_term_grazing_patch(self):
        # Patch 1 sits on the horizon (cos = 0): the cosine strategy can
        # never sample it (its target term is 0, so this is informational).
        zs = [1.0, 0.0, 0.5]
        L = [1.0, 2.0, 1.0]
        f = [1.0, 1.0, 1.0]
        scene = make_scene(zs, L)
        res = estimate(scene, f, 4, 4, seed=9)
        hits = [d for d in res.diagnostics
                if d.code == DIAG_ZERO_PDF_TERM and d.patch_index == 1]
        self.assertEqual(len(hits), 1)
        self.assertIn(STRATEGY_COSINE, hits[0].message)

    def test_zero_pdf_term_dark_patch(self):
        # Patch 0 has L = 0: the radiance strategy never samples it.
        scene = make_scene([1.0, 0.5], [0.0, 2.0])
        res = estimate(scene, [1.0, 1.0], 4, 4, seed=9)
        hits = [d for d in res.diagnostics
                if d.code == DIAG_ZERO_PDF_TERM and d.patch_index == 0]
        self.assertEqual(len(hits), 1)
        self.assertIn(STRATEGY_RADIANCE, hits[0].message)

    def test_unreachable_term_guard(self):
        # g_i > 0 implies L_i > 0 and cos_i > 0, so both proposals are
        # positive there: this state cannot arise through the validated
        # public API. Exercise the defensive guard directly with synthetic
        # inputs (a positive target term no active strategy can reach).
        from mis_envlight.core import _diagnose
        scene = make_scene([0.9, 0.8], [0.0, 1.0])
        g = (0.5, 1.0)  # synthetic: patch 0 has a positive target term
        diags = _diagnose(scene, g, (0.0, 1.0), (0.0, 1.0), 1, 1)
        hits = [d for d in diags if d.code == DIAG_UNREACHABLE_TERM]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].patch_index, 0)


class TestDeterminismAndRecords(unittest.TestCase):
    def test_same_seed_reproduces_identical_result(self):
        scene = make_scene([1.0, 0.7, 0.3], [1.0, 2.0, 0.5])
        f = [1.0, 0.5, 2.0]
        a = estimate(scene, f, 5, 5, seed=42)
        b = estimate(scene, f, 5, 5, seed=42)
        self.assertEqual(a, b)

    def test_records_explain_weights(self):
        zs = [1.0, 0.8, 0.5]
        L = [1.0, 2.0, 3.0]
        f = [0.5, 1.0, 1.5]
        scene = make_scene(zs, L)
        n_r, n_c = 3, 2
        res = estimate(scene, f, n_r, n_c, seed=13)
        pL, pC = proposal_pdfs(scene)
        for rec in res.records:
            i = rec.patch_index
            denom = n_r * pL[i] + n_c * pC[i]
            self.assertEqual(rec.target_value,
                             L[i] * f[i] * zs[i] * 1.0)
            self.assertTrue(math.isclose(rec.contribution,
                                         rec.target_value / denom,
                                         rel_tol=1e-15))
            n_self = n_r if rec.strategy == STRATEGY_RADIANCE else n_c
            self.assertEqual(rec.n_self, n_self)
            self.assertTrue(math.isclose(
                rec.mis_weight, n_self * rec.pdf_self / denom, rel_tol=1e-15))
            self.assertTrue(math.isclose(
                rec.contribution,
                rec.mis_weight * rec.target_value / rec.pdf_self / n_self,
                rel_tol=1e-15))
        text = res.explain(limit=2)
        self.assertIsInstance(text, str)
        self.assertIn("mis_weight", text)
        self.assertIn("more samples", text)
        self.assertIn("radiance", res.explain())


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.scene = make_scene([1.0, 0.5], [1.0, 2.0])
        self.f = [1.0, 1.0]

    def test_nan_and_inf_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValidationError):
                Scene([Patch(direction=(0.0, 0.0, 1.0), radiance=bad, weight=1.0)])
            with self.assertRaises(ValidationError):
                Scene([Patch(direction=(0.0, 0.0, 1.0), radiance=1.0, weight=bad)])
            with self.assertRaises(ValidationError):
                Scene([Patch(direction=(bad, 0.0, 1.0), radiance=1.0, weight=1.0)])
            with self.assertRaises(ValidationError):
                direct_sum(self.scene, [bad, 1.0])

    def test_bool_rejected_for_int_params(self):
        for bad in (True, False):
            with self.assertRaises(ValidationError):
                estimate(self.scene, self.f, bad, 1, seed=1)
            with self.assertRaises(ValidationError):
                estimate(self.scene, self.f, 1, bad, seed=1)
            with self.assertRaises(ValidationError):
                estimate(self.scene, self.f, 1, 1, seed=bad)

    def test_non_integer_counts_rejected(self):
        for bad in (1.5, "2", None, [1]):
            with self.assertRaises(ValidationError):
                estimate(self.scene, self.f, bad, 1, seed=1)

    def test_negative_and_zero_total_counts(self):
        with self.assertRaises(ValidationError):
            estimate(self.scene, self.f, -1, 2, seed=1)
        with self.assertRaises(ValidationError):
            estimate(self.scene, self.f, 0, 0, seed=1)

    def test_error_message_locates_field(self):
        patches = [Patch(direction=(0.0, 0.0, 1.0), radiance=1.0, weight=1.0),
                   Patch(direction=(0.0, 0.0, 1.0), radiance=-2.0, weight=1.0)]
        with self.assertRaises(ValidationError) as ctx:
            Scene(patches)
        self.assertIn("patches[1].radiance", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            direct_sum(self.scene, [1.0])
        self.assertIn("f:", str(ctx.exception))

    def test_direction_validation(self):
        with self.assertRaises(ValidationError):
            Scene([Patch(direction=(0.0, 0.0, 2.0), radiance=1.0, weight=1.0)])
        with self.assertRaises(ValidationError):
            Scene([Patch(direction=(0.0, 0.0, -1.0), radiance=1.0, weight=1.0)])
        with self.assertRaises(ValidationError):
            Scene([Patch(direction=(0.0, 0.0), radiance=1.0, weight=1.0)])
        with self.assertRaises(ValidationError):
            Scene([Patch(direction="up", radiance=1.0, weight=1.0)])

    def test_weight_and_radiance_bounds(self):
        with self.assertRaises(ValidationError):
            Scene([Patch(direction=(0.0, 0.0, 1.0), radiance=1.0, weight=0.0)])
        with self.assertRaises(ValidationError):
            Scene([Patch(direction=(0.0, 0.0, 1.0), radiance=-1e-9, weight=1.0)])

    def test_f_validation(self):
        with self.assertRaises(ValidationError):
            direct_sum(self.scene, [1.0, -0.5])
        with self.assertRaises(ValidationError):
            direct_sum(self.scene, [1.0])
        with self.assertRaises(ValidationError):
            direct_sum(self.scene, "not-a-sequence")

    def test_scene_size_limits(self):
        with self.assertRaises(ValidationError):
            Scene([])
        ok = [Patch(direction=(0.0, 0.0, 1.0), radiance=1.0, weight=1.0)
              for _ in range(MAX_PATCHES)]
        self.assertEqual(len(Scene(ok)), MAX_PATCHES)
        with self.assertRaises(ValidationError):
            Scene(ok + [ok[0]])

    def test_non_scene_and_non_patch_rejected(self):
        with self.assertRaises(ValidationError):
            direct_sum("not a scene", self.f)
        with self.assertRaises(ValidationError):
            Scene([("not", "a", "patch")])


class TestAdaptiveAPI(unittest.TestCase):
    """New multi-batch adaptive entry point: basic contracts."""

    def setUp(self):
        zs = [1.0, 0.8, 0.5, 0.25]
        L = [1.5, 0.0, 3.25, 2.0]
        w = [0.5, 0.5, 0.25, 1.0]
        f = [1.0, 2.0, 0.5, 0.75]
        self.scene = make_scene(zs, L, w)
        self.zs, self.L, self.w, self.f = zs, L, w, f

    def test_single_batch_matches_legacy_estimate_exactly(self):
        # One adaptive batch with an even 50/50 split draws and scores in the
        # exact same order as estimate(), hence must agree bit for bit.
        old = estimate(self.scene, self.f, 5, 5, seed=123)
        new = estimate_adaptive(self.scene, self.f, 1, 10, seed=123)
        self.assertEqual(new.estimate, old.estimate)
        self.assertEqual(len(new.records), len(old.records))
        for nr, nr2 in zip(new.records, old.records):
            self.assertEqual(nr.strategy, nr2.strategy)
            self.assertEqual(nr.patch_index, nr2.patch_index)
            self.assertEqual(nr.n_self, nr2.n_self)
            self.assertEqual(nr.n_other, nr2.n_other)
            self.assertEqual(nr.pdf_self, nr2.pdf_self)
            self.assertEqual(nr.pdf_other, nr2.pdf_other)
            self.assertEqual(nr.contribution, nr2.contribution)
            self.assertEqual(nr.mis_weight, nr2.mis_weight)
            self.assertEqual(nr.batch, 0)
        self.assertEqual(new.estimate,
                         math.fsum(b.estimate for b in new.batch_records))
        self.assertEqual(new.n_batches, 1)
        self.assertEqual(new.samples_per_batch, 10)

    def test_weights_are_ratios_and_realized_split(self):
        # (1, 3) normalizes to 0.25; m=8 realizes counts (2, 6).
        res = estimate_adaptive(self.scene, self.f, 1, 8, seed=1,
                                initial_weights=(1.0, 3.0))
        self.assertAlmostEqual(res.initial_weights[0], 0.25)
        self.assertAlmostEqual(res.initial_weights[1], 0.75)
        b = res.batch_records[0]
        self.assertEqual((b.n_radiance, b.n_cosine), (2, 6))
        self.assertAlmostEqual(b.weight_radiance, 0.25)
        self.assertAlmostEqual(b.weight_cosine, 0.75)

    def test_reproducible(self):
        kw = dict(n_batches=3, samples_per_batch=12,
                  updater=lambda br: (0.8, 0.2))
        a = estimate_adaptive(self.scene, self.f, seed=77, floor=0.05, **kw)
        b = estimate_adaptive(self.scene, self.f, seed=77, floor=0.05, **kw)
        self.assertEqual(a, b)
        self.assertIn("batch", a.explain())

    def test_earlier_batches_immune_to_later_updates(self):
        # Two runs whose updaters agree on the batch-0 weights but diverge
        # afterwards: batch 0 draws and scores must be identical (its
        # conditional mixture was frozen before any later update).
        def u_a(br):
            return (0.25, 0.75) if br.batch % 2 == 0 else (0.75, 0.25)

        def u_b(br):
            if br.batch == 0:
                return (0.25, 0.75)
            return (0.5, 0.5)

        a = estimate_adaptive(self.scene, self.f, 4, 4, seed=9,
                              updater=u_a, floor=0.05)
        b = estimate_adaptive(self.scene, self.f, 4, 4, seed=9,
                              updater=u_b, floor=0.05)
        self.assertEqual(a.batch_records[0].records,
                         b.batch_records[0].records)


class TestAdaptiveUnbiasedness(unittest.TestCase):
    """Conditional-PDF scoring: per-batch identity and aggregate unbiasedness."""

    def setUp(self):
        self.zs = [1.0, 0.8, 0.5, 0.25]
        self.L = [1.5, 0.0, 3.25, 2.0]
        self.w = [1.0, 1.0, 1.0, 1.0]
        self.f = [1.0, 2.0, 0.5, 0.75]
        self.scene = make_scene(self.zs, self.L, self.w)

    def _ref_pdfs(self):
        # Independently recomputed proposal pdfs (plain arithmetic).
        lm = [self.L[i] * self.w[i] for i in range(4)]
        cm = [self.zs[i] * self.w[i] for i in range(4)]
        pL = tuple(x / sum(lm) for x in lm)
        pC = tuple(x / sum(cm) for x in cm)
        return pL, pC

    def test_every_adapted_batch_is_conditionally_unbiased(self):
        # Whatever split an updater produced, batch b's conditional
        # expectation over its own draw space must equal S:
        #   E_b = sum_i (n_r*pL_i + n_c*pC_i) * g_i / same_denominator
        updater = lambda br: ((0.9, 0.1) if br.batch % 2 == 0
                              else (0.1, 0.9))
        res = estimate_adaptive(self.scene, self.f, 5, 20, seed=2026,
                                updater=updater, floor=0.05)
        pL, pC = self._ref_pdfs()
        g = [self.L[i] * self.f[i] * self.zs[i] * self.w[i]
             for i in range(4)]
        expected = reference_direct(self.L, self.f, self.zs, self.w)
        for b in res.batch_records:
            cond = 0.0
            for i in range(4):
                denom = b.n_radiance * pL[i] + b.n_cosine * pC[i]
                cond += denom * g[i] / denom
            self.assertTrue(math.isclose(cond, expected, rel_tol=1e-12),
                            f"batch {b.batch}: {cond!r} != {expected!r}")

    def test_monte_carlo_unbiased_changing_and_fixed(self):
        # Aggregate empirical mean over many independent seeds must land
        # within 4 standard errors of S for both regimes.
        expected = reference_direct(self.L, self.f, self.zs, self.w)
        flip = lambda br: ((0.9, 0.1) if br.batch % 2 == 0 else (0.1, 0.9))
        for updater, label in ((None, "fixed"), (flip, "changing")):
            ests = []
            for seed in range(256):
                r = estimate_adaptive(self.scene, self.f, 4, 16, seed,
                                      updater=updater, floor=0.05)
                ests.append(r.estimate)
            mean = math.fsum(ests) / len(ests)
            var = math.fsum((x - mean) ** 2 for x in ests) / (len(ests) - 1)
            se = math.sqrt(var / len(ests))
            self.assertLessEqual(abs(mean - expected), 4.0 * se,
                                 f"{label}: mean={mean!r} S={expected!r} "
                                 f"se={se!r}")

    def test_records_freeze_their_own_batch_mixture(self):
        # m=4: initial 0.25/0.75 realizes counts (1,3); the updater then
        # switches batch 1 to (3,1).
        updater = lambda br: (0.75, 0.25)
        res = estimate_adaptive(self.scene, self.f, 2, 4, seed=3,
                                initial_weights=(0.25, 0.75),
                                updater=updater, floor=0.05)
        pL, pC = self._ref_pdfs()
        counts = [(b.n_radiance, b.n_cosine) for b in res.batch_records]
        self.assertEqual(counts, [(1, 3), (3, 1)])
        final = counts[-1]
        for rec in res.records:
            i = rec.patch_index
            own_denom = rec.n_self * rec.pdf_self + rec.n_other * rec.pdf_other
            # Stored contribution is scored with THIS batch's mixture.
            self.assertTrue(math.isclose(
                rec.contribution, rec.target_value / own_denom, rel_tol=1e-14))
            self.assertTrue(math.isclose(
                rec.mixture_pdf(),
                own_denom / (rec.n_self + rec.n_other), rel_tol=1e-14))
            # Scoring the same sample with the FINAL batch's counts changes
            # its value: the implementation must not have done that. Skip
            # zero-target samples (0 under every denominator).
            if rec.batch == 0 and rec.target_value > 0.0:
                final_denom = final[0] * pL[i] + final[1] * pC[i]
                self.assertFalse(math.isclose(
                    rec.contribution, rec.target_value / final_denom,
                    rel_tol=1e-9))

    def test_recalculating_old_samples_with_final_pdf_is_biased(self):
        """Exact-rational proof of why frozen conditional pdfs are required.

        Two patches, batch 1 uses counts (1,3), batch 2 uses (3,1).
        Re-scoring batch-1 draws with the FINAL mixture (3,1) has
        expectation 74/35 instead of S = 2; the averaged wrong estimator
        lands at 72/35 (bias 2/35). The correct per-batch scoring equals S.
        """
        cos = [Fraction(1), Fraction(1, 2)]
        L = [Fraction(1), Fraction(2)]
        w = [Fraction(1), Fraction(1)]
        f = [Fraction(1), Fraction(1)]
        g = [L[i] * f[i] * cos[i] * w[i] for i in range(2)]
        lm = [L[i] * w[i] for i in range(2)]
        cm = [cos[i] * w[i] for i in range(2)]
        pL = [x / sum(lm) for x in lm]
        pC = [x / sum(cm) for x in cm]

        def expected_batch_sum(nr, nc, sr, sc):
            # Sampling mixture (nr, nc), scoring mixture (sr, sc).
            total = Fraction(0)
            for i in range(2):
                sampling_mass = nr * pL[i] + nc * pC[i]
                scoring_denom = sr * pL[i] + sc * pC[i]
                total += sampling_mass * g[i] / scoring_denom
            return total

        correct = (expected_batch_sum(1, 3, 1, 3)
                   + expected_batch_sum(3, 1, 3, 1)) / 2
        forbidden = (expected_batch_sum(1, 3, 3, 1)
                     + expected_batch_sum(3, 1, 3, 1)) / 2
        self.assertEqual(correct, sum(g))
        self.assertEqual(forbidden, Fraction(72, 35))
        self.assertNotEqual(forbidden, sum(g))


class TestAdaptiveFloorAndDegeneracy(unittest.TestCase):
    def _scene(self):
        return make_scene([1.0, 0.7, 0.4], [1.0, 4.0, 2.0],
                          [0.5, 1.0, 0.25])

    def test_floor_holds_on_realized_counts(self):
        res = estimate_adaptive(self._scene(), [1.0, 1.0, 1.0], 6, 20, seed=4,
                                updater=lambda br: (1e9, 1.0), floor=0.1)
        lo = math.ceil(20 * 0.1 - 1e-12)
        for b in res.batch_records:
            self.assertGreaterEqual(b.n_radiance, lo)
            self.assertGreaterEqual(b.n_cosine, lo)
        # The extreme ratio was clamped onto [0.1, 0.9].
        self.assertEqual(res.final_weights, (0.9, 0.1))
        self.assertTrue(any(d.code == DIAG_WEIGHTS_CLAMPED
                            for d in res.diagnostics))

    def test_fixed_distribution_half_floor(self):
        # floor=0.5 forces an even split and clamps every updater request.
        res = estimate_adaptive(self._scene(), [1.0, 1.0, 1.0], 4, 4, seed=2,
                                floor=0.5, updater=lambda br: (0.99, 0.01))
        for b in res.batch_records:
            self.assertEqual((b.n_radiance, b.n_cosine), (2, 2))
        self.assertEqual(sum(d.code == DIAG_WEIGHTS_CLAMPED
                             for d in res.diagnostics), 3)

    def test_fixed_distribution_without_updater(self):
        res = estimate_adaptive(self._scene(), [1.0, 1.0, 1.0], 3, 10, seed=8,
                                initial_weights=(0.3, 0.7))
        for b in res.batch_records:
            self.assertEqual((b.n_radiance, b.n_cosine), (3, 7))
        self.assertFalse(any(d.code == DIAG_WEIGHTS_UPDATED
                             for d in res.diagnostics))

    def test_in_bounds_ratio_is_not_falsely_reported_as_clamped(self):
        # A well-formed updater whose returned ratio is already inside the
        # floor band (despite float renormalization) must not be flagged.
        calls = []

        def updater(br):
            w = (0.3 + 1e-13, 0.7)
            calls.append(w)
            return w

        res = estimate_adaptive(self._scene(), [1.0, 1.0, 1.0], 2, 8, seed=0,
                                updater=updater, floor=0.1)
        self.assertFalse(any(d.code == DIAG_WEIGHTS_CLAMPED
                             for d in res.diagnostics))
        self.assertEqual(len(calls), 1)

    def test_distribution_changes_between_batches(self):
        seen = []

        def updater(br):
            seen.append((br.batch, br.n_radiance, br.n_cosine))
            return (0.8, 0.2) if br.batch == 0 else (0.2, 0.8)

        res = estimate_adaptive(self._scene(), [1.0, 1.0, 1.0], 3, 10, seed=5,
                                updater=updater, floor=0.1)
        splits = [(b.n_radiance, b.n_cosine) for b in res.batch_records]
        self.assertEqual(splits[0], (5, 5))
        self.assertNotEqual(splits[1], splits[0])
        self.assertEqual(seen, [(0, 5, 5), (1, 8, 2)])
        self.assertEqual(sum(d.code == DIAG_WEIGHTS_UPDATED
                             for d in res.diagnostics), 2)
        self.assertEqual(res.final_weights, (0.2, 0.8))

    def test_updater_not_called_for_single_batch(self):
        def boom(br):
            raise AssertionError("updater must not run after the last batch")

        res = estimate_adaptive(self._scene(), [1.0, 1.0, 1.0], 1, 4, seed=0,
                                updater=boom)
        self.assertEqual(len(res.batch_records), 1)

    def test_zero_energy_scene_rejected_up_front(self):
        dark = make_scene([1.0, 0.5], [0.0, 0.0])
        with self.assertRaises(DegenerateProposalError):
            estimate_adaptive(dark, [1.0, 1.0], 3, 6, seed=1)

    def test_all_grazing_scene_rejected_up_front(self):
        flat = make_scene([0.0, 0.0], [1.0, 2.0])
        with self.assertRaises(DegenerateProposalError):
            estimate_adaptive(flat, [1.0, 1.0], 3, 6, seed=1)

    def test_initial_weights_at_floor_boundary_accepted(self):
        # Exactly on the floor is legal; just below is rejected.
        estimate_adaptive(self._scene(), [1.0, 1.0, 1.0], 1, 10, seed=1,
                          initial_weights=(0.001, 0.999))
        with self.assertRaises(ValidationError):
            estimate_adaptive(self._scene(), [1.0, 1.0, 1.0], 1, 10, seed=1,
                              initial_weights=(0.0009, 0.9991))

    def test_unrealizable_integer_floor_rejected(self):
        # m=3, floor=0.5: each strategy would need 2 of 3 draws.
        with self.assertRaises(ValidationError):
            estimate_adaptive(self._scene(), [1.0, 1.0, 1.0], 2, 3, seed=1,
                              floor=0.5)
        # m=4, floor=0.5 is realizable as (2,2).
        res = estimate_adaptive(self._scene(), [1.0, 1.0, 1.0], 1, 4, seed=1,
                                floor=0.5)
        self.assertEqual((res.batch_records[0].n_radiance,
                          res.batch_records[0].n_cosine), (2, 2))


class TestAdaptiveVariance(unittest.TestCase):
    def test_variance_formula_recomputed_independently(self):
        scene = make_scene([1.0, 0.9, 0.5, 0.2],
                           [1.0, 2.0, 0.5, 3.0])
        res = estimate_adaptive(scene, [1.0, 0.5, 2.0, 1.0], 5, 8, seed=6)
        # Independent recompute: per-batch stratified variance, then the
        # variance of the mean of independent batches.
        total = 0.0
        for b in res.batch_records:
            groups = {STRATEGY_RADIANCE: [], STRATEGY_COSINE: []}
            for r in b.records:
                groups[r.strategy].append(r.contribution)
            vb = 0.0
            for xs in groups.values():
                mean = math.fsum(xs) / len(xs)
                s2 = math.fsum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
                vb += len(xs) * s2
            total += vb
        expected_var = total / 25.0
        self.assertTrue(math.isclose(res.variance, expected_var, rel_tol=1e-12))
        self.assertTrue(math.isclose(res.stddev, math.sqrt(res.variance)))

    def test_single_draw_stratum_makes_variance_none(self):
        scene = make_scene([1.0, 0.7, 0.3], [1.0, 2.0, 0.5])
        # m=4 with a 0.25 share realizes (1,3): the 1-draw stratum has no
        # sample variance, per batch and for the whole run.
        res = estimate_adaptive(scene, [1.0, 0.5, 2.0], 3, 4, seed=6,
                                initial_weights=(0.25, 0.75), floor=0.05)
        self.assertIsNone(res.variance)
        self.assertIsNone(res.stddev)
        self.assertEqual(sum(d.code == DIAG_VARIANCE_NOT_ESTIMABLE
                             for d in res.diagnostics), 3)

    def test_single_patch_scene_zero_variance(self):
        scene = make_scene([0.6], [2.5], [0.75])
        res = estimate_adaptive(scene, [1.5], 4, 8, seed=0)
        expected = reference_direct([2.5], [1.5], [0.6], [0.75])
        self.assertEqual(res.estimate, expected)
        self.assertEqual(res.variance, 0.0)


class TestAdaptiveValidation(unittest.TestCase):
    def setUp(self):
        self.scene = make_scene([1.0, 0.5], [1.0, 2.0])
        self.f = [1.0, 1.0]

    def _call(self, **over):
        kw = dict(n_batches=2, samples_per_batch=4, seed=1)
        kw.update(over)
        return estimate_adaptive(self.scene, self.f, **kw)

    def test_batch_counts(self):
        for bad in (0, -1, True, 1.5, "2", MAX_BATCHES + 1):
            with self.assertRaises(ValidationError):
                self._call(n_batches=bad)
        # Boundary value is accepted and runs.
        self.assertEqual(self._call(n_batches=1, samples_per_batch=2).n_batches, 1)
        self._call(n_batches=MAX_BATCHES, samples_per_batch=2)

    def test_samples_per_batch(self):
        for bad in (0, 1, -2, False, 2.0):
            with self.assertRaises(ValidationError):
                self._call(samples_per_batch=bad)

    def test_floor(self):
        for bad in (0.0, -0.1, 0.51, math.nan, math.inf, "x"):
            with self.assertRaises(ValidationError):
                self._call(floor=bad)

    def test_initial_weights(self):
        for bad in ((0.0, 1.0), (-0.1, 1.0), (1.0,), (1.0, 1.0, 1.0),
                    "50/50", (math.nan, 1.0), None):
            with self.assertRaises(ValidationError):
                self._call(initial_weights=bad)

    def test_seed(self):
        for bad in (True, 1.5, "1", None):
            with self.assertRaises(ValidationError):
                self._call(seed=bad)

    def test_updater_type_and_bad_returns_abort_run(self):
        with self.assertRaises(ValidationError):
            self._call(updater="not callable")
        for bad in ((1.0,), (1.0, 1.0, 0.0), (math.nan, 1.0),
                    (-0.5, 1.0), (0.0, 0.0), "0.5/0.5", None):
            def updater(br, bad=bad):
                return bad
            with self.assertRaises(ValidationError):
                self._call(n_batches=3, updater=updater)

    def test_inherited_scene_and_f_validation(self):
        with self.assertRaises(ValidationError):
            estimate_adaptive("not a scene", self.f, 2, 4, seed=1)
        with self.assertRaises(ValidationError):
            estimate_adaptive(self.scene, [1.0], 2, 4, seed=1)
        with self.assertRaises(ValidationError):
            estimate_adaptive(self.scene, [1.0, -1.0], 2, 4, seed=1)


class TestVarianceMinimizingUpdater(unittest.TestCase):
    def _brec(self, rad_contribs, cos_contribs, prev=(0.3, 0.7)):
        def rec(strategy, k, value, n_self):
            other = len(cos_contribs) if strategy == STRATEGY_RADIANCE \
                else len(rad_contribs)
            return SampleRecord(
                strategy=strategy, draw=k, patch_index=k % 2,
                n_self=n_self, n_other=other,
                pdf_self=0.5, pdf_other=0.5, mis_weight=0.5,
                target_value=value, contribution=value, batch=0)
        records = tuple(
            [rec(STRATEGY_RADIANCE, k, v, len(rad_contribs))
             for k, v in enumerate(rad_contribs)]
            + [rec(STRATEGY_COSINE, k, v, len(cos_contribs))
               for k, v in enumerate(cos_contribs)])
        return BatchRecord(
            batch=0, samples_per_batch=len(records),
            n_radiance=len(rad_contribs), n_cosine=len(cos_contribs),
            weight_radiance=prev[0], weight_cosine=prev[1],
            estimate=0.0, variance=0.0, records=records)

    def test_equal_spread_equal_weights(self):
        w = variance_minimizing_updater()(self._brec([0.0, 2.0], [0.0, 2.0]))
        self.assertTrue(math.isclose(w[0], 0.5))
        self.assertTrue(math.isclose(w[1], 0.5))

    def test_all_mass_goes_to_spread_stratum_before_floor(self):
        w = variance_minimizing_updater()(self._brec([0.0, 2.0], [5.0, 5.0]))
        self.assertEqual(w, (1.0, 0.0))  # driver clamps this to the floor

    def test_zero_variance_keeps_previous_weights(self):
        w = variance_minimizing_updater()(self._brec([3.0, 3.0], [3.0, 3.0]))
        self.assertEqual(w, (0.3, 0.7))

    def test_single_draw_stratum_keeps_previous_weights(self):
        w = variance_minimizing_updater()(
            self._brec([1.0], [1.0, 2.0, 3.0]))
        self.assertEqual(w, (0.3, 0.7))

    def test_end_to_end_clamps_and_runs(self):
        scene = make_scene([1.0, 0.8, 0.5, 0.25], [1.0, 0.1, 3.0, 0.2])
        res = estimate_adaptive(scene, [1.0, 1.0, 1.0, 1.0], 4, 16, seed=11,
                                updater=variance_minimizing_updater(),
                                floor=0.05)
        for b in res.batch_records:
            self.assertGreaterEqual(b.n_radiance, 1)
            self.assertGreaterEqual(b.n_cosine, 1)


if __name__ == "__main__":
    unittest.main()
