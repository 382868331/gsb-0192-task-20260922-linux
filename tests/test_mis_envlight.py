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
    STRATEGY_RADIANCE,
    STRATEGY_COSINE,
    DIAG_ZERO_PDF_TERM,
    DIAG_UNREACHABLE_TERM,
    DIAG_VARIANCE_NOT_ESTIMABLE,
    DIAG_UNUSED_ZERO_MASS_STRATEGY,
    ValidationError,
    DegenerateProposalError,
    Patch,
    Scene,
    direct_sum,
    proposal_pdfs,
    enumerate_contributions,
    estimate,
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


if __name__ == "__main__":
    unittest.main()
