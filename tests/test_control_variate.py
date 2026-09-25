"""Tests for the two-phase control-variate estimator.

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
    STRATEGY_RADIANCE,
    STRATEGY_COSINE,
    DIAG_CONTROL_ZERO_PILOT_VARIANCE,
    DIAG_UNREACHABLE_TERM,
    DIAG_ZERO_PDF_TERM,
    DIAG_VARIANCE_NOT_ESTIMABLE,
    PHASE_PILOT,
    PHASE_MAIN,
    ValidationError,
    DegenerateProposalError,
    Patch,
    Scene,
    direct_sum,
    proposal_pdfs,
    estimate,
    estimate_with_control,
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


class TestExactControlIdentities(unittest.TestCase):
    def test_f_equals_g_recovers_exact_sum(self):
        # With g == f the per-sample control quotient equals the target
        # quotient, so the pilot finds beta == 1 exactly, the residual is
        # exactly zero, and the estimate is the exact sum for any seed.
        zs = [1.0, 0.8, 0.6, 0.3]
        L = [2.0, 0.5, 3.0, 1.0]
        f = [0.5, 1.5, 0.25, 2.0]
        scene = make_scene(zs, L)
        G = reference_direct(L, f, zs, [1.0] * 4)
        for seed in (1, 7, 12345):
            res = estimate_with_control(scene, f, f, G, 24, 24, 8, 8,
                                        seed=seed)
            self.assertEqual(res.beta, 1.0)
            self.assertEqual(res.estimate, G)
            self.assertEqual(res.exact, G)
            self.assertEqual(res.variance, 0.0)
            self.assertEqual(res.residual_estimate, 0.0)

    def test_f_equals_g_plus_constant_expectation_exact(self):
        """Exact rational check: E[controlled estimator] == S for fixed beta.

        Scene with f = g + c (c a nonzero constant, g with mixed signs).
        Everything (proposals, MIS denominators, residual expectation) is
        recomputed here with Fractions from the mathematical specification;
        beta is an arbitrary fixed rational.  The identity
        E[residual] + beta*G == S must hold exactly.
        """
        cos = [Fraction(1), Fraction(4, 5), Fraction(12, 13), Fraction(15, 17)]
        L = [Fraction(2), Fraction(1, 2), Fraction(3), Fraction(5, 4)]
        w = [Fraction(1, 4)] * 4
        g = [Fraction(1, 3), Fraction(-2), Fraction(5, 2), Fraction(1)]
        cst = Fraction(7, 10)
        f = [g[i] + cst for i in range(4)]  # f = g + constant
        t = [L[i] * f[i] * cos[i] * w[i] for i in range(4)]
        ct = [L[i] * g[i] * cos[i] * w[i] for i in range(4)]
        G = sum(ct)
        S = sum(t)
        lm = [L[i] * w[i] for i in range(4)]
        cm = [cos[i] * w[i] for i in range(4)]
        pL = [m / sum(lm) for m in lm]
        pC = [m / sum(cm) for m in cm]
        beta = Fraction(3, 7)  # arbitrary fixed coefficient
        for n_r, n_c in ((1, 1), (2, 1), (1, 3), (5, 7)):
            denom = [n_r * pL[i] + n_c * pC[i] for i in range(4)]
            residual_expectation = Fraction(0)
            for i in range(4):
                residual_expectation += n_r * pL[i] * (t[i] - beta * ct[i]) / denom[i]
                residual_expectation += n_c * pC[i] * (t[i] - beta * ct[i]) / denom[i]
            self.assertEqual(residual_expectation + beta * G, S,
                             f"identity failed for counts ({n_r}, {n_c})")

    def test_f_equals_g_plus_constant_stochastic(self):
        # Float companion of the exact identity: with g the control and
        # G its exact sum, the controlled estimator stays unbiased; the
        # residual estimates S - beta*G = c * sum(L*cos*w).
        zs = [1.0, 0.9, 0.7, 0.4]
        L = [1.0, 2.0, 0.5, 3.0]
        g = [0.5, -1.0, 2.0, 1.5]
        cst = 1.25
        f = [v + cst for v in g]
        scene = make_scene(zs, L)
        G = reference_direct(L, g, zs, [1.0] * 4)
        S = reference_direct(L, f, zs, [1.0] * 4)
        res = estimate_with_control(scene, f, g, G, 40, 40, 16, 16, seed=5)
        self.assertEqual(res.exact, direct_sum(scene, f))
        self.assertTrue(math.isclose(res.exact, S, rel_tol=1e-15))
        # Deterministic seed: within 4 estimated standard deviations.
        self.assertLessEqual(abs(res.estimate - S), 4.0 * res.stddev,
                             f"{res.estimate!r} vs {S!r}")


class TestPilotAndBeta(unittest.TestCase):
    def setUp(self):
        self.zs = [1.0, 0.8, 0.6, 0.3]
        self.L = [2.0, 0.5, 3.0, 1.0]
        self.f = [0.5, 1.5, 0.25, 2.0]
        self.scene = make_scene(self.zs, self.L)

    def test_beta_comes_from_pilot_only(self):
        # Recompute beta independently from the returned pilot records and
        # the main-phase counts: beta must be a function of the pilot
        # samples alone, fixed before the main phase starts.
        g = [0.4, 1.6, 0.3, 1.8]
        G = reference_direct(self.L, g, self.zs, [1.0] * 4)
        n_r, n_c = 24, 24
        res = estimate_with_control(self.scene, self.f, g, G,
                                    n_r, n_c, 8, 8, seed=42)
        pL, pC = proposal_pdfs(self.scene)
        num = den = 0.0
        for strategy, n_main in ((STRATEGY_RADIANCE, n_r),
                                 (STRATEGY_COSINE, n_c)):
            pairs = []
            for r in res.pilot_records:
                if r.strategy != strategy:
                    continue
                self.assertEqual(r.phase, PHASE_PILOT)
                d = n_r * pL[r.patch_index] + n_c * pC[r.patch_index]
                pairs.append((r.target_term / d, r.control_term / d))
            m = len(pairs)
            self.assertGreaterEqual(m, 2)
            mx = math.fsum(p[0] for p in pairs) / m
            my = math.fsum(p[1] for p in pairs) / m
            cov = math.fsum((x - mx) * (y - my) for x, y in pairs) / (m - 1)
            var = math.fsum((y - my) ** 2 for _, y in pairs) / (m - 1)
            num += n_main * cov
            den += n_main * var
        self.assertTrue(math.isclose(res.beta, num / den, rel_tol=1e-15),
                        f"{res.beta!r} != {num / den!r}")

    def test_zero_pilot_variance_sets_beta_zero(self):
        # All-zero control: every control quotient is 0, pilot variance is
        # zero, beta must be 0 with an explanatory diagnostic.
        res = estimate_with_control(self.scene, self.f,
                                    [0.0] * 4, 0.0, 10, 10, 4, 4, seed=3)
        self.assertEqual(res.beta, 0.0)
        hits = [d for d in res.diagnostics
                if d.code == DIAG_CONTROL_ZERO_PILOT_VARIANCE]
        self.assertEqual(len(hits), 1)
        self.assertIn("beta", hits[0].message)
        self.assertIn("0.0", hits[0].message)

    def test_no_pilot_samples_sets_beta_zero(self):
        g = [0.4, 1.6, 0.3, 1.8]
        G = reference_direct(self.L, g, self.zs, [1.0] * 4)
        res = estimate_with_control(self.scene, self.f, g, G, 10, 10,
                                    seed=3)
        self.assertEqual(res.beta, 0.0)
        self.assertEqual(res.pilot_records, ())
        codes = [d.code for d in res.diagnostics]
        self.assertIn(DIAG_CONTROL_ZERO_PILOT_VARIANCE, codes)

    def test_beta_zero_matches_plain_estimate_sample_by_sample(self):
        # With beta == 0 the main phase must reduce to plain MIS on the
        # same stream: records match estimate(..., same seed) one by one.
        res = estimate_with_control(self.scene, self.f,
                                    [0.0] * 4, 0.0, 5, 5, 4, 4, seed=99)
        plain = estimate(self.scene, self.f, 5, 5, seed=99)
        self.assertEqual(res.beta, 0.0)
        self.assertEqual(len(res.records), len(plain.records))
        for rc, rp in zip(res.records, plain.records):
            self.assertEqual(rc.phase, PHASE_MAIN)
            self.assertEqual(rc.strategy, rp.strategy)
            self.assertEqual(rc.draw, rp.draw)
            self.assertEqual(rc.patch_index, rp.patch_index)
            self.assertEqual(rc.pdf_self, rp.pdf_self)
            self.assertEqual(rc.pdf_other, rp.pdf_other)
            self.assertEqual(rc.mis_weight, rp.mis_weight)
            self.assertEqual(rc.target_term, rp.target_value)
            self.assertEqual(rc.contribution, rp.contribution)
        self.assertEqual(res.estimate, plain.estimate)
        self.assertEqual(res.variance, plain.variance)

    def test_pilot_stream_is_separate_from_main_stream(self):
        # Different pilot budgets must not disturb the main stream: with
        # beta forced to 0 (zero control) the main records are identical
        # whether the pilot drew 0 or 6 samples per strategy.
        a = estimate_with_control(self.scene, self.f, [0.0] * 4, 0.0,
                                  6, 6, 0, 0, seed=17)
        b = estimate_with_control(self.scene, self.f, [0.0] * 4, 0.0,
                                  6, 6, 6, 6, seed=17)
        self.assertEqual([r.patch_index for r in a.records],
                         [r.patch_index for r in b.records])
        self.assertEqual([r.contribution for r in a.records],
                         [r.contribution for r in b.records])
        self.assertEqual(a.estimate, b.estimate)
        # And the pilot records really were drawn in the second call.
        self.assertEqual(len(b.pilot_records), 12)

    def test_same_seed_reproduces_identical_result(self):
        g = [0.4, 1.6, 0.3, 1.8]
        G = reference_direct(self.L, g, self.zs, [1.0] * 4)
        a = estimate_with_control(self.scene, self.f, g, G, 12, 12, 4, 4,
                                  seed=7)
        b = estimate_with_control(self.scene, self.f, g, G, 12, 12, 4, 4,
                                  seed=7)
        self.assertEqual(a, b)


class TestBudgetAndResultShape(unittest.TestCase):
    def test_budget_includes_pilot(self):
        zs = [1.0, 0.7, 0.4]
        L = [1.0, 4.0, 2.0]
        f = [2.0, 0.5, 1.0]
        g = [1.5, 0.75, 1.25]
        scene = make_scene(zs, L)
        G = reference_direct(L, g, zs, [1.0] * 3)
        res = estimate_with_control(scene, f, g, G, 20, 12, 6, 2, seed=1)
        self.assertEqual(res.n_radiance, 20)
        self.assertEqual(res.n_cosine, 12)
        self.assertEqual(res.n_pilot_radiance, 6)
        self.assertEqual(res.n_pilot_cosine, 2)
        self.assertEqual(res.total_samples, 20 + 12 + 6 + 2)
        self.assertEqual(len(res.records), 32)
        self.assertEqual(len(res.pilot_records), 8)
        # estimate == residual + beta*G by construction.
        self.assertEqual(res.estimate,
                         math.fsum((res.residual_estimate,
                                    res.beta * res.control_exact)))
        self.assertEqual(res.control_exact, G)
        self.assertTrue(math.isclose(res.control_sum, G, rel_tol=1e-15))
        text = res.explain(limit=2)
        self.assertIn("beta", text)
        self.assertIn("control_term", text)
        self.assertIn("more samples", text)

    def test_variance_none_with_single_sample_strategy(self):
        scene = make_scene([1.0, 0.5], [1.0, 2.0])
        res = estimate_with_control(scene, [1.0, 1.0], [0.5, 0.5],
                                    1.0, 6, 1, 4, 4, seed=5)
        self.assertIsNone(res.variance)
        self.assertIsNone(res.stddev)
        codes = [d.code for d in res.diagnostics]
        self.assertIn(DIAG_VARIANCE_NOT_ESTIMABLE, codes)


class TestControlValidation(unittest.TestCase):
    def setUp(self):
        self.scene = make_scene([1.0, 0.5], [1.0, 2.0])
        self.f = [1.0, 1.0]

    def test_negative_control_entries_allowed(self):
        # g may have any sign; the residual keeps its sign too.
        res = estimate_with_control(self.scene, self.f, [-3.0, 2.5],
                                    -0.5, 4, 4, 2, 2, seed=1)
        self.assertIsInstance(res.estimate, float)

    def test_g_validation(self):
        with self.assertRaises(ValidationError):
            estimate_with_control(self.scene, self.f, [1.0], 0.0,
                                  2, 2, seed=1)
        with self.assertRaises(ValidationError):
            estimate_with_control(self.scene, self.f, "not-a-seq", 0.0,
                                  2, 2, seed=1)
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValidationError) as ctx:
                estimate_with_control(self.scene, self.f, [1.0, bad], 0.0,
                                      2, 2, seed=1)
            self.assertIn("g[1]", str(ctx.exception))

    def test_G_validation(self):
        for bad in (math.nan, math.inf, -math.inf, True, "1"):
            with self.assertRaises(ValidationError) as ctx:
                estimate_with_control(self.scene, self.f, [0.5, 0.5], bad,
                                      2, 2, seed=1)
            self.assertIn("G:", str(ctx.exception))

    def test_pilot_count_validation(self):
        with self.assertRaises(ValidationError):
            estimate_with_control(self.scene, self.f, [0.5, 0.5], 0.0,
                                  2, 2, -1, 0, seed=1)
        with self.assertRaises(ValidationError):
            estimate_with_control(self.scene, self.f, [0.5, 0.5], 0.0,
                                  2, 2, True, 0, seed=1)
        with self.assertRaises(ValidationError):
            estimate_with_control(self.scene, self.f, [0.5, 0.5], 0.0,
                                  2, 2, 1.5, 0, seed=1)

    def test_pilot_degenerate_proposal_rejected(self):
        dark = make_scene([1.0, 0.5], [0.0, 0.0])
        with self.assertRaises(DegenerateProposalError):
            estimate_with_control(dark, [1.0, 1.0], [0.5, 0.5], 0.0,
                                  0, 4, 1, 0, seed=1)
        grazing = make_scene([0.0, 0.0], [1.0, 2.0])
        with self.assertRaises(DegenerateProposalError):
            estimate_with_control(grazing, [1.0, 1.0], [0.5, 0.5], 0.0,
                                  4, 0, 0, 1, seed=1)


class TestControlDiagnostics(unittest.TestCase):
    def test_unreachable_term_guard_covers_control(self):
        # A nonzero control term no active strategy can reach must raise
        # the same zero-support alarm as a nonzero target term. (Not
        # reachable through the validated public API; exercise the guard
        # directly with synthetic inputs.)
        from mis_envlight.core import _diagnose
        scene = make_scene([0.9, 0.8], [0.0, 1.0])
        t = (0.0, 1.0)
        control = (-0.5, 1.0)  # patch 0: nonzero control, zero support
        diags = _diagnose(scene, t, (0.0, 1.0), (0.0, 1.0), 1, 1,
                          control=control)
        hits = [d for d in diags if d.code == DIAG_UNREACHABLE_TERM]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].patch_index, 0)
        self.assertIn("control term", hits[0].message)

    def test_zero_pdf_term_reports_control_term(self):
        # Patch 0 has L = 0: the radiance strategy never samples it; the
        # diagnostic message must also state the control term there.
        scene = make_scene([1.0, 0.5], [0.0, 2.0])
        res = estimate_with_control(scene, [1.0, 1.0], [3.0, 1.0], 5.0,
                                    4, 4, 2, 2, seed=9)
        hits = [d for d in res.diagnostics
                if d.code == DIAG_ZERO_PDF_TERM and d.patch_index == 0]
        self.assertEqual(len(hits), 1)
        self.assertIn(STRATEGY_RADIANCE, hits[0].message)
        self.assertIn("control term", hits[0].message)


class TestSeedSweepComparison(unittest.TestCase):
    """Finite-seed comparison of plain MIS vs. the controlled estimator.

    Any variance reduction observed here is a property of THIS scene,
    control and seed set -- the control-variate identity guarantees
    unbiasedness for any fixed beta, not a universal variance drop.
    """

    def test_controlled_estimator_over_finite_seeds(self):
        zs = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
        L = [0.5, 1.0, 3.0, 0.2, 2.0, 1.5]
        f = [1.0, 0.5, 1.5, 2.0, 0.25, 1.0]
        g = [0.9 * v + 0.1 for v in f]  # correlated control, any sign ok
        scene = make_scene(zs, L)
        G = reference_direct(L, g, zs, [1.0] * 6)
        S = reference_direct(L, f, zs, [1.0] * 6)

        plain_est, ctrl_est = [], []
        for seed in range(16):
            # Equal total budget: 64 samples each, pilot included.
            plain_est.append(
                estimate(scene, f, 32, 32, seed=seed).estimate)
            ctrl_est.append(
                estimate_with_control(scene, f, g, G, 24, 24, 8, 8,
                                      seed=seed).estimate)

        def mean(xs):
            return math.fsum(xs) / len(xs)

        def sample_var(xs):
            m = mean(xs)
            return math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)

        # Both estimators are unbiased: 16-seed means must sit within
        # 4 standard errors of the exact value.
        for ests in (plain_est, ctrl_est):
            se = math.sqrt(sample_var(ests) / len(ests))
            self.assertLessEqual(abs(mean(ests) - S), 4.0 * se,
                                 f"mean {mean(ests)!r} vs {S!r}")
        # On this scenario the controlled estimator's spread across seeds
        # is smaller. This is an observed, scenario-specific fact, not a
        # guarantee for other scenes/controls.
        self.assertLess(sample_var(ctrl_est), sample_var(plain_est))


if __name__ == "__main__":
    unittest.main()
