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
    DIAG_ZERO_PDF_TERM,
    DIAG_UNREACHABLE_TERM,
    DIAG_VARIANCE_NOT_ESTIMABLE,
    DIAG_PILOT_ZERO_VARIANCE,
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


def target_terms(scene, f):
    """Independent reference: plain loop over the mathematical definition."""
    return [scene.radiance[i] * f[i] * scene.cosine[i] * scene.weight[i]
            for i in range(len(scene))]


def reference_direct(radiances, f, cosines, weights):
    total = 0.0
    for i in range(len(radiances)):
        total += radiances[i] * f[i] * cosines[i] * weights[i]
    return total


class TestExactSumChecks(unittest.TestCase):
    """Small scenes where the control matches the target up to a constant."""

    def test_control_equal_target_terms_is_exact(self):
        # g_i = t_i exactly: beta comes out 1.0, every residual contribution
        # is exactly 0, and the estimate is exactly beta*G = S with zero
        # variance, regardless of the seed.
        zs = [1.0, 0.8, 0.5, 0.25]
        L = [1.5, 0.0, 3.25, 2.0]
        w = [0.5, 0.5, 0.25, 1.0]
        f = [1.0, 2.0, 0.5, 0.75]
        scene = make_scene(zs, L, w)
        t = target_terms(scene, f)
        exact = reference_direct(L, f, zs, w)
        for seed in (1, 7, 12345):
            res = estimate_with_control(scene, f, t, math.fsum(t),
                                        6, 6, 8, 8, seed)
            self.assertEqual(res.beta, 1.0)
            self.assertEqual(res.residual_estimate, 0.0)
            self.assertEqual(res.estimate, exact)
            self.assertEqual(res.variance, 0.0)
            self.assertEqual(res.stddev, 0.0)

    def test_target_equals_control_plus_constant(self):
        # f chosen so that t_i = g_i + K (the integrand equals the control
        # plus a constant, with signed g). The exact sum is G + n*K; the
        # estimate must land within 3 estimated standard deviations of it.
        zs = [1.0, 0.7, 0.4]
        L = [2.0, 1.0, 3.0]
        w = [0.5, 1.0, 0.25]
        g = [0.35, -0.2, 0.55]          # signed control terms
        K = 0.4
        f = [(g[i] + K) / (L[i] * zs[i] * w[i]) for i in range(3)]
        scene = make_scene(zs, L, w)
        exact = reference_direct(L, f, zs, w)
        self.assertTrue(math.isclose(exact, math.fsum(g) + 3 * K,
                                     rel_tol=1e-12))
        res = estimate_with_control(scene, f, g, math.fsum(g),
                                    16, 16, 32, 32, 20260925)
        self.assertEqual(res.exact, direct_sum(scene, f))
        self.assertLessEqual(abs(res.estimate - exact), 3.0 * res.stddev,
                             f"{res.estimate!r} vs {exact!r} "
                             f"(stddev {res.stddev!r}, beta {res.beta!r})")

    def test_expectation_identity_with_fixed_beta_exact_fraction(self):
        """Exact rational check: E[residual MIS] + beta*G == S for any beta.

        Everything (proposals, MIS denominator, residual expectation) is
        recomputed here with Fractions from the mathematical specification;
        no library code is involved.
        """
        cos = [Fraction(1), Fraction(4, 5), Fraction(12, 13), Fraction(15, 17)]
        L = [Fraction(2), Fraction(1, 2), Fraction(3), Fraction(5, 4)]
        w = [Fraction(1, 4)] * 4
        f = [Fraction(1, 3), Fraction(2), Fraction(5, 2), Fraction(1)]
        g = [Fraction(1, 10), Fraction(-1, 20), Fraction(2, 5), Fraction(0)]
        beta = Fraction(4, 5)
        G = sum(g)
        t = [L[i] * f[i] * cos[i] * w[i] for i in range(4)]
        lm = [L[i] * w[i] for i in range(4)]
        cm = [cos[i] * w[i] for i in range(4)]
        pL = [m / sum(lm) for m in lm]
        pC = [m / sum(cm) for m in cm]
        for n_r, n_c in ((1, 1), (2, 1), (1, 3), (5, 7)):
            denom = [n_r * pL[i] + n_c * pC[i] for i in range(4)]
            residual_expectation = Fraction(0)
            for i in range(4):
                r = t[i] - beta * g[i]
                residual_expectation += n_r * pL[i] * r / denom[i]
                residual_expectation += n_c * pC[i] * r / denom[i]
            total = residual_expectation + beta * G
            self.assertEqual(total, sum(t),
                             f"identity failed for counts ({n_r}, {n_c})")


class TestTwoPhaseStructure(unittest.TestCase):
    def setUp(self):
        self.zs = [1.0, 0.9, 0.6, 0.3]
        self.L = [1.0, 2.0, 0.5, 3.0]
        self.f = [1.0, 0.5, 2.0, 1.0]
        self.scene = make_scene(self.zs, self.L)
        t = target_terms(self.scene, self.f)
        # Signed, well-correlated but imperfect control.
        self.control = [t[0] * 1.02, t[1] * 0.98, t[2] - 0.01, t[3] * 1.01]
        self.control_sum = math.fsum(self.control)

    def test_pilot_and_main_streams_are_separate(self):
        # Same seed and main counts, different pilot counts: the main-phase
        # draws must be identical (main stream depends only on the seed),
        # while beta may differ.
        a = estimate_with_control(self.scene, self.f, self.control,
                                  self.control_sum, 4, 4, 12, 12, 77)
        b = estimate_with_control(self.scene, self.f, self.control,
                                  self.control_sum, 9, 7, 12, 12, 77)
        key = lambda rs: [(r.strategy, r.patch_index) for r in rs]
        self.assertEqual(key(a.records), key(b.records))
        self.assertNotEqual(a.beta, b.beta)  # beta follows the pilot only

    def test_beta_matches_pilot_only_recomputation(self):
        # beta is frozen after the pilot phase: recomputing the pooled
        # formula from the exposed pilot records must reproduce it exactly.
        n_pr, n_pc = 6, 4
        res = estimate_with_control(self.scene, self.f, self.control,
                                    self.control_sum, n_pr, n_pc, 10, 10, 5)
        self.assertEqual(len(res.pilot_records), n_pr + n_pc)
        pL, pC = proposal_pdfs(self.scene)
        by_strategy = {STRATEGY_RADIANCE: [], STRATEGY_COSINE: []}
        for rec in res.pilot_records:
            i = rec.patch_index
            denom = n_pr * pL[i] + n_pc * pC[i]
            x = rec.contribution
            y = rec.control_value / denom
            by_strategy[rec.strategy].append((x, y))
        num = den = 0.0
        for name in (STRATEGY_RADIANCE, STRATEGY_COSINE):
            pairs = by_strategy[name]
            n = len(pairs)
            if n < 2:
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            mx = math.fsum(xs) / n
            my = math.fsum(ys) / n
            cov = math.fsum((x - mx) * (y - my)
                            for x, y in zip(xs, ys)) / (n - 1)
            var = math.fsum((y - my) ** 2 for y in ys) / (n - 1)
            num += n * cov
            den += n * var
        self.assertNotEqual(den, 0.0)
        self.assertEqual(res.beta, num / den)
        # Pilot records carry the plain target term, not the residual.
        t = target_terms(self.scene, self.f)
        for rec in res.pilot_records:
            self.assertEqual(rec.target_value, t[rec.patch_index])

    def test_deterministic_per_seed(self):
        a = estimate_with_control(self.scene, self.f, self.control,
                                  self.control_sum, 4, 4, 8, 8, 42)
        b = estimate_with_control(self.scene, self.f, self.control,
                                  self.control_sum, 4, 4, 8, 8, 42)
        self.assertEqual(a, b)

    def test_budget_includes_pilot(self):
        res = estimate_with_control(self.scene, self.f, self.control,
                                    self.control_sum, 3, 5, 7, 9, 1)
        self.assertEqual(res.n_total, 3 + 5 + 7 + 9)
        self.assertEqual(res.n_pilot_radiance, 3)
        self.assertEqual(res.n_pilot_cosine, 5)
        self.assertEqual(len(res.pilot_records), 8)
        self.assertEqual(len(res.records), 16)

    def test_variance_definition_matches_records(self):
        # Variance 口径: sum_s n_s * s_s^2 over MAIN-phase residual
        # contributions, conditional on the frozen beta.
        res = estimate_with_control(self.scene, self.f, self.control,
                                    self.control_sum, 6, 6, 7, 5, 11)
        by_strategy = {}
        for rec in res.records:
            by_strategy.setdefault(rec.strategy, []).append(rec.contribution)
        variance = 0.0
        for own in by_strategy.values():
            n = len(own)
            mean = math.fsum(own) / n
            variance += n * math.fsum((x - mean) ** 2 for x in own) / (n - 1)
        self.assertTrue(math.isclose(res.variance, variance, rel_tol=1e-12))
        self.assertTrue(math.isclose(res.stddev, math.sqrt(variance),
                                     rel_tol=1e-12))

    def test_single_sample_main_strategy_variance_none(self):
        res = estimate_with_control(self.scene, self.f, self.control,
                                    self.control_sum, 4, 4, 1, 6, 3)
        self.assertIsNone(res.variance)
        self.assertIsNone(res.stddev)
        codes = [d.code for d in res.diagnostics]
        self.assertIn(DIAG_VARIANCE_NOT_ESTIMABLE, codes)


class TestPilotFallback(unittest.TestCase):
    def test_zero_control_gives_beta_zero_and_diagnostic(self):
        scene = make_scene([1.0, 0.7, 0.3], [1.0, 2.0, 0.5])
        f = [1.0, 0.5, 2.0]
        res = estimate_with_control(scene, f, [0.0, 0.0, 0.0], 0.0,
                                    4, 4, 8, 8, 9)
        self.assertEqual(res.beta, 0.0)
        hits = [d for d in res.diagnostics
                if d.code == DIAG_PILOT_ZERO_VARIANCE]
        self.assertEqual(len(hits), 1)
        self.assertIn("zero variance", hits[0].message)

    def test_constant_pilot_control_gives_beta_zero(self):
        # Single patch: every pilot control contribution is the same
        # constant, so the pilot variance is zero even with a nonzero
        # control; beta falls back to 0 with an explanation.
        scene = make_scene([0.6], [2.5], [0.75])
        res = estimate_with_control(scene, [1.5], [0.3], 0.3,
                                    5, 5, 4, 4, 2)
        self.assertEqual(res.beta, 0.0)
        codes = [d.code for d in res.diagnostics]
        self.assertIn(DIAG_PILOT_ZERO_VARIANCE, codes)
        # beta = 0: plain MIS on a single patch is exact.
        self.assertEqual(res.estimate, 2.5 * 1.5 * 0.6 * 0.75)

    def test_zero_pilot_counts_fall_back(self):
        scene = make_scene([1.0, 0.5], [1.0, 2.0])
        res = estimate_with_control(scene, [1.0, 1.0], [0.1, 0.2], 0.3,
                                    0, 0, 4, 4, 8)
        self.assertEqual(res.beta, 0.0)
        codes = [d.code for d in res.diagnostics]
        self.assertIn(DIAG_PILOT_ZERO_VARIANCE, codes)


class TestOldPathCompatibility(unittest.TestCase):
    def test_zero_control_reproduces_plain_estimate_sample_by_sample(self):
        zs = [1.0, 0.8, 0.5, 0.3]
        L = [1.0, 2.0, 0.5, 3.0]
        f = [1.0, 0.5, 2.0, 1.5]
        scene = make_scene(zs, L)
        n_r, n_c, seed = 6, 4, 99
        plain = estimate(scene, f, n_r, n_c, seed)
        ctrl = estimate_with_control(scene, f, [0.0] * 4, 0.0,
                                     3, 2, n_r, n_c, seed)
        self.assertEqual(ctrl.beta, 0.0)
        self.assertEqual(ctrl.estimate, plain.estimate)
        self.assertEqual(len(ctrl.records), len(plain.records))
        for a, b in zip(ctrl.records, plain.records):
            self.assertEqual(a.strategy, b.strategy)
            self.assertEqual(a.draw, b.draw)
            self.assertEqual(a.patch_index, b.patch_index)
            self.assertEqual(a.pdf_self, b.pdf_self)
            self.assertEqual(a.pdf_other, b.pdf_other)
            self.assertEqual(a.mis_weight, b.mis_weight)
            self.assertEqual(a.target_value, b.target_value)
            self.assertEqual(a.contribution, b.contribution)
            self.assertEqual(a.control_value, 0.0)
        self.assertEqual(ctrl.variance, plain.variance)


class TestControlDiagnostics(unittest.TestCase):
    def test_unreachable_control_term_reported(self):
        # Patch 0 is dark AND grazing (L = 0, cos = 0): no active strategy
        # can sample it. Its target term is 0, but its control term is not —
        # the residual estimator is biased there and must say so.
        scene = Scene([
            Patch(direction=(1.0, 0.0, 0.0), radiance=0.0, weight=1.0),
            Patch(direction=(0.0, 0.0, 1.0), radiance=1.0, weight=1.0),
            Patch(direction=unit(0.5), radiance=2.0, weight=1.0),
        ])
        f = [1.0, 1.0, 1.0]
        control = [0.5, 0.1, -0.2]
        res = estimate_with_control(scene, f, control, 0.4,
                                    4, 4, 6, 6, 13)
        hits = [d for d in res.diagnostics
                if d.code == DIAG_UNREACHABLE_TERM and d.patch_index == 0]
        self.assertEqual(len(hits), 1)
        self.assertIn("control term", hits[0].message)
        self.assertIn("0.5", hits[0].message)

    def test_zero_pdf_term_message_covers_control(self):
        # Patch 0 is dark (L = 0) but well-lit geometrically: the radiance
        # strategy never samples it; the diagnostic names the control term.
        scene = make_scene([1.0, 0.5], [0.0, 2.0])
        res = estimate_with_control(scene, [1.0, 1.0], [-0.3, 0.2], -0.1,
                                    4, 4, 4, 4, 9)
        hits = [d for d in res.diagnostics
                if d.code == DIAG_ZERO_PDF_TERM and d.patch_index == 0]
        self.assertEqual(len(hits), 1)
        self.assertIn(STRATEGY_RADIANCE, hits[0].message)
        self.assertIn("control term -0.3", hits[0].message)


class TestSeedSweep(unittest.TestCase):
    def test_control_reduces_mse_over_finite_seed_set(self):
        # Equal TOTAL budget (pilot counts against the control estimator).
        # Comparison is over the aggregate of 32 seeds: a single-seed
        # variance drop proves nothing, and no per-seed claim is made.
        zs = [1.0, 0.9, 0.75, 0.55, 0.35, 0.15]
        L = [1.0, 0.5, 2.0, 3.0, 0.25, 1.5]
        w = [1.0, 0.5, 1.0, 0.75, 1.25, 1.0]
        f = [1.0, 2.0, 0.5, 1.5, 0.75, 1.25]
        scene = make_scene(zs, L, w)
        t = target_terms(scene, f)
        # Strongly correlated signed control (~2% residual).
        control = [t[i] * (1.0 + 0.02 * ((i % 3) - 1)) for i in range(6)]
        control_sum = math.fsum(control)
        exact = reference_direct(L, f, zs, w)
        seeds = range(1000, 1032)
        plain_err2 = 0.0
        ctrl_err2 = 0.0
        for seed in seeds:
            plain = estimate(scene, f, 24, 24, seed)
            ctrl = estimate_with_control(scene, f, control, control_sum,
                                         8, 8, 16, 16, seed)
            self.assertEqual(ctrl.n_total, 48)
            plain_err2 += (plain.estimate - exact) ** 2
            ctrl_err2 += (ctrl.estimate - exact) ** 2
        n = len(list(seeds))
        mse_plain = plain_err2 / n
        mse_ctrl = ctrl_err2 / n
        self.assertLess(mse_ctrl, mse_plain,
                        f"control MSE {mse_ctrl!r} not below plain "
                        f"MSE {mse_plain!r} over {n} seeds")


class TestControlValidation(unittest.TestCase):
    def setUp(self):
        self.scene = make_scene([1.0, 0.5], [1.0, 2.0])
        self.f = [1.0, 1.0]
        self.control = [0.1, 0.2]

    def run_cv(self, **kw):
        args = dict(scene=self.scene, f=self.f, control=self.control,
                    control_sum=0.3, n_pilot_radiance=2, n_pilot_cosine=2,
                    n_radiance=2, n_cosine=2, seed=1)
        args.update(kw)
        return estimate_with_control(**args)

    def test_control_array_validation(self):
        with self.assertRaises(ValidationError):
            self.run_cv(control=[0.1])
        with self.assertRaises(ValidationError):
            self.run_cv(control="not-a-sequence")
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValidationError):
                self.run_cv(control=[0.1, bad])

    def test_control_sum_validation(self):
        for bad in (math.nan, math.inf, True, "0.3"):
            with self.assertRaises(ValidationError):
                self.run_cv(control_sum=bad)

    def test_pilot_count_validation(self):
        for bad in (True, -1, 1.5, "2"):
            with self.assertRaises(ValidationError):
                self.run_cv(n_pilot_radiance=bad)
            with self.assertRaises(ValidationError):
                self.run_cv(n_pilot_cosine=bad)

    def test_seed_validation(self):
        for bad in (True, 1.5, "7"):
            with self.assertRaises(ValidationError):
                self.run_cv(seed=bad)

    def test_pilot_degenerate_proposal(self):
        dark = make_scene([1.0, 0.5], [0.0, 0.0])
        with self.assertRaises(DegenerateProposalError):
            estimate_with_control(dark, self.f, self.control, 0.3,
                                  1, 0, 0, 4, 1)
        grazing = make_scene([0.0, 0.0], [1.0, 2.0])
        with self.assertRaises(DegenerateProposalError):
            estimate_with_control(grazing, self.f, self.control, 0.3,
                                  0, 1, 4, 0, 1)

    def test_error_message_locates_field(self):
        with self.assertRaises(ValidationError) as ctx:
            self.run_cv(control=[0.1, math.nan])
        self.assertIn("control[1]", str(ctx.exception))
        with self.assertRaises(ValidationError) as ctx:
            self.run_cv(control_sum=math.inf)
        self.assertIn("control_sum", str(ctx.exception))

    def test_negative_control_entries_accepted(self):
        res = self.run_cv(control=[-0.5, 0.7], control_sum=0.2)
        self.assertIsInstance(res.estimate, float)


if __name__ == "__main__":
    unittest.main()
