"""Discrete environment-light multi-strategy importance estimation.

Estimates the hemispheric sum

    S = sum_i  L_i * f_i * cos_i * w_i

over a scene of at most ``MAX_PATCHES`` discrete patches, using multiple
importance sampling (balance heuristic) with two proposal distributions:

  * radiance strategy: p_i proportional to L_i * w_i
  * cosine strategy:   q_i proportional to cos_i * w_i

Under the scene contract the patches have equal solid angle, so including
``w_i`` does not change the normalized proposals; it is kept so that
non-uniform weights remain correct.

Public contract (see README.md for the full specification):

  * numeric inputs must be finite (NaN/Infinity are rejected);
  * integer parameters reject bool;
  * invalid input raises ValidationError with a message locating the field;
  * requesting samples from a strategy whose total proposal mass is zero
    raises DegenerateProposalError;
  * terms that no active strategy can sample are reported as diagnostics,
    never silently dropped.

A two-phase control-variate extension (``estimate_with_control``) accepts a
caller-provided control array ``g`` (per-patch control terms, any sign) with
its exact discrete integral ``G``: a pilot phase on a separate random stream
estimates a fixed coefficient ``beta``, then the main phase estimates
``sum_i (t_i - beta*g_i)`` by plain MIS and adds back ``beta*G``.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "MAX_PATCHES",
    "NORM_TOLERANCE",
    "STRATEGY_RADIANCE",
    "STRATEGY_COSINE",
    "DIAG_ZERO_PDF_TERM",
    "DIAG_UNREACHABLE_TERM",
    "DIAG_VARIANCE_NOT_ESTIMABLE",
    "DIAG_UNUSED_ZERO_MASS_STRATEGY",
    "DIAG_PILOT_ZERO_VARIANCE",
    "MisEnvLightError",
    "ValidationError",
    "DegenerateProposalError",
    "Patch",
    "Scene",
    "Diagnostic",
    "SampleRecord",
    "EstimateResult",
    "ControlEstimateResult",
    "ContributionTable",
    "direct_sum",
    "proposal_pdfs",
    "enumerate_contributions",
    "estimate",
    "estimate_with_control",
]

#: Maximum number of patches in a scene (interface bound from the task).
MAX_PATCHES = 64

#: Tolerance on |direction| - 1 when validating unit directions.
NORM_TOLERANCE = 1e-9

STRATEGY_RADIANCE = "radiance"
STRATEGY_COSINE = "cosine"

#: Diagnostic codes (stable strings, compare against these constants).
DIAG_ZERO_PDF_TERM = "ZERO_PDF_TERM"
DIAG_UNREACHABLE_TERM = "UNREACHABLE_TERM"
DIAG_VARIANCE_NOT_ESTIMABLE = "VARIANCE_NOT_ESTIMABLE"
DIAG_UNUSED_ZERO_MASS_STRATEGY = "UNUSED_ZERO_MASS_STRATEGY"
DIAG_PILOT_ZERO_VARIANCE = "PILOT_ZERO_VARIANCE"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class MisEnvLightError(Exception):
    """Base class for all errors raised by mis_envlight."""


class ValidationError(MisEnvLightError, ValueError):
    """An input violates the documented contract.

    The message always names the offending field (e.g. ``patches[3].radiance``)
    so the caller can locate the problem. Nothing is silently corrected.
    """


class DegenerateProposalError(MisEnvLightError):
    """A requested sampling strategy has zero total probability mass."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_real(name, value, *, minimum=None, strict_minimum=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{name}: expected a real number, got {type(value).__name__}")
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise ValidationError(f"{name}: must be finite, got {value!r}")
    if minimum is not None:
        if strict_minimum and not v > minimum:
            raise ValidationError(f"{name}: must be > {minimum}, got {v!r}")
        if not strict_minimum and not v >= minimum:
            raise ValidationError(f"{name}: must be >= {minimum}, got {v!r}")
    return v


def _check_count(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            f"{name}: expected an integer sample count, got {type(value).__name__}")
    if value < 0:
        raise ValidationError(f"{name}: must be >= 0, got {value}")
    return value


def _check_direction(name, value):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError(
            f"{name}: expected a sequence of 3 numbers, got {type(value).__name__}")
    if len(value) != 3:
        raise ValidationError(f"{name}: expected 3 components, got {len(value)}")
    x = _check_real(f"{name}[0]", value[0])
    y = _check_real(f"{name}[1]", value[1])
    z = _check_real(f"{name}[2]", value[2])
    norm = math.sqrt(x * x + y * y + z * z)
    if abs(norm - 1.0) > NORM_TOLERANCE:
        raise ValidationError(
            f"{name}: expected a unit vector, |v| = {norm!r} "
            f"(tolerance {NORM_TOLERANCE})")
    if z < 0.0:
        raise ValidationError(
            f"{name}: z must be >= 0 (upper hemisphere), got {z!r}")
    return (x, y, z)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Patch:
    """One hemispheric patch.

    direction: unit 3-vector (x, y, z) with z >= 0 (upper hemisphere).
    radiance:  finite, >= 0.
    weight:    finite, > 0 (solid-angle weight of the patch).
    """
    direction: tuple
    radiance: float
    weight: float


class Scene:
    """Validated, immutable set of 1..MAX_PATCHES hemispheric patches."""

    __slots__ = ("directions", "radiance", "weight", "cosine",
                 "lum_mass", "cos_mass", "lum_total", "cos_total")

    def __init__(self, patches):
        if isinstance(patches, (str, bytes)) or not isinstance(patches, Sequence):
            raise ValidationError(
                f"patches: expected a sequence of Patch, got {type(patches).__name__}")
        patches = list(patches)
        if not patches:
            raise ValidationError("patches: scene must contain at least 1 patch")
        if len(patches) > MAX_PATCHES:
            raise ValidationError(
                f"patches: at most {MAX_PATCHES} patches allowed, got {len(patches)}")
        directions, radiance, weight, cosine = [], [], [], []
        for i, p in enumerate(patches):
            if not isinstance(p, Patch):
                raise ValidationError(
                    f"patches[{i}]: expected Patch, got {type(p).__name__}")
            d = _check_direction(f"patches[{i}].direction", p.direction)
            lum = _check_real(f"patches[{i}].radiance", p.radiance, minimum=0.0)
            w = _check_real(f"patches[{i}].weight", p.weight,
                            minimum=0.0, strict_minimum=True)
            directions.append(d)
            radiance.append(lum)
            weight.append(w)
            cosine.append(d[2])
        self.directions = tuple(directions)
        self.radiance = tuple(radiance)
        self.weight = tuple(weight)
        self.cosine = tuple(cosine)
        self.lum_mass = tuple(l * w for l, w in zip(self.radiance, self.weight))
        self.cos_mass = tuple(c * w for c, w in zip(self.cosine, self.weight))
        self.lum_total = math.fsum(self.lum_mass)
        self.cos_total = math.fsum(self.cos_mass)

    def __len__(self):
        return len(self.radiance)

    def __repr__(self):
        return f"Scene({len(self)} patches)"


def _check_scene(scene):
    if not isinstance(scene, Scene):
        raise ValidationError(
            f"scene: expected a Scene, got {type(scene).__name__}")


def _validate_f(scene, f):
    if isinstance(f, (str, bytes)) or not isinstance(f, Sequence):
        raise ValidationError(
            f"f: expected a sequence of {len(scene)} finite non-negative numbers, "
            f"got {type(f).__name__}")
    if len(f) != len(scene):
        raise ValidationError(f"f: expected {len(scene)} entries, got {len(f)}")
    return tuple(_check_real(f"f[{i}]", v, minimum=0.0) for i, v in enumerate(f))


def _validate_control(scene, control):
    """Control array: per-patch control term values g_i (finite, any sign).

    Unlike f, control entries may be negative: the control is a signed
    approximation to the target summand, not a physical quantity.
    """
    if isinstance(control, (str, bytes)) or not isinstance(control, Sequence):
        raise ValidationError(
            f"control: expected a sequence of {len(scene)} finite numbers, "
            f"got {type(control).__name__}")
    if len(control) != len(scene):
        raise ValidationError(
            f"control: expected {len(scene)} entries, got {len(control)}")
    return tuple(_check_real(f"control[{i}]", v)
                 for i, v in enumerate(control))


def _check_strategies(scene, n_radiance, n_cosine):
    n_radiance = _check_count("n_radiance", n_radiance)
    n_cosine = _check_count("n_cosine", n_cosine)
    if n_radiance + n_cosine == 0:
        raise ValidationError("n_radiance + n_cosine: at least one sample is required")
    if n_radiance > 0 and scene.lum_total == 0.0:
        raise DegenerateProposalError(
            "radiance strategy requested (n_radiance > 0) but sum(L*w) is zero; "
            "the proposal distribution is undefined")
    if n_cosine > 0 and scene.cos_total == 0.0:
        raise DegenerateProposalError(
            "cosine strategy requested (n_cosine > 0) but sum(cos*w) is zero; "
            "the proposal distribution is undefined")
    return n_radiance, n_cosine


def _target_terms(scene, f):
    return tuple(l * fi * c * w
                 for l, fi, c, w in zip(scene.radiance, f, scene.cosine, scene.weight))


# ---------------------------------------------------------------------------
# Deterministic reference
# ---------------------------------------------------------------------------

def direct_sum(scene, f):
    """Exact target value S = sum_i L_i * f_i * cos_i * w_i (math.fsum)."""
    _check_scene(scene)
    fv = _validate_f(scene, f)
    return math.fsum(_target_terms(scene, fv))


def proposal_pdfs(scene):
    """Return (pdf_radiance, pdf_cosine) as tuples of per-patch probabilities.

    A strategy whose total mass is zero yields an all-zero tuple; requesting
    samples from it is rejected separately by _check_strategies.
    """
    _check_scene(scene)
    if scene.lum_total > 0.0:
        pdf_radiance = tuple(m / scene.lum_total for m in scene.lum_mass)
    else:
        pdf_radiance = (0.0,) * len(scene)
    if scene.cos_total > 0.0:
        pdf_cosine = tuple(m / scene.cos_total for m in scene.cos_mass)
    else:
        pdf_cosine = (0.0,) * len(scene)
    return pdf_radiance, pdf_cosine


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Diagnostic:
    """A non-fatal finding about the scene/strategy combination.

    code is one of the DIAG_* constants; patch_index is None when the
    diagnostic is not tied to a single patch.
    """
    code: str
    message: str
    patch_index: int | None = None


def _diagnose(scene, g, pdf_radiance, pdf_cosine, n_radiance, n_cosine,
              control=None):
    """Zero-support diagnostics for target terms g and (optional) control terms.

    control, when given, holds the per-patch control term values (any sign);
    a nonzero control term that no active strategy can sample biases the
    control-variate estimator just like a nonzero unreachable target term.
    """
    diags = []
    active = []
    if n_radiance > 0:
        active.append((STRATEGY_RADIANCE, pdf_radiance))
    if n_cosine > 0:
        active.append((STRATEGY_COSINE, pdf_cosine))
    for i in range(len(scene)):
        zero = [name for name, pdf in active if pdf[i] == 0.0]
        if not zero:
            continue
        c = control[i] if control is not None else 0.0
        if len(zero) == len(active) and (g[i] > 0.0 or c != 0.0):
            # Both-strategies-zero-but-term-nonzero: the estimator cannot
            # see this term and is biased. For the plain target this is
            # unreachable through the public API with validated inputs
            # (g_i > 0 implies L_i > 0 and cos_i > 0; kept as a defensive
            # guard), but a caller-supplied control term CAN be nonzero
            # where every proposal has zero support.
            parts = []
            if g[i] > 0.0:
                parts.append(f"target term {g[i]!r} > 0")
            if c != 0.0:
                parts.append(f"control term {c!r} != 0")
            diags.append(Diagnostic(
                DIAG_UNREACHABLE_TERM,
                f"patch {i}: {' and '.join(parts)} but every active "
                f"strategy has zero probability there; the estimator cannot "
                f"see this term and will be biased",
                i))
        else:
            # Zero-probability term: at least one active strategy never
            # samples this patch. Harmless when the target term is zero or
            # another active strategy covers it; reported for transparency.
            covered = any(name not in zero for name, _ in active)
            control_note = f", control term {c!r}" if control is not None else ""
            diags.append(Diagnostic(
                DIAG_ZERO_PDF_TERM,
                f"patch {i}: zero probability under "
                f"{' and '.join(zero)!r} (target term {g[i]!r}{control_note}"
                f"{', covered by another active strategy' if covered else ''})",
                i))
    if n_radiance == 0 and scene.lum_total == 0.0:
        diags.append(Diagnostic(
            DIAG_UNUSED_ZERO_MASS_STRATEGY,
            "radiance strategy unused and sum(L*w) is zero"))
    if n_cosine == 0 and scene.cos_total == 0.0:
        diags.append(Diagnostic(
            DIAG_UNUSED_ZERO_MASS_STRATEGY,
            "cosine strategy unused and sum(cos*w) is zero"))
    return diags


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleRecord:
    """Per-sample explanation of one MIS draw.

    contribution = target_value / (n_radiance*pdf_radiance + n_cosine*pdf_cosine)
    mis_weight   = n_self * pdf_self / (same denominator)   (balance heuristic)
    The estimator is the sum of all records' contributions.

    control_value is the per-patch control term g_i at the sampled patch
    (0.0 when no control variate is in use). In the control-variate main
    phase target_value is the residual term t_i - beta*g_i; in the pilot
    phase it is the plain target term t_i.
    """
    strategy: str
    draw: int
    patch_index: int
    n_self: int
    pdf_self: float
    pdf_other: float
    mis_weight: float
    target_value: float
    contribution: float
    control_value: float = 0.0


@dataclass(frozen=True)
class EstimateResult:
    """Result of estimate().

    variance is the estimated variance of the estimator (of the sum, not of a
    single sample); it is None when any active strategy has exactly one
    sample, because that strategy's variance contribution is then not
    estimable (see DIAG_VARIANCE_NOT_ESTIMABLE).
    """
    estimate: float
    variance: float | None
    stddev: float | None
    exact: float
    n_radiance: int
    n_cosine: int
    records: tuple
    diagnostics: tuple

    def explain(self, limit=None):
        """Human-readable per-sample weight explanation."""
        records = self.records if limit is None else self.records[:limit]
        lines = [
            f"estimate={self.estimate!r} exact={self.exact!r} "
            f"variance={self.variance!r} "
            f"(n_radiance={self.n_radiance}, n_cosine={self.n_cosine})",
            "idx  strategy  patch  target_g        pdf_self   pdf_other  "
            "mis_weight  contribution",
        ]
        for k, r in enumerate(records):
            lines.append(
                f"{k:<4} {r.strategy:<9} {r.patch_index:<6} "
                f"{r.target_value:<15.8g} {r.pdf_self:<10.4g} {r.pdf_other:<10.4g} "
                f"{r.mis_weight:<11.6g} {r.contribution:<12.8g}")
        shown = len(records)
        if shown < len(self.records):
            lines.append(f"... ({len(self.records) - shown} more samples)")
        return "\n".join(lines)


def _cdf(pdf):
    acc = 0.0
    out = []
    for p in pdf:
        acc += p
        out.append(acc)
    out[-1] = 1.0  # guard against accumulated round-off at the last bin
    return out


def _draw(cdf, rng):
    i = bisect.bisect_left(cdf, rng.random())
    return min(i, len(cdf) - 1)


def _strategy_variance(xs):
    n = len(xs)
    mean = math.fsum(xs) / n
    return math.fsum((x - mean) ** 2 for x in xs) / (n - 1)


def _run_mis(terms, control_terms, n_radiance, n_cosine,
             pdf_radiance, pdf_cosine, rng):
    """Draw MIS samples for per-patch terms; shared by all estimators.

    Draws the radiance strategy first, then cosine, consuming ``rng`` in
    that fixed order. Returns (records, contribs, control_contribs) where
    contribs / control_contribs map strategy name to per-sample
    contributions (control_contribs is None when control_terms is None).
    """
    records = []
    contribs = {}
    control_contribs = {} if control_terms is not None else None
    for strategy, count, pdf in (
            (STRATEGY_RADIANCE, n_radiance, pdf_radiance),
            (STRATEGY_COSINE, n_cosine, pdf_cosine)):
        if count == 0:
            continue
        cdf = _cdf(pdf)
        own = []
        own_control = []
        for draw in range(count):
            idx = _draw(cdf, rng)
            denom = n_radiance * pdf_radiance[idx] + n_cosine * pdf_cosine[idx]
            # denom > 0 here: the active strategy that sampled idx has
            # pdf > 0 at idx, and its count is positive.
            contrib = terms[idx] / denom
            p_self = pdf[idx]
            p_other = (pdf_cosine[idx] if strategy == STRATEGY_RADIANCE
                       else pdf_radiance[idx])
            c = control_terms[idx] if control_terms is not None else 0.0
            records.append(SampleRecord(
                strategy=strategy,
                draw=draw,
                patch_index=idx,
                n_self=count,
                pdf_self=p_self,
                pdf_other=p_other,
                mis_weight=count * p_self / denom,
                target_value=terms[idx],
                contribution=contrib,
                control_value=c,
            ))
            own.append(contrib)
            if control_terms is not None:
                own_control.append(c / denom)
        contribs[strategy] = own
        if control_terms is not None:
            control_contribs[strategy] = own_control
    return records, contribs, control_contribs


def _pooled_variance(contribs, diagnostics):
    """Estimated variance of the estimator: sum_s n_s * s_s^2.

    Returns None (and appends DIAG_VARIANCE_NOT_ESTIMABLE) when any active
    strategy has exactly one sample.
    """
    variance = 0.0
    for name, own in contribs.items():
        if len(own) == 1:
            diagnostics.append(Diagnostic(
                DIAG_VARIANCE_NOT_ESTIMABLE,
                f"strategy {name!r} has exactly 1 sample; its variance "
                f"contribution is not estimable, so no variance is reported"))
            return None
        variance += len(own) * _strategy_variance(own)
    return variance


def estimate(scene, f, n_radiance, n_cosine, seed):
    """Run the MIS estimator with a deterministic seed.

    scene:      Scene instance.
    f:          sequence of len(scene) finite, non-negative numbers.
    n_radiance: int >= 0, samples from the radiance proposal (~ L*w).
    n_cosine:   int >= 0, samples from the cosine proposal (~ cos*w).
    seed:       int (bool rejected); same seed reproduces identical results.

    All inputs are validated before any sampling happens, so a failed call
    leaves no partial state behind.
    """
    _check_scene(scene)
    fv = _validate_f(scene, f)
    n_radiance, n_cosine = _check_strategies(scene, n_radiance, n_cosine)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError(
            f"seed: expected an integer, got {type(seed).__name__}")

    g = _target_terms(scene, fv)
    pdf_radiance, pdf_cosine = proposal_pdfs(scene)
    diagnostics = _diagnose(scene, g, pdf_radiance, pdf_cosine,
                            n_radiance, n_cosine)

    rng = random.Random(seed)
    records, contribs, _ = _run_mis(g, None, n_radiance, n_cosine,
                                    pdf_radiance, pdf_cosine, rng)
    est = math.fsum(c for own in contribs.values() for c in own)
    variance = _pooled_variance(contribs, diagnostics)

    return EstimateResult(
        estimate=est,
        variance=variance,
        stddev=math.sqrt(variance) if variance is not None else None,
        exact=math.fsum(g),
        n_radiance=n_radiance,
        n_cosine=n_cosine,
        records=tuple(records),
        diagnostics=tuple(diagnostics),
    )


# ---------------------------------------------------------------------------
# Control-variate estimator (two-phase: pilot for beta, then main estimate)
# ---------------------------------------------------------------------------

#: Fixed 64-bit mixing constant (fractional bits of the golden ratio) used to
#: derive the pilot RNG seed from the caller's seed by pure integer
#: arithmetic, so both streams are deterministic across runs and processes.
_PILOT_SEED_MIX = 0x9E3779B97F4A7C15


def _pilot_seed(seed):
    """Derive the pilot-phase seed from the caller's seed (splitmix64 step).

    The pilot phase uses random.Random(_pilot_seed(seed)) while the main
    phase uses random.Random(seed): two independent streams, and the main
    stream is exactly the one the plain estimator uses.
    """
    return (seed * 6364136223846793005 + _PILOT_SEED_MIX) % 2**64


def _estimate_beta(pilot_contribs, pilot_control_contribs):
    """Optimal fixed beta from pilot samples, or None when not estimable.

    The estimator is sum_k (x_k - beta * y_k) + beta * G with per-sample
    target contributions x_k and control contributions y_k. Samples within
    a strategy are i.i.d., so Var of the total is sum_s n_s Var_s(x - beta*y),
    minimized by beta = (sum_s n_s Cov_s(x, y)) / (sum_s n_s Var_s(y)).
    Strategies with fewer than 2 pilot samples cannot contribute a variance
    and are skipped. Returns None when the denominator is zero (zero pilot
    variance of the control, or no strategy with >= 2 pilot samples).
    """
    num = 0.0
    den = 0.0
    for name, xs in pilot_contribs.items():
        ys = pilot_control_contribs[name]
        n = len(xs)
        if n < 2:
            continue
        mx = math.fsum(xs) / n
        my = math.fsum(ys) / n
        cov = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
        var = math.fsum((y - my) ** 2 for y in ys) / (n - 1)
        num += n * cov
        den += n * var
    if den == 0.0:
        return None
    return num / den


@dataclass(frozen=True)
class ControlEstimateResult:
    """Result of estimate_with_control().

    estimate          = residual_estimate + beta * control_sum.
    variance/stddev   estimated variance of the estimator, conditional on the
                      pilot-estimated beta (beta is frozen after the pilot
                      phase and treated as a constant; the exact add-back
                      beta*control_sum contributes no variance, and pilot
                      sampling noise is not propagated). Computed from the
                      MAIN phase residual contributions only, as
                      sum_s n_s * s_s^2 per strategy; None when any active
                      main strategy has exactly one sample.
    n_total           total sampling budget spent, pilot included.
    records           main-phase samples (target_value = residual term).
    pilot_records     pilot-phase samples (target_value = plain target term).
    """
    estimate: float
    variance: float | None
    stddev: float | None
    exact: float
    beta: float
    residual_estimate: float
    control_sum: float
    n_pilot_radiance: int
    n_pilot_cosine: int
    n_radiance: int
    n_cosine: int
    n_total: int
    records: tuple
    pilot_records: tuple
    diagnostics: tuple

    def explain(self, limit=None):
        """Human-readable two-phase summary plus per-sample weights."""
        records = self.records if limit is None else self.records[:limit]
        lines = [
            f"estimate={self.estimate!r} exact={self.exact!r} "
            f"variance={self.variance!r} beta={self.beta!r}",
            f"residual_estimate={self.residual_estimate!r} "
            f"control_sum={self.control_sum!r} "
            f"(n_pilot_radiance={self.n_pilot_radiance}, "
            f"n_pilot_cosine={self.n_pilot_cosine}, "
            f"n_radiance={self.n_radiance}, n_cosine={self.n_cosine}, "
            f"n_total={self.n_total})",
            "idx  strategy  patch  residual_term   control_term  pdf_self   "
            "pdf_other  mis_weight  contribution",
        ]
        for k, r in enumerate(records):
            lines.append(
                f"{k:<4} {r.strategy:<9} {r.patch_index:<6} "
                f"{r.target_value:<15.8g} {r.control_value:<13.8g} "
                f"{r.pdf_self:<10.4g} {r.pdf_other:<10.4g} "
                f"{r.mis_weight:<11.6g} {r.contribution:<12.8g}")
        shown = len(records)
        if shown < len(self.records):
            lines.append(f"... ({len(self.records) - shown} more samples)")
        return "\n".join(lines)


def estimate_with_control(scene, f, control, control_sum,
                          n_pilot_radiance, n_pilot_cosine,
                          n_radiance, n_cosine, seed):
    """Two-phase control-variate MIS estimator with a deterministic seed.

    scene:      Scene instance.
    f:          sequence of len(scene) finite, non-negative numbers.
    control:    sequence of len(scene) finite numbers (ANY sign) — the
                per-patch control terms g_i, a signed approximation to the
                target summands t_i = L_i*f_i*cos_i*w_i.
    control_sum: exact discrete integral G = sum_i g_i (finite, any sign);
                caller-provided and trusted, never recomputed here.
    n_pilot_radiance / n_pilot_cosine:
                pilot-phase sample counts (int >= 0), spent ONLY on
                estimating beta; they are part of the total budget.
    n_radiance / n_cosine:
                main-phase sample counts, same contract as estimate().
    seed:       int (bool rejected). The pilot phase draws from
                random.Random(_pilot_seed(seed)) and the main phase from
                random.Random(seed) — two separate streams, and the main
                stream is identical to the plain estimator's, so with a
                zero control (beta = 0) the main-phase samples reproduce
                estimate() sample by sample.

    Phase 1 (pilot): estimate a FIXED beta minimizing the pilot-estimated
    variance of the residual estimator. If the pilot control contributions
    have zero variance (or no strategy has >= 2 pilot samples), beta is set
    to 0.0 and DIAG_PILOT_ZERO_VARIANCE explains why.

    Phase 2 (main): estimate sum_i (t_i - beta*g_i) by plain MIS with the
    unchanged proposals and the unchanged balance-heuristic denominator
    (the residual is kept signed — never replaced by its absolute value),
    then add back the exact beta*control_sum. beta does NOT adapt to
    main-phase samples.

    All inputs are validated before any sampling happens, so a failed call
    leaves no partial state behind.
    """
    _check_scene(scene)
    fv = _validate_f(scene, f)
    cv = _validate_control(scene, control)
    control_sum = _check_real("control_sum", control_sum)
    n_pilot_radiance = _check_count("n_pilot_radiance", n_pilot_radiance)
    n_pilot_cosine = _check_count("n_pilot_cosine", n_pilot_cosine)
    if n_pilot_radiance > 0 and scene.lum_total == 0.0:
        raise DegenerateProposalError(
            "pilot radiance strategy requested (n_pilot_radiance > 0) but "
            "sum(L*w) is zero; the proposal distribution is undefined")
    if n_pilot_cosine > 0 and scene.cos_total == 0.0:
        raise DegenerateProposalError(
            "pilot cosine strategy requested (n_pilot_cosine > 0) but "
            "sum(cos*w) is zero; the proposal distribution is undefined")
    n_radiance, n_cosine = _check_strategies(scene, n_radiance, n_cosine)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError(
            f"seed: expected an integer, got {type(seed).__name__}")

    t = _target_terms(scene, fv)
    pdf_radiance, pdf_cosine = proposal_pdfs(scene)
    diagnostics = _diagnose(scene, t, pdf_radiance, pdf_cosine,
                            n_radiance, n_cosine, control=cv)

    # Phase 1: pilot samples from a separate stream, used only for beta.
    pilot_rng = random.Random(_pilot_seed(seed))
    pilot_records, pilot_x, pilot_y = _run_mis(
        t, cv, n_pilot_radiance, n_pilot_cosine,
        pdf_radiance, pdf_cosine, pilot_rng)
    beta = _estimate_beta(pilot_x, pilot_y)
    if beta is None:
        beta = 0.0
        diagnostics.append(Diagnostic(
            DIAG_PILOT_ZERO_VARIANCE,
            "pilot phase cannot estimate beta: the pilot control "
            "contributions have zero variance (or no strategy has >= 2 "
            "pilot samples); beta is fixed to 0.0, so the estimator "
            "reduces to plain MIS and the pilot budget is spent without "
            "effect"))

    # Phase 2: main samples from the plain estimator's stream; beta frozen.
    residual = tuple(ti - beta * ci for ti, ci in zip(t, cv))
    rng = random.Random(seed)
    records, contribs, _ = _run_mis(residual, cv, n_radiance, n_cosine,
                                    pdf_radiance, pdf_cosine, rng)
    residual_est = math.fsum(c for own in contribs.values() for c in own)
    est = residual_est + beta * control_sum
    variance = _pooled_variance(contribs, diagnostics)

    return ControlEstimateResult(
        estimate=est,
        variance=variance,
        stddev=math.sqrt(variance) if variance is not None else None,
        exact=math.fsum(t),
        beta=beta,
        residual_estimate=residual_est,
        control_sum=control_sum,
        n_pilot_radiance=n_pilot_radiance,
        n_pilot_cosine=n_pilot_cosine,
        n_radiance=n_radiance,
        n_cosine=n_cosine,
        n_total=(n_pilot_radiance + n_pilot_cosine
                 + n_radiance + n_cosine),
        records=tuple(records),
        pilot_records=tuple(pilot_records),
        diagnostics=tuple(diagnostics),
    )


# ---------------------------------------------------------------------------
# Deterministic enumeration (analysis / verification aid)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContributionTable:
    """Per-(strategy, patch) MIS contributions for the balance heuristic.

    radiance[i] is the contribution of one radiance-strategy sample that
    draws patch i: g_i / (n_radiance*pdf_radiance_i + n_cosine*pdf_cosine_i).
    It is None where the strategy never samples (pdf 0). The full estimator
    expectation is sum over active strategies of sum_i pdf_i * table_i,
    which analytically equals direct_sum(scene, f).
    """
    n_radiance: int
    n_cosine: int
    pdf_radiance: tuple
    pdf_cosine: tuple
    radiance: tuple | None
    cosine: tuple | None


def enumerate_contributions(scene, f, n_radiance, n_cosine):
    """Deterministic per-(strategy, patch) contribution table.

    Same validation and error semantics as estimate(), but no sampling.
    """
    _check_scene(scene)
    fv = _validate_f(scene, f)
    n_radiance, n_cosine = _check_strategies(scene, n_radiance, n_cosine)
    g = _target_terms(scene, fv)
    pdf_radiance, pdf_cosine = proposal_pdfs(scene)
    denom = tuple(n_radiance * pr + n_cosine * pc
                  for pr, pc in zip(pdf_radiance, pdf_cosine))

    def row(pdf):
        return tuple(g[i] / denom[i] if pdf[i] > 0.0 else None
                     for i in range(len(scene)))

    return ContributionTable(
        n_radiance=n_radiance,
        n_cosine=n_cosine,
        pdf_radiance=pdf_radiance,
        pdf_cosine=pdf_cosine,
        radiance=row(pdf_radiance) if n_radiance > 0 else None,
        cosine=row(pdf_cosine) if n_cosine > 0 else None,
    )
