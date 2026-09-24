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
    "DEFAULT_MIX_FLOOR",
    "STRATEGY_RADIANCE",
    "STRATEGY_COSINE",
    "DIAG_ZERO_PDF_TERM",
    "DIAG_UNREACHABLE_TERM",
    "DIAG_VARIANCE_NOT_ESTIMABLE",
    "DIAG_UNUSED_ZERO_MASS_STRATEGY",
    "DIAG_ZERO_MASS_COMPONENT",
    "MisEnvLightError",
    "ValidationError",
    "DegenerateProposalError",
    "Patch",
    "Scene",
    "Diagnostic",
    "SampleRecord",
    "EstimateResult",
    "ContributionTable",
    "BatchSpec",
    "BatchSampleRecord",
    "BatchResult",
    "BatchSummary",
    "BatchedEstimateResult",
    "BatchedEstimator",
    "direct_sum",
    "proposal_pdfs",
    "enumerate_contributions",
    "estimate",
    "estimate_batched",
]

#: Maximum number of patches in a scene (interface bound from the task).
MAX_PATCHES = 64

#: Tolerance on |direction| - 1 when validating unit directions.
NORM_TOLERANCE = 1e-9

#: Default lower bound on every mixture component: each batch must use a
#: radiance fraction alpha in [DEFAULT_MIX_FLOOR, 1 - DEFAULT_MIX_FLOOR],
#: so neither existing strategy is ever assigned zero probability.
DEFAULT_MIX_FLOOR = 0.05

STRATEGY_RADIANCE = "radiance"
STRATEGY_COSINE = "cosine"

#: Diagnostic codes (stable strings, compare against these constants).
DIAG_ZERO_PDF_TERM = "ZERO_PDF_TERM"
DIAG_UNREACHABLE_TERM = "UNREACHABLE_TERM"
DIAG_VARIANCE_NOT_ESTIMABLE = "VARIANCE_NOT_ESTIMABLE"
DIAG_UNUSED_ZERO_MASS_STRATEGY = "UNUSED_ZERO_MASS_STRATEGY"
DIAG_ZERO_MASS_COMPONENT = "ZERO_MASS_COMPONENT"


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


def _diagnose(scene, g, pdf_radiance, pdf_cosine, n_radiance, n_cosine):
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
        if len(zero) == len(active) and g[i] > 0.0:
            # Both-strategies-zero-but-target-nonzero: the estimator cannot
            # see this term and is biased. (Unreachable through the public
            # API with validated inputs, since g_i > 0 implies L_i > 0 and
            # cos_i > 0; kept as a defensive guard.)
            diags.append(Diagnostic(
                DIAG_UNREACHABLE_TERM,
                f"patch {i}: target term {g[i]!r} > 0 but every active "
                f"strategy has zero probability there; the estimator cannot "
                f"see this term and will be biased",
                i))
        else:
            # Zero-probability term: at least one active strategy never
            # samples this patch. Harmless when the target term is zero or
            # another active strategy covers it; reported for transparency.
            covered = any(name not in zero for name, _ in active)
            diags.append(Diagnostic(
                DIAG_ZERO_PDF_TERM,
                f"patch {i}: zero probability under "
                f"{' and '.join(zero)!r} (target term {g[i]!r}"
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
    records = []
    contribs = {}
    for strategy, count, pdf in (
            (STRATEGY_RADIANCE, n_radiance, pdf_radiance),
            (STRATEGY_COSINE, n_cosine, pdf_cosine)):
        if count == 0:
            continue
        cdf = _cdf(pdf)
        own = []
        for draw in range(count):
            idx = _draw(cdf, rng)
            denom = n_radiance * pdf_radiance[idx] + n_cosine * pdf_cosine[idx]
            # denom > 0 here: the active strategy that sampled idx has
            # pdf > 0 at idx, and its count is positive.
            contrib = g[idx] / denom
            p_self = pdf[idx]
            p_other = (pdf_cosine[idx] if strategy == STRATEGY_RADIANCE
                       else pdf_radiance[idx])
            records.append(SampleRecord(
                strategy=strategy,
                draw=draw,
                patch_index=idx,
                n_self=count,
                pdf_self=p_self,
                pdf_other=p_other,
                mis_weight=count * p_self / denom,
                target_value=g[idx],
                contribution=contrib,
            ))
            own.append(contrib)
        contribs[strategy] = own

    est = math.fsum(c for own in contribs.values() for c in own)

    variance = 0.0
    for name, own in contribs.items():
        if len(own) == 1:
            variance = None
            diagnostics.append(Diagnostic(
                DIAG_VARIANCE_NOT_ESTIMABLE,
                f"strategy {name!r} has exactly 1 sample; its variance "
                f"contribution is not estimable, so no variance is reported"))
            break
        variance += len(own) * _strategy_variance(own)

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


# ---------------------------------------------------------------------------
# Batched adaptive finite-mixture sampling
# ---------------------------------------------------------------------------
#
# Each batch draws from a *finite mixture* of the two existing proposals:
#
#     pdf_b(i) = alpha_b * p_i + (1 - alpha_b) * q_i
#
# where alpha_b is chosen per batch (it may change after every finished
# batch) and is constrained to  mix_floor <= alpha_b <= 1 - mix_floor, so
# neither component strategy is ever assigned zero probability.
#
# A draw from batch b contributes  g_i / pdf_b(i).  The pdf actually used
# (and alpha) are frozen onto the sample record at draw time; the final
# estimate only ever reads those frozen values. It must never recompute an
# old sample's pdf from a later batch's mixture.
#
# With N total draws the unbiased estimator is
#
#     S_hat = (1/N) * sum_samples g_i / pdf_b(i)
#
# Conditional on the sampling history (which fixes alpha_b), each draw has
# expectation S, hence the estimator is unbiased also under adaptive alphas.

@dataclass(frozen=True)
class BatchSpec:
    """One planned batch: n_samples draws from mixture weight alpha.

    alpha is the radiance-strategy fraction; cosine gets (1 - alpha).
    """
    n_samples: int
    alpha: float


@dataclass(frozen=True)
class BatchSampleRecord:
    """One draw, with the conditional mixture pdf frozen at draw time.

    pdf_mixture is pdf_b(patch_index) for the batch this draw belonged to;
    contribution = target_value / pdf_mixture. Later batches that change
    alpha never mutate these fields.
    """
    batch_index: int
    draw: int
    patch_index: int
    alpha: float
    pdf_radiance: float
    pdf_cosine: float
    pdf_mixture: float
    target_value: float
    contribution: float


@dataclass(frozen=True)
class BatchResult:
    """State of one finished batch, including its frozen conditional pdf."""
    batch_index: int
    n_samples: int
    alpha: float
    pdf: tuple
    records: tuple
    contribution_sum: float
    mean: float


@dataclass(frozen=True)
class BatchSummary:
    """Per-batch summary carried by a BatchedEstimateResult."""
    batch_index: int
    n_samples: int
    alpha: float
    pdf: tuple
    contribution_sum: float
    mean: float


@dataclass(frozen=True)
class BatchedEstimateResult:
    """Result of batched adaptive mixture sampling.

    estimate is the average (1/N) * sum of the frozen per-draw terms.
    variance estimates the variance of that average; it is None when any
    batch contains exactly one sample, whose within-batch variance is not
    estimable (see DIAG_VARIANCE_NOT_ESTIMABLE).
    """
    estimate: float
    variance: float | None
    stddev: float | None
    exact: float
    total_samples: int
    n_batches: int
    mix_floor: float
    records: tuple
    batches: tuple
    diagnostics: tuple

    def explain(self, limit=None):
        """Human-readable per-sample explanation with batch information."""
        records = self.records if limit is None else self.records[:limit]
        lines = [
            f"batched estimate={self.estimate!r} exact={self.exact!r} "
            f"variance={self.variance!r} "
            f"(batches={self.n_batches}, n={self.total_samples}, "
            f"mix_floor={self.mix_floor})",
            "batch  draw  patch  alpha    pdf_rad    pdf_cos    "
            "pdf_mix     target_g        contribution",
        ]
        for r in records:
            lines.append(
                f"{r.batch_index:<6} {r.draw:<5} {r.patch_index:<6} "
                f"{r.alpha:<8.4g} {r.pdf_radiance:<10.4g} "
                f"{r.pdf_cosine:<10.4g} {r.pdf_mixture:<11.4g} "
                f"{r.target_value:<15.8g} {r.contribution:<12.8g}")
        shown = len(records)
        if shown < len(self.records):
            lines.append(f"... ({len(self.records) - shown} more samples)")
        return "\n".join(lines)


def _check_mix_floor(mix_floor):
    if isinstance(mix_floor, bool) or not isinstance(mix_floor, (int, float)):
        raise ValidationError(
            "mix_floor: expected a real number, got "
            f"{type(mix_floor).__name__}")
    e = float(mix_floor)
    if math.isnan(e) or math.isinf(e):
        raise ValidationError(f"mix_floor: must be finite, got {mix_floor!r}")
    if not e > 0.0:
        raise ValidationError(
            f"mix_floor: must be > 0 (a positive probability floor is "
            f"required), got {e!r}")
    if e >= 0.5:
        raise ValidationError(
            "mix_floor: must be < 0.5, otherwise no feasible mixture "
            f"weight exists, got {e!r}")
    return e


def _check_alpha(alpha, mix_floor, *, name="alpha"):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValidationError(
            f"{name}: expected a real mixture weight, got {type(alpha).__name__}")
    a = float(alpha)
    if math.isnan(a) or math.isinf(a):
        raise ValidationError(f"{name}: must be finite, got {alpha!r}")
    hi = 1.0 - mix_floor
    if not (mix_floor <= a <= hi):
        raise ValidationError(
            f"{name}: mixture weight of the radiance strategy must stay in "
            f"[{mix_floor}, {hi}] to keep a positive probability floor on "
            f"both strategies, got {a!r}")
    return a


def _mixture_pdf(alpha, pdf_radiance, pdf_cosine, lum_total, cos_total):
    """Conditional mixture pdf, renormalized over the live components.

    With both strategies live this is alpha*p + (1-alpha)*q (mass 1). When
    one component has zero total mass it carries no weight and the *other*
    component's effective weight is renormalized to 1, so the mixture is a
    proper distribution regardless of alpha (which is then irrelevant).
    """
    w_r = alpha if lum_total > 0.0 else 0.0
    w_c = (1.0 - alpha) if cos_total > 0.0 else 0.0
    scale = w_r + w_c  # > 0: the constructor rejects both-zero scenes
    wr, wc = w_r / scale, w_c / scale
    return tuple(wr * pr + wc * pc
                 for pr, pc in zip(pdf_radiance, pdf_cosine))


class BatchedEstimator:
    """Stateful runner: one batch at a time, weights updated between batches.

    Typical use:::

        est = BatchedEstimator(scene, f, seed=1)
        b0 = est.run_batch(n_samples=16, alpha=0.5)      # explore
        a1 = choose_next_weight(b0)                      # any user rule
        est.run_batch(n_samples=16, alpha=a1)            # adapted batch
        result = est.result()

    A failed run_batch validates everything before touching the state or
    consuming random numbers, so a rejected batch leaves no partial update
    and does not perturb the reproducible stream.
    """

    __slots__ = ("scene", "f", "mix_floor", "g", "pdf_radiance",
                 "pdf_cosine", "_rng", "_records", "_batches",
                 "_diagnostics")

    def __init__(self, scene, f, *, seed, mix_floor=DEFAULT_MIX_FLOOR):
        _check_scene(scene)
        fv = _validate_f(scene, f)
        floor = _check_mix_floor(mix_floor)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValidationError(
                f"seed: expected an integer, got {type(seed).__name__}")
        if scene.lum_total == 0.0 and scene.cos_total == 0.0:
            raise DegenerateProposalError(
                "both strategies have zero total proposal mass "
                "(sum(L*w) = 0 and sum(cos*w) = 0); no mixture is defined")

        self.scene = scene
        self.f = fv
        self.mix_floor = floor
        self.g = _target_terms(scene, fv)
        self.pdf_radiance, self.pdf_cosine = proposal_pdfs(scene)
        self._rng = random.Random(seed)
        self._records = []
        self._batches = []
        diags = []
        if scene.lum_total == 0.0:
            diags.append(Diagnostic(
                DIAG_ZERO_MASS_COMPONENT,
                "radiance strategy has zero total mass; the mixture is "
                "driven by the cosine strategy regardless of alpha"))
        if scene.cos_total == 0.0:
            diags.append(Diagnostic(
                DIAG_ZERO_MASS_COMPONENT,
                "cosine strategy has zero total mass; the mixture is "
                "driven by the radiance strategy regardless of alpha"))
        self._diagnostics = diags

    @property
    def total_samples(self):
        return len(self._records)

    @property
    def n_batches(self):
        return len(self._batches)

    def run_batch(self, n_samples, alpha):
        """Draw n_samples from the mixture alpha*p + (1-alpha)*q.

        Validates fully before any state change or random draw, records the
        frozen conditional pdf on every sample, and returns BatchResult.
        """
        n = _check_count("n_samples", n_samples)
        if n < 1:
            raise ValidationError(
                f"n_samples: a batch needs at least 1 sample, got {n}")
        a = _check_alpha(alpha, self.mix_floor)

        pdf = _mixture_pdf(a, self.pdf_radiance, self.pdf_cosine,
                           self.scene.lum_total, self.scene.cos_total)
        cdf = _cdf(pdf)  # total mass is 1: at least one component is valid
        batch_index = len(self._batches)
        records = []
        terms = []
        for draw in range(n):
            idx = _draw(cdf, self._rng)
            pdf_here = pdf[idx]
            # pdf_here > 0: idx came from cdf, which only spans positive mass.
            contrib = self.g[idx] / pdf_here
            records.append(BatchSampleRecord(
                batch_index=batch_index,
                draw=draw,
                patch_index=idx,
                alpha=a,
                pdf_radiance=self.pdf_radiance[idx],
                pdf_cosine=self.pdf_cosine[idx],
                pdf_mixture=pdf_here,
                target_value=self.g[idx],
                contribution=contrib,
            ))
            terms.append(contrib)

        batch = BatchResult(
            batch_index=batch_index,
            n_samples=n,
            alpha=a,
            pdf=pdf,
            records=tuple(records),
            contribution_sum=math.fsum(terms),
            mean=math.fsum(terms) / n,
        )
        # Commit only after the whole batch has been computed.
        self._batches.append(batch)
        self._records.extend(records)
        return batch

    def result(self):
        """Collect all finished batches into a BatchedEstimateResult."""
        if not self._batches:
            raise ValidationError(
                "result: no batches have been run; call run_batch first")
        return _build_batched_result(
            g=self.g,
            mix_floor=self.mix_floor,
            records=self._records,
            batches=self._batches,
            base_diagnostics=self._diagnostics)


def _batch_diagnostics(g, batches, base_diagnostics):
    """Zero-pdf / unreachable findings over the union of frozen batch pdfs.

    A positive target term missed by every batch means bias; a zero-target
    patch that some batch cannot reach is merely informational.
    """
    diags = list(base_diagnostics)
    n_patches = len(g)
    for i in range(n_patches):
        missed = [b.batch_index for b in batches if b.pdf[i] == 0.0]
        if not missed:
            continue
        if len(missed) == len(batches) and g[i] > 0.0:
            diags.append(Diagnostic(
                DIAG_UNREACHABLE_TERM,
                f"patch {i}: target term {g[i]!r} > 0 but the frozen "
                f"mixture pdf is zero in every batch ({len(missed)}); "
                f"the estimator cannot see this term and will be biased",
                i))
        else:
            diags.append(Diagnostic(
                DIAG_ZERO_PDF_TERM,
                f"patch {i}: zero mixture pdf in batch(es) {missed} "
                f"(target term {g[i]!r})",
                i))
    return diags


def _build_batched_result(g, mix_floor, records, batches, base_diagnostics):
    n_total = len(records)
    # Estimate straight from the contributions frozen at draw time.
    est = math.fsum(r.contribution for r in records) / n_total

    diagnostics = _batch_diagnostics(g, batches, base_diagnostics)
    variance = 0.0
    for b in batches:
        xs = [r.contribution for r in b.records]
        if len(xs) == 1:
            variance = None
            diagnostics.append(Diagnostic(
                DIAG_VARIANCE_NOT_ESTIMABLE,
                f"batch {b.batch_index} has exactly 1 sample; its "
                f"within-batch variance is not estimable, so no variance "
                f"is reported"))
            break
        # Stratified variance of the global mean:
        # Var(mean) = (1/N^2) * sum_b n_b * s_b^2, with s_b^2 the
        # unbiased within-batch sample variance.
        variance += len(xs) * _strategy_variance(xs)
    if variance is not None:
        variance /= n_total * n_total

    summaries = tuple(
        BatchSummary(batch_index=b.batch_index, n_samples=b.n_samples,
                     alpha=b.alpha, pdf=b.pdf,
                     contribution_sum=b.contribution_sum, mean=b.mean)
        for b in batches)
    return BatchedEstimateResult(
        estimate=est,
        variance=variance,
        stddev=math.sqrt(variance) if variance is not None else None,
        exact=math.fsum(g),
        total_samples=n_total,
        n_batches=len(batches),
        mix_floor=mix_floor,
        records=tuple(records),
        batches=summaries,
        diagnostics=tuple(diagnostics),
    )


def _coerce_batch_spec(name, spec, mix_floor):
    if isinstance(spec, BatchSpec):
        n, alpha = spec.n_samples, spec.alpha
    elif isinstance(spec, (str, bytes)) or not isinstance(spec, Sequence):
        raise ValidationError(
            f"{name}: expected (n_samples, alpha) or BatchSpec, got "
            f"{type(spec).__name__}")
    elif len(spec) != 2:
        raise ValidationError(
            f"{name}: expected (n_samples, alpha) with 2 entries, got "
            f"{len(spec)}")
    else:
        n, alpha = spec
    n = _check_count(f"{name}.n_samples", n)
    if n < 1:
        raise ValidationError(
            f"{name}.n_samples: a batch needs at least 1 sample, got {n}")
    alpha = _check_alpha(alpha, mix_floor, name=f"{name}.alpha")
    return BatchSpec(n_samples=n, alpha=alpha)


def estimate_batched(scene, f, batches, seed, mix_floor=DEFAULT_MIX_FLOOR):
    """Run several mixture batches in one call; weights may differ per batch.

    batches: non-empty sequence of (n_samples, alpha) pairs or BatchSpec.
    All specs are validated up front (before any random number is drawn), so
    an invalid plan raises without leaving partial state.
    """
    _check_scene(scene)
    fv = _validate_f(scene, f)
    floor = _check_mix_floor(mix_floor)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError(
            f"seed: expected an integer, got {type(seed).__name__}")
    if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
        raise ValidationError(
            f"batches: expected a sequence of (n_samples, alpha) specs, got "
            f"{type(batches).__name__}")
    if len(batches) == 0:
        raise ValidationError("batches: at least one batch is required")
    specs = [_coerce_batch_spec(f"batches[{i}]", b, floor)
             for i, b in enumerate(batches)]

    runner = BatchedEstimator(scene, fv, seed=seed, mix_floor=floor)
    for spec in specs:
        runner.run_batch(spec.n_samples, spec.alpha)
    return runner.result()
