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
    "MAX_BATCHES",
    "NORM_TOLERANCE",
    "MIN_STRATEGY_WEIGHT",
    "STRATEGY_RADIANCE",
    "STRATEGY_COSINE",
    "DIAG_ZERO_PDF_TERM",
    "DIAG_UNREACHABLE_TERM",
    "DIAG_VARIANCE_NOT_ESTIMABLE",
    "DIAG_UNUSED_ZERO_MASS_STRATEGY",
    "DIAG_WEIGHTS_UPDATED",
    "DIAG_WEIGHTS_CLAMPED",
    "MisEnvLightError",
    "ValidationError",
    "DegenerateProposalError",
    "Patch",
    "Scene",
    "Diagnostic",
    "SampleRecord",
    "EstimateResult",
    "ContributionTable",
    "BatchRecord",
    "AdaptiveResult",
    "direct_sum",
    "proposal_pdfs",
    "enumerate_contributions",
    "estimate",
    "estimate_adaptive",
    "variance_minimizing_updater",
]

#: Maximum number of patches in a scene (interface bound from the task).
MAX_PATCHES = 64

#: Maximum number of batches in one adaptive run (interface bound).
MAX_BATCHES = 1024

#: Tolerance on |direction| - 1 when validating unit directions.
NORM_TOLERANCE = 1e-9

#: Lower bound on every strategy's sampling weight within a batch. Keeps the
#: per-batch mixture a genuine mix of both existing strategies; neither can
#: be driven to zero probability between batches. A weight pair is a ratio
#: (internally normalized); each normalized share must be >= this value.
MIN_STRATEGY_WEIGHT = 1e-3

STRATEGY_RADIANCE = "radiance"
STRATEGY_COSINE = "cosine"

#: Diagnostic codes (stable strings, compare against these constants).
DIAG_ZERO_PDF_TERM = "ZERO_PDF_TERM"
DIAG_UNREACHABLE_TERM = "UNREACHABLE_TERM"
DIAG_VARIANCE_NOT_ESTIMABLE = "VARIANCE_NOT_ESTIMABLE"
DIAG_UNUSED_ZERO_MASS_STRATEGY = "UNUSED_ZERO_MASS_STRATEGY"
DIAG_WEIGHTS_UPDATED = "WEIGHTS_UPDATED"
DIAG_WEIGHTS_CLAMPED = "WEIGHTS_CLAMPED"


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

    In an adaptive run every record additionally carries the index of the
    batch that produced it; the counts/pdfs above are the *conditional* ones
    in force when that batch was drawn and are frozen here, so later weight
    updates never recompute an earlier sample's contribution.
    """
    strategy: str
    draw: int
    patch_index: int
    n_self: int
    n_other: int
    pdf_self: float
    pdf_other: float
    mis_weight: float
    target_value: float
    contribution: float
    batch: int = 0

    def mixture_pdf(self):
        """Conditional mixture pdf of one draw, (n_self*pdf_self +
        n_other*pdf_other) / (n_self + n_other), frozen from sampling time."""
        total = self.n_self + self.n_other
        return (self.n_self * self.pdf_self
                + self.n_other * self.pdf_other) / total


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
                n_other=(n_cosine if strategy == STRATEGY_RADIANCE
                         else n_radiance),
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
# Multi-batch adaptive MIS
# ---------------------------------------------------------------------------
#
# A run consists of ``n_batches`` independent batches of ``samples_per_batch``
# draws. Within one batch the two existing strategies are mixed with fixed
# counts (n_radiance, n_cosine) and the ordinary balance heuristic, so each
# batch estimator E_b is unbiased for S on its own. The run estimator is the
# unweighted mean (1/B) * sum_b E_b, hence unbiased regardless of how the
# per-batch counts evolve.
#
# Between batches the caller may update the *sampling weights* (count split)
# for the next batch. Every sample is scored with the conditional mixture in
# force when it was drawn: its SampleRecord freezes that batch's counts and
# the (unchanging) per-strategy proposal pdfs. Aggregation only ever reads
# those frozen per-batch values; it never recomputes an old sample with the
# final weights (doing so would bias the estimator — see the exact-rational
# regression test).

@dataclass(frozen=True)
class BatchRecord:
    """One batch of an adaptive run (frozen at sampling time).

    weight_radiance/weight_cosine are the realized mixture shares, i.e. the
    actual integer count split (n_self / samples_per_batch). These are the
    conditional weights used to score this batch's samples; they are never
    rewritten by later updates.
    """
    batch: int
    samples_per_batch: int
    n_radiance: int
    n_cosine: int
    weight_radiance: float
    weight_cosine: float
    estimate: float
    variance: float | None
    records: tuple


@dataclass(frozen=True)
class AdaptiveResult:
    """Result of estimate_adaptive().

    estimate is the unweighted mean of the per-batch estimators. variance is
    the estimated variance of that mean (batches are sampled independently):
    (1/B^2) * sum_b Var(E_b). It is None when any active per-batch strategy
    stratum has a single sample, matching the single-shot convention.
    """
    estimate: float
    variance: float | None
    stddev: float | None
    exact: float
    n_batches: int
    samples_per_batch: int
    floor: float
    initial_weights: tuple
    final_weights: tuple
    batch_records: tuple
    records: tuple
    diagnostics: tuple

    def explain(self, limit=None):
        """Human-readable per-batch allocation summary."""
        lines = [
            f"estimate={self.estimate!r} exact={self.exact!r} "
            f"variance={self.variance!r} "
            f"(batches={self.n_batches}, samples/batch={self.samples_per_batch})",
            "batch  n_rad  n_cos  w_rad       w_cos       batch_est     batch_var",
        ]
        shown = self.batch_records if limit is None else self.batch_records[:limit]
        for b in shown:
            lines.append(
                f"{b.batch:<6} {b.n_radiance:<6} {b.n_cosine:<6} "
                f"{b.weight_radiance:<11.6g} {b.weight_cosine:<11.6g} "
                f"{b.estimate:<13.8g} {b.variance!r}")
        if limit is not None and limit < len(self.batch_records):
            lines.append(f"... ({len(self.batch_records) - limit} more batches)")
        return "\n".join(lines)


def _check_batch_params(n_batches, samples_per_batch, floor):
    n_batches = _check_count("n_batches", n_batches)
    if not 1 <= n_batches <= MAX_BATCHES:
        raise ValidationError(
            f"n_batches: must be in [1, {MAX_BATCHES}], got {n_batches}")
    m = _check_count("samples_per_batch", samples_per_batch)
    if m < 2:
        # Both strategies keep positive probability, so each needs >= 1 draw;
        # 2 is the smallest batch that admits the floor guarantee.
        raise ValidationError(
            f"samples_per_batch: must be >= 2 (both strategies stay active), "
            f"got {m}")
    fl = _check_real("floor", floor)
    if not 0.0 < fl <= 0.5:
        raise ValidationError(
            f"floor: must be in (0.0, 0.5], got {floor!r}")
    # The realized counts are integers; each strategy must receive at least
    # ceil(m*floor) draws for the bound to be realizable.
    lo = max(1, math.ceil(m * fl - 1e-12))
    if 2 * lo > m:
        raise ValidationError(
            f"floor={fl!r} is not realizable with samples_per_batch={m}: "
            f"each strategy would need {lo} draws but only {m} are available")
    return n_batches, m, fl, lo


def _check_initial_weights(weights, floor):
    name = "initial_weights"
    if isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence):
        raise ValidationError(
            f"{name}: expected a pair (w_radiance, w_cosine), "
            f"got {type(weights).__name__}")
    if len(weights) != 2:
        raise ValidationError(f"{name}: expected exactly 2 entries, got {len(weights)}")
    wr = _check_real(f"{name}[0]", weights[0])
    wc = _check_real(f"{name}[1]", weights[1])
    if wr <= 0.0 or wc <= 0.0:
        raise ValidationError(
            f"{name}: both weights must be > 0, got ({wr!r}, {wc!r})")
    s = wr + wc
    ar, ac = wr / s, wc / s
    if ar < floor or ac < floor:
        raise ValidationError(
            f"{name}: normalized shares ({ar!r}, {ac!r}) violate the "
            f"positive-probability floor {floor!r}")
    return ar, ac


def _split_counts(m, ar, lo):
    """Integer count split; each strategy gets at least lo draws."""
    n_r = int(math.floor(m * ar + 0.5))
    n_r = min(m - lo, max(lo, n_r))
    return n_r, m - n_r


def _coerce_updated_weights(raw, batch_index, floor):
    """Validate an updater return, then enforce the floor by clamping.

    Malformed values (wrong type/length, NaN, inf, negative, both zero) are
    rejected: the failure aborts the whole run before it is returned, so no
    caller can observe a half-updated run. Out-of-floor but well-formed
    shares are clamped onto [floor, 1-floor] (the documented positivity
    guarantee) and surface as a DIAG_WEIGHTS_CLAMPED diagnostic.
    """
    name = f"updater return (after batch {batch_index})"
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValidationError(
            f"{name}: expected a pair (w_radiance, w_cosine), "
            f"got {type(raw).__name__}")
    if len(raw) != 2:
        raise ValidationError(f"{name}: expected exactly 2 entries, got {len(raw)}")
    wr = _check_real(f"{name}[0]", raw[0], minimum=0.0)
    wc = _check_real(f"{name}[1]", raw[1], minimum=0.0)
    total = wr + wc
    if total <= 0.0:
        raise ValidationError(f"{name}: weights must not both be zero")
    ar, ac = wr / total, wc / total
    cr = min(1.0 - floor, max(floor, ar))
    cc = 1.0 - cr
    # Clamping actually changes the value iff the radiance share lies
    # outside [floor, 1-floor]; compare the decision inputs directly rather
    # than the renormalized floats (1-cr vs ac can differ by round-off even
    # when nothing was clamped).
    clamped = ar < floor or ar > 1.0 - floor
    return (cr, cc), clamped


def _run_batch(scene, g, pdf_radiance, pdf_cosine, rng,
               batch_index, m, ar, lo):
    """Draw and score one batch with its own frozen conditional mixture."""
    n_r, n_c = _split_counts(m, ar, lo)
    records = []
    per_strategy = {STRATEGY_RADIANCE: [], STRATEGY_COSINE: []}
    for strategy, count, pdf in (
            (STRATEGY_RADIANCE, n_r, pdf_radiance),
            (STRATEGY_COSINE, n_c, pdf_cosine)):
        cdf = _cdf(pdf)
        for draw in range(count):
            idx = _draw(cdf, rng)
            denom = n_r * pdf_radiance[idx] + n_c * pdf_cosine[idx]
            contrib = g[idx] / denom
            p_self = pdf[idx]
            p_other = (pdf_cosine[idx] if strategy == STRATEGY_RADIANCE
                       else pdf_radiance[idx])
            rec = SampleRecord(
                strategy=strategy,
                draw=draw,
                patch_index=idx,
                n_self=count,
                n_other=(n_c if strategy == STRATEGY_RADIANCE else n_r),
                pdf_self=p_self,
                pdf_other=p_other,
                mis_weight=count * p_self / denom,
                target_value=g[idx],
                contribution=contrib,
                batch=batch_index,
            )
            records.append(rec)
            per_strategy[strategy].append(contrib)
    est = math.fsum(r.contribution for r in records)
    variance = 0.0
    not_estimable = False
    for own in per_strategy.values():
        if len(own) == 1:
            not_estimable = True
            break
        variance += len(own) * _strategy_variance(own)
    return BatchRecord(
        batch=batch_index,
        samples_per_batch=m,
        n_radiance=n_r,
        n_cosine=n_c,
        weight_radiance=n_r / m,
        weight_cosine=n_c / m,
        estimate=est,
        variance=(None if not_estimable else variance),
        records=tuple(records),
    )


def estimate_adaptive(scene, f, n_batches, samples_per_batch, seed, *,
                      initial_weights=(0.5, 0.5),
                      floor=MIN_STRATEGY_WEIGHT,
                      updater=None):
    """Multi-batch MIS with weights that may change between batches.

    scene, f:        as in estimate().
    n_batches:       int in [1, MAX_BATCHES].
    samples_per_batch: int >= 2; each batch allocates its draws between the
                     two strategies and every batch keeps both active.
    seed:            int (bool rejected); same seed reproduces the whole run.
    initial_weights: positive ratio pair (w_radiance, w_cosine); normalized
                     internally, each share must be >= floor.
    floor:           positive-probability lower bound in (0, 0.5]; weights
                     returned by the updater are clamped onto
                     [floor, 1-floor] with a DIAG_WEIGHTS_CLAMPED diagnostic.
    updater:         callable(BatchRecord) -> (w_radiance, w_cosine), invoked
                     after every batch except the last; None keeps the
                     distribution fixed. Malformed returns raise
                     ValidationError and abort the run.

    Both strategies must be non-degenerate for the whole run (the floor keeps
    both active every batch), so a zero-mass proposal is rejected up front
    with DegenerateProposalError before any sampling happens.
    """
    _check_scene(scene)
    fv = _validate_f(scene, f)
    n_batches, m, fl, lo = _check_batch_params(n_batches, samples_per_batch, floor)
    ar, ac = _check_initial_weights(initial_weights, fl)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError(
            f"seed: expected an integer, got {type(seed).__name__}")
    if updater is not None and not callable(updater):
        raise ValidationError(
            f"updater: expected a callable or None, got {type(updater).__name__}")
    # Both strategies are active in every batch by construction, so both
    # proposal masses must be positive.
    _check_strategies(scene, 1, 1)

    g = _target_terms(scene, fv)
    pdf_radiance, pdf_cosine = proposal_pdfs(scene)
    diagnostics = list(_diagnose(scene, g, pdf_radiance, pdf_cosine, 1, 1))

    rng = random.Random(seed)
    batch_records = []
    cur = (ar, ac)
    for b in range(n_batches):
        brec = _run_batch(scene, g, pdf_radiance, pdf_cosine, rng,
                          b, m, cur[0], lo)
        if brec.variance is None:
            diagnostics.append(Diagnostic(
                DIAG_VARIANCE_NOT_ESTIMABLE,
                f"batch {b}: a strategy stratum has exactly 1 sample; its "
                f"variance contribution is not estimable"))
        batch_records.append(brec)
        if b < n_batches - 1 and updater is not None:
            raw = updater(brec)
            (nwr, nwc), clamped = _coerce_updated_weights(raw, b, fl)
            diagnostics.append(Diagnostic(
                DIAG_WEIGHTS_UPDATED,
                f"after batch {b}: next-batch weights "
                f"({nwr:.6g}, {nwc:.6g})"))
            if clamped:
                diagnostics.append(Diagnostic(
                    DIAG_WEIGHTS_CLAMPED,
                    f"after batch {b}: updater weights {tuple(raw)!r} violated "
                    f"the floor {fl!r}; clamped to ({nwr:.6g}, {nwc:.6g})"))
            cur = (nwr, nwc)

    est = math.fsum(b.estimate for b in batch_records) / n_batches
    if any(b.variance is None for b in batch_records):
        variance = None
    else:
        variance = math.fsum(b.variance for b in batch_records) / (
            n_batches * n_batches)
    all_records = tuple(r for b in batch_records for r in b.records)

    return AdaptiveResult(
        estimate=est,
        variance=variance,
        stddev=math.sqrt(variance) if variance is not None else None,
        exact=math.fsum(g),
        n_batches=n_batches,
        samples_per_batch=m,
        floor=fl,
        initial_weights=(ar, ac),
        final_weights=(batch_records[-1].weight_radiance,
                       batch_records[-1].weight_cosine),
        batch_records=tuple(batch_records),
        records=all_records,
        diagnostics=tuple(diagnostics),
    )


def variance_minimizing_updater():
    """Built-in updater: allocate the next batch proportional to the current
    batch's empirical per-strategy standard deviation of a single draw's
    balance-heuristic contribution (w_s ~ sigma_s).

    This is the standard variance-minimizing allocation heuristic; it is not
    a global optimum guarantee because each draw's value itself depends on
    the allocation. If a stratum cannot be estimated (fewer than 2 draws) the
    previous weights are kept. Zero empirical variance drives a weight to
    zero; the driver then clamps it back to the positivity floor.
    """
    def update(brec):
        by_strategy = {STRATEGY_RADIANCE: [], STRATEGY_COSINE: []}
        for r in brec.records:
            by_strategy[r.strategy].append(r.contribution)
        sigmas = []
        for xs in (by_strategy[STRATEGY_RADIANCE], by_strategy[STRATEGY_COSINE]):
            if len(xs) < 2:
                return (brec.weight_radiance, brec.weight_cosine)
            sigmas.append(math.sqrt(max(0.0, _strategy_variance(xs))))
        total = math.fsum(sigmas)
        if total <= 0.0:
            return (brec.weight_radiance, brec.weight_cosine)
        return (sigmas[0] / total, sigmas[1] / total)

    return update
