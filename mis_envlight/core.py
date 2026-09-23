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
    "STRATEGY_RADIANCE",
    "STRATEGY_COSINE",
    "DIAG_ZERO_PDF_TERM",
    "DIAG_UNREACHABLE_TERM",
    "DIAG_VARIANCE_NOT_ESTIMABLE",
    "DIAG_UNUSED_ZERO_MASS_STRATEGY",
    "MisEnvLightError",
    "ValidationError",
    "DegenerateProposalError",
    "Patch",
    "Scene",
    "Diagnostic",
    "SampleRecord",
    "EstimateResult",
    "ContributionTable",
    "direct_sum",
    "proposal_pdfs",
    "enumerate_contributions",
    "estimate",
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
