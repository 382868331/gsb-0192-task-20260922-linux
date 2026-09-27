"""Discrete environment-light multi-strategy importance estimation.

Public API is defined in mis_envlight.core; see README.md for the contract.
"""

from .core import (
    MAX_PATCHES,
    NORM_TOLERANCE,
    DEFAULT_NORMAL,
    STRATEGY_RADIANCE,
    STRATEGY_COSINE,
    DIAG_ZERO_PDF_TERM,
    DIAG_UNREACHABLE_TERM,
    DIAG_VARIANCE_NOT_ESTIMABLE,
    DIAG_UNUSED_ZERO_MASS_STRATEGY,
    MisEnvLightError,
    ValidationError,
    DegenerateProposalError,
    Patch,
    Scene,
    Diagnostic,
    SampleRecord,
    EstimateResult,
    ContributionTable,
    direct_sum,
    proposal_pdfs,
    enumerate_contributions,
    estimate,
)

__all__ = [
    "MAX_PATCHES",
    "NORM_TOLERANCE",
    "DEFAULT_NORMAL",
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
