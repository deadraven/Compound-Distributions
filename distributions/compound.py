"""
compound.py — Compound distributions built from simple ones via Monte Carlo.

Key functions
-------------
add(d1, d2)        : distribution of X + Y  (X ~ d1, Y ~ d2, independent)
sum_of_n(d, n)     : distribution of X1 + X2 + ... + Xn  (all iid ~ d)

Both return a CompoundDistribution object that supports .sample() and .pdf().

PDF estimation uses kernel density estimation (scipy.stats.gaussian_kde)
so it is always approximate; sample() is exact (by construction).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from .simple import Distribution


# Number of samples used internally to build the KDE for pdf()
_KDE_SAMPLES = 200_000


class CompoundDistribution(Distribution):
    """
    A distribution defined implicitly by a sampling function.

    Parameters
    ----------
    sampler     : callable(n) -> np.ndarray  —  exact sampler
    description : human-readable label
    kde_samples : how many samples to use when building the KDE for pdf()
    """

    def __init__(
        self,
        sampler,
        description: str = "CompoundDistribution",
        kde_samples: int = _KDE_SAMPLES,
    ) -> None:
        self._sampler = sampler
        self._description = description
        # Fit KDE once on a large batch so pdf() calls are cheap afterwards
        self._kde = stats.gaussian_kde(sampler(kde_samples))

    def sample(self, n: int) -> np.ndarray:
        """Draw n exact samples."""
        return self._sampler(n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Approximate PDF via kernel density estimation."""
        return self._kde(np.asarray(x))

    def __repr__(self) -> str:
        return self._description


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add(d1: Distribution, d2: Distribution) -> CompoundDistribution:
    """
    Return the distribution of X + Y where X ~ d1, Y ~ d2 (independent).

    Uses Monte Carlo: draw paired samples and sum them.

    Parameters
    ----------
    d1, d2 : any Distribution subclass (Gaussian, Uniform, Exponential, …)

    Returns
    -------
    CompoundDistribution representing d1 + d2
    """
    def _sampler(n: int) -> np.ndarray:
        return d1.sample(n) + d2.sample(n)

    return CompoundDistribution(
        sampler=_sampler,
        description=f"({d1} + {d2})",
    )


def sum_of_n(d: Distribution, n: int) -> CompoundDistribution:
    """
    Return the distribution of X1 + X2 + ... + Xn where each Xi ~ d (iid).

    By the Central Limit Theorem, this approaches Normal as n grows.

    Parameters
    ----------
    d : base Distribution
    n : number of iid terms to sum

    Returns
    -------
    CompoundDistribution representing the n-fold convolution of d with itself
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    def _sampler(size: int) -> np.ndarray:
        # Draw (size, n) matrix and sum along the n axis — fully vectorised
        return d.sample(size * n).reshape(size, n).sum(axis=1)

    return CompoundDistribution(
        sampler=_sampler,
        description=f"sum_of_{n}({d})",
    )
