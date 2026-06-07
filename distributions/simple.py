"""
simple.py — Base class and simple probability distributions.

Each distribution exposes:
  - sample(n)  : draw n iid samples as a 1-D numpy array
  - pdf(x)     : evaluate the probability density at points x
  - __repr__   : human-readable description
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from scipy import stats


class Distribution(ABC):
    """Abstract base for all distributions."""

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Draw n iid samples."""

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the PDF at points x."""

    @abstractmethod
    def __repr__(self) -> str: ...


class Gaussian(Distribution):
    """
    Normal distribution N(mu, sigma^2).

    Parameters
    ----------
    mu    : mean
    sigma : standard deviation (> 0)
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        self.mu = mu
        self.sigma = sigma
        self._dist = stats.norm(loc=mu, scale=sigma)

    def sample(self, n: int) -> np.ndarray:
        return self._dist.rvs(size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.pdf(x)

    def __repr__(self) -> str:
        return f"Gaussian(mu={self.mu}, sigma={self.sigma})"


class Uniform(Distribution):
    """
    Uniform distribution U(a, b).

    Parameters
    ----------
    a : lower bound
    b : upper bound (> a)
    """

    def __init__(self, a: float = 0.0, b: float = 1.0) -> None:
        if b <= a:
            raise ValueError(f"b must be > a, got a={a}, b={b}")
        self.a = a
        self.b = b
        # scipy Uniform parameterised as loc=a, scale=b-a
        self._dist = stats.uniform(loc=a, scale=b - a)

    def sample(self, n: int) -> np.ndarray:
        return self._dist.rvs(size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.pdf(x)

    def __repr__(self) -> str:
        return f"Uniform(a={self.a}, b={self.b})"


class Exponential(Distribution):
    """
    Exponential distribution Exp(lambda).

    Parameters
    ----------
    lam : rate parameter lambda (> 0)
    """

    def __init__(self, lam: float = 1.0) -> None:
        if lam <= 0:
            raise ValueError(f"lam must be > 0, got {lam}")
        self.lam = lam
        # scipy uses scale = 1/lambda
        self._dist = stats.expon(scale=1.0 / lam)

    def sample(self, n: int) -> np.ndarray:
        return self._dist.rvs(size=n)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._dist.pdf(x)

    def __repr__(self) -> str:
        return f"Exponential(lam={self.lam})"

class TwoPoint(Distribution):
    """
    Discrete distribution: P(X = a) = p, P(X = b) = 1-p.
    Convolving any unimodal dist with this creates a bimodal result.
    """
    def __init__(self, a: float, b: float, p: float = 0.5) -> None:
        self.a, self.b, self.p = a, b, p

    def sample(self, n: int) -> np.ndarray:
        # Bernoulli selector between a and b
        return np.where(np.random.rand(n) < self.p, self.a, self.b)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        # Discrete — no true pdf; return zeros (use sample/KDE only)
        return np.zeros_like(x, dtype=float)

    def __repr__(self) -> str:
        return f"TwoPoint(a={self.a}, b={self.b}, p={self.p})"
        
