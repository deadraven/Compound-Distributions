# Compound Distributions

A Python library for building and analysing compound probability distributions via Monte Carlo convolution. Simple distributions (Gaussian, Uniform, Exponential, TwoPoint) can be freely combined using `add()` and `sum_of_n()`, and the results behave like first-class distributions — they can be sampled, have their PDF evaluated, and chained into further compounds.

---

## Project Structure

```
compound_distributions/
├── distributions/
│   ├── __init__.py       # public exports
│   ├── simple.py         # Gaussian, Uniform, Exponential, TwoPoint
│   └── compound.py       # CompoundDistribution, add(), sum_of_n()
├── outputs/              # saved plots land here
├── notebook.ipynb        # interactive demo with 5 worked examples
└── README.md
```

---

## Quick Start

```python
from distributions import Gaussian, Uniform, Exponential, TwoPoint, add, sum_of_n

# Simple distributions
g = Gaussian(mu=0, sigma=1)
u = Uniform(a=0, b=1)
e = Exponential(lam=2)

# Compound: sum of any two
c = add(g, u)
print(c.sample(5))        # exact Monte Carlo samples
print(c.pdf([0.5, 1.0]))  # KDE-estimated density

# Compound: n iid copies (demonstrates CLT)
s = sum_of_n(u, 12)

# Chaining — result is still a Distribution
d = add(add(g, u), e)
```

---

## API

### Simple distributions

| Class | Parameters | Mean | Variance |
|---|---|---|---|
| `Gaussian(mu, sigma)` | μ, σ > 0 | μ | σ² |
| `Uniform(a, b)` | a < b | (a+b)/2 | (b−a)²/12 |
| `Exponential(lam)` | λ > 0 | 1/λ | 1/λ² |
| `TwoPoint(a, b, p)` | p ∈ (0,1) | pa+(1−p)b | p(1−p)(b−a)² |

Every class exposes:
- `sample(n) -> np.ndarray` — draw n iid samples (exact)
- `pdf(x) -> np.ndarray` — evaluate density at points x

### Compound functions

```python
add(d1, d2)       # distribution of X + Y, X ~ d1, Y ~ d2 (independent)
sum_of_n(d, n)    # distribution of X1 + X2 + ... + Xn, all iid ~ d
```

Both return a `CompoundDistribution` — a full `Distribution` subclass that supports `sample()`, `pdf()`, and further compounding.

PDF is estimated via **kernel density estimation** (scipy `gaussian_kde`) over 200k Monte Carlo samples, fitted once at construction. `sample()` is always exact.

---

## Mean and Variance of Compound Distributions

For any two **independent** random variables X and Y:

```
E[X + Y]   = E[X] + E[Y]
Var(X + Y) = Var(X) + Var(Y)
```

The full general formula includes a covariance term:

```
Var(X + Y) = Var(X) + Var(Y) + 2·Cov(X, Y)
```

Since `add()` always draws X and Y independently, `Cov(X, Y) = 0` and the simpler form holds exactly. The Monte Carlo estimates confirm this to ~3 decimal places.

For `sum_of_n(d, n)` where each Xᵢ ~ d with mean μ and variance σ²:

```
E[sum]   = n·μ
Var[sum] = n·σ²
```

---

## Getting Bimodal Distributions

Convolving two unimodal distributions (e.g. two Gaussians) always produces a unimodal result — the means add and the peaks merge. To get a **bimodal** compound, convolve with a `TwoPoint` distribution, which acts as a "splitter":

```python
g  = Gaussian(mu=0, sigma=1)
tp = TwoPoint(a=-10, b=10)   # P(X=-10) = P(X=10) = 0.5
bimodal = add(g, tp)         # two Gaussian bumps centred at -10 and +10
```

This works because convolving with a two-point distribution creates two shifted copies of the original distribution — one shifted by `a`, one by `b`.

**Note:** `add(g1, g2)` where g1 ~ N(−10,1) and g2 ~ N(10,1) is NOT bimodal. It equals N(0, √2) — the means cancel. This is a sum, not a mixture.

---

## Central Limit Theorem

`sum_of_n(d, n)` demonstrates the CLT: for any distribution d with finite mean and variance, the standardised sum converges to N(0,1) as n → ∞ — regardless of the shape of d.

```python
# Even a bimodal base distribution converges
base = add(Gaussian(0,1), TwoPoint(-10, 10))  # Var = 101
s50  = sum_of_n(base, 50)                      # nearly Gaussian
```

### Why it works — sketch of the proof

**1. Variance addition** establishes that the standardised sum Sₙ = (ΣXᵢ − nμ)/(σ√n) has mean 0 and variance 1 for all n.

**2. Characteristic functions** carry the shape information. The characteristic function of Sₙ factorises by independence:

```
φ_{Sn}(t) = [φ_{Zi}(t)]^n
```

where Zᵢ = (Xᵢ − μ)/(σ√n). Taylor-expanding to second order:

```
φ_{Zi}(t) = 1 − t²/(2n) + O(n^{-3/2})
```

**3. Taking n → ∞** using (1 + x/n)ⁿ → eˣ:

```
φ_{Sn}(t) → e^{−t²/2}
```

which is exactly the characteristic function of N(0,1). By the Lévy continuity theorem, distributional convergence follows.

The variance rule is necessary (it pins the first two moments) but not sufficient — the characteristic function argument proves convergence in full.

---

## Memory Warning: Avoid Deep Nesting

Each `CompoundDistribution` stores a KDE and its sampler calls the inner distribution's sampler. Nesting compounds multiplies sample sizes:

```python
# BAD — triple nesting blows up memory for large n
bimodal   = add(g, tp)
bimodal10 = sum_of_n(bimodal, 10)   # inner sampler called with 200k × 10
result    = sum_of_n(bimodal10, 50) # inner sampler called with 100k × 50 × 10 = 50M

# GOOD — single level of nesting
base   = add(g, tp)
result = sum_of_n(base, n)          # inner sampler called with 100k × n
```

Keep nesting to one level when working with large n.

---

## Dependencies

```
numpy
scipy
matplotlib
seaborn
jupyter
```

Install with:

```bash
pip install numpy scipy matplotlib seaborn jupyter --break-system-packages
```
