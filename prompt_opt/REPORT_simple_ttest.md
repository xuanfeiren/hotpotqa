# Validating the optimizer-improvement assumption — a one-sided t-test

**Assumption to validate.** For some $(\gamma, \delta_0)$,

$$
\Pr\!\big[\,\mu(\theta') > \mu(\theta) + \gamma\,\big] \;\ge\; \delta_0,
$$

where $\mu(\cdot)$ is the *true* (population) reward of a prompt and $\theta, \theta'$ are a parent and its child prompt produced in one optimizer step.

## 1. Setup (what we observe vs. what the assumption is about)

For each (parent, child) pair $i = 1, \ldots, N$, we observe

$$
D_i = \widehat r(\theta'_i) - \widehat r(\theta_i), \qquad \mathbb{E}[D_i] = \Delta_i := \mu(\theta'_i) - \mu(\theta_i),
$$

i.e. $D_i$ is an *unbiased noisy estimate* of the (unobservable) true gap $\Delta_i \in [-1, 1]$. The minibatch is the same for parent and child within each step, so the noise has zero mean. We have $N = 551$ pairs across three independent runs and treat them as i.i.d. observations.

## 2. Reduction to a test on E[Δ] (Markov)

Because $\Delta \le 1$,

$$
\mathbb{E}[\Delta] \;\le\; \gamma \cdot \Pr(\Delta \le \gamma) \;+\; 1 \cdot \Pr(\Delta > \gamma) \;=\; \gamma + (1 - \gamma)\,\Pr(\Delta > \gamma).
$$

Hence

$$
\Pr(\Delta > \gamma) \;\ge\; \delta_0 \;\;\Longleftarrow\;\; \mathbb{E}[\Delta] \;\ge\; c(\gamma, \delta_0) := \gamma + (1 - \gamma)\,\delta_0.
$$

So if we can lower-bound $\mathbb{E}[\Delta]$ by $c(\gamma, \delta_0)$ at 95% confidence, the assumption holds for that $(\gamma, \delta_0)$.

## 3. One-sided t-test

$D_i$ is unbiased for $\Delta_i$, so $\bar D = \tfrac{1}{N}\sum_i D_i$ is unbiased for $\mathbb{E}[\Delta]$. Test

$$
H_0:\ \mathbb{E}[\Delta] \le c(\gamma, \delta_0) \quad \text{vs.} \quad H_1:\ \mathbb{E}[\Delta] > c(\gamma, \delta_0),
$$

$$
t \;=\; \frac{\bar D - c(\gamma, \delta_0)}{s / \sqrt{N}}.
$$

Reject $H_0$ at level 0.05 iff $t > t^{\star}_{0.05,\,N-1} \approx 1.648$. Equivalently, $(\gamma, \delta_0)$ is certified iff $c(\gamma, \delta_0) < L$, where

$$
L \;:=\; \bar D - t^{\star}_{0.05,\,N-1}\cdot \tfrac{s}{\sqrt{N}}
$$

is the 95% one-sided lower confidence bound on $\mathbb{E}[\Delta]$.

## 4. Result on our 3-run data

$$
N = 551, \qquad \bar D = 0.0580, \qquad s = 0.2546, \qquad SE = 0.0108, \qquad L = 0.0401.
$$

The certifiable region is $\{(\gamma, \delta_0):\ \gamma + (1-\gamma)\delta_0 < 0.0401\}$.

### 4.1 Frontier: largest $\delta_0$ we can certify at each $\gamma$

| $\gamma$ | $\delta_0^{\max}$ certifiable @ 95% |
|:---:|:---:|
| 0.000 | **0.040** |
| 0.005 | 0.035 |
| 0.010 | 0.030 |
| 0.015 | 0.026 |
| 0.020 | 0.021 |
| 0.025 | 0.016 |
| 0.030 | 0.010 |
| 0.035 | 0.005 |
| ≥ 0.040 | — (vacuous) |

### 4.2 Grid of $(\gamma, \delta_0)$ pass/fail at 95%

| $\gamma\backslash\delta_0$ | 0.005 | 0.010 | 0.015 | 0.020 | 0.025 | 0.030 | 0.035 | 0.040 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.000 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 0.005 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| 0.010 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| 0.015 | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | — |
| 0.020 | ✓ | ✓ | ✓ | ✓ | — | — | — | — |
| 0.025 | ✓ | ✓ | ✓ | — | — | — | — | — |
| 0.030 | ✓ | ✓ | — | — | — | — | — | — |
| 0.035 | ✓ | — | — | — | — | — | — | — |

## 5. Strongest certified claims

At 95% confidence we can certify any $(\gamma, \delta_0)$ on the frontier; in particular,

$$
\Pr\!\big[\mu(\theta') > \mu(\theta)\big] \;\ge\; 0.040,
$$

$$
\Pr\!\big[\mu(\theta') > \mu(\theta) + 0.020\big] \;\ge\; 0.020,
$$

$$
\Pr\!\big[\mu(\theta') > \mu(\theta) + 0.035\big] \;\ge\; 0.005.
$$

For $\gamma \ge 0.04$ the unconditional test is vacuous: $\bar D$ is pulled down by the many already-saturated parents ($\mu(\theta) \approx 1$), so the Markov reduction $\bar D \ge \gamma + (1-\gamma)\delta_0$ cannot hold. This is exactly the regime the original assumption excludes via the side condition $\mu(\theta) \le B - \gamma$ — without that conditioning, the unconditional Markov reduction can only certify weak $(\gamma, \delta_0)$ on the frontier $\gamma + (1-\gamma)\delta_0 < 0.040$.
