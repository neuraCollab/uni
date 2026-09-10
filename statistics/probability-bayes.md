# Probability & Bayes' Theorem

## What

The math of quantifying uncertainty: axioms of probability, conditional probability, independence, and Bayes' theorem — the tool for updating beliefs given new evidence. Bayes' theorem (specifically the "false positive" medical-test flavor) is the single most commonly asked probability question in DS/ML interviews.

## Probability axioms (Kolmogorov)

For a sample space `Ω` and event `A ⊆ Ω`:

1. `P(A) ≥ 0` — probability is non-negative.
2. `P(Ω) = 1` — something in the sample space happens with certainty.
3. For mutually exclusive events `A₁, A₂, ...`: `P(A₁ ∪ A₂ ∪ ...) = P(A₁) + P(A₂) + ...` (countable additivity).

Everything else (e.g. `P(Aᶜ) = 1 − P(A)`, `P(A ∪ B) = P(A) + P(B) − P(A ∩ B)`) follows from these three.

## Conditional probability

Probability of `A` given that `B` has occurred:

```
P(A | B) = P(A ∩ B) / P(B),   provided P(B) > 0
```

Rearranged, this gives the **multiplication rule**: `P(A ∩ B) = P(A | B) · P(B)`.

## Independence

`A` and `B` are independent iff knowing `B` tells you nothing about `A`:

```
P(A | B) = P(A)   <=>   P(A ∩ B) = P(A) · P(B)
```

**Common trap**: independence and mutual exclusivity are opposite in spirit. If `A` and `B` are mutually exclusive (`P(A ∩ B) = 0`) and both have nonzero probability, they are *not* independent — knowing one occurred tells you the other definitely didn't.

## Law of total probability

If `B₁, B₂, ..., Bₙ` partition the sample space (mutually exclusive, collectively exhaustive), then for any event `A`:

```
P(A) = Σᵢ P(A | Bᵢ) · P(Bᵢ)
```

This is the workhorse for computing an "unconditional" probability by conditioning on every way it could happen — and it's the denominator you compute right before applying Bayes' theorem.

## Bayes' theorem

### Derivation

From the multiplication rule, `P(A ∩ B)` can be written two ways:

```
P(A | B) · P(B) = P(A ∩ B) = P(B | A) · P(A)
```

Divide through by `P(B)`:

```
P(A | B) = P(B | A) · P(A) / P(B)
```

Expand the denominator with the law of total probability (`P(B) = P(B|A)P(A) + P(B|Aᶜ)P(Aᶜ)`) to get the full form:

```
P(A | B) = P(B | A) · P(A) / [P(B | A) · P(A) + P(B | Aᶜ) · P(Aᶜ)]
```

Naming convention: `P(A)` = **prior**, `P(B | A)` = **likelihood**, `P(A | B)` = **posterior**, `P(B)` = **evidence** (normalizing constant).

### Worked example: the classic false-positive problem

> A disease affects 1% of the population. A test for it is 95% sensitive (catches 95% of people who have the disease) and 90% specific (correctly clears 90% of people who don't have it). A random person tests positive. What's the probability they actually have the disease?

Define events: `D` = has disease, `T` = tests positive.

- Prior: `P(D) = 0.01`, so `P(Dᶜ) = 0.99`
- Sensitivity (true positive rate): `P(T | D) = 0.95`
- Specificity: `P(Tᶜ | Dᶜ) = 0.90`, so the false positive rate `P(T | Dᶜ) = 0.10`

Apply Bayes:

```
P(D | T) = P(T | D) P(D) / [P(T | D) P(D) + P(T | Dᶜ) P(Dᶜ)]
         = (0.95 × 0.01) / [(0.95 × 0.01) + (0.10 × 0.99)]
         = 0.0095 / (0.0095 + 0.099)
         = 0.0095 / 0.1085
         ≈ 0.0876  →  about 8.8%
```

**The punchline** (this is the point of the question): even with a 95%-accurate test, a positive result only means an ~8.8% chance of actually having the disease, because the disease is rare — the base rate dominates. Most positives are false positives simply because there are so many more healthy people than sick people being tested. This is why screening low-prevalence conditions needs confirmatory follow-up testing, and it's the intuition behind "base rate neglect" / "base rate fallacy."

## Expectation, variance, covariance, correlation

**Expectation** (mean): `E[X] = Σ x · P(X = x)` (discrete) or `∫ x f(x) dx` (continuous). Linear: `E[aX + bY] = aE[X] + bE[Y]` — always true, even if `X, Y` are dependent.

**Variance**: spread around the mean.
```
Var(X) = E[(X − E[X])²] = E[X²] − (E[X])²
```
For independent `X, Y`: `Var(X + Y) = Var(X) + Var(Y)`. If they're *not* independent, you need the covariance term: `Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)`.

**Covariance**: how two variables move together.
```
Cov(X, Y) = E[(X − E[X])(Y − E[Y])] = E[XY] − E[X]E[Y]
```
Positive → tend to move together; negative → move oppositely; zero → linearly unrelated (but not necessarily independent — see below).

**Correlation**: covariance normalized to `[-1, 1]`, so it's comparable across variables with different scales.
```
ρ(X, Y) = Cov(X, Y) / (σ_X · σ_Y)
```

### Correlation vs. causation (interview talking point)

`Cov(X, Y) = 0` does **not** imply independence — it only rules out *linear* relationships. A classic example: `X ~ Uniform(-1, 1)`, `Y = X²`. These are strongly (nonlinearly) dependent, yet `Cov(X, Y) = 0`.

More importantly, in any interview involving observational data, be ready to name why correlation ≠ causation:

- **Confounding variable**: a third variable drives both (e.g. ice cream sales and drowning deaths both rise with summer heat).
- **Reverse causation**: `Y` could be causing `X` instead of the assumed direction.
- **Selection bias**: the sample itself was filtered in a way that induces a spurious association.
- **Coincidence / multiple comparisons**: with enough variables, some will correlate by chance.

To establish causation you generally need a randomized controlled experiment (see [`ab-testing.md`](./ab-testing.md)) or quasi-experimental methods (instrumental variables, diff-in-diff, regression discontinuity) that are usually out of scope for a stats-fundamentals interview round but worth name-dropping.

## Common interview questions

1. Derive Bayes' theorem from the definition of conditional probability.
2. A test is 99% accurate for a disease with 1/10,000 prevalence — given a positive result, what's `P(disease)`? (Same pattern as above — practice until it's automatic.)
3. Two dice are rolled; what's `P(sum = 7 | first die = 3)`? (Tests conditional probability with a concrete sample space.)
4. If `Cov(X, Y) = 0`, are `X` and `Y` independent? Give a counterexample.
5. Explain the difference between independence and mutual exclusivity.
6. Why does `Var(X + Y) ≠ Var(X) + Var(Y)` in general?
7. Give three reasons two correlated variables might not have a causal relationship.

## Common mistakes

- Confusing `P(A | B)` with `P(B | A)` — this is literally what Bayes' theorem exists to correct ("prosecutor's fallacy").
- Forgetting the base rate (prior) and reasoning only from sensitivity/specificity.
- Assuming zero correlation implies independence.
- Adding variances of dependent variables without the covariance cross-term.
- Treating mutually exclusive events as if they were independent (they're the opposite when both have positive probability).

## See also

- [`distributions-clt.md`](./distributions-clt.md) — expectation/variance in the context of specific distributions
- [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md)
- [`ab-testing.md`](./ab-testing.md) — where "correlation vs. causation" becomes an experiment-design problem
