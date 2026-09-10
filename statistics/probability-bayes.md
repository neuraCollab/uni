# Probability & Bayes' Theorem

## What

The math of quantifying uncertainty: axioms of probability, conditional probability, independence, and Bayes' theorem — the tool for updating beliefs given new evidence. Bayes' theorem (specifically the "false positive" medical-test flavor) is the single most commonly asked probability question in DS/ML interviews.

## Probability axioms (Kolmogorov)

For a sample space $\Omega$ and event $A \subseteq \Omega$:

1. $P(A) \geq 0$ — probability is non-negative.
2. $P(\Omega) = 1$ — something in the sample space happens with certainty.
3. For mutually exclusive events $A_1, A_2, \ldots$: $P(A_1 \cup A_2 \cup \ldots) = P(A_1) + P(A_2) + \ldots$ (countable additivity).

Everything else (e.g. $P(A^c) = 1 - P(A)$, $P(A \cup B) = P(A) + P(B) - P(A \cap B)$) follows from these three.

## Conditional probability

Probability of $A$ given that $B$ has occurred:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad \text{provided } P(B) > 0$$

Rearranged, this gives the **multiplication rule**: $P(A \cap B) = P(A \mid B) \cdot P(B)$.

## Independence

$A$ and $B$ are independent iff knowing $B$ tells you nothing about $A$:

$$P(A \mid B) = P(A) \iff P(A \cap B) = P(A) \cdot P(B)$$

**Common trap**: independence and mutual exclusivity are opposite in spirit. If $A$ and $B$ are mutually exclusive ($P(A \cap B) = 0$) and both have nonzero probability, they are *not* independent — knowing one occurred tells you the other definitely didn't.

## Law of total probability

If $B_1, B_2, \ldots, B_n$ partition the sample space (mutually exclusive, collectively exhaustive), then for any event $A$:

$$P(A) = \sum_i P(A \mid B_i) \cdot P(B_i)$$

This is the workhorse for computing an "unconditional" probability by conditioning on every way it could happen — and it's the denominator you compute right before applying Bayes' theorem.

## Bayes' theorem

### Derivation

From the multiplication rule, $P(A \cap B)$ can be written two ways:

$$P(A \mid B) \cdot P(B) = P(A \cap B) = P(B \mid A) \cdot P(A)$$

Divide through by $P(B)$:

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

Expand the denominator with the law of total probability ($P(B) = P(B \mid A)P(A) + P(B \mid A^c)P(A^c)$) to get the full form:

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B \mid A) \cdot P(A) + P(B \mid A^c) \cdot P(A^c)}$$

Naming convention: $P(A)$ = **prior**, $P(B \mid A)$ = **likelihood**, $P(A \mid B)$ = **posterior**, $P(B)$ = **evidence** (normalizing constant).

### Worked example: the classic false-positive problem

> A disease affects 1% of the population. A test for it is 95% sensitive (catches 95% of people who have the disease) and 90% specific (correctly clears 90% of people who don't have it). A random person tests positive. What's the probability they actually have the disease?

Define events: $D$ = has disease, $T$ = tests positive.

- Prior: $P(D) = 0.01$, so $P(D^c) = 0.99$
- Sensitivity (true positive rate): $P(T \mid D) = 0.95$
- Specificity: $P(T^c \mid D^c) = 0.90$, so the false positive rate $P(T \mid D^c) = 0.10$

Apply Bayes:

$$
\begin{aligned}
P(D \mid T) &= \frac{P(T \mid D) P(D)}{P(T \mid D) P(D) + P(T \mid D^c) P(D^c)} \\
&= \frac{0.95 \times 0.01}{(0.95 \times 0.01) + (0.10 \times 0.99)} \\
&= \frac{0.0095}{0.0095 + 0.099} \\
&= \frac{0.0095}{0.1085} \\
&\approx 0.0876 \to \text{about } 8.8\%
\end{aligned}
$$

**The punchline** (this is the point of the question): even with a 95%-accurate test, a positive result only means an ~8.8% chance of actually having the disease, because the disease is rare — the base rate dominates. Most positives are false positives simply because there are so many more healthy people than sick people being tested. This is why screening low-prevalence conditions needs confirmatory follow-up testing, and it's the intuition behind "base rate neglect" / "base rate fallacy."

## Expectation, variance, covariance, correlation

**Expectation** (mean): $E[X] = \sum x \cdot P(X = x)$ (discrete) or $\int x f(x) \, dx$ (continuous). Linear: $E[aX + bY] = aE[X] + bE[Y]$ — always true, even if $X, Y$ are dependent.

**Variance**: spread around the mean.
$$Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$
For independent $X, Y$: $Var(X + Y) = Var(X) + Var(Y)$. If they're *not* independent, you need the covariance term: $Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)$.

**Covariance**: how two variables move together.
$$Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$$
Positive → tend to move together; negative → move oppositely; zero → linearly unrelated (but not necessarily independent — see below).

**Correlation**: covariance normalized to $[-1, 1]$, so it's comparable across variables with different scales.
$$\rho(X, Y) = \frac{Cov(X, Y)}{\sigma_X \cdot \sigma_Y}$$

### Correlation vs. causation (interview talking point)

$Cov(X, Y) = 0$ does **not** imply independence — it only rules out *linear* relationships. A classic example: $X \sim \text{Uniform}(-1, 1)$, $Y = X^2$. These are strongly (nonlinearly) dependent, yet $Cov(X, Y) = 0$.

More importantly, in any interview involving observational data, be ready to name why correlation ≠ causation:

- **Confounding variable**: a third variable drives both (e.g. ice cream sales and drowning deaths both rise with summer heat).
- **Reverse causation**: $Y$ could be causing $X$ instead of the assumed direction.
- **Selection bias**: the sample itself was filtered in a way that induces a spurious association.
- **Coincidence / multiple comparisons**: with enough variables, some will correlate by chance.

To establish causation you generally need a randomized controlled experiment (see [`ab-testing.md`](./ab-testing.md)) or quasi-experimental methods (instrumental variables, diff-in-diff, regression discontinuity) that are usually out of scope for a stats-fundamentals interview round but worth name-dropping.

## Common interview questions

1. Derive Bayes' theorem from the definition of conditional probability.
2. A test is 99% accurate for a disease with 1/10,000 prevalence — given a positive result, what's $P(\text{disease})$? (Same pattern as above — practice until it's automatic.)
3. Two dice are rolled; what's $P(\text{sum} = 7 \mid \text{first die} = 3)$? (Tests conditional probability with a concrete sample space.)
4. If $Cov(X, Y) = 0$, are $X$ and $Y$ independent? Give a counterexample.
5. Explain the difference between independence and mutual exclusivity.
6. Why does $Var(X + Y) \neq Var(X) + Var(Y)$ in general?
7. Give three reasons two correlated variables might not have a causal relationship.

## Common mistakes

- Confusing $P(A \mid B)$ with $P(B \mid A)$ — this is literally what Bayes' theorem exists to correct ("prosecutor's fallacy").
- Forgetting the base rate (prior) and reasoning only from sensitivity/specificity.
- Assuming zero correlation implies independence.
- Adding variances of dependent variables without the covariance cross-term.
- Treating mutually exclusive events as if they were independent (they're the opposite when both have positive probability).

## See also

- [`distributions-clt.md`](./distributions-clt.md) — expectation/variance in the context of specific distributions
- [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md)
- [`ab-testing.md`](./ab-testing.md) — where "correlation vs. causation" becomes an experiment-design problem
