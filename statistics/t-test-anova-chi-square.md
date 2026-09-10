# t-test, ANOVA, Chi-Square (and Non-Parametric Alternatives)

## What

The standard toolkit for comparing groups: t-tests (compare 1 or 2 group means), ANOVA (compare 3+ group means), and chi-square tests (categorical/count data). Plus the non-parametric alternatives you reach for when the normality assumption behind these tests doesn't hold.

## t-tests

All t-tests rely on the **t-distribution** — like a normal distribution but with heavier tails, converging to normal as degrees of freedom (roughly, sample size) grows. It's used instead of a plain z-test whenever the population variance is unknown and estimated from the sample (which is almost always the real-world case).

### One-sample t-test

Tests whether a single sample's mean differs from a known/hypothesized value `μ₀`.

```
t = (x̄ − μ₀) / (s / √n)
```

**Example**: "Is the average delivery time different from the advertised 30 minutes?"

### Two-sample (independent) t-test

Tests whether two *independent* groups have different means.

```
t = (x̄₁ − x̄₂) / SE(x̄₁ − x̄₂)
```

**Example**: "Do users in test vs. control have different average session length?" — the standard A/B test workhorse for continuous metrics (see [`ab-testing.md`](./ab-testing.md)).

**Assumptions**: independence between (and within) groups, approximate normality of each group's mean (often via CLT — see [`distributions-clt.md`](./distributions-clt.md), so raw data needn't be normal if `n` is large), and — for the *standard* (Student's) version — **equal variances** between groups.

**Welch's t-test**: a correction used when the two groups have *unequal* variances (very common in practice — e.g. comparing a small control group to a large treatment group, or metrics with different spreads). Welch's version adjusts the standard error formula and uses an approximate (often non-integer) degrees of freedom instead of assuming equal variance. **Rule of thumb**: default to Welch's t-test unless you have good reason to believe variances are equal — it's nearly as powerful when variances *are* equal, and much safer when they aren't. (This is what most stats software, e.g. `scipy.stats.ttest_ind(equal_var=False)`, recommends as the default.)

### Paired t-test

Tests whether the mean *difference* between paired/matched observations (same subject measured twice, or naturally paired units) is zero.

```
t = d̄ / (s_d / √n)     where dᵢ = x₁ᵢ − x₂ᵢ
```

**Example**: "Did the same users' spend change before vs. after a UI redesign?" Paired designs remove between-subject variance, giving more power than treating the two measurements as independent groups — use a paired test whenever the data is naturally paired.

## ANOVA (Analysis of Variance)

Tests whether **3 or more** group means differ, using a single omnibus test.

### Why not just run pairwise t-tests?

With `k` groups, there are `k(k-1)/2` possible pairs. Running a t-test on every pair inflates the overall false-positive rate exactly like the multiple comparisons problem (see [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md)) — e.g. with 5 groups (10 pairwise tests) at `α = 0.05` each, the chance of at least one false "significant" pair by pure chance is well above 5%. ANOVA controls this by testing all groups simultaneously with one test at the stated `α`.

### F-statistic intuition

ANOVA compares the variance *between* group means to the variance *within* groups:

```
F = (variance between groups) / (variance within groups)
  = MSB / MSW
```

- Large `F` → the group means are spread out relative to the natural noise within each group → evidence the groups really differ.
- `F ≈ 1` → between-group spread is no bigger than you'd expect from within-group noise alone → no evidence of a difference.

`H₀`: all group means are equal. Rejecting `H₀` tells you **at least one** group differs from the rest — it does **not** tell you *which* group(s). That requires a follow-up **post-hoc test** (e.g. Tukey's HSD) with its own multiple-comparisons correction.

**Assumptions**: independence, approximate normality within each group, and (for standard one-way ANOVA) equal variances across groups (analogous to the t-test's equal-variance assumption; Welch's ANOVA is the corresponding fix).

## Chi-square test

For **categorical**/count data rather than continuous means. Two common flavors, same underlying statistic:

```
χ² = Σ (observed − expected)² / expected
```

### Goodness-of-fit

Tests whether a single categorical variable's observed distribution matches a hypothesized/expected distribution.

**Example**: "Is this die fair?" — compare observed roll counts per face against the expected uniform 1/6 each.

### Independence / contingency table (test of association)

Tests whether two categorical variables are independent, using a contingency table of observed counts.

**Example**: "Is conversion rate independent of which landing-page variant a user saw?" — a 2x2 (or larger) contingency table of {variant} x {converted/didn't}. This is the standard test for A/B tests measured by a **conversion rate** (categorical outcome) rather than a continuous metric.

**Assumptions**: independent observations, and expected cell counts not too small (common rule of thumb: expected count ≥ 5 per cell; otherwise use Fisher's exact test instead).

## Quick decision table

| Comparing... | Test |
|---|---|
| 1 group mean vs. a fixed value | One-sample t-test |
| 2 independent group means | Two-sample t-test (Welch's if variances unequal) |
| 2 paired/matched measurements | Paired t-test |
| 3+ group means | ANOVA (+ post-hoc test to find which pair) |
| Categorical distribution vs. expected | Chi-square goodness-of-fit |
| Association between 2 categorical variables | Chi-square test of independence |

## Non-parametric alternatives (when normality is violated)

Non-parametric tests make no assumption about the underlying distribution's shape — useful with small samples, heavy skew, or outliers where the CLT hasn't "kicked in" enough to trust a t-test/ANOVA.

### Shapiro-Wilk test

Not an alternative to a t-test itself — it's a test **for normality**, often run first as a diagnostic to decide whether a parametric test is even appropriate. `H₀`: the sample comes from a normal distribution. A small p-value here is evidence *against* normality (pushing you toward a non-parametric alternative below).

### Kolmogorov-Smirnov (KS) test

Compares an empirical distribution to a reference distribution (one-sample KS), or compares two samples' empirical distributions to each other (two-sample KS), based on the largest gap between their cumulative distribution functions (CDFs). More general than Shapiro-Wilk (can test against any reference distribution, not just normal) and, in its two-sample form, doubles as a general "are these two samples drawn from the same distribution" test — not restricted to comparing means.

### Mann-Whitney U test (a.k.a. Wilcoxon rank-sum test)

Non-parametric alternative to the **independent two-sample t-test**. Instead of comparing means directly, it ranks all observations from both groups together and tests whether one group's ranks tend to be systematically higher/lower than the other's — i.e., it tests whether one distribution is stochastically greater than the other, not strictly "equal means." Doesn't require normality, more robust to outliers, but generally has somewhat less power than a t-test when the data genuinely *is* normal.

### Wilcoxon signed-rank test

Non-parametric alternative to the **paired t-test** — tests whether the median of the paired differences is zero, using signed ranks of the differences rather than assuming they're normally distributed.

## Common interview questions

1. When do you use a paired t-test instead of a two-sample t-test?
2. Why can't you just run multiple pairwise t-tests instead of ANOVA?
3. What does a significant ANOVA result actually tell you (and not tell you)?
4. What's Welch's correction and when should you use it by default?
5. Walk through the difference between chi-square goodness-of-fit and chi-square independence tests.
6. Your data is heavily skewed with `n = 15` — what test would you use to compare two groups, and why?
7. What does the Shapiro-Wilk test's null hypothesis assume, and what do you do if you reject it?

## Common mistakes

- Running many pairwise t-tests instead of ANOVA (inflated Type I error — see multiple comparisons in [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md)).
- Treating "ANOVA is significant" as "I know which groups differ" without a post-hoc test.
- Defaulting to Student's t-test when group variances are visibly unequal instead of Welch's.
- Using chi-square with small expected cell counts (should use Fisher's exact test instead).
- Applying a paired test to data that isn't actually paired (or vice versa) — the two use different variance calculations and give different power/results.
- Assuming normality without checking (or reasoning about CLT/sample size) before defaulting to a t-test/ANOVA over a non-parametric alternative.

## See also

- [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md) — multiple comparisons problem, p-values
- [`confidence-intervals.md`](./confidence-intervals.md) — t-distribution critical values
- [`distributions-clt.md`](./distributions-clt.md) — why t-tests still work on non-normal raw data via CLT
- [`ab-testing.md`](./ab-testing.md) — picking the right test for a live experiment
