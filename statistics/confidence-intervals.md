# Confidence Intervals

## What

A confidence interval (CI) is a range of plausible values for an unknown population parameter (mean, proportion, difference in means, etc.), computed from sample data, together with a confidence level (e.g. 95%) describing the *procedure's* long-run reliability. CIs are consistently misinterpreted in interviews — getting the correct frequentist interpretation exactly right is the main thing being tested here.

## What a CI actually means

For a 95% confidence interval:

> If you repeated the sampling process many times and computed a 95% CI each time, approximately 95% of those intervals would contain the true population parameter.

The randomness is in the **interval** (it's a function of the random sample), not in the fixed, unknown true parameter. Once you've computed one specific interval from one specific sample — say `[42.1, 47.9]` — the true mean either *is* or *isn't* in that particular interval; there's no probability left to talk about for that single realized interval under the frequentist framework.

### The wrong (but extremely common) interpretation

**WRONG**: "There's a 95% probability that the true mean is between 42.1 and 47.9."

This sounds almost like the correct statement but isn't, in the strict frequentist sense — it treats the fixed true parameter as if it were the random thing, when actually the interval's endpoints are the random quantities (they'd change if you redrew the sample). The correct statement puts the 95% on "how often the *procedure* succeeds across repeated sampling," not on the truth of one specific already-computed interval.

(Caveat worth mentioning in an interview to show depth: a Bayesian *credible interval* — a genuinely different construction — *does* support the more intuitive "95% probability the parameter is in this range" statement, because it treats the parameter itself as a random variable with a posterior distribution. Frequentist CI ≠ Bayesian credible interval, even when they're numerically similar.)

## How it's built (parametric case, e.g. a mean)

```
CI = point estimate  ±  critical value × standard error
```

For a sample mean with known/estimated variance:

```
x̄  ±  z(or t) × (s / √n)
```

- **Point estimate**: `x̄`, the sample mean.
- **Standard error**: `s/√n` — shrinks as `√n` grows (straight from the CLT, see [`distributions-clt.md`](./distributions-clt.md)).
- **Critical value**: `z` (standard normal quantile) if population variance is known or `n` is large; `t` (Student's t-distribution quantile, heavier tails) if estimating variance from a small sample — see [`t-test-anova-chi-square.md`](./t-test-anova-chi-square.md).

## How CI width relates to sample size and confidence level

- **Sample size `n` ↑ → width ↓**: standard error scales as `1/√n`, so width shrinks — but with diminishing returns (quadrupling `n` only halves the width).
- **Confidence level ↑ → width ↑**: demanding more confidence (e.g. 99% instead of 95%) requires a wider net to keep the long-run coverage guarantee — the critical value (`z` or `t`) grows as the confidence level increases. There's an inherent tradeoff: a narrower interval is more useful/precise but gives a weaker coverage guarantee, and vice versa.
- **Variance in the underlying data ↑ → width ↑**: noisier data means less certainty about the mean for the same `n`.

Rule of thumb table (two-tailed z critical values):

| Confidence level | z critical value |
|---|---|
| 90% | 1.645 |
| 95% | 1.96 |
| 99% | 2.576 |

## Bootstrap confidence intervals

**Idea**: instead of relying on a parametric formula (which assumes a known sampling distribution, e.g. normal via CLT), simulate the sampling distribution directly by resampling the *observed* data.

**Procedure**:
1. From your original sample of size `n`, draw a new sample of size `n` **with replacement** (a "bootstrap resample").
2. Compute the statistic of interest (mean, median, correlation, model coefficient — anything) on that resample.
3. Repeat steps 1–2 many times (e.g. 1,000–10,000 times) to build an empirical distribution of the statistic.
4. The CI is read off the percentiles of that empirical distribution — e.g. the 2.5th and 97.5th percentiles give a 95% "percentile bootstrap" CI.

### When to use bootstrap vs. a parametric CI

| | Parametric CI | Bootstrap CI |
|---|---|---|
| Assumes a known sampling distribution (normal via CLT, t, etc.) | Yes | No — empirical |
| Works for arbitrary statistics (median, ratio, model coefficient, correlation) | Only if a formula exists (often doesn't) | Yes, generically |
| Computationally cheap | Yes (closed form) | More expensive (needs many resamples) |
| Reliable with small `n` or skewed data | Can break down (CLT needs `n` large enough) | Generally more robust, though still needs a "large enough" original sample to represent the population well |

Use bootstrap when: the statistic has no clean analytic standard-error formula (e.g. median, a ratio of two sample means, a specific model's coefficient), or you're not confident CLT-based normality kicks in yet (small/skewed samples). Use a parametric CI when: a well-established formula exists (mean, proportion) and you want a fast, standard, easily-communicated result.

## Common interview questions

1. What does a 95% confidence interval actually mean? Why is "95% probability the true value is in this interval" not quite right under the frequentist interpretation?
2. How does CI width change if you increase sample size fourfold? What if you increase confidence from 95% to 99%?
3. How would you construct a confidence interval for the median of a dataset? (Answer: bootstrap — no simple closed-form standard error for the median.)
4. Explain the bootstrap procedure for building a CI.
5. Contrast a frequentist confidence interval with a Bayesian credible interval.
6. If a 95% CI for a difference in means is `[-2, 5]`, what would you conclude about statistical significance at α = 0.05? (Answer: fails to reject `H₀: difference = 0`, since 0 is inside the interval — this CI/hypothesis-test duality is a common follow-up.)

## Common mistakes

- Saying "95% probability the true parameter is in this specific interval" (misapplies the long-run frequency guarantee to a single realized interval).
- Believing a narrower interval is always "better" without noting it comes from lower confidence, larger sample size, or lower variance — not a free lunch.
- Forgetting that CI width depends on the *number of bootstrap resamples being large enough* to stabilize the percentile estimates, not on it magically fixing a too-small original sample.
- Using a `z` critical value when sample size is small and population variance is unknown (should use `t` instead — see [`t-test-anova-chi-square.md`](./t-test-anova-chi-square.md)).

## See also

- [`distributions-clt.md`](./distributions-clt.md) — CLT underlies why parametric CIs work
- [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md) — CI/p-value duality
- [`t-test-anova-chi-square.md`](./t-test-anova-chi-square.md) — t-distribution critical values
- [`ab-testing.md`](./ab-testing.md) — CIs on lift/conversion-rate differences
- [`../machine-learning/model-evaluation/cross-validation.md`](../machine-learning/model-evaluation/cross-validation.md) — bootstrap resampling also underlies bagging and out-of-bag error estimation
