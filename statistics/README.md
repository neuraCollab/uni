# Statistics

Interview-cram notes on the statistical reasoning that shows up in DS/ML interviews independent of any specific model or algorithm — probability, hypothesis testing, and experimentation. This section exists because interviews (especially at companies with a strong experimentation/A-B-testing culture) routinely probe statistical fundamentals on their own, separate from ML-specific rounds: a candidate who can build a great model but can't correctly explain what a p-value means, or can't spot a flawed A/B test, is a red flag. These notes are written to be reviewed quickly before an interview, not read as a textbook.

## Contents

- [`probability-bayes.md`](./probability-bayes.md) — probability axioms, conditional probability, independence, Bayes' theorem (with the classic false-positive worked example), law of total probability, expectation/variance/covariance/correlation, correlation vs. causation
- [`distributions-clt.md`](./distributions-clt.md) — Bernoulli, Binomial, Poisson, Normal, Uniform, Exponential — what each models in the real world — and the Central Limit Theorem
- [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md) — null/alternative hypotheses, the correct definition of a p-value (and the misinterpretation everyone makes), significance level, Type I/II errors, power, one- vs. two-tailed tests, multiple comparisons/Bonferroni
- [`confidence-intervals.md`](./confidence-intervals.md) — the correct frequentist interpretation of a CI, how width relates to sample size and confidence level, bootstrap confidence intervals
- [`t-test-anova-chi-square.md`](./t-test-anova-chi-square.md) — t-tests (one-sample/two-sample/paired, Welch's correction), ANOVA, chi-square (goodness-of-fit/independence), and non-parametric alternatives (Shapiro-Wilk, KS test, Mann-Whitney/Wilcoxon)
- [`ab-testing.md`](./ab-testing.md) — the standard industry A/B testing workflow and its classic pitfalls: peeking, multiple metrics, sample ratio mismatch, network effects, novelty effect

## Which test do I use? (quick cheat sheet)

| Question | Test |
|---|---|
| Compare 1 group's mean to a fixed value | One-sample t-test |
| Compare means of 2 independent groups | Two-sample t-test (Welch's if variances unequal) |
| Compare means of 2 paired/matched measurements | Paired t-test |
| Compare means of 3+ groups | ANOVA (+ post-hoc test) |
| Test association between 2 categorical variables | Chi-square test of independence |
| Test a categorical variable against an expected distribution | Chi-square goodness-of-fit |
| Test whether a sample is normally distributed | Shapiro-Wilk test |
| Compare a sample to a reference distribution, or two samples to each other | Kolmogorov-Smirnov (KS) test |
| Compare 2 groups without assuming normality | Mann-Whitney U test |
| Compare 2 paired measurements without assuming normality | Wilcoxon signed-rank test |

## Suggested reading order

1. [`probability-bayes.md`](./probability-bayes.md) — the foundation everything else builds on
2. [`distributions-clt.md`](./distributions-clt.md) — CLT is the bridge from raw data to inference
3. [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md) — the core inferential framework
4. [`confidence-intervals.md`](./confidence-intervals.md) — the estimation counterpart to hypothesis testing
5. [`t-test-anova-chi-square.md`](./t-test-anova-chi-square.md) — the concrete test toolkit
6. [`ab-testing.md`](./ab-testing.md) — everything above applied end-to-end in a real experiment

## Related sections

- [`../machine-learning/model-evaluation/cross-validation.md`](../machine-learning/model-evaluation/cross-validation.md) — sampling variability and resampling ideas (bootstrap, CV) recur in model evaluation too
