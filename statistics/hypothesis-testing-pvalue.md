# Hypothesis Testing & the p-value

## What

The formal framework for deciding whether observed data provides enough evidence against a default assumption ("null hypothesis"). This file covers the null/alternative framing, the *correct* definition of a p-value (and the misinterpretation that trips up almost everyone), significance level, Type I/II errors, power, one- vs two-tailed tests, and the multiple comparisons problem.

## Null and alternative hypotheses

- **Null hypothesis (`H₀`)**: the default, "nothing interesting is happening" claim — e.g. "the new checkout page has the same conversion rate as the old one," "this coin is fair," "these two drugs have the same effect."
- **Alternative hypothesis (`H₁` / `Hₐ`)**: what you'd conclude if you reject `H₀` — e.g. "the new checkout page has a *different* conversion rate."

You never "prove `H₀` true." You either **reject `H₀`** (data is inconsistent enough with it) or **fail to reject `H₀`** (not enough evidence to rule it out) — the asymmetry matters: absence of evidence against `H₀` is not evidence for `H₀`.

## p-value — the correct definition

> The p-value is the probability of observing data **at least as extreme** as what was actually observed, **assuming the null hypothesis is true**.

In symbols: `p = P(data this extreme or more | H₀ true)`.

A small p-value means: "if `H₀` were really true, data like this would be surprising/rare" — which is evidence (not proof) against `H₀`.

### The most common interview trap: misinterpreting the p-value

**WRONG**: "The p-value is the probability that the null hypothesis is true."
**WRONG**: "p = 0.03 means there's a 97% chance the alternative hypothesis is correct."
**WRONG**: "p = 0.20 means there's an 80% chance the null is true."

**Why it's wrong**: the p-value is computed by *conditioning on `H₀` being true* — `P(data | H₀)`. It says nothing directly about `P(H₀ | data)`, which is what people intuitively (and incorrectly) think it means. Confusing `P(data | H₀)` with `P(H₀ | data)` is exactly the same logical error as confusing `P(B|A)` with `P(A|B)` in Bayes' theorem (see [`probability-bayes.md`](./probability-bayes.md)) — getting `P(H₀ | data)` actually requires a prior on `H₀` and a full Bayesian calculation, which frequentist hypothesis testing deliberately doesn't do.

If you say only one sentence about p-values in an interview, say this: **"p-value is P(data this extreme | H₀ true), not P(H₀ true | data)."**

## Significance level (α) and the decision rule

Before running the test, you choose a threshold `α` (commonly 0.05) — the significance level. Decision rule:

```
if p-value < α:  reject H₀ ("statistically significant")
else:             fail to reject H₀
```

`α` is really "the rate of false alarms I'm willing to tolerate if `H₀` is actually true" — see Type I error below. It is a policy choice made *before* seeing the data, not derived from the data.

## Type I and Type II errors

| | `H₀` is actually True | `H₀` is actually False |
|---|---|---|
| **Reject `H₀`** | Type I error (false positive), rate = `α` | Correct (true positive) |
| **Fail to reject `H₀`** | Correct (true negative) | Type II error (false negative), rate = `β` |

- **Type I error (`α`)**: concluding there's an effect when there isn't one. Directly controlled by your choice of significance threshold.
- **Type II error (`β`)**: missing a real effect. Depends on effect size, sample size, variance, and `α` — not directly set by the experimenter the way `α` is.

There's an inherent tradeoff: lowering `α` (to reduce false positives) increases `β` (more false negatives) for a fixed sample size — the only way to reduce both simultaneously is to collect more data.

## Statistical power

```
Power = 1 − β = P(reject H₀ | H₀ is actually false)
```

The probability of correctly detecting a real effect when one exists. Power increases with: larger sample size, larger true effect size, lower variance in the data, and higher `α` (looser significance threshold). Power analysis — computing the sample size needed to hit a target power (conventionally 80%) for a given expected effect size — should be done **before** running an experiment, not after (see [`ab-testing.md`](./ab-testing.md)).

## One-tailed vs. two-tailed tests

- **Two-tailed**: tests whether the parameter differs from the null value in *either* direction (`H₁: μ ≠ μ₀`). Rejection region is split across both tails of the distribution. The default choice unless you have a specific, pre-registered directional hypothesis.
- **One-tailed**: tests a specific direction only (`H₁: μ > μ₀` or `μ < μ₀`). Puts the entire `α` in one tail, making it "easier" to reach significance in that direction — but you gain zero power to detect an effect in the opposite direction, and you've implicitly asserted you don't care if the effect goes the other way.

**Interview trap**: switching from two-tailed to one-tailed *after seeing the data* (because the effect happened to go the "right" direction) is p-hacking — it invalidates the test's error-rate guarantees. The tail choice must be decided in advance.

## Multiple comparisons problem

If you run many independent hypothesis tests at `α = 0.05` each, the probability of at least one false positive climbs fast:

```
P(≥ 1 false positive across m tests) = 1 − (1 − α)^m
```

At `m = 20` independent tests, that's `1 − 0.95²⁰ ≈ 64%` — you're more likely than not to get a spurious "significant" result somewhere, purely by chance. This is the statistical basis for "if you torture the data enough, it will confess."

**Bonferroni correction**: the simplest fix — test each of `m` hypotheses at a stricter threshold `α/m` instead of `α`, so the family-wise error rate stays at (or below) `α`. Simple and conservative (can be overly strict, reducing power, especially when tests are correlated). Alternatives worth name-dropping: Holm-Bonferroni (less conservative, same guarantee), Benjamini-Hochberg (controls false discovery rate instead of family-wise error rate — standard in large-scale testing like genomics or many simultaneous A/B metrics).

This problem is exactly why running an A/B test with a dozen "secondary metrics" and reporting whichever one came back significant is misleading — see [`ab-testing.md`](./ab-testing.md).

## Common interview questions

1. Define the p-value precisely. What does p = 0.03 actually mean?
2. What's wrong with saying "there's a 95% chance the alternative hypothesis is true" after getting p < 0.05?
3. What's the difference between Type I and Type II error? Give a real-world example of each (e.g. in a medical diagnosis context).
4. What is statistical power, and what four things increase it?
5. Why is switching from a two-tailed to a one-tailed test after seeing the data a problem?
6. You run 50 hypothesis tests at α = 0.05 each. How many "significant" results would you expect by pure chance? How do you correct for this?
7. Explain the tradeoff between α and β for a fixed sample size.

## Common mistakes

- Misinterpreting the p-value as `P(H₀ true | data)` (the single most common mistake — see above).
- Treating "fail to reject `H₀`" as "proved `H₀` is true" — it's only "insufficient evidence to reject."
- "p-hacking": trying multiple tests/subgroups/metrics until something crosses `α = 0.05`, without correcting for multiple comparisons.
- Choosing one-tailed vs. two-tailed, or the significance threshold, *after* looking at the data.
- Treating statistical significance as automatically meaning practical/business significance — a huge sample can make a trivially small effect "statistically significant."

## See also

- [`probability-bayes.md`](./probability-bayes.md) — the `P(B|A)` vs `P(A|B)` confusion underlying p-value misinterpretation
- [`confidence-intervals.md`](./confidence-intervals.md)
- [`t-test-anova-chi-square.md`](./t-test-anova-chi-square.md)
- [`ab-testing.md`](./ab-testing.md) — where these concepts get applied end-to-end
