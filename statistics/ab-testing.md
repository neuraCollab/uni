# A/B Testing

## What

The standard industry framework for running controlled online experiments to measure the causal effect of a change (a new feature, UI variant, pricing, ranking algorithm) on a metric of interest. This is where probability, hypothesis testing, and confidence intervals all get applied end-to-end — and it's a heavily tested topic at any company with a strong experimentation culture.

## The standard process

### 1. Formulate a hypothesis

State clearly, before looking at any data: what change you're testing, what metric it should move, and in which direction — e.g. "Changing the checkout button color from grey to green will increase the checkout conversion rate." This becomes your $H_1$; $H_0$ is "no difference in conversion rate between button colors."

### 2. Pick a primary metric

One (or a small, pre-declared, prioritized set of) metric that defines success — e.g. conversion rate, revenue per user, click-through rate. Committing to a primary metric *before* the test avoids "metric shopping" after the fact — see the multiple-comparisons pitfall below.

Also worth defining upfront: **guardrail metrics** (things that must *not* get worse — e.g. page load time, unsubscribe rate) even if they're not the primary success metric.

### 3. Compute required sample size (power analysis) — BEFORE running the test

Given: the minimum detectable effect (MDE) you care about, the metric's baseline variance, your chosen significance level $\alpha$ (typically 0.05), and your target power (typically 0.80), compute the sample size needed per arm. This should always be done **before** the experiment starts, not after — deciding "how long to run" partway through based on the data you're seeing is exactly the peeking problem below.

Smaller MDE, higher desired power, or lower α all *increase* the required sample size — this is the direct analytical link back to Type I/II errors and power (see [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md)).

### 4. Randomization: control vs. treatment split

Users (or sessions, or whatever the experimental unit is) are randomly assigned to control (existing experience) or treatment (new experience), typically 50/50, though unequal splits (e.g. 90/10) are common when de-risking a change. Randomization is what lets you interpret a measured difference as *causal* rather than merely correlational — it breaks the link to confounding variables (see [`probability-bayes.md`](./probability-bayes.md)).

Key requirement: assignment must be independent of any variable that could also affect the outcome — usually done via a consistent hash of a stable user ID, so the same user always lands in the same arm across sessions.

### 5. Run for a full cycle

Run the test for a pre-committed duration (not until it "looks significant") — ideally spanning at least one full business cycle (e.g. a full week, to average over day-of-week effects) and long enough for any novelty effect to settle. Don't stop early because you're excited, and don't stop late because you're chasing significance either.

### 6. Analyze results — which test applies

| Metric type | Test |
|---|---|
| Continuous (revenue, session length, time-on-page) | Two-sample t-test (Welch's by default) — see [`t-test-anova-chi-square.md`](./t-test-anova-chi-square.md) |
| Binary/conversion rate | Chi-square test of independence (or equivalently a two-proportion z-test) |
| 3+ variants | ANOVA (continuous) / chi-square (categorical), + post-hoc pairwise tests with correction |
| Non-normal / small-sample / heavily skewed continuous metric | Mann-Whitney U (non-parametric) |

Report the result as both a p-value/significance decision **and** a confidence interval on the effect size (lift) — the CI communicates practical magnitude, not just "significant or not" (see [`confidence-intervals.md`](./confidence-intervals.md)).

## Common pitfalls

### Peeking at results early and stopping when significant

Checking the p-value repeatedly during the test and stopping as soon as it crosses $\alpha$ inflates the true false-positive rate far above the nominal 5% — because you're effectively running many implicit hypothesis tests (one per peek) and taking the first "win," which is exactly the multiple comparisons problem in disguise. With continuous monitoring, the false positive rate can climb to 20-30%+ even though every individual peek used $\alpha = 0.05$. **Fix**: pre-commit to a sample size/duration and only look once (a fixed-horizon test), or use a sequential testing method explicitly designed for repeated looks (e.g. group sequential designs, always-valid p-values/mSPRT).

### Not accounting for multiple metrics / multiple tests

Testing a dozen secondary metrics and reporting whichever one came back significant is p-hacking — with enough metrics, something will cross $\alpha = 0.05$ by chance alone (see [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md)). Fix: declare one primary metric upfront; apply a multiple-comparisons correction (Bonferroni, Benjamini-Hochberg) to any secondary/exploratory metrics you do report.

### Sample ratio mismatch (SRM)

If you intended a 50/50 split but the actual observed split is, say, 54/46, something is broken in the randomization or logging pipeline (e.g. one variant's page loads slower and users bounce before being logged, biasing who gets counted). An SRM invalidates the whole experiment's causal interpretation — always run a simple chi-square goodness-of-fit check on the actual arm sizes before trusting any result.

### Network effects / interference between test and control

The core assumption behind a standard A/B test (SUTVA — stable unit treatment value assumption) is that one user's assigned arm doesn't affect another user's outcome. This breaks in social/marketplace/two-sided products — e.g. a treatment that makes sellers list more inventory can affect prices/availability seen by *control* buyers too, contaminating the control group and biasing the measured effect toward zero. Fix: cluster-based randomization (randomize by geographic region, marketplace shard, or social cluster instead of by individual user) when interference is a concern.

### Novelty effect vs. true lasting effect

Users often react to *any* change — positively or negatively — simply because it's new, and that reaction fades. A short test can overstate (or understate) the true steady-state effect. Fix: run long enough to let novelty wear off, or explicitly compare early-window vs. late-window effect size within the same test to detect a decaying novelty signal.

## Common interview questions

1. Walk through the end-to-end process of designing and running an A/B test.
2. Why must sample size be computed before running the test, not after?
3. What happens to the false-positive rate if you check significance every day and stop as soon as p < 0.05? Why?
4. What is sample ratio mismatch and how would you detect it?
5. How would you A/B test a feature on a marketplace/social product where network effects are a concern?
6. Your test shows a large lift in week 1 but it shrinks by week 3 — what's going on?
7. Would you use a t-test or chi-square test to analyze a conversion-rate experiment?

## Common mistakes

- Peeking and stopping early ("optional stopping") without a sequential-testing correction.
- Skipping the pre-experiment power analysis and running with "whatever traffic shows up."
- Reporting a secondary metric as the headline result because it happened to be significant when the primary metric wasn't.
- Ignoring SRM checks and trusting a result from a broken randomization/logging pipeline.
- Treating a short-term novelty bump as a permanent effect.
- Randomizing by individual user in a setting with strong network effects (should randomize by cluster instead).

## See also

- [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md) — p-values, power, multiple comparisons
- [`confidence-intervals.md`](./confidence-intervals.md) — reporting effect size with uncertainty
- [`t-test-anova-chi-square.md`](./t-test-anova-chi-square.md) — choosing the right test for your metric
- [`probability-bayes.md`](./probability-bayes.md) — why randomization supports causal claims
- [`../machine-learning/model-evaluation/cross-validation.md`](../machine-learning/model-evaluation/cross-validation.md) — offline model evaluation as a complement to online A/B testing
