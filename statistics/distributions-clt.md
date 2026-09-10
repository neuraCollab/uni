# Distributions & the Central Limit Theorem

## What

The handful of probability distributions that show up constantly in interviews, what real-world processes each one models, and the Central Limit Theorem (CLT) — the result that justifies using normal-distribution-based statistics (confidence intervals, t-tests, z-tests) even when the underlying data isn't normally distributed.

## Why this matters in interviews

A very common question format is: *"What distribution would you use to model X?"* — website clicks per hour, time until a server fails, number of defective items in a batch, human height. Being able to map a real scenario to the right distribution (and explain *why*, in terms of its generating assumptions) is the actual skill being tested — not memorizing PDFs.

## Discrete distributions

### Bernoulli

Single trial, two outcomes (success/failure) with probability $p$ of success. $X \in \{0, 1\}$.
- $E[X] = p$, $Var(X) = p(1-p)$
- **Models**: a single coin flip, a single user converting or not, a single ad click.

### Binomial

Sum of $n$ independent, identical Bernoulli trials — "number of successes in $n$ trials." Parameters $n, p$.
- $E[X] = np$, $Var(X) = np(1-p)$
- **Models**: number of heads in 20 coin flips, number of conversions out of 1,000 site visitors, number of defective items in a batch of fixed size.
- **Interview flag**: requires *fixed* number of trials, each independent with the *same* success probability. If $p$ varies per trial or trials aren't independent, binomial no longer strictly applies.

### Poisson

Number of events in a fixed interval of time/space, given events happen independently at a constant average rate $\lambda$, and (in principle) infinitely many opportunities for a rare event to occur.
- $E[X] = \lambda$, $Var(X) = \lambda$ (mean equals variance — a useful diagnostic/interview fact)
- **Models**: number of customer support tickets per hour, number of typos per page, number of server requests per second, number of earthquakes per year. Classic "rare, rate-based counting" distribution.
- **Relation to Binomial**: Poisson is the limit of Binomial as $n \to \infty$, $p \to 0$, with $np = \lambda$ held fixed — i.e., Poisson approximates Binomial when $n$ is large and $p$ is small (many trials, rare success).

## Continuous distributions

### Uniform

Every value in $[a, b]$ is equally likely.
- $E[X] = (a+b)/2$, $Var(X) = (b-a)^2/12$
- **Models**: a random number generator, arrival time within a scheduled window when you have no other information (maximum-entropy assumption), or the fractional part of an unrelated continuous measurement.

### Normal (Gaussian)

Bell curve, parameterized by mean $\mu$ and variance $\sigma^2$. Defined by the property that it's the limiting distribution of sums/averages of many independent random effects (this *is* the CLT — see below).
- **Models**: measurement error, human height/weight (approximately — sum of many small genetic/environmental factors), and — critically — **sample means and sums of almost any underlying distribution**, once the sample is large enough, thanks to the CLT.
- 68-95-99.7 rule: ~68% of mass within $1\sigma$ of $\mu$, ~95% within $2\sigma$, ~99.7% within $3\sigma$.

### Exponential

Time *between* events in a Poisson process — continuous analog of "waiting time." Parameter $\lambda$ (rate).
- $E[X] = 1/\lambda$, $Var(X) = 1/\lambda^2$
- **Models**: time until the next customer arrives, time until a machine part fails (under a constant-hazard assumption), time between earthquakes.
- **Memoryless property**: $P(X > s + t \mid X > s) = P(X > t)$ — the distribution doesn't "remember" how long you've already waited. This is a favorite interview gotcha: it means, e.g., "given a lightbulb has already lasted 1000 hours, its remaining lifetime distribution is identical to a brand-new bulb's" — counterintuitive but a direct consequence of the constant-hazard-rate assumption. (Real component wear-out violates this — that's why Weibull exists, but that's usually out of scope.)

### Quick "which distribution" cheat sheet

| Scenario | Distribution |
|---|---|
| Single yes/no outcome | Bernoulli |
| Count of successes in $n$ fixed trials | Binomial |
| Count of rare events in fixed time/space | Poisson |
| Time between independent rare events | Exponential |
| Sum/average of many independent effects | Normal |
| No information beyond a range | Uniform |

## The Central Limit Theorem (CLT)

### Statement

Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with finite mean $\mu$ and finite variance $\sigma^2$ (the underlying distribution can be *anything* — binomial, exponential, wildly skewed, doesn't matter). Then as $n \to \infty$, the standardized sample mean converges in distribution to a standard normal:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \to N(0, 1)$$

Equivalently: $\bar{X}_n$ is approximately $N(\mu, \sigma^2/n)$ for large $n$, regardless of the shape of the original distribution.

### Why it matters

The CLT is the reason so much of classical inference (confidence intervals, t-tests, z-tests) can assume normality of a *sample mean* even when raw, individual data points are clearly not normal (e.g. heavily skewed revenue-per-user data, binary conversion data). You don't need the population to be normal — you need the sample size to be "large enough" (commonly cited rule of thumb: $n \geq 30$, though this depends heavily on how skewed the underlying distribution is — more skew needs a larger $n$).

This is *the* theoretical justification behind A/B testing on non-normal metrics: even if individual user revenue is heavily right-skewed, the *difference in sample means* between test and control is approximately normal for reasonably large sample sizes, so a t-test is still valid.

### Intuition example: averaging dice rolls

A single fair die roll is uniform on $\{1, \ldots, 6\}$ — not remotely bell-shaped, it's flat.

- Roll **1** die many times and plot the outcomes → flat/uniform histogram.
- Roll **2** dice, average them, repeat many times → the histogram of the averages already looks triangular (values near 3.5 much more common than near 1 or 6).
- Roll **30** dice, average them, repeat many times → the histogram of averages is visibly bell-shaped and tightly concentrated around $\mu = 3.5$, with spread shrinking as $\sigma/\sqrt{n}$.

The underlying per-roll distribution never changes (still uniform) — but the distribution of the *average* converges to normal as $n$ grows. That's the CLT in miniature.

### Standard error shrinks with √n

Since $Var(\bar{X}_n) = \sigma^2/n$, the standard deviation of the sample mean (the **standard error**) is $\sigma/\sqrt{n}$. Quadrupling your sample size only halves your standard error — a key intuition for sample-size/power planning (see [`confidence-intervals.md`](./confidence-intervals.md) and [`ab-testing.md`](./ab-testing.md)).

## Common interview questions

1. What distribution would you use to model the number of website visits in an hour? Why?
2. Explain the memoryless property of the exponential distribution with an example.
3. State the Central Limit Theorem. Does the underlying data need to be normal?
4. Why does a t-test remain valid on skewed conversion/revenue data given a large enough sample?
5. What's the relationship between the Binomial and Poisson distributions?
6. Why is the standard error proportional to $1/\sqrt{n}$ rather than $1/n$?

## Common mistakes

- Thinking CLT means *individual data points* become normally distributed as $n$ grows — it's the *sampling distribution of the mean* that becomes normal, not the raw data.
- Applying CLT-based inference to very small samples without checking robustness (rule-of-thumb $n \geq 30$ is not a hard law — heavier skew needs more).
- Confusing Poisson's rate parameter $\lambda$ (events per interval) with Exponential's rate parameter $\lambda$ (they're related — same underlying process, different question: "how many events?" vs. "how long until the next one?").
- Forgetting Binomial requires independent trials with constant $p$ — misapplying it to sequentially dependent events.

## See also

- [`confidence-intervals.md`](./confidence-intervals.md)
- [`hypothesis-testing-pvalue.md`](./hypothesis-testing-pvalue.md)
- [`t-test-anova-chi-square.md`](./t-test-anova-chi-square.md)
- [`../machine-learning/model-evaluation/cross-validation.md`](../machine-learning/model-evaluation/cross-validation.md) — sampling variability shows up again when estimating model performance
