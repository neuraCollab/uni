# Recommender Systems: Collaborative Filtering, Matrix Factorization, ALS/iALS

## What is it / why?

The recommendation problem: predict a rating or preference `r_ui` (explicit) or
`p_ui` (implicit) for a user-item pair `(u, i)`, using a historical
**user-item interaction matrix** `R` that is almost entirely missing — any
one user has only rated/interacted with a tiny fraction of the item catalog.
The whole field is about filling in (or ranking) the missing entries of that
sparse matrix well enough to surface items a user hasn't seen yet but would
like.

Two families of approaches dominate the classical (pre-deep-learning) toolkit:
**collaborative filtering** (use the interaction patterns of other
users/items, no content needed) and **matrix factorization** (learn a compact
latent-factor representation of users and items that reconstructs `R`).

## Explicit vs. implicit feedback

- **Explicit feedback** — the user directly and deliberately rates something
  (1-5 stars, thumbs up/down, a review score). Clean signal, but rare: most
  users don't bother rating most things.
- **Implicit feedback** — inferred from behavior: clicks, views, watch time,
  purchases, add-to-cart. This is the far more common real-world case,
  because it's collected automatically as a byproduct of normal usage — no
  extra effort required from the user.

**Why implicit feedback is harder to model:** with explicit ratings, a low
score is a genuine negative example — the user told you they disliked it.
With implicit feedback there is no such thing: you only ever observe
*positive* signal (a click happened) or *absence of signal* (no click). An
item a user never interacted with might be something they'd hate, or
something they simply never saw — you can't tell the difference from the data
alone. This ambiguity — no real negatives, only positives and unlabeled
non-events — is the central modeling challenge of implicit-feedback systems
(see iALS below for the standard way of coping with it).

## Collaborative filtering (memory-based / neighborhood methods)

Two symmetric variants:
- **item2item** — recommend items similar to items the user already liked.
- **user2user** — recommend items liked by users similar to this user.

**Similarity-weighted prediction.** For item2item, compute a similarity score
between every pair of items, then predict a user's rating for item `i` as a
similarity-weighted average of their ratings on similar items `j`:

```
N(i,j) = sum_u S(i,j,u) / sum_u |S(i,j,u)|
```

i.e. aggregate a pairwise similarity/agreement signal `S(i,j,u)` (computed
per user `u` who rated both `i` and `j`) into a single normalized
item-item similarity `N(i,j)`, then predict a rating by taking a weighted
average of the target user's ratings on items similar to `i`, weighted by
that similarity. (The user2user variant is the mirror image: similarity
between users instead of items, aggregated over co-rated items.)

**Pearson correlation as the similarity measure.** `S_{i,j}` is standardly
taken to be the **Pearson correlation** between items `i` and `j`, computed
over the users who rated both (or between users, computed over items they
both rated). Why Pearson and not raw cosine/dot-product similarity: Pearson
first centers each user's (or item's) ratings around their own mean, so it
measures whether two items' ratings move *together relative to each rater's
personal baseline* rather than their absolute rating scale. A user who rates
everything 4-5 stars and a user who rates everything 1-3 stars can still show
high Pearson similarity if their *relative* preferences agree — cosine
similarity on raw ratings would incorrectly treat them as dissimilar just
because of the scale offset. You can also weight the correlation by variance
or deviation from the mean to downweight noisy/low-signal raters.

**Content-based recommendation** is the non-collaborative alternative: instead
of leaning on other users' interaction patterns, recommend based on
item/user **features or metadata** (genre, text description, category,
demographics). It doesn't need any interaction history for an item/user to
work, which makes it the standard fallback for **cold start** — collaborative
filtering has nothing to go on for a brand-new item or user, while
content-based methods just need the item's/user's attributes.

## Matrix factorization

**Idea:** decompose the sparse rating matrix `R` (users × items) into two
low-rank latent-factor matrices:

```
R ≈ X · Yᵀ
```

where `X` is `n_users × k` (user factors) and `Y` is `n_items × k` (item
factors), `k` a small latent dimensionality. The predicted rating for user
`u`, item `i` is just the dot product of their latent vectors:

```
r̂_ui = x_u · y_i
```

Instead of relying on raw neighborhood similarity, the model learns `k`
latent dimensions that jointly explain the observed ratings — implicitly
capturing things like genre affinity, without anyone hand-labeling them.

**Classical named algorithms worth recognizing (Netflix-Prize-era lineage):**
- **RankSVD** — SVD-style matrix factorization fit for the rating-prediction
  task directly (rather than a true SVD of a dense matrix, which is undefined
  when most entries are missing).
- **SVD++** — extends RankSVD by also folding in *implicit* feedback signals
  (e.g. the mere fact that a user rated an item at all, regardless of the
  rating value) into the user's latent representation, even when the target
  being predicted is still an explicit rating.
- **timeSVD++** — adds temporal dynamics on top of SVD++ (user preferences
  and item popularity drift over time; factors are allowed to evolve).
- **SLIM** (Sparse Linear Methods) — a different family: learns an
  item-item similarity/weight matrix directly by solving a sparse regression
  problem, rather than factorizing into latent user/item vectors.

## Alternating Least Squares (ALS)

**Algorithm:**
1. Fix `Y` (item factors), solve for `X` (user factors): with `Y` fixed, the
   loss becomes a per-user regularized least-squares problem with a closed-form
   solution.
2. Fix `X`, solve for `Y` the same way, per item.
3. Alternate 1-2 until convergence.

**Why this works well:** jointly optimizing `X` and `Y` together is
non-convex (the loss has products of the two unknowns). But *fixing either
one* turns the problem into an ordinary regularized least-squares problem in
the other — convex, closed-form, easy. This alternating structure is exactly
the same pattern as the [EM algorithm](probabilistic-ml/em-algorithm.md)
alternating between an E-step and an M-step, each easy to solve while holding
the other fixed. The key difference: ALS adds **L2 regularization** on both
`X` and `Y` to keep the factors from overfitting to what is usually a very
sparse set of observed entries.

## Implicit ALS (iALS)

Plain ALS assumes explicit, precise ratings and only sums the loss over
observed entries. Real-world implicit signals (clicks, views, purchases)
violate both assumptions, so iALS reframes the problem:

| | ALS (explicit) | iALS (implicit) |
|---|---|---|
| Typical signal | Ratings 1-5, stars, likes | Clicks, views, purchases |
| What "zero" / missing means | A genuinely missing/unknown value | Absence of interaction is itself an (uncertain) negative signal, not just missing data |
| Target variable | Precise numeric rating | Binary preference `p_ui ∈ {0,1}` |
| Loss computed over | Only observed/known ratings | **All** user-item pairs |
| Weighting | Equal weight for every observed rating | Weighted by a confidence term `c_ui` |
| Complexity per optimization step | `O(|R|·k² + N·k³)` | `O(|R|·k² + (N+M)·k³)` |

(`|R|` = number of observed ratings/interactions, `N` = number of users, `M`
= number of items, `k` = latent dimensionality.)

**Confidence weighting, in words:** since implicit feedback gives no real
negatives, iALS treats *every* user-item pair as having some preference
`p_ui ∈ {0,1}` (1 if any interaction happened, 0 otherwise), but weights how
much to trust that label with a confidence `c_ui` that grows with
interaction **intensity/frequency** — a user who watched a show five times or
clicked an item repeatedly gets a much higher-confidence positive than a
single stray click, and unobserved pairs get low (but nonzero) confidence
rather than being ignored outright.

**Why summing over all pairs is tractable.** Naively, iALS's loss must be
summed over *every* user-item pair — including the huge number of pairs with
no interaction — which looks like it should cost `O(N·M)` per step, infeasible
at industrial scale. The standard trick that makes iALS practical is an
algebraic decomposition of that sum into (a) a data-independent global term
that only depends on the current factor matrices and can be precomputed once
per iteration, plus (b) a sparse correction term evaluated only over the
`|R|` actually-observed interactions. This is what brings the per-step cost
down to `O(|R|·k² + (N+M)·k³)` instead of scaling with the full dense
`N × M` matrix — the reason iALS scales to industrial recommenders with
millions of items.

## When to use what

- **Collaborative filtering (memory-based)** — simple, interpretable ("users
  like you also liked..."), good baseline, no training step beyond computing
  similarities. Struggles as the matrix gets sparser (harder to find
  reliable overlap between users/items) and with cold start.
- **Matrix factorization / ALS / iALS** — handles sparsity much better by
  learning dense latent structure instead of relying on direct overlap; was
  the standard production recommender approach for years (fast to train,
  scales well, closed-form sub-steps).
- **Modern deep-learning recommenders** — two-tower models, sequence/session
  models (e.g. transformer-based next-item prediction), are the current
  state of the art for large-scale industrial systems, handling richer
  features and sequential context. ALS/iALS remain a strong, fast,
  well-understood baseline — still very much worth knowing cold for an
  interview, and often still used as a fast candidate-generation stage even
  when a heavier model does final ranking.

## Common interview questions

- How would you handle cold-start users/items? *(Content-based fallback using
  item/user metadata, popularity-based defaults, or a hybrid of
  collaborative + content signals until enough interaction data accumulates.)*
- Why is implicit feedback harder to model than explicit feedback? *(No true
  negatives — only observed positives and ambiguous absence-of-signal.)*
- How do you evaluate a recommender system offline? Ranking metrics like
  Precision@K, Recall@K, NDCG are the standard tools (see
  [Classification Metrics](model-evaluation/classification-metrics.md) for
  the precision/recall building blocks; ranking-specific metrics are a
  distinct topic beyond that page).
- What's the difference between memory-based (collaborative filtering) and
  model-based (matrix factorization/ALS) collaborative filtering? *(Memory-based
  computes similarities directly from raw interaction data at query time
  with no trained model; model-based learns compact latent factors offline,
  which generalizes better under sparsity.)*
- Walk through why ALS's sub-problems are convex when the joint problem isn't.
- What does the confidence weight `c_ui` represent in iALS, and why do you
  need it when there are no real negative labels?

## Common mistakes

- Using raw cosine/dot-product similarity on ratings instead of Pearson
  correlation — ignores that different users have different rating baselines
  and scales.
- Treating "no interaction" in implicit feedback as a hard negative with the
  same confidence as an observed interaction — it's the reason iALS needs a
  confidence-weighted loss instead of naive binary classification.
- Trying to compute the implicit-ALS loss by literally summing over every
  user-item pair — forgetting the sparse-plus-global-term decomposition that
  makes each step tractable.
- Applying plain (explicit) ALS directly to implicit signals (clicks, views)
  without adapting the loss/weighting — explicit ALS assumes missing = truly
  unknown, which implicit data violates.

## See also

- [EM Algorithm](probabilistic-ml/em-algorithm.md) — the alternating-optimization
  pattern ALS mirrors.
- [Classification Metrics](model-evaluation/classification-metrics.md) —
  precision/recall building blocks used in ranking evaluation.
