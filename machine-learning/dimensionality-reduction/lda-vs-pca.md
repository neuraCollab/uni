# LDA vs. PCA

## What is it?

Two classic linear dimensionality-reduction techniques that are frequently
confused because both produce a linear projection into fewer dimensions —
but they optimize for completely different objectives.

- **PCA (Principal Component Analysis)** — **unsupervised**. Finds the
  directions of maximum variance in the data, ignoring any labels.
- **LDA (Linear Discriminant Analysis)** — **supervised**. Finds the
  directions that best *separate* known classes.

## PCA — how it works

1. **Center the data** (subtract the mean of each feature).
2. **Compute the covariance matrix** of the features.
3. **Eigendecompose** it — eigenvectors are the **principal components**
   (orthogonal directions), eigenvalues are the variance captured along each.
4. **Sort** components by descending eigenvalue.
5. **Project** onto the top-`k` components (`X_reduced = X_centered @ V_k`).

PCA has no concept of class labels — it purely asks "which directions
preserve the most spread in the data?" `explained_variance_ratio_` tells you
how much of the total variance each retained component accounts for.

**Uses:** general-purpose dimensionality reduction, visualization, noise
reduction, decorrelating features before a distance-based or linear model
(also shows up as a lens for understanding Ridge — see
[Regularization](../linear-models/regularization.md#ridge-regression-l2)).

## LDA — how it works

1. Compute the **within-class scatter matrix** `S_W` — how spread out each
   class is around its own mean (averaged/summed over classes).
2. Compute the **between-class scatter matrix** `S_B` — how far apart the
   class means are from the overall mean.
3. Find the projection directions `w` that **maximize the ratio**
   `(wᵀ S_B w) / (wᵀ S_W w)` — i.e., maximize between-class separation
   *relative to* within-class spread. Solved as a generalized eigenvalue
   problem on `S_W^(-1) S_B`.
4. Project onto the top eigenvectors.

**Assumptions:** classical LDA assumes each class is (approximately)
Gaussian-distributed and that all classes **share the same covariance
matrix** — only the class means differ. When that assumption is badly
violated, Quadratic Discriminant Analysis (QDA, which fits a separate
covariance per class) or a different method is usually more appropriate.

**Key constraint:** LDA can produce at most `n_classes - 1` discriminant
components — `S_B` has rank at most `n_classes - 1` since it's built from the
differences between `n_classes` means and one grand mean. With 3 classes
(e.g. the iris dataset), LDA tops out at 2 dimensions — conveniently exactly
enough for a 2D scatter plot.

## PCA vs. LDA — the core interview distinction

| | PCA | LDA |
|---|---|---|
| Supervision | Unsupervised — never looks at `y` | Supervised — explicitly uses class labels |
| Objective | Maximize **total variance** captured | Maximize **between-class / within-class** scatter ratio |
| Max output dims | `min(n_features, n_samples)` | `n_classes - 1` |
| Goal | Best low-dim *representation* of the data | Best low-dim *separation* for classification |
| Can hurt classification? | Yes — the direction of maximum variance isn't necessarily the direction that separates classes; PCA can throw away a low-variance-but-highly-discriminative direction | By construction, optimized for exactly this |
| Typical use | Visualization, compression, preprocessing before unsupervised or general models | Preprocessing specifically before a classifier, or as a linear classifier itself |

The classic illustration (used in scikit-learn's own iris demo): project the
150-sample, 4-feature iris dataset down to 2D with both methods. PCA's top 2
components capture the most spread in sepal/petal measurements overall, but
mixes the *versicolor* and *virginica* classes more than LDA does — because
PCA doesn't know those are different classes. LDA's 2 components (the max
possible for 3 classes) are chosen specifically to pull the three species'
clusters apart, giving visibly cleaner class separation in the scatter plot.

LDA can also be used **directly as a classifier** (not just a
dimensionality-reduction step): assign a new point to the class whose
Gaussian (under the shared-covariance assumption) gives it the highest
posterior probability — this reduces to a linear decision boundary, which is
why it's called *linear* discriminant analysis. See
[Logistic Regression](../linear-models/logistic-regression.md) for the
more commonly used discriminative alternative — LDA is generative
(models `P(x|y)` then applies Bayes' rule), logistic regression is
discriminative (models `P(y|x)` directly), and they coincide asymptotically
under LDA's Gaussian-shared-covariance assumptions.

## When to use / when not

**Use PCA when:** you have no labels, or you want a general-purpose
representation useful for multiple downstream tasks, or you're doing
exploratory visualization/noise reduction.

**Use LDA when:** you have labels, your end goal is classification, and you
want a reduction that's explicitly optimized to keep classes separable
rather than merely to preserve variance.

**Avoid LDA when:** classes are strongly non-Gaussian or have very different
covariance structures (violates the shared-covariance assumption — consider
QDA instead), or when `n_classes` is small relative to how many dimensions
you actually need (LDA's `n_classes - 1` ceiling can be too restrictive).

**Avoid PCA when:** you specifically need the reduced space to help
downstream classification and have labels available — you're leaving useful
signal on the table by ignoring them.

## Common interview questions

- What's the fundamental difference in objective between PCA and LDA?
- Why can LDA only produce `n_classes - 1` components?
- What assumptions does LDA make about the data, and what breaks them?
- Could PCA ever *hurt* a classification task compared to using raw
  features? How?
- Is LDA a generative or discriminative model? How does that compare to
  logistic regression?
- How would you pick the number of PCA components to keep?
  (`explained_variance_ratio_` cumulative sum / elbow, or cross-validate
  downstream performance.)
- What happens to LDA when the within-class scatter matrix `S_W` is singular
  (e.g. `n_features > n_samples`)? (Need regularization —
  `shrinkage` in scikit-learn's `LinearDiscriminantAnalysis` — or PCA first
  to reduce dimensionality below `n_samples`.)

## Common mistakes

- Treating LDA as "just supervised PCA" without acknowledging it optimizes a
  genuinely different objective (separability ratio, not raw variance).
- Forgetting the `n_classes - 1` component ceiling and being confused when
  `LinearDiscriminantAnalysis(n_components=5)` errors out with only 3 classes.
- Using LDA when classes clearly have very different spreads/covariances
  (violates the shared-covariance assumption) without considering QDA.
- Not scaling features before PCA — since PCA maximizes variance, a feature
  with a larger raw scale dominates the components regardless of its actual
  informativeness. (LDA is less scale-sensitive in principle since it
  normalizes by within-class scatter, but scaling is still good practice.)
- Using PCA as a preprocessing step for classification and being surprised
  it doesn't improve accuracy — it was never optimizing for that.

## Example

Adapted from scikit-learn's classic iris PCA-vs-LDA comparison:

```python
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
X, y = iris.data, iris.target

pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)
print("PCA explained variance ratio:", pca.explained_variance_ratio_)

lda = LinearDiscriminantAnalysis(n_components=2)  # max possible: 3 classes - 1 = 2
X_lda = lda.fit(X, y).transform(X)

# Scatter-plot X_pca and X_lda side by side, colored by y: LDA's projection
# separates the three iris species more cleanly than PCA's, because LDA was
# explicitly optimized to do so and PCA wasn't.
```

See also: [Classification Metrics](../model-evaluation/classification-metrics.md)
for evaluating a classifier built on top of either projection.

For the full derivation of LDA/QDA from Bayes' rule and Gaussian
class-conditional densities, see
[Generative Classification: GDA, QDA, LDA](../probabilistic-ml/generative-classification-gda-lda.md).
