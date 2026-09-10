# Text Features: TF-IDF

## What is it?

**TF-IDF (Term Frequency - Inverse Document Frequency)** is a way to turn
text into numeric feature vectors by weighting each word (or n-gram) by how
often it appears in a given document, discounted by how common that word is
across the whole corpus. It's one of the oldest and still most widely used
baselines for text classification and information retrieval.

## Why?

Raw word counts overweight common words that appear in almost every
document ("the", "is", domain-generic filler) and underweight rare words
that are actually discriminative for a specific document's topic. TF-IDF
corrects for this by down-weighting terms that show up everywhere.

## How does it work?

For a term `t` in document `d`, within a corpus of `N` documents:

```
TF(t, d)  = (count of t in d) / (total terms in d)          # how locally important
IDF(t)    = log( N / (1 + count of documents containing t) )  # how globally rare
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

(scikit-learn's exact smoothing/normalization defaults differ slightly —
`+1` smoothing on the denominator and log terms to avoid division by zero
and to keep IDF from being undefined for terms in zero documents — but the
core idea is the same.)

**Why log-scale IDF:** raw inverse document frequency (`N / df`) would let a
term that appears in only 1 document out of a million dominate with a weight
of a million, while a term in half the documents gets a weight of 2 — wildly
disproportionate to the actual difference in "rarity." The log compresses
that range, so IDF grows slowly as a term gets rarer instead of exploding,
keeping the weighting sane on corpora of any size.

Each document ends up as a vector of length `|vocabulary|`, where most
entries are 0 (a document only contains a tiny fraction of the total
vocabulary) — hence "sparse vector representation." scikit-learn's
`TfidfVectorizer` stores this as a `scipy.sparse` matrix rather than a dense
array, which is essential for memory: a 5,000-word vocabulary over 100,000
documents would be 500M dense floats (~4GB) but is typically well under 1%
non-zero in sparse form.

### Pipeline pattern

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df["content"])   # fit vocabulary + IDF on TRAIN only
X_val = vectorizer.transform(val_df["content"])            # transform only -- no refit

model = LogisticRegression()
model.fit(X_train, train_df["label"])
preds = model.predict(X_val)
print("Accuracy:", accuracy_score(val_df["label"], preds))
```

Note the same leakage principle as anywhere else in the pipeline (see
[Data Leakage](model-evaluation/data-leakage.md)): `fit_transform` on train,
`transform` only on validation/test — the vocabulary and document
frequencies must come from the training split alone. `max_features` caps
the vocabulary to the `N` highest-TF-IDF-scoring terms, both controlling
dimensionality and acting as light regularization against extremely rare
terms.

## Limitations vs. dense embeddings

- **No semantic similarity.** TF-IDF treats "car" and "automobile" as
  completely unrelated dimensions — there's no notion that two different
  words can mean similar things. Dense embeddings (word2vec, GloVe, or
  transformer-based sentence embeddings) place semantically similar words/
  documents near each other in a continuous space, so a model can generalize
  across synonyms and paraphrases.
- **Vocabulary-size-dependent.** The feature space is exactly the corpus
  vocabulary (or `max_features` cap) — a term never seen in training is
  simply invisible at inference (`transform` silently drops out-of-vocabulary
  words), and vocabulary size grows with corpus diversity, unlike a fixed-
  dimensional embedding.
- **No word order / limited context** — a bag-of-words model (TF-IDF's
  basis) ignores word order beyond whatever's captured by including n-grams;
  it can't represent "not good" as different from "good" without explicit
  bigram features, and even then can't capture longer-range dependencies the
  way a sequence model can.
- **Sparse, high-dimensional** — usually fine for linear models
  (Logistic Regression, linear SVM — see [SVM](kernel-methods/svm.md), which
  is a strong fit for exactly this feature type) but awkward for architectures
  expecting dense fixed-size input.

## When to use / when not

**TF-IDF is still a reasonable baseline when:**
- You need a fast, cheap, interpretable model (feature = literal word,
  coefficient = its learned importance — easy to debug).
- The dataset is small-to-medium and domain vocabulary is fairly
  fixed/specific (legal, medical, or other jargon-heavy text where
  transformer embeddings pretrained on general web text may not transfer
  cleanly without fine-tuning).
- You want a strong, well-understood baseline to compare a fancier
  embeddings-based model against before justifying the added complexity.
- Latency/resource budget is tight — no GPU, no large model to serve.

**Prefer dense embeddings when:**
- Semantic similarity matters (search/retrieval, paraphrase detection,
  clustering by topic rather than exact wording).
- The corpus is large and diverse enough that out-of-vocabulary words are a
  real, frequent problem.
- You have the compute budget to fine-tune or serve a transformer-based
  embedding model, and the accuracy gain justifies it.
- Labeled data is scarce — pretrained embeddings carry semantic knowledge
  learned from a much larger corpus than your task's training set, whereas
  TF-IDF only ever "knows" what's in your own corpus.

## Common interview questions

- Write out the TF-IDF formula and explain each term's purpose.
- Why is IDF log-scaled instead of a raw ratio?
- Why must the vectorizer be fit on the training split only?
- What's the practical difference between `fit_transform` and `transform`
  on a `TfidfVectorizer`, and why does misuse of the two cause data leakage?
- Compare TF-IDF to word embeddings — what does TF-IDF fundamentally lack?
- Why does TF-IDF pair naturally with linear models (logistic regression,
  linear SVM) specifically?
- How would you handle an out-of-vocabulary word at inference time under
  TF-IDF? Under a subword/BPE-based embedding model?
- What does `max_features` control, and what's the tradeoff in setting it
  too low vs. too high?

## Common mistakes

- Calling `fit_transform` on both train and validation/test sets separately
  (instead of `fit_transform` on train, `transform` on val/test) — this
  builds a *different* vocabulary/IDF per split, silently breaking the
  feature space's consistency and, if done on a combined dataset first,
  leaking test-set document frequencies into the IDF weights.
- Not lowercasing / not stripping punctuation consistently, inflating the
  vocabulary with near-duplicate tokens (`"Car"` vs. `"car"` vs. `"car."`).
- Treating TF-IDF cosine similarity as true semantic similarity — it
  measures lexical overlap, not meaning; two paraphrased sentences with no
  shared words score zero similarity.
- Using a huge, unbounded vocabulary (no `max_features` / `min_df` /
  `max_df`) on a large corpus, blowing up memory and training time on mostly
  uninformative rare terms.
- Forgetting that `TfidfVectorizer.transform` silently drops any word not
  seen during `fit` — no error, just a missing feature, which can hide bugs
  in a train/serve vocabulary mismatch.

## Example

Adapted from a real title+text binary classification pipeline
(`AI/3 lab/tf-idf.py` in the source coursework — TF-IDF + logistic
regression on news-article text):

```python
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df["content"] = df["title"] + " " + df["text"]
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df["content"])
X_val = vectorizer.transform(val_df["content"])

model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_df["label"])

y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(val_df["label"], y_pred))
print(classification_report(val_df["label"], y_pred))

joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")  # must ship together -- the model
                                            # is meaningless without the exact
                                            # fitted vocabulary/IDF weights
```

See also: [Logistic Regression](linear-models/logistic-regression.md),
[SVM](kernel-methods/svm.md) (a strong classifier choice for sparse TF-IDF
features), [Classification Metrics](model-evaluation/classification-metrics.md),
[Data Leakage](model-evaluation/data-leakage.md).
