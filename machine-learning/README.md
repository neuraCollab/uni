# Machine Learning

Index for the ML section. Organized by topic, not by algorithm family, so you can jump straight to what an interviewer is probing for.

For each concept the notes try to answer: **What is it? Why do we need it? How does it work? When should you use it — and when shouldn't you? What do interviewers ask? What do people get wrong?**

## Linear Models

The strongest-covered corner of this KB — start here if you only have time for one section.

- [Regularization: OLS → Ridge → Lasso → Elastic Net → LARS/OMP](linear-models/regularization.md)
- [Bayesian Regression (BayesRidge, ARD)](linear-models/bayesian-regression.md)
- [Generalized Linear Models (Tweedie, Poisson, Gamma)](linear-models/generalized-linear-models.md)
- [Robust Regression (Theil-Sen, Huber, RANSAC)](linear-models/robust-regression.md)
- [Online Learning: SGD & Passive-Aggressive](linear-models/online-learning-sgd-pa.md)
- [Quantile Regression](linear-models/quantile-regression.md)
- [Logistic Regression](linear-models/logistic-regression.md)
- Code: [`linear-models/code/`](linear-models/code/)

## Probabilistic ML Foundations

The math underneath "why this loss function, why this model family" — start here for the rigorous answer to questions the Linear Models section takes as given.

- [MLE and Loss Functions](probabilistic-ml/mle-and-loss-functions.md) — deriving MSE, MAE, and log-loss as maximum-likelihood consequences of assumed noise distributions.
- [Entropy and KL Divergence](probabilistic-ml/entropy-and-kl-divergence.md) — Shannon entropy, KL divergence, the exponential family, and the Koopman-Pitman-Darmois maximum-entropy theorem.
- [Generative Classification: GDA, QDA, LDA](probabilistic-ml/generative-classification-gda-lda.md)
- [Naive Bayes](probabilistic-ml/naive-bayes.md)
- [Bayesian Inference](probabilistic-ml/bayesian-inference.md)
- [EM Algorithm](probabilistic-ml/em-algorithm.md)

## Kernel Methods

- [Support Vector Machines](kernel-methods/svm.md)
- [Kernel Ridge Regression](kernel-methods/kernel-ridge-regression.md)

## Dimensionality Reduction

- [LDA vs PCA](dimensionality-reduction/lda-vs-pca.md)

## Trees & Ensembles

- [Decision Trees](trees-ensembles/decision-trees.md)
- [Random Forest](trees-ensembles/random-forest.md)
- [Gradient Boosting: CatBoost / LightGBM](trees-ensembles/gradient-boosting-catboost-lgbm.md)
- [Stacking & Blending](trees-ensembles/stacking-blending.md)
- Code: [`trees-ensembles/code/`](trees-ensembles/code/) — includes a fixed real data-leakage bug, see [Data Leakage](model-evaluation/data-leakage.md)

## Clustering

- [Overview: k-means baseline + how CURE/FOREL/ISODATA/hierarchical differ](clustering/overview.md)
- [CURE, FOREL, ISODATA, Hierarchical — algorithm detail](clustering/kmeans-hierarchical-cure-forel-isodata.md)
- [Distance Metrics](clustering/distance-metrics.md)
- [Clustering Evaluation Metrics](clustering/clustering-evaluation-metrics.md)
- Code: [`clustering/code/`](clustering/code/) — 6 clustering algorithms from scratch behind a shared `BaseClusterer`

## Preprocessing

- [Missing Values & Imputation](preprocessing/missing-values-imputation.md) — 10 strategies including MICE
- [Scaling & Categorical Encoding](preprocessing/scaling-categorical.md)
- [Feature Engineering](preprocessing/feature-engineering.md)
- Code: [`preprocessing/code/`](preprocessing/code/)

## Model Evaluation

- [Classification Metrics](model-evaluation/classification-metrics.md) — precision/recall/F1/confusion matrix, log-loss, pairwise AUC-ROC, Average Precision
- [Regression Metrics](model-evaluation/regression-metrics.md) — MSE/RMSE/R²/MAE, MAPE/SMAPE/WAPE, RMSLE
- [Probability Calibration](model-evaluation/calibration.md) — histogram/isotonic calibration, ECE/MCE, Brier Score
- [Cross-Validation](model-evaluation/cross-validation.md)
- [Data Leakage](model-evaluation/data-leakage.md) — anchored on a real leakage bug found and fixed in this repo
- [Bias-Variance Tradeoff](model-evaluation/bias-variance-tradeoff.md)

## Cross-cutting

- [Hyperparameter Optimization](hyperparameter-optimization.md) — grid/random/Bayesian (TPE), Optuna conditional search spaces, Population Based Training
- [Imbalanced Data](imbalanced-data.md)
- [k-Nearest Neighbors](knn.md)
- [Text Features: TF-IDF](text-features-tfidf.md)

## Recommender Systems

- [Collaborative Filtering, Matrix Factorization, ALS/iALS](recommender-systems.md) — explicit vs. implicit feedback, item2item/user2user CF, Pearson-based similarity, ALS and its EM-like alternating structure, the ALS-vs-iALS comparison table

## Quick revision order

If you're cramming: **regularization → logistic regression → classification metrics → cross-validation → data leakage → bias-variance → trees/ensembles → clustering overview → hyperparameter optimization.** That's the path most DS/ML interviews actually probe.

Related: [Statistics](../statistics/README.md) for the probability/hypothesis-testing foundations these notes assume · [Deep Learning](../deep-learning/README.md) for neural-network-specific material.
