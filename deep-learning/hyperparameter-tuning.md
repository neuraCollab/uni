# Hyperparameter Tuning for Deep Learning

> This note covers hyperparameter search **specific to deep learning training runs**. For the general search-strategy background (grid/random/Bayesian search concepts, applicable to any model type), see [`../machine-learning/hyperparameter-optimization.md`](../machine-learning/hyperparameter-optimization.md). Source example: the archived `tensorFlow/hyperParameters.py`, a Keras-Tuner Hyperband search over a small MLP's width and learning rate.

## What is it?

Searching over a model's hyperparameters — values chosen before training rather than learned by it (learning rate, batch size, network width/depth, dropout rate, weight decay, choice of optimizer) — to find a configuration that generalizes well.

## Why?

Deep learning hyperparameters strongly affect both whether training converges at all and how well the trained model generalizes ([`optimization-sgd-adam.md`](fundamentals/optimization-sgd-adam.md), [`regularization-overfitting.md`](regularization-overfitting.md)). Unlike classical ML models, deep nets have no simple closed-form or fast-to-fit way to pick these — they have to be searched empirically.

## How does it work?

### Grid / random / Bayesian search, applied to DL hyperparameters

- **Grid search**: exhaustively try every combination of a fixed set of values per hyperparameter. Simple but scales combinatorially — impractical once you have more than 2-3 hyperparameters at more than a couple of values each.
- **Random search**: sample combinations randomly from each hyperparameter's range. Empirically often outperforms grid search at equal budget, because it explores each individual hyperparameter's range more densely (grid search "wastes" trials on combinations that only vary unimportant dimensions).
- **Bayesian optimization**: build a probabilistic model of "hyperparameters -> validation performance" from trials run so far, and use it to pick the next, most-promising point to try (balancing exploration of uncertain regions against exploitation of known-good regions). More sample-efficient than grid/random, at the cost of more bookkeeping/complexity.

Typical DL hyperparameters to search: **learning rate** (usually log-scale), **batch size**, **architecture width/depth** (e.g. the archived example tunes `units` in `[32, 512]`), **dropout rate**, **weight decay**, and sometimes optimizer choice itself.

```python
hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
```

### Why DL hyperparameter search is uniquely expensive

For a classical ML model (e.g. a random forest or linear model), a single "trial" — fit the model with one hyperparameter setting — might take seconds to minutes. For a deep network, **one trial is an entire training run**, potentially hours or days, since the same expensive gradient-descent process (many epochs over the full dataset) has to happen from scratch for every hyperparameter combination you want to evaluate. This makes naive grid/random search over a large space computationally prohibitive — you simply cannot afford hundreds of full training runs the way you might afford hundreds of `RandomForest.fit()` calls.

### Hyperband / early-stopping-based pruning

**Hyperband** (used in the archived example via `keras_tuner.Hyperband`) addresses the cost problem directly: instead of always running every candidate configuration to full convergence, it runs many configurations for a small number of epochs first, evaluates their early validation performance, and **kills the obviously-worse ones early** — reallocating the compute budget that would've been wasted on them toward training the more promising survivors for more epochs. This is a form of successive halving: start with many cheap, short trials; keep only the top fraction; give those more budget; repeat.

```python
tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,              # fraction of configs kept at each round of pruning
    directory='my_dir', project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
```

Crucially, the archived example's final step **retrains the best-found configuration from scratch** for the full epoch budget (`best_epoch`, found from where validation accuracy peaked during search) — the pruning-based search is only for finding good hyperparameters cheaply; the actual deployed model is trained properly afterward, not taken directly from a truncated search trial.

```python
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)
best_epoch = ... # epoch where val_accuracy peaked
hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)
```

This general pattern — combine early stopping with an adaptive resource-allocation search strategy, then retrain the winner properly — generalizes directly to PyTorch training loops via libraries like Optuna or Ray Tune, which implement the same Hyperband/ASHA-style pruning independent of the training framework.

## When to use

Any time you have compute budget to spare and the default hyperparameters aren't clearly good enough — start with a coarse random search over the most impactful hyperparameters (learning rate first, almost always), then narrow with Bayesian search or Hyperband-style pruning once you have a rough sense of good ranges. For quick iteration or small models, manual tuning based on a handful of runs and the range test described in [`optimization-sgd-adam.md`](fundamentals/optimization-sgd-adam.md) is often sufficient and far cheaper than a full automated search.

## Common interview questions

- **Why is hyperparameter search harder/more expensive for deep learning than for classical ML models?** Each trial requires a full, expensive training run rather than a cheap model fit, so you can't afford as many trials.
- **How does Hyperband save compute compared to plain random/grid search?** It runs many configurations briefly, discards the clearly-worse ones early based on partial training curves, and only spends the full training budget on the survivors.
- **Why retrain the best configuration from scratch after the search, instead of just keeping the best trial's model?** Search trials are often truncated (early-stopped) for efficiency and may use held-out splits/callbacks tuned for search rather than final deployment; retraining with the winning hyperparameters for the full/proper schedule gives the actual best model.
- **Random search vs. grid search — why does random often win at equal budget?** Grid search wastes trials on combinations that vary hyperparameters that don't matter much; random search explores each hyperparameter's marginal range more effectively for the same trial count.
- **What hyperparameter would you tune first, and why?** Learning rate — it has the largest effect on whether training converges at all, before architecture or regularization choices even become meaningful to compare.

## Common mistakes

- Running full grid search over many hyperparameters for a deep model — computationally infeasible past a couple of dimensions.
- Tuning hyperparameters against the test set instead of a separate validation set, leaking test information into the model selection process.
- Treating a search trial's truncated/early-stopped result as the final deployed model instead of retraining the winning configuration properly.
- Searching hyperparameters independently one at a time instead of jointly, when they interact (e.g. optimal learning rate depends on batch size — see [`optimization-sgd-adam.md`](fundamentals/optimization-sgd-adam.md)).
