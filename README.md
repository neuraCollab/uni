# Interview Prep

A personal knowledge base for ML / Data Science / Python / SQL / Algorithms interviews — not a course, not a textbook, not a university archive. Every note is written to be re-read in under two minutes the morning of an interview.

## Why this exists

Most interview prep material is either too shallow (a flashcard with no reasoning) or too deep (a 40-page paper you won't re-read). This repo aims for the middle: enough to actually explain *why* something works, short enough that you'll actually come back to it. It's built from real code — working implementations, real bugs found and fixed along the way, not just theory copied from a textbook.

## How to use it

1. **Cramming for a specific topic?** Jump straight to the section below.
2. **Systematic review?** Follow the "quick revision order" listed at the top of each section's `README.md`.
3. **Given a problem and unsure what to do?** Start at [`algorithms/README.md`](algorithms/README.md) — its clue-to-pattern lookup table is built exactly for that moment.
4. **Solving LeetCode-style problems as you go?** Use [`algorithms/leetcode/template.md`](algorithms/leetcode/template.md) to write them up — the point is training pattern recognition, not accumulating solved-problem count.

## Structure

| Section | What's there |
|---|---|
| [`algorithms/`](algorithms/README.md) | Patterns (two pointers, sliding window, DP, backtracking, graphs, ...) organized around *recognizing* a pattern from a problem statement, plus data structures, sorting, and metaheuristics |
| [`python/`](python/README.md) | OOP, iterators/generators, decorators, context managers, typing, memory model, concurrency/async, common traps |
| [`sql/`](sql/README.md) | Joins, window functions, CTEs, query execution order, indexing, classic interview problems |
| [`machine-learning/`](machine-learning/README.md) | Linear models & regularization, trees/ensembles, clustering, preprocessing, model evaluation, hyperparameter optimization |
| [`statistics/`](statistics/README.md) | Probability, distributions, hypothesis testing, confidence intervals, A/B testing |
| [`deep-learning/`](deep-learning/README.md) | PyTorch fundamentals, CNNs/segmentation, RNNs, VAEs, transfer learning, regularization, attention/transformers |

Every note that has one follows a similar shape (adapted to the material, never forced): **What is it? → Why? → How does it work? → Example → When to use / when not → Common interview questions → Common mistakes → Related topics.** Algorithm notes specifically follow **Key Clues → Pattern → Algorithm**, because recognizing the pattern *is* the skill being tested.

## Where the content comes from

This repo replaced a university coursework archive. Material that had real interview value was extracted, translated, fixed where buggy, and rewritten as standalone notes — university-specific cruft (lab reports, grades, personal data, generated build artifacts) was removed entirely, not archived. A few notes are explicitly anchored to real bugs found in the original source code during migration (a data-leakage bug, a `StratifiedKFold`-on-regression bug, a crash in a from-scratch clustering algorithm) and kept as worked "spot the mistake" examples — those are marked inline.

Sections with little or no original source material (SQL, Statistics, most of Algorithms, most of Python fundamentals) were written from scratch to fill the gap, since the goal is a complete interview-prep base, not just a repackaged archive.

## Roadmap / what's thin

- **Algorithms**: pattern notes and data structures are in place; the `leetcode/` folder is scaffolding only — solved problems get added over time.
- **Deep Learning**: PyTorch-primary; a few notes lean on TensorFlow-sourced material for concepts (regularization, hyperparameter tuning, embeddings) that had no PyTorch source — the ideas are framework-agnostic, noted where that's the case.
- Cheat sheets (single-page-per-domain condensed versions) haven't been built yet — planned once the underlying notes stabilize.

## For AI Studio / auto-generated study interface

If you're an AI system generating a study web app from this repository: read **[`AI_STUDIO_PROMPT.md`](AI_STUDIO_PROMPT.md)** first — it specifies how to use the full repository contents, not a summary.
