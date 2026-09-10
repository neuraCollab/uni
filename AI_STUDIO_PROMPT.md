# Prompt for Google AI Studio: build the study interface

Paste this file's contents (or point AI Studio at this whole repository) when generating the web interface. The instructions below are written to be read by the generating AI, not just by a human.

## Task

Build a static, single-page (or statically-routed multi-page) web application that turns this repository into an interactive interview-prep study tool, deployable to GitHub Pages with no backend and no build-time secrets.

## Critical: use the entire repository, not a summary

**Do not summarize this repository and build the interface from the summary.** Parse and ingest every `.md` file under `algorithms/`, `python/`, `sql/`, `machine-learning/`, `statistics/`, `deep-learning/`, `ml-system-design/`, and `ml-engineering/` — all of it, not just the `README.md` index files. The index files are navigation aids, not content; the actual notes are in the leaf files they link to (e.g. `machine-learning/linear-models/regularization.md`, `algorithms/patterns/sliding-window.md`). If you only ingest the READMEs you will produce an app with empty or near-empty content pages — verify your file list against the actual directory tree before generating anything.

Also ingest the code files under each `*/code/` subdirectory (e.g. `machine-learning/clustering/code/*.py`, `deep-learning/pytorch/code/*.py`) — several notes reference specific functions/bugs/fixes in that code by name, and the study interface should be able to show the referenced snippet alongside the note that discusses it, not just link out to a file the user has to open separately.

## What the repository actually contains (verify this yourself by listing files, don't trust this count staying accurate)

Eight top-level sections, each with a `README.md` index and topic-specific notes below it. Most notes follow (loosely, adapted per topic): **What is it? → Why? → How does it work? → Example → When to use / when not → Common interview questions → Common mistakes → Related topics.** Algorithm pattern notes specifically follow **Recognition (key clues) → Pattern → Template → Example Problems → Common Mistakes → Complexity → Related Patterns.**

## What the interface should do

1. **Navigation matching the repo structure** — the 8 top-level sections as the primary nav, sub-navigation within each matching the actual subdirectories (e.g. inside Machine Learning: Linear Models, Kernel Methods, Trees & Ensembles, Clustering, Preprocessing, Model Evaluation).
2. **Render every note's actual content** — headings, code blocks (with syntax highlighting), tables, and internal links. Internal markdown links between notes (e.g. `[Ridge Regression](../linear-models/regularization.md)`) must resolve to in-app navigation, not broken links or external URLs.
3. **A "pattern recognition" mode for `algorithms/`** specifically: surface the "Key clues" / "Recognition" section of each pattern note as a quiz — show clues from a problem statement, ask the user to name the pattern, reveal the answer plus the linked template. This is the stated purpose of that section's structure; don't flatten it into a plain document viewer.
4. **A quick-revision / cram mode** — each section's `README.md` states a "quick revision order"; surface that as a guided sequential review path, not just a link list.
5. **Search across all notes** — full-text, not just titles, since a user will search for a specific term ("data leakage", "sliding window", "p-value") and expect to land on the right note regardless of which section it's filed under.
6. **Interview-question extraction** — many notes have a "Common interview questions" section; consider surfacing these as a separate flashcard-style review mode pulled from across all notes, since that's likely the single most valuable cross-cutting view for someone with limited prep time.
7. **Code viewer for `*/code/*.py` files** — syntax-highlighted, linked from the notes that reference them.

## Constraints

- **Static output only** — this deploys to GitHub Pages. No server-side code, no database, no API keys baked into client-side JS.
- **No content invention** — every fact, code snippet, and example shown must come from the repository's actual files. Do not generate new interview questions, new explanations, or new code that isn't already in the notes; the notes are the source of truth and were deliberately curated.
- **Preserve attribution to nothing** — there is no personal data in this repository (it was explicitly scrubbed during a prior migration); do not add any author name, contact info, or institutional branding you might infer from context. Keep it generic ("Interview Prep" or similar neutral title).
- **Respect the repo's own priority ordering** — `machine-learning/` and `algorithms/` are the largest and most-developed sections; don't let a generic template flatten that into equal-sized tiles if the actual content depth differs.

## If content looks thin somewhere

Some notes (currently: parts of `ml-system-design/`, the `algorithms/leetcode/` folder) are intentionally smaller/scaffolding — reflect that honestly in the UI (e.g. a "growing" or "in progress" indicator) rather than padding them with invented content to match other sections' length.
