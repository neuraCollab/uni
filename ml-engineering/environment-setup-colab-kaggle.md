# Environment Setup: Colab & Kaggle

## What

Google Colab and Kaggle Notebooks are free, hosted, ephemeral Jupyter environments — a fresh VM is provisioned per session with no persistent disk (Colab) or a sandboxed container (Kaggle). Because nothing persists, every session needs a short setup ritual: mount storage, install packages, and place API credentials. This note covers that ritual and, more importantly, frames it against the "real" environment-reproducibility problem discussed elsewhere in this KB.

## Why

- Interview-relevant framing: Colab/Kaggle setup is a miniature, low-stakes version of the "reproducible environment" problem every ML system faces — but it solves it the *wrong* way for production (manual, per-session, unpinned), which is exactly why understanding the contrast matters.
- Practically, being fluent in this setup is table stakes for take-home ML exercises and quick prototyping, which frequently happen in Colab/Kaggle.

## How does it work?

### Mounting Google Drive (Colab)

Colab's VM disk is wiped when the runtime disconnects, so persistent files (datasets, saved models, helper scripts) live on Drive and get mounted each session:

```python
from google.colab import drive
drive.mount("/content/drive")

# Reuse a shared folder of helper scripts across notebooks:
import sys
sys.path.append("/content/drive/My Drive/colab")
from dataset_analysis import explore_dataset
```

### Kaggle API credentials (Colab or local)

The Kaggle API needs `kaggle.json` (an API token downloaded from Kaggle account settings) placed at `~/.kaggle/kaggle.json` with `600` permissions before `kaggle datasets download` / `kaggle competitions download` will work:

```python
import os, subprocess
from google.colab import files

uploaded = files.upload()                      # upload kaggle.json via widget
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
subprocess.run(["mv", "kaggle.json", kaggle_dir + os.sep], check=True)
subprocess.run(["chmod", "600", os.path.join(kaggle_dir, "kaggle.json")], check=True)
subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
```

Full script: [`code/kaggle_setup.py`](./code/kaggle_setup.py).

### Typical notebook boilerplate

A real session chains these together: mount Drive -> copy/run the Kaggle setup script -> `pip install` any extra packages -> import shared helper modules from Drive. See the original `%run kaggle_setup.py` pattern in the source template this note is based on.

## Reproducibility: why this doesn't scale to production

This is the actual interview point — contrast the two ends of the spectrum:

| | Colab/Kaggle setup | Production (Docker + pinned deps) |
|---|---|---|
| Dependency versions | Whatever's preinstalled + ad-hoc `pip install` | Pinned exactly in `requirements.txt`/lockfile, baked into an image |
| Persistence | None — repeat setup every session | Image is immutable and versioned (tag/digest) |
| Credentials | Manually uploaded each session | Injected via secrets manager / env vars at deploy time, never uploaded by hand |
| Reproducible across machines? | Not really — depends on Google's current base image | Yes, by construction (same image = same environment) |
| Appropriate for | Experimentation, prototyping, competitions | Training pipelines, serving, anything that needs to run identically twice |

The rule of thumb: Colab/Kaggle-style setup is fine for **experimentation only**. The moment a notebook's logic needs to run unattended, repeatedly, or in a team (a training pipeline, a serving container), it needs to graduate to pinned dependencies and a container image — see the Docker section in [`README.md`](./README.md#docker) and the notebook -> script -> container -> orchestrated-deployment path described there.

## Common interview questions

1. Why doesn't a Colab notebook environment persist between sessions, and what problem does Drive-mounting solve?
2. What's wrong with relying on "whatever's `pip install`ed in this Colab session" for a production training pipeline?
3. How would you turn a working Colab notebook into something reproducible for a teammate or a CI pipeline?
4. Why does the Kaggle API require `chmod 600` on the credentials file?
5. Where should API credentials live in a production system, versus in a personal notebook?

## Common mistakes

- Hardcoding personal file paths (a specific Drive folder, a specific username) into a notebook that's then shared or committed to a repo.
- Treating "it works in my Colab notebook" as equivalent to "it's reproducible" — no pinned versions means it can silently break next session when Google updates the base image.
- Committing `kaggle.json` (or any credentials file) to version control.
- Not setting file permissions (`600`) on credential files, which some CLIs (including Kaggle's) will outright refuse to use.
- Relying on Drive-mounted helper scripts as if they were a proper installed package — no versioning, easy to silently diverge between notebooks.

## See also

- [`README.md`](./README.md#docker) — pinned dependencies and Docker as the production alternative
- [`deployment-systemd-daemons.md`](./deployment-systemd-daemons.md) — the opposite end of "process management," for long-running production services
- [`../ml-system-design/README.md`](../ml-system-design/README.md#training-pipeline) — reproducibility (seeding, pinned dependencies) in the training pipeline context
