"""
Kaggle API credential setup, meant to be run as a cell in a Google Colab
notebook. Uploads a personal kaggle.json (downloaded from
https://www.kaggle.com/settings -> "Create New API Token"), installs it
where the Kaggle CLI/API expects it, and installs the `kaggle` package.

This is an experimentation-only reproducibility pattern: it gets a fresh
Colab VM into a working state for the session, but nothing here is pinned
or persisted. See ../environment-setup-colab-kaggle.md for how this
contrasts with pinned requirements.txt / Docker images used for production
reproducibility.

NOTE: ported from a personal Colab template. It never contained any
hardcoded username or personal filesystem path (Colab always runs as root
at a fixed /root/.kaggle location) — nothing to strip here, included only
for completeness alongside backup_daemon.py.
"""
import os
import subprocess

from google.colab import files

# Opens a file-picker widget in the notebook output; select your own
# kaggle.json here. Nothing is embedded in this script.
uploaded = files.upload()

kaggle_dir = os.path.expanduser("~/.kaggle")  # /root/.kaggle on Colab
os.makedirs(kaggle_dir, exist_ok=True)

subprocess.run(["mv", "kaggle.json", kaggle_dir + os.sep], check=True)
# Kaggle's client refuses to run if the credentials file is group/world
# readable, so permissions must be tightened after every fresh upload.
subprocess.run(["chmod", "600", os.path.join(kaggle_dir, "kaggle.json")], check=True)
subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
