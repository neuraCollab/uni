"""
Backup daemon: periodically copies a source directory to a timestamped
destination directory, managed as a long-running systemd service.

Demonstrates the pattern used to run any long-lived Python process
(a backup job here, but the same shape applies to a model-serving worker,
a queue consumer, etc.) as a supervised OS service:
  - config loaded from an external file (never hardcoded paths)
  - logging to a file instead of stdout
  - SIGTERM/SIGINT handled for a graceful shutdown

See ../deployment-systemd-daemons.md for the full write-up and the paired
.service unit file.

NOTE: ported from a university lab exercise. The original hardcoded a
personal Windows/OneDrive path in `if __name__ == "__main__"` and in the
accompanying backup_config.ini. Both have been replaced below with a
generic config path that can be overridden via an environment variable —
no real username or personal path is referenced anywhere in this file.
"""
import os
import time
import shutil
import logging
import configparser
import signal
import sys
from datetime import datetime

# Generic, environment-overridable config location — replaces the original
# hardcoded personal path (e.g. "C:/Users/<name>/OneDrive/.../backup_config.ini").
DEFAULT_CONFIG_PATH = os.environ.get(
    "BACKUP_DAEMON_CONFIG", "/etc/backup_daemon/config.ini"
)


def load_config(config_file: str) -> dict:
    """Read [Settings] section from an INI file.

    Expected keys: source_dir, backup_dir, backup_interval, log_file.
    Example config.ini:

        [Settings]
        source_dir = /path/to/backup/source
        backup_dir = /path/to/backup/destination
        backup_interval = 3600
        log_file = /var/log/backup_daemon.log
    """
    config = configparser.ConfigParser()
    if not config.read(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return {
        "source_dir": config.get("Settings", "source_dir"),
        "backup_dir": config.get("Settings", "backup_dir"),
        "backup_interval": config.getint("Settings", "backup_interval"),
        "log_file": config.get("Settings", "log_file"),
    }


def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )


def backup_data(source_dir: str, backup_dir: str) -> None:
    """Copy source_dir into a new timestamped subdirectory of backup_dir."""
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H.%M.%S_%d.%m.%Y")
    backup_subdir = os.path.join(backup_dir, timestamp)
    shutil.copytree(source_dir, backup_subdir)
    logging.info("Backup created: %s", backup_subdir)


def handle_signal(signum, frame) -> None:
    """Handle SIGTERM/SIGINT for a graceful shutdown.

    systemd sends SIGTERM on `systemctl stop`; Kubernetes sends SIGTERM to
    a pod's main process before SIGKILL on termination. Catching it here
    lets the process log its own shutdown and exit(0) instead of being
    killed mid-copy (which would leave a corrupt/partial backup).
    """
    logging.info("Received signal %s - daemon stopping...", signum)
    sys.exit(0)


def run_daemon(config_file: str) -> None:
    config = load_config(config_file)
    setup_logging(config["log_file"])
    logging.info("Daemon started...")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while True:
        backup_data(config["source_dir"], config["backup_dir"])
        time.sleep(config["backup_interval"])


if __name__ == "__main__":
    run_daemon(DEFAULT_CONFIG_PATH)
