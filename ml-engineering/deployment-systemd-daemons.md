# Deployment: systemd Daemons & Long-Running Services

## What

`systemd` is the standard Linux init/service manager. A `.service` unit file tells it how to start, restart, and supervise a long-running process — turning "a Python script" into "a managed service" that survives reboots, crashes, and log rotation without a human babysitting a terminal.

This matters for ML engineering because **a model-serving process is exactly this pattern**: a long-running loop (or an HTTP server loop) that needs to start on boot, restart on crash, log somewhere durable, and shut down cleanly when the orchestrator asks it to. Everything below generalizes directly from "backup daemon" to "inference server process."

## Why

- Running `python app.py` in a terminal dies when the terminal closes, doesn't restart on crash, and has no standard logging/monitoring hooks.
- Production services (a REST inference endpoint, a queue consumer, a scheduled retraining job) need: start-on-boot, auto-restart, controlled shutdown, and centralized logs (`journalctl`).
- The same signal-handling discipline you build here is required inside a Docker container managed by Kubernetes — the orchestrator's shutdown mechanism *is* a Unix signal, so understanding systemd's model transfers directly.

## How does it work?

### Anatomy of a `.service` unit file

```ini
[Unit]
Description=Backup Daemon

[Service]
User=appuser
ExecStart=/usr/bin/python3 /opt/backup_daemon/backup_daemon.py
Restart=always
Environment=BACKUP_DAEMON_CONFIG=/etc/backup_daemon/config.ini

[Install]
WantedBy=multi-user.target
```

- **`[Unit]`** — metadata and dependency ordering (`Description`, `After=network.target`, etc.).
- **`[Service]`**
  - `ExecStart=` — the exact command systemd runs to start the service (absolute paths, no shell expansion by default).
  - `User=` — run as an unprivileged service account, not root — least-privilege principle.
  - `Restart=always` (or `on-failure`) — systemd relaunches the process if it exits/crashes. Pair with `RestartSec=` to avoid a crash-restart tight loop.
  - `Environment=` / `EnvironmentFile=` — inject config without hardcoding paths into the script (see `code/backup_daemon.py`, which reads `BACKUP_DAEMON_CONFIG` instead of a baked-in path).
- **`[Install]`** — `WantedBy=multi-user.target` registers the service to start at normal multi-user boot when enabled.

### Graceful shutdown: SIGTERM vs SIGKILL

When you run `systemctl stop myservice`, systemd sends **SIGTERM** first, waits `TimeoutStopSec` (default 90s), and only then sends **SIGKILL** if the process hasn't exited. A process that doesn't handle SIGTERM gets killed mid-operation — mid-write, mid-request, mid-backup.

```python
import signal, sys, logging

def handle_signal(signum, frame):
    logging.info("Received signal %s - shutting down gracefully...", signum)
    # flush buffers, close DB connections, finish in-flight requests, etc.
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)   # Ctrl+C / local dev
```

**This is directly relevant to containerized/orchestrated ML deployments.** Kubernetes' pod termination sequence is the same two-step pattern: it sends SIGTERM to the container's PID 1, waits `terminationGracePeriodSeconds` (default 30s), then SIGKILLs. A model server that ignores SIGTERM will drop in-flight prediction requests and get hard-killed on every rolling deploy or autoscale-down event — exactly the bug this pattern prevents.

### Config-from-file, not hardcoded paths

`configparser` + an external `.ini`/`.env`/`.yaml` file keeps the code portable across dev/staging/prod without editing source:

```ini
[Settings]
source_dir = /path/to/backup/source
backup_dir = /path/to/backup/destination
backup_interval = 3600
log_file = /var/log/backup_daemon.log
```

The same idea in an ML server: model path, batch size, port, and feature-store URL all belong in config/env vars, never hardcoded — this is what makes the same container image deployable to dev and prod unchanged (see [`environment-setup-colab-kaggle.md`](./environment-setup-colab-kaggle.md) for the contrasting *non*-reproducible, experimentation-only pattern, and [`README.md`](./README.md#docker) for the Docker angle on this).

### Managing the service

```bash
sudo cp backup_daemon.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now backup_daemon.service   # enable = start on boot, --now = start immediately
sudo systemctl status backup_daemon.service
journalctl -u backup_daemon.service -f              # tail logs
```

## Example

Full working example (paths genericized — see note at the bottom):

- [`code/backup_daemon.py`](./code/backup_daemon.py) — the daemon: `configparser`-based config, file logging, SIGTERM/SIGINT handling, an infinite loop that does the actual work (`shutil.copytree` on an interval).
- Paired unit file (not duplicated here, structure shown above) wires it up with `User=`, `ExecStart=`, and `Restart=always`.

## Common interview questions

1. What happens if a process ignores SIGTERM? Why does that matter for a service behind a load balancer?
2. What's the difference between `Restart=always` and `Restart=on-failure`?
3. How does Kubernetes' pod termination relate to Unix signals?
4. Why put config in an external file/env var instead of hardcoding it in the script?
5. How would you view/debug the logs of a systemd-managed service?
6. Why run a service as a dedicated non-root user?

## Common mistakes

- Not handling SIGTERM at all — the process gets SIGKILLed and loses in-flight work (dropped requests, partial writes, corrupt backups).
- Hardcoding absolute paths/credentials into the script instead of reading them from config/env — breaks portability across environments and (as in the original lab exercise this note is based on) can leak a personal username or folder path into version control.
- Setting `Restart=always` with no `RestartSec=`, causing a crash loop that hammers the CPU/logs.
- Running the service as root when it doesn't need root privileges.
- Forgetting `systemctl daemon-reload` after editing a unit file — systemd keeps using the cached old definition.
- Logging to stdout only instead of a file/`journald`-friendly sink, losing history on restart.

## See also

- [`README.md`](./README.md) — Docker, CI/CD, and the notebook -> script -> container -> orchestrated-deployment path
- [`../ml-system-design/README.md`](../ml-system-design/README.md#model-serving) — model serving, canary/shadow/blue-green rollout strategies
- [`environment-setup-colab-kaggle.md`](./environment-setup-colab-kaggle.md) — the opposite end of the reproducibility spectrum (throwaway experimentation environments)
