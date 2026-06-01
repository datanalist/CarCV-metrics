#!/usr/bin/env python3
"""Remote experiment orchestrator for CARS evaluation.

Subcommands:
  deploy   — rsync project + run setup_server.sh on each host
  run      — launch evaluate.py for assigned experiments (fire-and-forget)
  collect  — rsync results + plots back to results_local/{host}/

Common flags (available on every subcommand):
  --dry-run        print planned ssh/rsync commands without executing
  --config PATH    YAML config (default: configs/remote_experiments.yaml)

Ordering: run `deploy` once before the first `run` — setup_server.sh creates
the remote `logs/` directory that fire-and-forget nohup writes into.

Loopback smoke-test:
  Alias `localhost-test` in ~/.ssh/config, list it under servers in the YAML,
  then `python deploy/scripts/run_remote.py deploy --dry-run` prints the exact
  commands that would run — copy any one into a shell to test manually.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NoReturn

import yaml

VALID_EXPERIMENTS = {
    "trafficcamnet",
    "vehiclemakenet",
    "vehicletypenet",
    "facedetect",
    "lpdnet",
    "lprnet",
    "nomeroff_lpd",
    "nomeroff_ocr",
}
REQUIRED_REMOTE_KEYS = {"deploy_dir", "venv", "results_local"}

# Charset allowed for values interpolated into remote shell strings.
# Permits ~ / . _ - and alnum — enough for paths like ~/cars-eval, /opt/x,
# rejects every shell metachar (;, &, |, $, (), {}, quotes, spaces, NL, etc.).
SAFE_SHELL_PATH = re.compile(r"^[A-Za-z0-9._~/-]+$")

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "remote_experiments.yaml"

RSYNC_EXCLUDES = [
    ".git",
    "venv",
    "__pycache__",
    "*.pyc",
    "data",
    "models",
    "results",
    "plots",
    "results_collected",
    ".claude",
]


def _die(msg: str) -> NoReturn:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def load_config(path: Path) -> dict:
    try:
        text = path.read_text()
    except FileNotFoundError:
        _die(f"config file not found: {path}")
    except (IsADirectoryError, PermissionError, OSError) as e:
        _die(f"cannot read config {path}: {e}")
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as e:
        _die(f"invalid YAML in {path}: {e}")
    return parsed or {}


def _coerce_port(server: dict) -> int:
    raw = server.get("port", 22)
    if raw is None:
        return 22
    # bool is an int subclass — reject explicitly so port: true/false doesn't
    # silently coerce to 1/0.
    if isinstance(raw, bool):
        _die(
            f"invalid port {raw!r} on host {server.get('host')!r}; must be an integer"
        )
    try:
        port = int(raw)
    except (TypeError, ValueError):
        _die(
            f"invalid port {raw!r} on host {server.get('host')!r}; must be an integer"
        )
    if not 1 <= port <= 65535:
        _die(
            f"port {port} on host {server.get('host')!r} out of range; must be 1..65535"
        )
    return port


def _validate_shell_safe(value: str, field: str) -> None:
    if not isinstance(value, str) or not SAFE_SHELL_PATH.fullmatch(value):
        _die(
            f"{field}={value!r} contains characters disallowed in remote shell "
            f"interpolation; allowed charset: {SAFE_SHELL_PATH.pattern}"
        )


def validate_config(cfg: dict) -> None:
    """Fail fast on any config issue — before opening any SSH connection."""
    if not isinstance(cfg, dict):
        _die(f"config root must be a YAML mapping; got {type(cfg).__name__}")

    servers = cfg.get("servers") or []
    remote = cfg.get("remote") or {}

    missing = REQUIRED_REMOTE_KEYS - set(remote.keys())
    if missing:
        _die(f"remote section missing required keys: {sorted(missing)}")

    _validate_shell_safe(remote["deploy_dir"], "remote.deploy_dir")
    _validate_shell_safe(remote["venv"], "remote.venv")

    results_local = remote["results_local"]
    if results_local is None:
        _die("remote.results_local must be a non-null relative path")
    if not isinstance(results_local, str):
        _die(
            f"remote.results_local must be a string; got "
            f"{type(results_local).__name__}: {results_local!r}"
        )
    if Path(results_local).is_absolute():
        _die(
            f"remote.results_local must be a relative path; got {results_local!r}. "
            "Use a repo-relative path like 'results_collected'."
        )
    _validate_shell_safe(results_local, "remote.results_local")

    seen: set[tuple[str, int, str]] = set()
    for server in servers:
        host = server.get("host")
        if not host:
            _die(f"server entry missing required 'host' key: {server!r}")
        # host appears in shell-interpolated rsync src strings (`user@host:path`)
        # and as a path segment under results_local. Apply the same charset.
        _validate_shell_safe(host, "servers[].host")
        port = _coerce_port(server)
        user = server.get("user") or os.environ.get("USER", "root")
        key = (host, port, user)
        if key in seen:
            _die(
                f"duplicate server in config: host={host}, "
                f"port={port}, user={user}"
            )
        seen.add(key)

        experiments = server.get("experiments") or []
        exp_seen: set[str] = set()
        for exp in experiments:
            if exp not in VALID_EXPERIMENTS:
                allowed = ", ".join(sorted(VALID_EXPERIMENTS))
                _die(
                    f"unknown experiment {exp!r} on host {host!r}; "
                    f"allowed: {allowed}"
                )
            if exp in exp_seen:
                _die(
                    f"duplicate experiment {exp!r} on host {host!r}; "
                    "each experiment writes to logs/{exp}.log and would clobber"
                )
            exp_seen.add(exp)


def _server_addr(server: dict) -> tuple[str, int, str, str | None]:
    host = server["host"]
    port = _coerce_port(server)
    user = server.get("user") or os.environ.get("USER", "root")
    identity = server.get("identity_file")
    if identity:
        identity = str(Path(identity).expanduser())
    return host, port, user, identity


def build_ssh_cmd(
    host: str, port: int, user: str, remote_cmd: str, identity: str | None = None
) -> list[str]:
    cmd = ["ssh", "-p", str(port)]
    if identity:
        cmd.extend(["-i", identity])
    cmd.extend([f"{user}@{host}", remote_cmd])
    return cmd


def build_rsync_cmd(
    src: str,
    dst: str,
    port: int,
    excludes: list[str] | None = None,
    identity: str | None = None,
    extras: list[str] | None = None,
) -> list[str]:
    ssh_opts = f"ssh -p {port}" + (f" -i {identity}" if identity else "")
    cmd = ["rsync", "-avz", "-e", ssh_opts]
    for ex in excludes or []:
        cmd.append(f"--exclude={ex}")
    if extras:
        cmd.extend(extras)
    cmd.extend([src, dst])
    return cmd


def execute_or_print(cmd: list[str], dry_run: bool) -> int:
    """Single seam for shell execution — mockable in tests, dry-runnable in CLI."""
    if dry_run:
        print(f"[DRY-RUN] {shlex.join(cmd)}")
        return 0
    try:
        return subprocess.run(cmd, check=False).returncode
    except FileNotFoundError as e:
        print(f"[ERROR] command not found: {e}", file=sys.stderr)
        return 127


def deploy_one_host(server: dict, remote: dict, dry_run: bool) -> tuple[str, int]:
    host, port, user, identity = _server_addr(server)
    deploy_dir = remote["deploy_dir"]

    src = f"{ROOT}/deploy/"
    dst = f"{user}@{host}:{deploy_dir}/"
    rc1 = execute_or_print(
        build_rsync_cmd(src, dst, port, RSYNC_EXCLUDES, identity), dry_run
    )
    if rc1 != 0 and not dry_run:
        return host, rc1  # don't run setup on broken deploy

    setup_cmd = f"cd {deploy_dir} && bash scripts/setup_server.sh"
    rc2 = execute_or_print(
        build_ssh_cmd(host, port, user, setup_cmd, identity), dry_run
    )
    return host, max(rc1, rc2)


def run_one_host(server: dict, remote: dict, dry_run: bool) -> tuple[str, int]:
    host, port, user, identity = _server_addr(server)
    experiments = server.get("experiments") or []
    if not experiments:
        print(f"[{host}] WARNING: empty experiments list, skipping run")
        return host, 0

    deploy_dir = remote["deploy_dir"]
    venv = remote["venv"]
    prelude = f"cd {deploy_dir} && source {venv}/bin/activate"
    # Group command `{ ... }` runs in the SAME shell (not subshell),
    # so the cd from prelude stays in effect for each backgrounded nohup.
    # Without the group, `cmd && nohup &` puts the entire AND-chain into a
    # subshell — and the next `;`-separated nohup loses the cd and fails on
    # `logs/...: No such file or directory`.
    bg_jobs = " ".join(
        f"nohup python evaluation/evaluate.py --models {exp} "
        f"< /dev/null > logs/{exp}.log 2>&1 &"
        for exp in experiments
    )
    remote_cmd = f"{prelude} && {{ {bg_jobs} disown -a; }}"
    return host, execute_or_print(
        build_ssh_cmd(host, port, user, remote_cmd, identity), dry_run
    )


def collect_one_host(server: dict, remote: dict, dry_run: bool) -> tuple[str, int]:
    host, port, user, identity = _server_addr(server)
    deploy_dir = remote["deploy_dir"]
    results_local = Path(remote["results_local"])

    target_results = ROOT / results_local / host / "results"
    target_plots = ROOT / results_local / host / "plots"
    # No pre-mkdir: rsync --mkpath creates parents on demand. A failed collect
    # then leaves nothing behind, instead of empty `host/results/` dirs that
    # look like "collected but empty".
    rc1 = execute_or_print(
        build_rsync_cmd(
            f"{user}@{host}:{deploy_dir}/results/",
            f"{target_results}/",
            port,
            identity=identity,
            extras=["--mkpath"],
        ),
        dry_run,
    )
    rc2 = execute_or_print(
        build_rsync_cmd(
            f"{user}@{host}:{deploy_dir}/plots/",
            f"{target_plots}/",
            port,
            identity=identity,
            extras=["--mkpath"],
        ),
        dry_run,
    )
    return host, max(rc1, rc2)


def dispatch(action_fn, cfg: dict, dry_run: bool) -> int:
    servers = cfg.get("servers") or []
    remote = cfg.get("remote") or {}

    print(f"[PLAN] {action_fn.__name__} on {len(servers)} host(s); dry_run={dry_run}")
    for s in servers:
        host, port, user, identity = _server_addr(s)
        id_note = f", identity={identity}" if identity else ""
        print(
            f"  - {host} (port {port}, user {user}{id_note}): "
            f"{s.get('experiments') or []}"
        )

    if not servers:
        print("[PLAN] no servers configured — nothing to do", file=sys.stderr)
        return 0

    worst = 0
    with ThreadPoolExecutor(max_workers=max(1, len(servers))) as pool:
        futures = {pool.submit(action_fn, s, remote, dry_run): s for s in servers}
        for fut in as_completed(futures):
            srv = futures[fut]
            try:
                host, rc = fut.result()
            except Exception as exc:
                host, rc = srv.get("host", "<unknown>"), 1
                print(f"[{host}] UNCAUGHT: {exc}", file=sys.stderr)
            status = "OK" if rc == 0 else f"FAIL (rc={rc})"
            print(f"[{host}] {status}")
            worst = max(worst, rc)
    return worst


def main(argv: list[str] | None = None) -> int:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--dry-run",
        action="store_true",
        help="print ssh/rsync commands without executing them",
    )
    parent.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG.relative_to(ROOT)})",
    )

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("deploy", parents=[parent], help="rsync + setup on each host")
    sub.add_parser("run", parents=[parent], help="launch evaluate.py per assignment")
    sub.add_parser("collect", parents=[parent], help="rsync results to results_local/{host}/")

    args = p.parse_args(argv)

    cfg = load_config(args.config)
    validate_config(cfg)

    action = {
        "deploy": deploy_one_host,
        "run": run_one_host,
        "collect": collect_one_host,
    }[args.cmd]
    return dispatch(action, cfg, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
