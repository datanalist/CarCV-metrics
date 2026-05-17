#!/usr/bin/env python3
"""Remote experiment orchestrator for CarCV-metrics.

Usage:
    python deploy/scripts/run_remote.py [--config PATH] <deploy|run|collect>

Commands:
    deploy   Sync deploy/ to all servers and run setup_server.sh
    run      Launch assigned experiments in parallel (fire-and-forget)
    collect  rsync results and plots from all servers to local
"""

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

_SCRIPTS_DIR = Path(__file__).parent
DEPLOY_ROOT = _SCRIPTS_DIR.parent       # <project>/deploy/
PROJECT_ROOT = DEPLOY_ROOT.parent       # <project>/

VALID_EXPERIMENTS = frozenset(
    ["trafficcamnet", "vehiclemakenet", "vehicletypenet", "lpdnet", "lprnet"]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _validate(servers: list[dict]) -> None:
    bad: set[str] = set()
    for s in servers:
        for exp in s.get("experiments") or []:
            if exp not in VALID_EXPERIMENTS:
                bad.add(exp)
    if bad:
        sys.exit(
            f"Unknown experiments: {sorted(bad)}. "
            f"Valid: {sorted(VALID_EXPERIMENTS)}"
        )


def _ssh(host: str, cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", host, cmd], capture_output=True, text=True)


def _rsync(src: str, dst: str, excludes: list[str] | None = None) -> subprocess.CompletedProcess:
    args = ["rsync", "-avz", "--progress"]
    for exc in excludes or []:
        args += ["--exclude", exc]
    args += [src, dst]
    return subprocess.run(args, capture_output=True, text=True)


# ── Commands ──────────────────────────────────────────────────────────────────

def _deploy_one(host: str, deploy_dir: str) -> bool:
    src = str(DEPLOY_ROOT) + "/"
    dst = f"{host}:{deploy_dir}/"
    log.info("[%s] rsync → %s", host, deploy_dir)

    result = _rsync(
        src, dst,
        excludes=["venv", "__pycache__", "*.pyc", "data", "models", "results", "plots"],
    )
    if result.returncode != 0:
        log.error("[%s] rsync failed:\n%s", host, result.stderr)
        return False

    log.info("[%s] running setup_server.sh", host)
    result = _ssh(host, f"cd {deploy_dir} && bash scripts/setup_server.sh")
    if result.returncode != 0:
        log.error("[%s] setup failed:\n%s", host, result.stderr)
        return False

    log.info("[%s] deploy complete", host)
    return True


def cmd_deploy(config: dict) -> int:
    servers = config["servers"]
    deploy_dir = config["remote"]["deploy_dir"]
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=len(servers)) as pool:
        futures = {
            pool.submit(_deploy_one, s["host"], deploy_dir): s["host"]
            for s in servers
        }
        for fut in as_completed(futures):
            if not fut.result():
                failed.append(futures[fut])

    if failed:
        log.error("Deploy failed on: %s", failed)
        return 1
    return 0


def _run_one(host: str, experiments: list[str], deploy_dir: str, venv: str) -> bool:
    if not experiments:
        log.warning("[%s] no experiments assigned, skipping", host)
        return True

    activate = f"source {deploy_dir}/{venv}/bin/activate"
    nohup_cmds = " && ".join(
        f"nohup python {deploy_dir}/evaluation/evaluate.py --models {exp} "
        f"> {deploy_dir}/logs/{exp}.log 2>&1 &"
        for exp in experiments
    )
    cmd = f"cd {deploy_dir} && {activate} && {nohup_cmds}"

    log.info("[%s] launching: %s", host, experiments)
    result = _ssh(host, cmd)
    if result.returncode != 0:
        log.error("[%s] SSH error:\n%s", host, result.stderr)
        return False

    log.info("[%s] experiments started (fire-and-forget)", host)
    return True


def cmd_run(config: dict) -> int:
    servers = config["servers"]
    remote = config["remote"]
    deploy_dir = remote["deploy_dir"]
    venv = remote.get("venv", "venv")
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=len(servers)) as pool:
        futures = {
            pool.submit(
                _run_one, s["host"], s.get("experiments") or [], deploy_dir, venv
            ): s["host"]
            for s in servers
        }
        for fut in as_completed(futures):
            if not fut.result():
                failed.append(futures[fut])

    if failed:
        log.error("Run failed on: %s", failed)
        return 1
    return 0


def _collect_one(host: str, deploy_dir: str, local_base: Path) -> bool:
    host_dir = local_base / host
    host_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    for subdir in ("results", "plots"):
        src = f"{host}:{deploy_dir}/{subdir}/"
        dst = str(host_dir / subdir) + "/"
        log.info("[%s] collecting %s/", host, subdir)
        result = _rsync(src, dst)
        if result.returncode != 0:
            log.warning("[%s] rsync %s warning:\n%s", host, subdir, result.stderr)
            ok = False

    return ok


def cmd_collect(config: dict) -> int:
    servers = config["servers"]
    remote = config["remote"]
    deploy_dir = remote["deploy_dir"]
    results_local = remote.get("results_local", "results_collected")
    local_base = PROJECT_ROOT / results_local
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=len(servers)) as pool:
        futures = {
            pool.submit(_collect_one, s["host"], deploy_dir, local_base): s["host"]
            for s in servers
        }
        for fut in as_completed(futures):
            if not fut.result():
                failed.append(futures[fut])

    if failed:
        log.warning("Collect had warnings on: %s", failed)
        return 1
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    default_config = PROJECT_ROOT / "configs" / "remote_experiments.yaml"

    parser = argparse.ArgumentParser(
        description="Remote experiment orchestrator for CarCV-metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        metavar="PATH",
        help="Path to remote_experiments.yaml",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("deploy", help="Sync project files to all servers and run setup")
    sub.add_parser("run", help="Launch experiments in parallel on all servers (fire-and-forget)")
    sub.add_parser("collect", help="Collect results and plots from all servers")

    args = parser.parse_args()
    config = _load_config(args.config)
    _validate(config.get("servers", []))

    dispatch = {"deploy": cmd_deploy, "run": cmd_run, "collect": cmd_collect}
    sys.exit(dispatch[args.command](config))


if __name__ == "__main__":
    main()
