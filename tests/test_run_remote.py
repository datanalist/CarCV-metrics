"""Unit tests for deploy/scripts/run_remote.py (no network, no GPU)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy" / "scripts"))

import run_remote  # noqa: E402


def _minimal_cfg(experiments=None, results_local="results_collected", servers=None):
    return {
        "servers": servers
        or [
            {"host": "h", "port": 22, "user": "root", "experiments": experiments or []}
        ],
        "remote": {
            "deploy_dir": "~/x",
            "venv": "venv",
            "results_local": results_local,
        },
    }


# ── Original 5 (preserved per KEEP) ───────────────────────────────────────────

def test_validate_rejects_unknown_experiment(capsys):
    cfg = _minimal_cfg(experiments=["trafficcamnet", "foobar"])
    with pytest.raises(SystemExit) as exc:
        run_remote.validate_config(cfg)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "foobar" in err
    assert "trafficcamnet" in err


def test_validate_accepts_all_valid_experiments():
    cfg = _minimal_cfg(experiments=sorted(run_remote.VALID_EXPERIMENTS))
    run_remote.validate_config(cfg)  # must not raise


def test_build_ssh_cmd_inserts_port_and_user():
    cmd = run_remote.build_ssh_cmd("ssh4.qudata.ai", 19356, "root", "echo hi")
    assert cmd == ["ssh", "-p", "19356", "root@ssh4.qudata.ai", "echo hi"]


def test_build_rsync_cmd_uses_port_and_excludes():
    cmd = run_remote.build_rsync_cmd(
        "src/", "user@h:dst/", 19356, ["__pycache__", "venv"]
    )
    assert cmd[:4] == ["rsync", "-avz", "-e", "ssh -p 19356"]
    assert "--exclude=__pycache__" in cmd
    assert "--exclude=venv" in cmd
    assert cmd[-2:] == ["src/", "user@h:dst/"]


def test_dry_run_does_not_call_subprocess(monkeypatch, capsys):
    def must_not_run(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called in dry-run")

    monkeypatch.setattr(subprocess, "run", must_not_run)
    rc = run_remote.execute_or_print(
        ["ssh", "-p", "22", "root@host", "ls"], dry_run=True
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "[DRY-RUN]" in captured
    assert "ssh -p 22 root@host ls" in captured


# ── New 3 (added by amended spec, iter 1) ─────────────────────────────────────

def test_validate_rejects_dangerous_deploy_dir(capsys):
    """validate_config must reject deploy_dir containing shell metacharacters
    before any SSH attempt, neutralizing remote injection."""
    cfg = _minimal_cfg()
    cfg["remote"]["deploy_dir"] = "; rm -rf ~"
    with pytest.raises(SystemExit) as exc:
        run_remote.validate_config(cfg)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "deploy_dir" in err


def test_validate_accepts_tilde_path():
    """Charset must allow tilde so remote shell can expand ~/path."""
    cfg = _minimal_cfg()
    cfg["remote"]["deploy_dir"] = "~/cars-eval"
    cfg["remote"]["venv"] = "venv"
    run_remote.validate_config(cfg)  # must not raise


def test_duplicate_host_rejected(capsys):
    cfg = _minimal_cfg(
        servers=[
            {"host": "h1", "port": 22, "user": "root", "experiments": []},
            {"host": "h1", "port": 22, "user": "root", "experiments": []},
        ],
    )
    with pytest.raises(SystemExit) as exc:
        run_remote.validate_config(cfg)
    assert exc.value.code == 2
    assert "duplicate" in capsys.readouterr().err.lower()


def test_absolute_results_local_rejected(capsys):
    cfg = _minimal_cfg(results_local="/tmp/abs_path")
    with pytest.raises(SystemExit) as exc:
        run_remote.validate_config(cfg)
    assert exc.value.code == 2
    assert "results_local" in capsys.readouterr().err


# ── Round 2 hardening (post-review): boundary & command-assembly coverage ─────

def test_load_config_missing_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        run_remote.load_config(tmp_path / "does-not-exist.yaml")
    assert exc.value.code == 2
    assert "not found" in capsys.readouterr().err


def test_load_config_invalid_yaml(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("servers: [\n  - host:\n!!!not valid")
    with pytest.raises(SystemExit) as exc:
        run_remote.load_config(bad)
    assert exc.value.code == 2
    assert "invalid YAML" in capsys.readouterr().err


def test_validate_rejects_non_mapping_cfg(capsys):
    with pytest.raises(SystemExit) as exc:
        run_remote.validate_config(["not", "a", "mapping"])
    assert exc.value.code == 2
    assert "mapping" in capsys.readouterr().err


def test_validate_rejects_null_results_local(capsys):
    cfg = _minimal_cfg()
    cfg["remote"]["results_local"] = None
    with pytest.raises(SystemExit) as exc:
        run_remote.validate_config(cfg)
    assert exc.value.code == 2
    assert "results_local" in capsys.readouterr().err


def test_validate_rejects_dangerous_host(capsys):
    cfg = _minimal_cfg(
        servers=[{"host": "h;evil", "port": 22, "user": "root", "experiments": []}]
    )
    with pytest.raises(SystemExit) as exc:
        run_remote.validate_config(cfg)
    assert exc.value.code == 2
    assert "host" in capsys.readouterr().err


def test_validate_rejects_dangerous_results_local(capsys):
    cfg = _minimal_cfg(results_local="results; rm -rf /")
    with pytest.raises(SystemExit) as exc:
        run_remote.validate_config(cfg)
    assert exc.value.code == 2
    assert "results_local" in capsys.readouterr().err


def test_validate_rejects_duplicate_experiment(capsys):
    cfg = _minimal_cfg(experiments=["trafficcamnet", "trafficcamnet"])
    with pytest.raises(SystemExit) as exc:
        run_remote.validate_config(cfg)
    assert exc.value.code == 2
    err = capsys.readouterr().err.lower()
    assert "duplicate" in err and "trafficcamnet" in err


@pytest.mark.parametrize("bad_port", [0, -1, 70000, True, False, "22 bogus"])
def test_validate_rejects_bad_port(capsys, bad_port):
    cfg = _minimal_cfg(
        servers=[{"host": "h", "port": bad_port, "user": "root", "experiments": []}]
    )
    with pytest.raises(SystemExit) as exc:
        run_remote.validate_config(cfg)
    assert exc.value.code == 2


def test_run_one_host_wraps_jobs_in_brace_group(monkeypatch):
    """run_one_host must use `{ … disown -a; }` so cd survives backgrounded jobs."""
    captured = {}

    def fake_exec(cmd, dry_run):
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr(run_remote, "execute_or_print", fake_exec)
    server = {"host": "h", "port": 22, "user": "root",
              "experiments": ["trafficcamnet", "lpdnet"]}
    remote = {"deploy_dir": "~/x", "venv": "venv", "results_local": "rc"}
    run_remote.run_one_host(server, remote, dry_run=False)

    remote_cmd = captured["cmd"][-1]
    assert "cd ~/x && source venv/bin/activate" in remote_cmd
    # the load-bearing group: opening `{ `, trailing `disown -a; }`
    assert "&& { " in remote_cmd
    assert "disown -a; }" in remote_cmd
    assert "< /dev/null > logs/trafficcamnet.log" in remote_cmd
    assert "< /dev/null > logs/lpdnet.log" in remote_cmd


def test_run_one_host_skips_empty_experiments(capsys, monkeypatch):
    called = []
    monkeypatch.setattr(
        run_remote, "execute_or_print", lambda *a, **kw: called.append(a) or 0
    )
    server = {"host": "h", "port": 22, "user": "root", "experiments": []}
    remote = {"deploy_dir": "~/x", "venv": "venv", "results_local": "rc"}
    _, rc = run_remote.run_one_host(server, remote, dry_run=False)
    assert rc == 0
    assert not called
    assert "skipping run" in capsys.readouterr().out


def test_deploy_one_host_skips_setup_on_rsync_failure(monkeypatch):
    calls = []

    def fake_exec(cmd, dry_run):
        calls.append(cmd)
        return 23 if cmd[0] == "rsync" else 0

    monkeypatch.setattr(run_remote, "execute_or_print", fake_exec)
    server = {"host": "h", "port": 22, "user": "root", "experiments": []}
    remote = {"deploy_dir": "~/x", "venv": "venv", "results_local": "rc"}
    _, rc = run_remote.deploy_one_host(server, remote, dry_run=False)
    assert rc == 23
    # only rsync was attempted; setup_server.sh ssh was skipped
    assert len(calls) == 1
    assert calls[0][0] == "rsync"


def test_collect_one_host_uses_mkpath_and_no_premkdir(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        run_remote, "execute_or_print", lambda cmd, dry: (calls.append(cmd), 1)[1]
    )
    monkeypatch.setattr(run_remote, "ROOT", tmp_path)
    server = {"host": "h1", "port": 22, "user": "root", "experiments": []}
    remote = {"deploy_dir": "~/x", "venv": "venv", "results_local": "rc"}
    _, rc = run_remote.collect_one_host(server, remote, dry_run=False)
    assert rc == 1
    # both rsync calls include --mkpath
    assert all("--mkpath" in c for c in calls)
    # no leftover empty host dirs from pre-mkdir
    assert not (tmp_path / "rc" / "h1").exists()


def test_dispatch_aggregates_max_rc(capsys):
    def action(server, remote, dry_run):
        return server["host"], {"a": 0, "b": 5, "c": 2}[server["host"]]

    cfg = {
        "servers": [
            {"host": "a", "port": 22, "user": "root"},
            {"host": "b", "port": 22, "user": "root"},
            {"host": "c", "port": 22, "user": "root"},
        ],
        "remote": {"deploy_dir": "~/x", "venv": "venv", "results_local": "rc"},
    }
    worst = run_remote.dispatch(action, cfg, dry_run=False)
    assert worst == 5


def test_dispatch_isolates_uncaught_exception(capsys):
    def action(server, remote, dry_run):
        if server["host"] == "boom":
            raise RuntimeError("kaboom")
        return server["host"], 0

    cfg = {
        "servers": [
            {"host": "ok1", "port": 22, "user": "root"},
            {"host": "boom", "port": 22, "user": "root"},
            {"host": "ok2", "port": 22, "user": "root"},
        ],
        "remote": {"deploy_dir": "~/x", "venv": "venv", "results_local": "rc"},
    }
    worst = run_remote.dispatch(action, cfg, dry_run=False)
    err = capsys.readouterr().err
    # boom host gets UNCAUGHT, but ok1/ok2 still report OK and worst = 1
    assert worst == 1
    assert "UNCAUGHT" in err
    assert "kaboom" in err


def test_dispatch_empty_servers_logs_explicit_message(capsys):
    cfg = {"servers": [], "remote": {"deploy_dir": "~/x", "venv": "v", "results_local": "rc"}}
    worst = run_remote.dispatch(lambda *a: ("x", 0), cfg, dry_run=False)
    assert worst == 0
    assert "no servers configured" in capsys.readouterr().err
