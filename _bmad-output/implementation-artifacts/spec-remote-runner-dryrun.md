---
title: 'Remote Runner: orchestrator + dry-run + YAML с host/port/user'
type: 'feature'
created: '2026-05-17'
status: 'done'
baseline_commit: 'a5cf5837e6f080ac905bb10175e5fc63c8bffdb7'
context:
  - _bmad-output/project-context.md
  - _bmad-output/implementation-artifacts/deferred-work.md
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `deploy/scripts/deploy_to_servers.sh` и `collect_results.sh` прошиты на `qudata2`/`qudata5` без портов/пользователей, нет dry-run и нет способа объявить произвольный набор серверов с распределением экспериментов.

**Approach:** Декларативный `configs/remote_experiments.yaml` (поля `host/port/user/experiments`) и Python-оркестратор `deploy/scripts/run_remote.py` с подкомандами `deploy / run / collect` и общим `--dry-run`, который печатает `ssh`/`rsync`-команды вместо выполнения. Без изменений в `evaluate.py` и metrics-коде.

## Boundaries & Constraints

**Always:**
- Хосты и эксперименты — только из YAML; никаких hardcoded имён в Python.
- SSH/rsync — через `subprocess.run` (паттерн уже в `deploy/scripts/*.sh`); paramiko не используется.
- Параллелизм серверов — `ThreadPoolExecutor(max_workers=len(servers))`; на одном сервере N экспериментов запускаются цепочкой `nohup ... & disown` в одной SSH-сессии.
- Логи: `{deploy_dir}/logs/{experiment}.log` на сервере.
- `run` — fire-and-forget (завершается после старта фоновых процессов).
- Сбор — в `{remote.results_local}/{host}/results/` и `.../plots/`.
- Допустимые имена экспериментов — `trafficcamnet`, `vehiclemakenet`, `vehicletypenet`, `lpdnet`, `lprnet` (совпадение с `evaluate.py --models`).
- `--dry-run` доступен на всех трёх подкомандах: печатает каждый `ssh`/`rsync` + план параллелизма, exit 0, ничего не исполняет.
- YAML-сервер: `host` обязателен, `port` default `22`, `user` default `$USER`.

**Ask First:**
- Пустой `experiments: []` на сервере — предупредить и пропустить.

**Never:**
- Не добавлять `torch`/`torchvision`/`nvidia-*` в `deploy/requirements.txt`.
- Не трогать `deploy/evaluation/evaluate.py`, `metrics.py`, `visualize.py`.
- Не реализовывать retry, scheduling, мониторинг прогресса, агрегацию `SUMMARY.md`, notebooks — в [[deferred-work]] Goal 2/3.
- Не писать в `~/.ssh/config` из кода.
- Без hardcoded путей вида `/home/vallo/...`.

## I/O & Edge-Case Matrix

| Scenario | Behavior | Error |
|----------|----------|-------|
| `deploy` валидный конфиг | `rsync` + `setup_server.sh` на каждом хосте параллельно | один хост падает → лог `[host] FAIL`, остальные продолжают; exit 1 |
| `run` валидный конфиг | Параллельно по серверам, эксперименты фоном | SSH-error логируется, остальные продолжают |
| `collect` | `rsync` результатов и `plots/` с каждого хоста | rsync-warning без остановки |
| Неизвестное имя эксперимента | Validation падает до первого SSH с перечислением допустимых | exit 2 |
| Сервер недоступен (SSH timeout) | Warning, продолжение | для `deploy` exit ≠ 0 |
| Без `--config` | Используется `configs/remote_experiments.yaml` | — |
| `--dry-run` | Печать команд + плана, `subprocess.run` не вызывается | exit 0 |
| YAML без `port`/`user` | Default `port=22`, `user=$USER` | — |

</frozen-after-approval>

## Hardening Rules (added 2026-05-17 — bad_spec loopback iteration 1)

- **Validate-then-interpolate** для всех YAML-значений, попадающих в remote shell-строку. Pre-flight `validate_config(cfg)` принимает только символы из набора `^[A-Za-z0-9._~/-]+$` для `deploy_dir` и `venv`. Это допускает реальные пути (`~/cars-eval`, `/opt/cars-eval`, `cars-eval`) и отвергает все shell-метасимволы (`;`, `&`, `|`, `$`, `()`, `{}`, кавычки, пробелы, переносы строк и т.д.). `shlex.quote` НЕ применяется — он сломал бы раскрытие тильды на сервере.
- Имена экспериментов уже whitelist'ятся через `VALID_EXPERIMENTS` (5 значений) — этого достаточно как valid identifier для shell.
- `results_local` обязан быть относительным путём; абсолютный — fail-fast с понятной ошибкой.
- Pre-flight `validate_config(cfg)` (расширение бывшего `validate_experiments`) проверяет: каждый `server` имеет `host`; `remote.{deploy_dir,venv,results_local}` присутствуют; `port` коэрсится в int (catch с понятной ошибкой); нет дубликатов `(host, port, user)` в `servers`; `deploy_dir` и `venv` проходят charset-валидацию. Все ошибки — exit 2 до первого SSH.
- В `run` команда на сервере начинается с `cd {deploy_dir} && source {venv}/bin/activate && ...`. `logs/` создаётся `setup_server.sh` во время `deploy`; user обязан запустить `deploy` до первого `run` (документировать). Каждый `nohup` берёт stdin из `</dev/null`: `nohup python ... < /dev/null > logs/{exp}.log 2>&1 & disown`.

## Code Map

- `configs/remote_experiments.yaml` -- НОВЫЙ: схема с `servers[].{host,port,user,experiments,identity_file?}` + `remote.{deploy_dir,venv,results_local}`. Поле `identity_file` — опциональное, добавлено iter-1 для qudata (разные ключи на серверах); не валидируется через SAFE_SHELL_PATH (trust model: YAML авторится оператором).
- `deploy/scripts/run_remote.py` -- НОВЫЙ: CLI-оркестратор
- `deploy/scripts/setup_server.sh` -- MODIFIED iter-1: добавлены (1) `--system-site-packages` (нужно чтобы venv видел системный torch — `[Remote eval servers are variable]` memory), (2) идемпотентность (skip `uv venv` если каталог существует). Изменение флага `--system-site-packages` пост-фактум требует ручного `rm -rf venv` на каждом хосте — documented in inline comment.
- `deploy/scripts/deploy_to_servers.sh`, `collect_results.sh` -- DEPRECATED reference, не трогать
- `deploy/evaluation/evaluate.py` -- READ-ONLY: target на сервере (`--models {name}`)
- `pyproject.toml` -- MODIFIED: добавлена зависимость `pyyaml>=6.0` (нужна `run_remote.py` для локального чтения YAML; оркестратор не уходит на remote, так что это локальная dependency).
- `uv.lock` -- AUTO: обновлён после `uv sync` для pyyaml.
- `tests/test_run_remote.py` -- НОВЫЙ: pytest для валидации, сборки команд, command-assembly seams и dispatch (без сети)

## Tasks & Acceptance

**Execution:**
- [x] `configs/remote_experiments.yaml` -- CREATE -- Sample-конфиг с placeholder-хостами `gpu-server-1/2`, полями `host/port/user/experiments`, комментарии на каждое поле
- [x] `deploy/scripts/run_remote.py` -- CREATE -- argparse: parent `--dry-run` и `--config PATH`; subcommands `deploy`, `run`, `collect`; функции `load_config`, `validate_config`, `build_ssh_cmd`, `build_rsync_cmd`, `execute_or_print`, `deploy_one_host`, `run_one_host`, `collect_one_host`, `main`. Все YAML-значения, попадающие в shell-строку, проходят charset-валидацию (`^[A-Za-z0-9._~/-]+$` для путей; whitelist для имён экспериментов). Поведение полностью соответствует Hardening Rules.
- [x] `tests/test_run_remote.py` -- CREATE -- Юнит-тесты: (a) валидация неизвестного имени → exit 2; (b) `build_ssh_cmd` вставляет `-p` и `user@host`; (c) `--dry-run` не вызывает `subprocess.run` (monkeypatch); (d) `validate_config` отвергает `deploy_dir` с shell-метасимволами (например `"; rm -rf ~"`) → exit 2; (e) дубликат host в YAML → exit 2; (f) абсолютный `results_local` → exit 2

**Acceptance Criteria:**
- Given валидный конфиг 2 серверов × 2-3 эксперимента, when `run_remote.py run --dry-run`, then напечатаны все ssh-команды и `subprocess.run` не вызван ни разу.
- Given `experiments: [foobar]`, when любая подкоманда, then validation падает с exit 2 и перечислением допустимых имён до первого SSH.
- Given сервер `port: 19356, user: root`, when `deploy --dry-run`, then команды содержат `-p 19356` и `root@<host>`.
- Given `collect --dry-run`, then rsync команды с `results_collected/{host}/results/` и `.../plots/`.
- Given `pytest tests/test_run_remote.py`, then все тесты passed без сети и GPU.

## Design Notes

**YAML-схема:**
```yaml
servers:
  - host: gpu-server-1
    port: 22       # default 22
    user: root     # default $USER
    experiments: [trafficcamnet, vehiclemakenet]
remote:
  deploy_dir: ~/cars-eval
  venv: venv
  results_local: results_collected
```

**Mockable seam для dry-run:**
```python
def execute_or_print(cmd: list[str], dry_run: bool) -> int:
    if dry_run:
        print(f"[DRY-RUN] {shlex.join(cmd)}")
        return 0
    return subprocess.run(cmd, check=False).returncode
```
Эта функция — единственная точка вызова shell; monkeypatch её в тестах.

**Шаблон запуска N экспериментов одной SSH-сессией (с hardening):**
```bash
cd {deploy_dir} && source {venv}/bin/activate && { 
  nohup python evaluation/evaluate.py --models {e1} < /dev/null > logs/{e1}.log 2>&1 &
  nohup python evaluation/evaluate.py --models {e2} < /dev/null > logs/{e2}.log 2>&1 &
  disown -a;
}
```
Здесь `deploy_dir`, `venv` и `e*` уже прошли валидацию через `validate_config`. `< /dev/null` нужен для корректного detach из SSH-канала. `logs/` готовится `setup_server.sh` на этапе `deploy`.

**Почему `{ ... }` обязательно** (а не `cmd1 ; cmd2 ; cmd3`): `cmd1 && cmd2 && nohup ... &` помещает всю AND-цепочку (включая `cd`) в подоболочку. После `;` родительский shell остаётся в домашней директории и `logs/...` не существует. Group command `{ ... }` запускается в родительском shell, сохраняя `cwd`.

## Verification

**Commands:**
- `python deploy/scripts/run_remote.py --help` -- subcommands `deploy/run/collect`, общие `--dry-run` и `--config`
- `python deploy/scripts/run_remote.py deploy --dry-run` -- печать команд, exit 0
- `python -c "import yaml; yaml.safe_load(open('configs/remote_experiments.yaml'))"` -- ok
- `pytest tests/test_run_remote.py -v` -- 3+ теста passed

## Spec Change Log

### Iteration 1 — 2026-05-17 — bad_spec loopback

**Triggering findings** (3 adversarial reviewers consolidated):
- [CRITICAL] Shell injection: YAML-значения `deploy_dir/venv/exp` интерполировались в remote shell-строку без квотирования.
- [HIGH] Нет pre-flight schema validation (`host` отсутствует → KeyError в worker; невалидный `port` → TypeError; пустой `remote:` → KeyError).
- [HIGH] `nohup ... & disown` без `</dev/null` риcкует не отделиться от SSH-канала; первый запуск падает на отсутствии `logs/`.
- [HIGH] Абсолютный `results_local` молча отбрасывает `ROOT` через Path-семантику.
- [MEDIUM] Дубликаты хостов гонятся за один `deploy_dir`.

**Amended sections** (non-frozen):
- Добавлен раздел **Hardening Rules** с **validate-then-interpolate** подходом (charset regex для `deploy_dir`/`venv`, whitelist для экспериментов). `shlex.quote` отвергнут — ломает тильду на сервере.
- Tasks обновлены: `validate_experiments` → `validate_config` (расширенная схема с charset, дубликатами, абс/отн `results_local`).
- Test suite расширен с 3 до 8 кейсов (добавлены: рассогласование `deploy_dir` валидацией, duplicate host, absolute results_local).
- Design Notes nohup-шаблон содержит `< /dev/null` и опирается на pre-flight валидацию (без `shlex.quote` обёрток).

**Known-bad state avoided:**
- Remote root code execution через сконструированный YAML-конфиг.
- Опаковые KeyError/TypeError из worker-thread'ов вместо понятных ошибок до SSH.
- Зомби-процессы на сервере из-за неотделённого SSH-канала.

**KEEP instructions** (что сохранить при re-derivation):
- argparse с parent-parser и 3 subcommands.
- `execute_or_print(cmd, dry_run)` как единая mockable seam.
- `ThreadPoolExecutor(max_workers=len(servers))`.
- `dispatch` с `[PLAN]` preamble и `[host] OK/FAIL` строками.
- `RSYNC_EXCLUDES` список (`.git/venv/__pycache__/...`).
- Exit code 2 для всех validation ошибок (stderr + `raise SystemExit(2)`).
- 5 уже работающих pytest кейсов (валидация имени, valid all, build_ssh_cmd, build_rsync_cmd, dry-run-no-subprocess) сохраняются; добавляются ещё 3.

## Review Findings

### Round 2 — 2026-05-17 — adversarial review of post-hardening version

Layers: Blind Hunter (diff-only) + Edge Case Hunter (diff+project) + Acceptance Auditor (diff+spec+context).

**Resolution summary:**
- 4 decision-needed → all accepted under trust model / spec contract (no code change).
- 13 patches surfaced → **12 applied**, 1 dismissed on re-examination (rsync rc=23 was a false positive — code already correct).
- 9 dismissed as noise upfront (false positives, scope-mismatch, cosmetic).
- Tests: 9 → **29 passing** (+14 new for boundary/command-assembly/dispatch coverage including parametrized port validation).

**Decision-needed (4) — resolved 2026-05-17, all accepted under trust model / spec contract:**
- [x] [Review][Decision] `run` silently reports `OK` on undeployed/broken-venv state — **RESOLVED: (a) accept fire-and-forget per spec.** "user обязан запустить deploy до первого run" уже задокументирован; реальные данные/модели — Goal 2 territory. [run_remote.py:245-268]
- [x] [Review][Decision] `SAFE_SHELL_PATH` permits `..` traversal — **RESOLVED: (b) accept trust model.** YAML авторится оператором, qudata-серверы — собственные под root, нет flow с чужим YAML. Self-attack нереалистична. [run_remote.py:89]
- [x] [Review][Decision] `identity_file` undeclared + unvalidated — **RESOLVED: (c) accept trust model, оставить как есть.** Документировать что поле существует, но валидация не применяется (то же обоснование что и SAFE_SHELL_PATH). [run_remote.py:184-186]
- [x] [Review][Decision] `USER` env unset → silent fallback `"root"` — **RESOLVED: (c) accept current behavior.** qudata = root-auth, локальный запуск с TTY всегда имеет `$USER`, CI/cron не предполагаются. [run_remote.py:161, 182]

**Patch (12 applied + 1 dismissed on re-examination) — fixed 2026-05-17, all tests passing (29/29):**
- [x] [Review][Patch] Validate `host` and `results_local` through SAFE_SHELL_PATH — added `_validate_shell_safe` calls в `validate_config` для обоих полей. [run_remote.py validate_config]
- [x] [Review][Patch] `Path(results_local).is_absolute()` raises TypeError on `results_local: null` before friendly error — REQUIRED_REMOTE_KEYS catches missing key, not explicit null. [run_remote.py:149-150]
- [x] [Review][Patch] `load_config` raises raw `FileNotFoundError`/`IsADirectoryError`/`yaml.YAMLError` instead of `_die` — wrap read+parse. [run_remote.py:113-114]
- [x] [Review][Patch] `validate_config(cfg=non-mapping)` raises `AttributeError` — top-level list/string in YAML survives `or {}` in load_config. Add `isinstance(cfg, dict)` check. [run_remote.py:137]
- [x] [Review][Patch] `_coerce_port` accepts 0, negative, &gt;65535, bool (True/False→1/0), and `" 22 "` — add bounds check + reject bool explicitly. [run_remote.py:117-126]
- [x] [Review][Patch] `dispatch` returns 0 silently on empty `servers` — operator can't distinguish "no work" from "all OK". Add explicit log line. [run_remote.py:316-317]
- [x] [Review][Patch] Duplicate experiments per server clobber log files — `[trafficcamnet, trafficcamnet]` writes both bg jobs to same `logs/trafficcamnet.log`. Dedupe in `validate_config` or fail-fast on dup. [run_remote.py:170-176, 260-264]
- [ ] ~~[Review][Patch] `deploy_one_host` runs `setup_server.sh` after rsync rc=23~~ — **DISMISSED on re-examination**: код УЖЕ корректен (`if rc1 != 0 and not dry_run: return host, rc1` skip setup). Edge Hunter ошибся в анализе ветки. Верифицировано новым тестом `test_deploy_one_host_skips_setup_on_rsync_failure` (rsync rc=23 → setup не вызывается). [run_remote.py deploy_one_host]
- [x] [Review][Patch] `collect_one_host` pre-creates `target_results/target_plots` then ignores rsync failure — leaves empty per-host dirs that look like "collected but empty". Skip mkdir, let rsync create on success. [run_remote.py:278-280]
- [x] [Review][Patch] `setup_server.sh` idempotent branch reuses existing venv without re-applying `--system-site-packages` — flag changes silently never take effect. Add comment + (optionally) detect mismatch. [deploy/scripts/setup_server.sh:14-19]
- [x] [Review][Patch] No tests for `run_one_host`/`deploy_one_host`/`collect_one_host`/`dispatch` command-assembly + concurrent failure path — all dangerous shell-string builders are uncovered. Add: assert `{ … ; disown -a; }` group is built, assert dispatch's `worst = max(...)` with mixed rcs, assert exception in `fut.result()` doesn't kill peers. [tests/test_run_remote.py]
- [x] [Review][Patch] `ThreadPoolExecutor(max_workers=len(servers))` fragile if guard removed — `max_workers=0` raises ValueError. Use `max(1, len(servers))` or document the guard's load-bearing role. [run_remote.py:316-320]
- [x] [Review][Patch] Code Map missing `pyproject.toml` (added pyyaml) and `setup_server.sh` (was marked READ-ONLY but modified for system-site-packages + idempotency) — amend spec Code Map to reflect actual touched files. [spec Code Map section]

### Round 2 — Dismissed as noise (9)

- B5 "deploy syncs only `deploy/` but invokes `evaluation/evaluate.py`" — false positive; path is `deploy/evaluation/evaluate.py`, so after `rsync deploy/ → deploy_dir/`, the remote path `deploy_dir/evaluation/evaluate.py` is correct (Edge Hunter independently verified).
- B6 "`disown -a` race with SSH session exit" — overstated; `nohup` redirects SIGHUP, real-world qudata runs confirm experiments start cleanly.
- B12+E17 "RSYNC_EXCLUDES silently excludes operator-named `data`/`models` dirs" — convention-driven; deploy is for code, not data.
- E10 "validator error message shows regex blob, not guidance" — cosmetic UX, not bug.
- E19 "main exits with worst rc, could collide with `_die`'s exit 2" — theoretical; ssh/rsync rc don't collide with 2 in practice.
- A4 "dispatch [PLAN] preamble format ambiguity" — spec wording satisfied.
- A5 "dry-run absolute target path vs. literal `results_collected/{host}/...` AC wording" — literal substring is present in absolute path.
- A6 "9 tests instead of promised 8" — additive (positive tilde companion); harmless.
- A7 "Empty experiments warning only in `run`, not deploy/collect" — out of context; deploy/collect don't consume experiments.
- A8 "--dry-run/--config availability before subcommand" — spec verification commands use `subcommand --flag` form, which works.
