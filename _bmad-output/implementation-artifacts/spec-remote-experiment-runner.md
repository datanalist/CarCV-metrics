---
title: 'Remote Experiment Runner'
type: 'feature'
created: '2026-05-16'
status: 'in-review'
baseline_commit: 'a5cf5837e6f080ac905bb10175e5fc63c8bffdb7'
context:
  - _bmad-output/project-context.md
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Текущие скрипты деплоя и запуска экспериментов (`deploy_to_servers.sh`, `collect_results.sh`) жёстко прошиты на конкретные серверы (`qudata2`, `qudata5`) и не поддерживают параллельный запуск нескольких экспериментов на произвольном наборе серверов.

**Approach:** Создать YAML-конфиг с описанием серверов и назначением экспериментов, и Python-скрипт `run_remote.py` с тремя отдельными командами: `deploy` (синхронизация файлов), `run` (параллельный запуск экспериментов), `collect` (сбор результатов).

## Boundaries & Constraints

**Always:**
- Серверы и эксперименты читаются исключительно из YAML-конфига; никаких хардкодированных имён хостов в коде.
- SSH-соединения — через subprocess (`ssh`, `rsync`); paramiko не используется (существующая инфраструктура проекта использует bash SSH).
- Параллелизм на уровне серверов — `ThreadPoolExecutor`; параллелизм экспериментов на одном сервере — фоновые SSH-процессы (`nohup ... &`).
- Логи каждого эксперимента на сервере пишутся в `~/cars-eval/logs/{experiment}.log`.
- `run` завершается сразу после запуска фоновых процессов (fire-and-forget); ожидание — на усмотрение пользователя.
- Результаты собираются в `results_collected/{host}/results/` и `results_collected/{host}/plots/`.
- Все имена экспериментов должны совпадать с choices в `evaluate.py`: `trafficcamnet`, `vehiclemakenet`, `vehicletypenet`, `lpdnet`, `lprnet`.

**Ask First:**
- Если у сервера нет доступных экспериментов (список пустой или `null`) — предупредить и пропустить.
- Если SSH-соединение с сервером недоступно — предупредить и продолжить для остальных серверов (не прерывать всю операцию).

**Never:**
- Не добавлять `torch`, `torchvision`, `nvidia-*` в `deploy/requirements.txt` и конфиги.
- Не использовать hardcoded пути вида `/home/vallo/...` или имена серверов в Python-коде.
- Не изменять `evaluate.py`, `metrics.py`, `visualize.py` в рамках этой задачи.
- Не реализовывать scheduling, мониторинг прогресса или повторный запуск при сбое (вне scope).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Deploy все серверы | `run_remote.py deploy` с N серверами в конфиге | rsync + setup на каждом сервере параллельно | Неуспешный сервер логируется, остальные продолжают; exit code ≠ 0 если хоть один упал |
| Запуск экспериментов | `run_remote.py run` | Эксперименты запускаются параллельно: каждый сервер параллельно, все эксперименты на одном сервере — фоном через `&` | SSH-ошибка логируется, остальные продолжают |
| Сбор результатов | `run_remote.py collect` | `results_collected/{host}/results/` и `.../plots/` заполнены | rsync-ошибка логируется, warn без остановки |
| Неизвестный эксперимент в конфиге | experiment name не входит в valid set | Ошибка с перечислением допустимых имён ДО SSH | Validation на старте, не SSH |
| Сервер недоступен | SSH timeout | Предупреждение с именем сервера, продолжение | Не падать целиком |
| `--config` не указан | аргумент отсутствует | Использовать `configs/remote_experiments.yaml` по умолчанию | — |

</frozen-after-approval>

## Code Map

- `configs/remote_experiments.yaml` — НОВЫЙ: декларативный конфиг серверов и назначений экспериментов
- `deploy/scripts/run_remote.py` — НОВЫЙ: оркестратор с subcommands deploy / run / collect
- `deploy/scripts/deploy_to_servers.sh` — СУЩЕСТВУЮЩИЙ (не трогать): устаревший hardcoded вариант, остаётся для обратной совместимости
- `deploy/scripts/collect_results.sh` — СУЩЕСТВУЮЩИЙ (не трогать): аналогично
- `deploy/evaluation/evaluate.py` — СУЩЕСТВУЮЩИЙ (read-only): точка входа на сервере, принимает `--models`
- `deploy/scripts/setup_server.sh` — СУЩЕСТВУЮЩИЙ: вызывается из `deploy` команды

## Tasks & Acceptance

**Execution:**
- [x] `configs/remote_experiments.yaml` -- CREATE -- Конфиг с секциями `servers` (host + experiments list) и `remote` (deploy_dir, venv, results_local)
- [x] `deploy/scripts/run_remote.py` -- CREATE -- CLI с subcommands: `deploy` (rsync + setup через SSH), `run` (параллельный запуск экспериментов фоном), `collect` (rsync результатов)

**Acceptance Criteria:**
- Given конфиг с 2 серверами по 2 эксперимента, when `run_remote.py run`, then на каждом сервере в фоне запущены 2 процесса (`evaluate.py --models X &`) через SSH, итого 4 SSH-вызова параллельно.
- Given эксперимент с некорректным именем в конфиге, when любая команда, then валидация падает до первого SSH-соединения с сообщением о допустимых именах.
- Given один сервер недоступен, when `run_remote.py deploy`, then остальные серверы деплоятся успешно, exit code = 1.
- Given `run_remote.py collect`, then файлы появляются в `results_collected/{host}/results/` и `results_collected/{host}/plots/`.
- Given `--config` не передан, when любая команда, then используется `configs/remote_experiments.yaml`.

## Design Notes

**Структура `configs/remote_experiments.yaml`:**
```yaml
servers:
  - host: gpu-server-1
    experiments: [trafficcamnet, vehiclemakenet]
  - host: gpu-server-2
    experiments: [vehicletypenet, lpdnet, lprnet]

remote:
  deploy_dir: ~/cars-eval
  venv: venv
  results_local: results_collected
```

**Команда запуска эксперимента на сервере (SSH):**
```bash
ssh {host} "cd {deploy_dir} && source {venv}/bin/activate && \
  nohup python evaluation/evaluate.py --models {exp} \
  > logs/{exp}.log 2>&1 &"
```

**Параллелизм:**
- Уровень серверов: `ThreadPoolExecutor(max_workers=len(servers))`
- Уровень экспериментов на одном сервере: одна SSH-сессия, несколько `nohup ... &` в одном скрипте

## Verification

**Commands:**
- `python deploy/scripts/run_remote.py --help` -- expected: subcommands deploy, run, collect отображены
- `python deploy/scripts/run_remote.py run --config /dev/stdin <<< 'servers: [{host: badhost, experiments: [trafficcamnet]}]'` -- expected: SSH timeout warning, exit 1
- `python -c "import yaml; yaml.safe_load(open('configs/remote_experiments.yaml'))"` -- expected: no errors

**Manual checks:**
- После `deploy`: директория `~/cars-eval/` существует на целевом сервере, `venv/` создан
- После `run`: процессы `evaluate.py` видны в `ps aux` на сервере
- После `collect`: `results_collected/{host}/results/*.json` существуют локально

## Spec Change Log
