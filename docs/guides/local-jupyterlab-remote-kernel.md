# Подключение локального JupyterLab к удалённому kernel

Этот гайд описывает рабочую схему, где:

- ноутбуки лежат локально в проекте;
- `JupyterLab` тоже запущен локально;
- код из ячеек исполняется на удалённой машине;
- переключение делается через custom kernelspec, а не через полный remote Jupyter workflow.

Все примеры ниже используют только placeholders. Не вставляй в файл реальные токены, URL, логины, ключи и приватные пути.

## Когда это нужно

Используй этот подход, если:

- хочешь редактировать ноутбуки локально в своём репозитории;
- GPU, CUDA, модели или окружение живут на удалённой машине;
- не хочешь переносить весь проект в remote JupyterLab;
- хочешь, чтобы notebook metadata ссылалась на kernelspec вроде `remote-gpu`.

## Как это работает

Схема такая:

1. Локальный `JupyterLab` создаёт `connection_file`.
2. Kernelspec `remote-gpu` запускает локальный shell-скрипт.
3. Скрипт:
   - читает порты из `connection_file`;
   - **(опционально)** синкает локальный проект на remote через `rsync` и экспортирует путь через env var — так ноутбук видит актуальный код без ручного staging;
   - копирует этот файл на удалённую машину;
   - поднимает SSH port forwarding для каналов `shell`, `iopub`, `stdin`, `control`, `hb`;
   - экспортирует на удалённой машине writable runtime-пути;
   - запускает `ipykernel_launcher` на remote.
4. Для frontend-а Jupyter kernel выглядит как обычный, но фактически он исполняется удалённо.

## Предварительные требования

### Локальная машина

- установлен Python;
- установлен `JupyterLab`;
- есть SSH-клиент;
- есть доступ к `~/.local/share/jupyter/kernels/`.

Если у тебя Python-окружение ведётся через `uv`, базовый набор обычно такой:

```bash
uv add jupyterlab ipykernel
```

### Удалённая машина

- работает SSH-доступ;
- доступен Python с `ipykernel`;
- есть writable каталог для runtime-данных;
- есть место, куда можно положить временный `connection_file`.

Проверка remote Python и `ipykernel`:

```bash
ssh <REMOTE_HOST_ALIAS> 'python -c "import sys, ipykernel; print(sys.executable); print(ipykernel.__version__)"'
```

## Рекомендуемая структура

### Локально

```text
~/.local/share/jupyter/kernels/remote-gpu/
├── kernel.json
└── launch-remote-kernel.sh
```

### Опционально

- `~/.ssh/config` с alias для удалённой машины;
- отдельный helper-скрипт для диагностики;
- отдельный remote Python/venv, если нельзя использовать системный.

## Шаг 1. Настрой SSH alias

Лучше использовать alias из `~/.ssh/config`, а не зашивать `host`, `port`, `user`, `IdentityFile` прямо в launcher.

Пример:

```sshconfig
Host <REMOTE_HOST_ALIAS>
    HostName <REMOTE_SSH_HOST>
    Port <REMOTE_SSH_PORT>
    User <REMOTE_USER>
    IdentityFile <SSH_KEY_PATH>
```

Проверка:

```bash
ssh -vv <REMOTE_HOST_ALIAS> 'echo ok'
```

Если alias не работает стабильно, дальше не иди. Сначала добейся обычного SSH без вопросов про host key, пароль и интерактивные подтверждения.

## Шаг 2. Подготовь writable runtime-каталоги на remote

Это не опция. На многих remote-конфигурациях kernel падает не из-за Python, а из-за того, что:

- `/tmp` недоступен;
- `HOME` read-only;
- `IPython` не может открыть history/cache;
- `tempfile.gettempdir()` не находит usable temp dir.

Надёжный вариант: вынести runtime в `/dev/shm/...`.

Пример:

```bash
ssh <REMOTE_HOST_ALIAS> '
  mkdir -p \
    /dev/shm/<REMOTE_USER>/jupyter-runtime \
    /dev/shm/<REMOTE_USER>/home \
    /dev/shm/<REMOTE_USER>/.ipython/profile_default
'
```

Если `/dev/shm` недоступен, используй любой другой writable каталог, но проверь его заранее.

## Шаг 3. Создай custom kernelspec

Создай каталог:

```bash
mkdir -p ~/.local/share/jupyter/kernels/remote-gpu
```

### `kernel.json`

```json
{
  "argv": [
    "/bin/bash",
    "/home/<LOCAL_USER>/.local/share/jupyter/kernels/remote-gpu/launch-remote-kernel.sh",
    "{connection_file}"
  ],
  "display_name": "Python (remote-gpu)",
  "language": "python",
  "metadata": {
    "debugger": true
  }
}
```

Что важно:

- `argv[2]` получает путь к локальному `connection_file`;
- имя kernelspec должно совпадать с тем, что ожидает notebook metadata;
- `display_name` может быть любым, `name` фактически определяется именем каталога kernelspec.

## Шаг 4. Напиши launcher-скрипт

Ниже минимальный рабочий шаблон. Он:

- читает порты из `connection_file`;
- создаёт remote runtime dirs;
- копирует `connection_file` на remote;
- поднимает SSH-туннели для всех 5 каналов;
- задаёт корректные env vars;
- запускает удалённый `ipykernel`.

```bash
#!/usr/bin/env bash
set -euo pipefail

LOCAL_CONNECTION_FILE="$1"
REMOTE_HOST="<REMOTE_HOST_ALIAS>"
REMOTE_RUNTIME_DIR="/dev/shm/<REMOTE_USER>/jupyter-runtime"
REMOTE_HOME="/dev/shm/<REMOTE_USER>/home"
REMOTE_IPYTHONDIR="/dev/shm/<REMOTE_USER>/.ipython"
REMOTE_CONNECTION_FILE="${REMOTE_RUNTIME_DIR}/$(basename "${LOCAL_CONNECTION_FILE}")"
REMOTE_PYTHON="<REMOTE_PYTHON_EXECUTABLE>"

read_json_field() {
  python - "$1" "$2" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data[sys.argv[2]])
PY
}

SHELL_PORT="$(read_json_field "${LOCAL_CONNECTION_FILE}" shell_port)"
IOPUB_PORT="$(read_json_field "${LOCAL_CONNECTION_FILE}" iopub_port)"
STDIN_PORT="$(read_json_field "${LOCAL_CONNECTION_FILE}" stdin_port)"
CONTROL_PORT="$(read_json_field "${LOCAL_CONNECTION_FILE}" control_port)"
HB_PORT="$(read_json_field "${LOCAL_CONNECTION_FILE}" hb_port)"

ssh "${REMOTE_HOST}" "
  mkdir -p \
    '${REMOTE_RUNTIME_DIR}' \
    '${REMOTE_HOME}' \
    '${REMOTE_IPYTHONDIR}/profile_default'
"

scp "${LOCAL_CONNECTION_FILE}" "${REMOTE_HOST}:${REMOTE_CONNECTION_FILE}"

TUNNEL_PID=""
cleanup() {
  if [[ -n "${TUNNEL_PID}" ]]; then
    kill "${TUNNEL_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ssh -N \
  -L "${SHELL_PORT}:127.0.0.1:${SHELL_PORT}" \
  -L "${IOPUB_PORT}:127.0.0.1:${IOPUB_PORT}" \
  -L "${STDIN_PORT}:127.0.0.1:${STDIN_PORT}" \
  -L "${CONTROL_PORT}:127.0.0.1:${CONTROL_PORT}" \
  -L "${HB_PORT}:127.0.0.1:${HB_PORT}" \
  "${REMOTE_HOST}" &
TUNNEL_PID="$!"

exec ssh "${REMOTE_HOST}" "
  export TMPDIR='${REMOTE_RUNTIME_DIR}'
  export TMP='${REMOTE_RUNTIME_DIR}'
  export TEMP='${REMOTE_RUNTIME_DIR}'
  export HOME='${REMOTE_HOME}'
  export IPYTHONDIR='${REMOTE_IPYTHONDIR}'
  '${REMOTE_PYTHON}' -m ipykernel_launcher -f '${REMOTE_CONNECTION_FILE}'
"
```

Сделай файл исполняемым:

```bash
chmod +x ~/.local/share/jupyter/kernels/remote-gpu/launch-remote-kernel.sh
```

## Шаг 4.5. Автосинхронизация проекта (опционально)

По умолчанию remote kernel не видит твой локальный репозиторий — он живёт на удалённой машине и работает с её filesystem. Если notebook читает `configs/*.yaml`, импортирует `scripts/common.py` или хочет `ROOT` указать на код проекта, нужно либо держать полную копию репо на remote, либо синкать перед каждым запуском kernel.

Самое надёжное — встроить `rsync` прямо в launcher: перед стартом kernel он копирует свежий срез проекта на remote и экспортирует путь через env var. Тогда:

- правишь файлы локально → рестарт kernel → изменения уже на remote;
- notebook использует детерминированный путь через env var вместо относительного `Path("..")`, который на remote резолвится непредсказуемо;
- большие артефакты (`data/`, `models/`, `.venv/`, `.git/`) исключаются и остаются на remote независимо от sync.

### Расширенный launcher с rsync

```bash
#!/usr/bin/env bash
set -euo pipefail

LOCAL_CONNECTION_FILE="$1"
REMOTE_HOST="<REMOTE_HOST_ALIAS>"
REMOTE_RUNTIME_DIR="/dev/shm/<REMOTE_USER>/jupyter-runtime"
REMOTE_HOME="/dev/shm/<REMOTE_USER>/home"
REMOTE_IPYTHONDIR="/dev/shm/<REMOTE_USER>/.ipython"
REMOTE_CONNECTION_FILE="${REMOTE_RUNTIME_DIR}/$(basename "${LOCAL_CONNECTION_FILE}")"
REMOTE_PYTHON="<REMOTE_PYTHON_EXECUTABLE>"

# Auto-sync: локальный проект → remote
LOCAL_PROJECT_ROOT="<LOCAL_PROJECT_ROOT>"         # например: /home/<LOCAL_USER>/MyProject
REMOTE_PROJECT_ROOT="<REMOTE_PROJECT_ROOT>"       # например: /dev/shm/<REMOTE_USER>/myproject

# Отдельные крупные артефакты (датасеты/метаданные), которые лежат не в репе,
# но нужны kernel-у. Оставь пустым, если не нужно.
LOCAL_EXTRA_FILE="<LOCAL_ABS_PATH_TO_BIG_FILE_OR_EMPTY>"
REMOTE_EXTRA_FILE="<REMOTE_ABS_PATH_TO_BIG_FILE_OR_EMPTY>"

# ... (чтение портов из connection_file, как в базовом launcher) ...

ssh "${REMOTE_HOST}" "
  mkdir -p \
    '${REMOTE_RUNTIME_DIR}' \
    '${REMOTE_HOME}' \
    '${REMOTE_IPYTHONDIR}/profile_default' \
    '${REMOTE_PROJECT_ROOT}'
"

rsync -az --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='node_modules/' \
  --exclude='data/' \
  --exclude='models/' \
  --exclude='results/' \
  --exclude='outputs/' \
  "${LOCAL_PROJECT_ROOT}/" "${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/" >&2

if [[ -n "${LOCAL_EXTRA_FILE}" && -f "${LOCAL_EXTRA_FILE}" ]]; then
  rsync -az "${LOCAL_EXTRA_FILE}" "${REMOTE_HOST}:${REMOTE_EXTRA_FILE}" >&2
fi

scp "${LOCAL_CONNECTION_FILE}" "${REMOTE_HOST}:${REMOTE_CONNECTION_FILE}"

# ... (SSH-туннели, как в базовом launcher) ...

exec ssh "${REMOTE_HOST}" "
  export TMPDIR='${REMOTE_RUNTIME_DIR}'
  export TMP='${REMOTE_RUNTIME_DIR}'
  export TEMP='${REMOTE_RUNTIME_DIR}'
  export HOME='${REMOTE_HOME}'
  export IPYTHONDIR='${REMOTE_IPYTHONDIR}'
  export PROJECT_ROOT='${REMOTE_PROJECT_ROOT}'
  export REMOTE_KERNEL=1
  cd '${REMOTE_PROJECT_ROOT}'
  '${REMOTE_PYTHON}' -m ipykernel_launcher -f '${REMOTE_CONNECTION_FILE}'
"
```

Ключевые решения:

- `--delete` гарантирует, что удалённая копия зеркалит локальную. Если удалил файл локально — он пропадёт и на remote. Временные файлы на remote клади **вне** `REMOTE_PROJECT_ROOT`.
- Исключения датасетов/моделей (`data/`, `models/`) — чтобы не гонять гигабайты каждый рестарт. Эти артефакты стейджишь отдельно (например, `scp` из orchestrator-скрипта) либо держишь постоянно на remote.
- `PROJECT_ROOT` и `REMOTE_KERNEL` — env vars, которые notebook читает, чтобы понять где он запущен и откуда грузить файлы.
- rsync работает по incremental-протоколу, поэтому типичный рестарт занимает секунды (первый запуск дольше).

### Использование в notebook

В первой ячейке notebook-а переопредели `ROOT` и пути к данным через env:

```python
import os
from pathlib import Path

IS_REMOTE = bool(os.environ.get("REMOTE_KERNEL"))
ROOT = Path(os.environ.get("PROJECT_ROOT") or Path("..").resolve())

if IS_REMOTE:
    DATA_DIR = Path("<REMOTE_DATA_DIR>")
else:
    DATA_DIR = Path("<LOCAL_DATA_DIR>")
```

Один и тот же notebook теперь работает:

- с локальным Python kernel → `IS_REMOTE=False`, пути локальные;
- с `remote-gpu` kernel → `IS_REMOTE=True`, пути remote, код автоматически свежий.

### Что не надо синкать через launcher

- ONNX/TensorRT модели и большие датасеты — заливай один раз вручную через `scp`/`rsync` либо оркестрируй отдельным скриптом. Их пересылка на каждый рестарт kernel убьёт iteration time.
- Временные outputs kernel-а — держи их в `/dev/shm/<REMOTE_USER>/outputs/` или любом каталоге вне `REMOTE_PROJECT_ROOT`, иначе `--delete` затрёт их при следующем рестарте.
- Секреты (`.env`, keys) — явно исключи через `--exclude='.env'` и подобные, особенно если remote — shared-машина.

### Быстрая проверка sync

После рестарта kernel в первой ячейке:

```python
import os, socket
from pathlib import Path
print("hostname:", socket.gethostname())
print("PROJECT_ROOT:", os.environ.get("PROJECT_ROOT"))
print("contents:", sorted(p.name for p in Path(os.environ["PROJECT_ROOT"]).iterdir())[:10])
```

Ожидаешь увидеть remote hostname и свежий список файлов проекта.

## Шаг 5. Проверь, что kernelspec виден локально

```bash
jupyter kernelspec list
```

Ожидаешь в списке что-то вроде:

```text
remote-gpu    /home/<LOCAL_USER>/.local/share/jupyter/kernels/remote-gpu
```

Если `remote-gpu` не виден, Jupyter вообще не сможет привязать notebook к удалённому kernel.

## Шаг 6. Быстрый smoke-test без ноутбука

Перед тем как открывать `.ipynb`, проверь сам kernelspec:

```bash
jupyter console --kernel remote-gpu
```

Если консоль стартует, выполни:

```python
import os, sys, socket, tempfile

print("python:", sys.executable)
print("host:", socket.gethostname())
print("HOME:", os.environ.get("HOME"))
print("TMPDIR:", os.environ.get("TMPDIR"))
print("tempdir:", tempfile.gettempdir())
```

Тут важно увидеть:

- hostname удалённой машины, а не локальной;
- remote Python, а не локальный интерпретатор;
- writable `HOME` и `TMPDIR`.

Если нужна GPU-проверка:

```python
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

## Шаг 7. Привяжи ноутбук к kernelspec

У ноутбука в metadata должно быть что-то вроде:

```json
{
  "kernelspec": {
    "display_name": "Python (remote-gpu)",
    "language": "python",
    "name": "remote-gpu"
  }
}
```

Если `metadata.kernelspec.name` не совпадает с локальным kernelspec, Jupyter покажет ошибку missing kernel или тихо переключится не туда.

## Шаг 8. Запусти локальный JupyterLab

Пример:

```bash
uv run jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Дальше:

1. открой нужный ноутбук;
2. выбери kernel `remote-gpu`, если он не подхватился автоматически;
3. запусти test cell;
4. проверь hostname и Python.

## Проверка, что выполнение реально удалённое

Минимальная test cell:

```python
import os, socket, sys, platform

print("HOSTNAME:", socket.gethostname())
print("PYTHON:", sys.executable)
print("CWD:", os.getcwd())
print("PLATFORM:", platform.platform())
```

Если видишь:

- remote hostname;
- remote filesystem layout;
- remote Python path;

значит схема работает корректно.

## Практические замечания

- Notebook остаётся локальным, kernel живёт удалённо.
- При обрыве SSH-сессии kernel обычно теряется.
- После sleep/смены сети чаще всего ломаются `heartbeat` и `control`.
- На нестабильной сети лучше часто сохранять notebook.
- Если remote Python не тот, укажи абсолютный путь в `REMOTE_PYTHON`, а не просто `python`.
- Auto-sync через `rsync --delete` односторонний: правки на remote не возвращаются локально. Всегда редактируй исходники локально.
- Первый rsync заливает весь проект; дальше incremental — обычно доли секунды.
- Для тяжёлых артефактов (датасеты, ONNX-модели, веса) используй отдельный orchestration-скрипт, а не launcher.
- `ControlMaster auto` + `ControlPersist 10m` в `~/.ssh/config` сильно ускоряют последовательные `ssh`/`scp`/`rsync` из launcher-а.

## Что не надо хранить в документации и репозитории

Никогда не коммить:

- реальные токены Jupyter;
- реальные URL remote Jupyter;
- реальные host/user/private key path;
- `scp`/`ssh` команды с настоящими учётными данными;
- экспорт секретов в открытом виде.

Используй только placeholders:

- `<REMOTE_HOST_ALIAS>`
- `<REMOTE_SSH_HOST>`
- `<REMOTE_SSH_PORT>`
- `<REMOTE_USER>`
- `<LOCAL_USER>`
- `<SSH_KEY_PATH>`
- `<REMOTE_PYTHON_EXECUTABLE>`
- `<REMOTE_JUPYTER_URL>`
- `<JUPYTER_TOKEN>`
- `<LOCAL_PROJECT_ROOT>`
- `<REMOTE_PROJECT_ROOT>`
- `<LOCAL_DATA_DIR>` / `<REMOTE_DATA_DIR>`
- `<LOCAL_ABS_PATH_TO_BIG_FILE_OR_EMPTY>` / `<REMOTE_ABS_PATH_TO_BIG_FILE_OR_EMPTY>`

## Troubleshooting

### `remote-gpu` не появляется в списке kernel-ов

Проверь:

- существует ли `~/.local/share/jupyter/kernels/remote-gpu/`;
- валиден ли `kernel.json`;
- показывает ли `jupyter kernelspec list` этот kernelspec.

### Kernel стартует и сразу падает

Обычно причина одна из трёх:

- неверный путь к launcher-скрипту;
- нет `chmod +x`;
- скрипт падает на SSH/scp/парсинге JSON.

Что проверить:

```bash
bash -x ~/.local/share/jupyter/kernels/remote-gpu/launch-remote-kernel.sh <PATH_TO_CONNECTION_FILE>
```

### Есть SSH, но kernel не поднимается

Проверь remote Python:

```bash
ssh <REMOTE_HOST_ALIAS> 'python -c "import ipykernel, sys; print(sys.executable)"'
```

Если запускается не то окружение, явно укажи нужный путь в:

```bash
REMOTE_PYTHON="<REMOTE_PYTHON_EXECUTABLE>"
```

### Ошибки вида `No usable temporary directory found`

Это значит, что remote kernel не может использовать стандартные temp dirs.

Лечится экспортом:

```bash
export TMPDIR='...'
export TMP='...'
export TEMP='...'
export HOME='...'
export IPYTHONDIR='...'
```

Если не сделать это сразу, kernel может падать ещё до первой ячейки.

### Ошибки `history.sqlite`, `readonly database`, `Permission denied`

Причина почти всегда в том, что `HOME` или `IPYTHONDIR` недоступны для записи.

Решение: перенести их в writable runtime dir, например в `/dev/shm/...`.

### Ноутбук висит на `Connecting`

Проверь:

- все ли 5 каналов проброшены;
- жив ли SSH-процесс с `-N -L ...`;
- не умер ли remote `ipykernel_launcher`.

Каналы должны быть все:

- `shell`
- `iopub`
- `stdin`
- `control`
- `hb`

### Ошибка `Address already in use`

Это значит, что один из портов уже занят локально или остался stale SSH tunnel.

Проверка:

```bash
ss -ltnp | grep -E ':(<PORT1>|<PORT2>|<PORT3>|<PORT4>|<PORT5>)\b'
```

Обычно помогает:

- убить старый `ssh`;
- перезапустить kernel, чтобы Jupyter сгенерировал новый `connection_file`.

### Metadata ноутбука ссылается не на тот kernelspec

Симптом: notebook открывается, но kernel missing или подхватывается другой interpreter.

Проверь:

- `metadata.kernelspec.name`;
- имя локального каталога kernelspec.

Они должны совпадать.

### После смены сети kernel “умер”

Причина: отвалился SSH forwarding.

Проверяй:

```bash
ssh <REMOTE_HOST_ALIAS> 'echo ok'
```

Если SSH нестабилен, перезапуск kernel почти всегда быстрее, чем пытаться оживлять полумёртвую сессию.

### Rsync падает или kernel стартует очень долго

Симптомы: каждый рестарт kernel занимает десятки секунд или минут, в логах launcher видны строки `rsync: ...`.

Проверь:

- первый sync всегда медленный (полная заливка); последующие — incremental и быстрые;
- проверь, что `data/`, `models/`, `.venv/` реально исключены — одна забытая папка может весить гигабайты;
- запусти вручную из терминала и посмотри реальную скорость:

```bash
rsync -az --stats --exclude='.git/' \
  <LOCAL_PROJECT_ROOT>/ <REMOTE_HOST_ALIAS>:<REMOTE_PROJECT_ROOT>/
```

- если тормозит SSH handshake — включи в `~/.ssh/config` `ControlMaster auto` + `ControlPersist 10m`, чтобы соединение переиспользовалось между вызовами.

### `ModuleNotFoundError` при `from scripts.xxx import ...` на remote kernel

Причина: `ROOT` в notebook указывает не на синхронизированный проект, а в случайный remote-каталог (например, `HOME`).

Лечится тем, что notebook явно берёт путь из env:

```python
ROOT = Path(os.environ.get("PROJECT_ROOT") or Path("..").resolve())
sys.path.insert(0, str(ROOT / "scripts"))
```

Проверь, что launcher действительно экспортирует `PROJECT_ROOT` в `ssh ... "export PROJECT_ROOT=..."` блоке.

### Файл удалился на remote после рестарта kernel

Причина: `rsync --delete` стирает всё, чего нет локально.

Никогда не создавай в `REMOTE_PROJECT_ROOT` временные файлы, которые должны пережить рестарт. Пиши их в:

- `/dev/shm/<REMOTE_USER>/outputs/` — эфемерный tmpfs;
- `${HOME}/persistent/...` — вне sync-корня;
- либо убери `--delete`, если тебя устраивает, что remote-копия может расходиться с локальной.

## Отладка по слоям

Если всё сломано, не дебажь всё сразу. Иди по слоям:

1. обычный SSH на remote;
2. наличие remote Python + `ipykernel`;
3. writable remote runtime dirs;
4. корректный `kernel.json`;
5. launcher-скрипт без rsync (базовая версия);
6. rsync вручную: `rsync -az <LOCAL_PROJECT_ROOT>/ <REMOTE_HOST_ALIAS>:<REMOTE_PROJECT_ROOT>/`;
7. launcher-скрипт с rsync и export env;
8. `jupyter console --kernel remote-gpu` + проверка `PROJECT_ROOT`/`REMOTE_KERNEL` через `os.environ`;
9. уже потом запуск notebook.

Это самый быстрый способ понять, где именно рвётся цепочка.
