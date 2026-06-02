# Validation Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Закрыть три оставшиеся модели валидационной кампании (TrafficCamNet на BDD100K, VehicleMakeNet на VMMRdb, FaceDetect на WIDER FACE), получив для каждой зафиксированные метрики и вердикт pass/fail по заданным порогам.

**Architecture:** Переиспользуем зрелый пайплайн `deploy/` (evaluate.py + run_remote.py + metrics.py). Новый код минимален: чистые функции конвертации датасетов выносятся в stdlib-only модуль `deploy/evaluation/dataset_prep.py` (юнит-тестируется локально без ML-зависимостей через `uv run --with pytest`); скрипты загрузки тянут данные (wget с Nextcloud-шары для BDD100K/VMMRdb, HuggingFace для WIDER FACE) и через эти функции готовят `labels.json`/манифесты в формате, который уже понимают существующие эвалуаторы; новый эвалуатор `eval_facedetect` добавляется в evaluate.py. Прогоны идут последовательно на одном удалённом GPU через `run_remote.py deploy/run/collect`.

**Tech Stack:** Python 3.12, onnxruntime-gpu, OpenCV, NumPy, HuggingFace `datasets`, NVIDIA TAO ONNX-модели (NGC), pytest, uv, SSH/rsync-оркестрация.

---

## Prerequisites (состояние на старте)

- **База git — ГОТОВА.** `main` приведён к зрелой базе (`c22c879`: инфраструктура exp + дизайн-доки + удаление BMAD, запушен в origin). Рабочая ветка кампании `exp/validation-campaign-2026-06` отведена от `main` и сейчас является текущей. Вся работа этого плана ведётся в ней последовательно.
- **Удалённый сервер ротируется.** Контейнер пересоздаётся между запусками: host/port и host key меняются, `~/datasets`/`~/models` пустые, onnxruntime/uv отсутствуют (torch+CUDA есть). Перед каждым прогоном:
  1. Обновить `configs/remote_experiments.yaml` (host/port/identity актуального контейнера).
  2. Снять stale host key: `ssh-keygen -R '[<host>]:<port>'`.
  3. Первый ssh — с `StrictHostKeyChecking=accept-new` (или один раз вручную принять ключ).
  Bootstrap (venv + onnxruntime + requirements, скачивание данных/весов) — идемпотентные скрипты, запускаемые в начале каждого прогона; на сохранённое состояние между прогонами не рассчитываем.
- **Локальные тесты.** Запускаются командой `uv run --with pytest python -m pytest <path> -v` из корня репозитория (локальной venv нет, ML-стек локально не нужен — тесты импортируют только stdlib-модуль `dataset_prep`).
- **Honest-measure.** FAIL по порогам — валидный окончательный результат. Никакого тюнинга порогов/дообучения ради PASS.

---

## File Structure

**Создаётся:**
- `deploy/evaluation/dataset_prep.py` — чистые функции конвертации аннотаций (stdlib only): `bdd100k_to_labels`, `vmmrdb_make`, `iter_vmmrdb_samples`, `widerface_faces_to_detections`. Импортируется тестами и download-скриптами.
- `deploy/scripts/download_datasets_campaign.sh` — загрузка+конвертация BDD100K, VMMRdb, WIDER FACE на удалённом сервере.
- `tests/test_dataset_prep.py` — юнит-тесты конвертеров.

**Модифицируется:**
- `deploy/evaluation/evaluate.py` — добавить `eval_facedetect`, запись `facedetect` в `EVAL_CONFIGS`, обобщить `eval_vehiclemakenet` на `cfg["meta_file"]`, переключить `vehiclemakenet` на `data/vmmrdb`, переключить `trafficcamnet` conf_thr на in-domain 0.4.
- `deploy/scripts/run_remote.py` — добавить `facedetect` в `VALID_EXPERIMENTS`.
- `deploy/scripts/download_models.sh` — добавить загрузку FaceNet ONNX с NGC.
- `deploy/scripts/setup_server.sh` — добавить каталоги `data/{vmmrdb,widerface}`, `models/facenet`, `results/facedetect`.
- `configs/remote_experiments.yaml` — актуальный сервер + список экспериментов на каждый прогон.

**Изоляция результатов:** у каждой модели свой `results/<model>/metrics.json` (+ графики). Сбор — в `results_collected/<host>/results/<model>/`.

---

## Архитектурные решения (отступления от дизайна, осознанные)

1. **VMMRdb обрабатывается в download-скрипте, а не «веткой в эвалуаторе».** Дизайн предлагал добавить ветку чтения каталогов-по-классам прямо в `eval_vehiclemakenet`. Вместо этого download-скрипт строит манифест `[{file_name, brand}]` (тот же формат, что mad-cars `sample_5k.json`) из имён каталогов VMMRdb, а эвалуатор читает его через новый cfg-ключ `meta_file`. Это DRY (переиспользуем уже протестированный classification-путь), а логика извлечения марки и сэмплирования покрывается юнит-тестами с `tmp_path`. Единственное изменение эвалуатора — `cfg.get("meta_file", "sample_5k.json")`.
2. **Чистые конвертеры — отдельный stdlib-only модуль `dataset_prep.py`.** Причина: `evaluate.py` импортирует cv2/onnxruntime на верхнем уровне, которых нет в локальном окружении; вынос конвертеров позволяет гонять их тесты локально через `uv run --with pytest` без ML-стека.

---

# Секция A — TrafficCamNet на BDD100K

Эвалуатор `eval_trafficcamnet` уже существует и читает `data/bdd100k/labels.json` (список `{image_id, file_name, detections:[{category, box2d:{x1,y1,x2,y2}}]}`) + `data/bdd100k/images/`. Class-mapping BDD→4 класса уже зашит в `TRAFFICCAMNET_GT_VOCAB`. Нужно: (1) конвертер нативных val-аннотаций BDD100K в этот формат, (2) download-скрипт, (3) прогон.

## Task A1: Конвертер BDD100K (TDD, локально)

**Files:**
- Create: `deploy/evaluation/dataset_prep.py`
- Test: `tests/test_dataset_prep.py`

- [ ] **Step 1: Написать падающий тест**

`tests/test_dataset_prep.py`:
```python
"""Юнит-тесты чистых конвертеров датасетов (stdlib only, без ML-зависимостей)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy" / "evaluation"))

from dataset_prep import bdd100k_to_labels  # noqa: E402


def test_bdd100k_to_labels_basic():
    native = [
        {
            "name": "0000f77c-6257be58.jpg",
            "attributes": {"weather": "clear", "scene": "city street",
                           "timeofday": "daytime"},
            "labels": [
                {"category": "car", "box2d": {"x1": 45.2, "y1": 254.5,
                                              "x2": 357.8, "y2": 487.9}},
                {"category": "traffic sign", "box2d": {"x1": 1.0, "y1": 2.0,
                                                       "x2": 3.0, "y2": 4.0}},
                # лейбл полосы без box2d (только poly2d) — должен быть отброшен
                {"category": "lane", "poly2d": [{"vertices": [[0, 0]]}]},
            ],
        },
        {"name": "b1c66a42-6f7d68ca.jpg", "attributes": {}, "labels": []},
    ]
    out = bdd100k_to_labels(native)
    assert len(out) == 2
    first = out[0]
    assert first["image_id"] == "0000f77c-6257be58"
    assert first["file_name"] == "0000f77c-6257be58.jpg"
    assert first["weather"] == "clear"
    # только два box2d-лейбла, категории в нижнем регистре
    assert len(first["detections"]) == 2
    assert first["detections"][0]["category"] == "car"
    assert first["detections"][1]["category"] == "traffic sign"
    assert first["detections"][0]["box2d"] == {"x1": 45.2, "y1": 254.5,
                                               "x2": 357.8, "y2": 487.9}
    # второе изображение — без детекций, но присутствует
    assert out[1]["detections"] == []
```

- [ ] **Step 2: Запустить тест — убедиться, что падает**

Run: `uv run --with pytest python -m pytest tests/test_dataset_prep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dataset_prep'`.

- [ ] **Step 3: Минимальная реализация**

`deploy/evaluation/dataset_prep.py`:
```python
"""Чистые конвертеры аннотаций датасетов в формат пайплайна.

Только stdlib — модуль импортируется и тестами (без ML-зависимостей), и
download-скриптами на удалённом сервере. Не импортировать cv2/numpy/onnx.
"""
from __future__ import annotations

from pathlib import Path


def bdd100k_to_labels(native: list) -> list:
    """Нативные BDD100K val-аннотации → формат labels.json пайплайна.

    Вход: список {name, attributes, labels:[{category, box2d:{x1,y1,x2,y2}}]}.
    Лейблы без box2d (полосы/области — только poly2d) отбрасываются.
    Выход: список {image_id, file_name, detections:[{category, box2d}], ...}.
    """
    items = []
    for ex in native:
        name = ex.get("name") or ""
        if not name:
            continue
        detections = []
        for lab in ex.get("labels", []):
            box = lab.get("box2d")
            cat = (lab.get("category") or "").lower()
            if cat and box:
                detections.append({
                    "category": cat,
                    "box2d": {"x1": box["x1"], "y1": box["y1"],
                              "x2": box["x2"], "y2": box["y2"]},
                })
        attrs = ex.get("attributes", {}) or {}
        items.append({
            "image_id": Path(name).stem,
            "file_name": name,
            "detections": detections,
            "weather": attrs.get("weather", ""),
            "timeofday": attrs.get("timeofday", ""),
            "scene": attrs.get("scene", ""),
        })
    return items
```

- [ ] **Step 4: Запустить тест — убедиться, что проходит**

Run: `uv run --with pytest python -m pytest tests/test_dataset_prep.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add deploy/evaluation/dataset_prep.py tests/test_dataset_prep.py
git commit -m "feat(eval): BDD100K→labels.json конвертер (dataset_prep) + тест"
```

## Task A2: Download-скрипт BDD100K

**Files:**
- Create: `deploy/scripts/download_datasets_campaign.sh`

- [ ] **Step 1: Создать скрипт с секцией BDD100K**

`deploy/scripts/download_datasets_campaign.sh` (запускается на удалённом сервере из `deploy/`):
```bash
#!/bin/bash
# Загрузка и конвертация датасетов валидационной кампании 2026-06:
#   - BDD100K val 10K       → data/bdd100k   (TrafficCamNet)
#   - VMMRdb (NGC-20 марки)  → data/vmmrdb    (VehicleMakeNet)   [Task B*]
#   - WIDER FACE val         → data/widerface (FaceDetect)       [Task C*]
#
# Идемпотентен: повторный запуск пропускает уже готовые датасеты.
# Запуск (на сервере, после `run_remote.py deploy`):
#   cd ~/cars-eval && source venv/bin/activate
#   bash scripts/download_datasets_campaign.sh [bdd100k|vmmrdb|widerface|all]
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

WHICH="${1:-all}"

# Nextcloud публичная шара: download-API возвращает zip по path+files.
NC_BASE="https://next.ogoun.name/index.php/s/6T4zxG4eek3AdGG/download?path=/DATASETS"

# ── BDD100K ──────────────────────────────────────────────────────────────────
if [ "$WHICH" = "bdd100k" ] || [ "$WHICH" = "all" ]; then
  echo "=== [BDD100K] TrafficCamNet ==="
  if [ -f data/bdd100k/labels.json ] && \
     [ "$(ls data/bdd100k/images 2>/dev/null | wc -l)" -ge 9000 ]; then
    echo "  [skip] data/bdd100k уже готов"
  else
    mkdir -p data/bdd100k_raw data/bdd100k/images
    if [ ! -f data/bdd100k_raw/bdd100k.zip ]; then
      echo "  скачиваю bdd100k.zip с Nextcloud…"
      wget -q --show-progress "${NC_BASE}&files=bdd100k.zip" -O data/bdd100k_raw/bdd100k.zip
    fi
    echo "  распаковка…"
    unzip -q -o data/bdd100k_raw/bdd100k.zip -d data/bdd100k_raw
    python - <<'EOF'
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, "evaluation")
from dataset_prep import bdd100k_to_labels

raw = Path("data/bdd100k_raw")
# Нативные val-аннотации (имя стабильно во всех раскладках BDD100K).
ann = next(raw.glob("**/bdd100k_labels_images_val.json"), None)
assert ann is not None, f"bdd100k_labels_images_val.json не найден в {raw} (см. unzip -l)"
native = json.loads(ann.read_text())
items = bdd100k_to_labels(native)

# Val-изображения: каталог .../images/100k/val
val_img_dir = next((p for p in raw.glob("**/images/100k/val") if p.is_dir()), None)
assert val_img_dir is not None, f"каталог images/100k/val не найден в {raw}"

out_img = Path("data/bdd100k/images")
out_img.mkdir(parents=True, exist_ok=True)
linked = 0
present = []
for it in items:
    src = val_img_dir / it["file_name"]
    if not src.exists():
        continue
    dst = out_img / it["file_name"]
    if not dst.exists():
        dst.symlink_to(src.resolve())
    linked += 1
    present.append(it)

Path("data/bdd100k/labels.json").write_text(json.dumps(present))
print(f"BDD100K val: {len(present)} изображений с аннотациями, "
      f"{sum(len(i['detections']) for i in present)} детекций (slинковано {linked})")
EOF
  fi
  echo "  готово: $(du -sh data/bdd100k 2>/dev/null | cut -f1)"
fi
```

- [ ] **Step 2: Проверить корректность bash-синтаксиса**

Run: `bash -n deploy/scripts/download_datasets_campaign.sh`
Expected: без вывода (синтаксис валиден).

- [ ] **Step 3: Commit**

```bash
git add deploy/scripts/download_datasets_campaign.sh
git commit -m "feat(deploy): download-скрипт кампании — секция BDD100K"
```

## Task A3: Прогон TrafficCamNet на удалённом GPU

**Files:**
- Modify: `deploy/evaluation/evaluate.py` (conf_thr 0.2→0.4 для in-domain BDD100K)
- Modify: `configs/remote_experiments.yaml` (актуальный сервер, experiments: [trafficcamnet])

- [ ] **Step 1: Переключить conf_thr на in-domain**

В `deploy/evaluation/evaluate.py`, в `EVAL_CONFIGS["trafficcamnet"]`, заменить:
```python
        "conf_thr": 0.2,
```
на:
```python
        # BDD100K — in-domain (traffic-cam), ближе к training-условиям → 0.4
        # (было 0.2 под cross-domain COCO street-level суррогат).
        "conf_thr": 0.4,
```

- [ ] **Step 2: Прописать актуальный сервер в конфиг**

Узнать host/port/ключ текущего контейнера. Отредактировать `configs/remote_experiments.yaml`: в единственном элементе `servers` выставить актуальные `host`, `port`, `identity_file`, и `experiments: [trafficcamnet]` (только текущая модель — прогоны последовательны).

Затем снять stale host key (подставить старый порт из предыдущего значения конфига):
```bash
ssh-keygen -R "[<host>]:<old_port>" 2>/dev/null || true
```

- [ ] **Step 3: Dry-run — проверить план команд**

Run: `python3 deploy/scripts/run_remote.py deploy --dry-run`
Expected: печатает `[PLAN] deploy_one_host on 1 host(s)` и rsync/ssh-команды с актуальным host/port. Ошибок валидации конфига нет.

- [ ] **Step 4: Deploy + bootstrap окружения**

```bash
python3 deploy/scripts/run_remote.py deploy
```
Expected: rsync `deploy/` на сервер, затем `setup_server.sh` создаёт venv (`--system-site-packages`, torch виден с хоста), ставит requirements, печатает `ORT providers: [...'CUDAExecutionProvider'...]`.

- [ ] **Step 5: Скачать веса и датасет на сервере**

Подключиться к серверу и выполнить (идемпотентно):
```bash
ssh -p <port> -i ~/.ssh/qudata root@<host> \
  'cd ~/cars-eval && source venv/bin/activate && \
   bash scripts/download_models.sh && \
   bash scripts/download_datasets_campaign.sh bdd100k'
```
Expected: TrafficCamNet ONNX скачан в `models/trafficcamnet/`, `data/bdd100k/labels.json` создан с ~9000+ изображениями. Если `bdd100k_labels_images_val.json` или `images/100k/val` не найдены — скрипт упадёт с понятным assert; тогда выполнить `unzip -l data/bdd100k_raw/bdd100k.zip | head -50`, скорректировать glob-пути в Task A2 и повторить.

- [ ] **Step 6: Запустить эвалуацию**

```bash
python3 deploy/scripts/run_remote.py run
```
Expected: fire-and-forget nohup пишет в `logs/trafficcamnet.log`. Дождаться завершения (наблюдать `ssh … 'tail -f ~/cars-eval/logs/trafficcamnet.log'` до строки `TrafficCamNet (car): P=… R=… F1=…`).

- [ ] **Step 7: Собрать результаты**

```bash
python3 deploy/scripts/run_remote.py collect
```
Expected: `results_collected/<host>/results/trafficcamnet/metrics.json` появился.

- [ ] **Step 8: Проверить результат**

Run: `python3 -c "import json; d=json.load(open([p for p in __import__('glob').glob('results_collected/*/results/trafficcamnet/metrics.json')][0])); print(d['metrics']['precision'], d['metrics']['recall'], d['thresholds'])"`
Expected: печатает числовые P/R и dict со статусами PASS/FAIL по порогам precision≥0.90, recall≥0.85, f1≥0.87. Любой исход (PASS или FAIL) — валиден; зафиксировать как есть.

- [ ] **Step 9: Commit**

```bash
git add deploy/evaluation/evaluate.py configs/remote_experiments.yaml results_collected/
git commit -m "feat(campaign): TrafficCamNet на BDD100K — прогон + метрики"
```

---

# Секция B — VehicleMakeNet на VMMRdb

Эвалуатор `eval_vehiclemakenet` уже умеет классификацию + `normalize_brand` + пропуск out-of-distribution (не из 20 NGC-марок). Читает манифест `[{file_name, brand}]`. Нужно: (1) обобщить чтение манифеста на `cfg["meta_file"]`, (2) функции извлечения марки + сэмплирования VMMRdb (TDD), (3) секция download-скрипта, (4) прогон.

## Task B1: Обобщить eval_vehiclemakenet + функции VMMRdb (TDD)

**Files:**
- Modify: `deploy/evaluation/evaluate.py:335` (meta_file)
- Modify: `deploy/evaluation/dataset_prep.py` (добавить `vmmrdb_make`, `iter_vmmrdb_samples`)
- Test: `tests/test_dataset_prep.py` (добавить тесты)

- [ ] **Step 1: Написать падающие тесты для VMMRdb-функций**

Добавить в `tests/test_dataset_prep.py`:
```python
from dataset_prep import vmmrdb_make, iter_vmmrdb_samples  # noqa: E402


def test_vmmrdb_make_first_token_lowercased():
    assert vmmrdb_make("Honda_Accord_2003") == "honda"
    assert vmmrdb_make("BMW_3_Series_2010") == "bmw"
    # многословные марки сворачиваются к первому токену (normalize_brand добьёт)
    assert vmmrdb_make("Mercedes_Benz_C_Class_2008") == "mercedes"


def test_iter_vmmrdb_samples_caps_per_class(tmp_path):
    # каталоги-по-классам с изображениями
    honda = tmp_path / "Honda_Civic_2005"
    honda.mkdir()
    for i in range(5):
        (honda / f"img{i}.jpg").write_bytes(b"x")
    bmw = tmp_path / "BMW_X5_2012"
    bmw.mkdir()
    (bmw / "a.jpg").write_bytes(b"x")

    samples = iter_vmmrdb_samples(tmp_path, per_class_cap=3)
    # Honda обрезана до 3, BMW — 1 → всего 4 пары
    assert len(samples) == 4
    makes = sorted({make for _, make in samples})
    assert makes == ["bmw", "honda"]
    # детерминированный отбор: одни и те же файлы при повторе
    assert iter_vmmrdb_samples(tmp_path, per_class_cap=3) == samples
    # элементы — (Path, make)
    p, m = samples[0]
    assert isinstance(p, Path) and isinstance(m, str)
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `uv run --with pytest python -m pytest tests/test_dataset_prep.py -v`
Expected: FAIL — `ImportError: cannot import name 'vmmrdb_make'`.

- [ ] **Step 3: Реализовать функции в dataset_prep.py**

Добавить в `deploy/evaluation/dataset_prep.py`:
```python
def vmmrdb_make(dirname: str) -> str:
    """Имя каталога VMMRdb '<make>_<model>_<year>' → марка (первый токен, lower)."""
    return dirname.split("_")[0].lower().strip()


def iter_vmmrdb_samples(root: Path, per_class_cap: int) -> list:
    """Обойти каталоги-по-классам VMMRdb, вернуть детерминированный список
    (image_path, make) с не более чем per_class_cap изображений на каталог.

    Сортировка по имени каталога и по имени файла → воспроизводимый отбор.
    """
    root = Path(root)
    samples = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        make = vmmrdb_make(class_dir.name)
        imgs = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for img in imgs[:per_class_cap]:
            samples.append((img, make))
    return samples
```

- [ ] **Step 4: Запустить — убедиться, что проходит**

Run: `uv run --with pytest python -m pytest tests/test_dataset_prep.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Обобщить eval_vehiclemakenet на meta_file**

В `deploy/evaluation/evaluate.py`, в `eval_vehiclemakenet`, заменить:
```python
    meta_path = data_dir / "sample_5k.json"
    if not meta_path.exists():
        log.error(f"mad-cars metadata not found: {meta_path}")
        return {"error": "dataset not found"}
```
на:
```python
    meta_path = data_dir / cfg.get("meta_file", "sample_5k.json")
    if not meta_path.exists():
        log.error(f"VehicleMakeNet metadata not found: {meta_path}")
        return {"error": "dataset not found"}
```

И в `EVAL_CONFIGS["vehiclemakenet"]` заменить:
```python
        "data_dir": "data/mad_cars",
```
на:
```python
        # Кампания 2026-06: VMMRdb (каталоги-по-классам → manifest.json),
        # вместо mad-cars-суррогата. Марка = первый токен имени каталога.
        "data_dir": "data/vmmrdb",
        "meta_file": "manifest.json",
```

- [ ] **Step 6: Прогнать все тесты (регрессия)**

Run: `uv run --with pytest python -m pytest tests/ -v`
Expected: PASS — новые тесты dataset_prep + существующие (`test_normalize_brand`, `test_typenet_mapping`, `test_run_remote`). Если какой-то существующий тест требует numpy/cv2 и падает на импорте — запустить только релевантные: `uv run --with pytest python -m pytest tests/test_dataset_prep.py tests/test_run_remote.py -v` и отметить это в отчёте.

- [ ] **Step 7: Commit**

```bash
git add deploy/evaluation/dataset_prep.py deploy/evaluation/evaluate.py tests/test_dataset_prep.py
git commit -m "feat(eval): VMMRdb make-extraction + сэмплирование + meta_file в eval_vehiclemakenet"
```

## Task B2: Секция VMMRdb в download-скрипте

**Files:**
- Modify: `deploy/scripts/download_datasets_campaign.sh`

- [ ] **Step 1: Добавить секцию VMMRdb**

Добавить в `deploy/scripts/download_datasets_campaign.sh` перед финальной строкой (после секции BDD100K):
```bash
# ── VMMRdb ───────────────────────────────────────────────────────────────────
if [ "$WHICH" = "vmmrdb" ] || [ "$WHICH" = "all" ]; then
  echo "=== [VMMRdb] VehicleMakeNet ==="
  if [ -f data/vmmrdb/manifest.json ] && \
     [ "$(ls data/vmmrdb/images 2>/dev/null | wc -l)" -ge 1000 ]; then
    echo "  [skip] data/vmmrdb уже готов"
  else
    mkdir -p data/vmmrdb_raw data/vmmrdb/images
    if [ ! -f data/vmmrdb_raw/VMMRdb.zip ]; then
      echo "  скачиваю VMMRdb.zip с Nextcloud…"
      wget -q --show-progress "${NC_BASE}&files=VMMRdb.zip" -O data/vmmrdb_raw/VMMRdb.zip
    fi
    echo "  распаковка…"
    unzip -q -o data/vmmrdb_raw/VMMRdb.zip -d data/vmmrdb_raw
    PER_CLASS_CAP="${VMMRDB_PER_CLASS_CAP:-50}" python - <<'EOF'
import json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, "evaluation")
from dataset_prep import iter_vmmrdb_samples, vmmrdb_make

raw = Path("data/vmmrdb_raw")
# Корень с каталогами-по-классам: каталог, где много подкаталогов вида make_model_year.
candidates = [raw] + [p for p in raw.iterdir() if p.is_dir()]
root = max(candidates, key=lambda d: sum(1 for x in d.iterdir() if x.is_dir())
           if d.is_dir() else 0)
print(f"VMMRdb корень классов: {root}")

cap = int(os.environ["PER_CLASS_CAP"])
samples = iter_vmmrdb_samples(root, per_class_cap=cap)

# Импортируем правила марок из эвалуатора, чтобы заранее отсеять out-of-distribution
# (не из 20 NGC-марок) — экономит копирование изображений. Эвалуатор всё равно
# повторно отфильтрует через normalize_brand.
sys.path.insert(0, "evaluation")
from evaluate import normalize_brand, NGC_MAKES_LOWER

out_img = Path("data/vmmrdb/images")
out_img.mkdir(parents=True, exist_ok=True)
records, kept = [], 0
for img_path, make in samples:
    if normalize_brand(make) not in NGC_MAKES_LOWER:
        continue
    fname = f"{kept:06d}_{img_path.name}"
    dst = out_img / fname
    if not dst.exists():
        dst.symlink_to(img_path.resolve())
    records.append({"file_name": fname, "brand": make})
    kept += 1

Path("data/vmmrdb/manifest.json").write_text(json.dumps(records, ensure_ascii=False))
print(f"VMMRdb manifest: {kept} изображений в 20 NGC-марках (cap={cap}/класс)")
EOF
  fi
  echo "  готово: $(du -sh data/vmmrdb 2>/dev/null | cut -f1)"
fi
```

> Примечание: секция импортирует `normalize_brand`/`NGC_MAKES_LOWER` из `evaluate.py`. На сервере это безопасно (cv2/onnx установлены). Локально эту секцию не гоняем — её предфильтр продублирован защитой в самом эвалуаторе.

- [ ] **Step 2: Проверить bash-синтаксис**

Run: `bash -n deploy/scripts/download_datasets_campaign.sh`
Expected: без вывода.

- [ ] **Step 3: Commit**

```bash
git add deploy/scripts/download_datasets_campaign.sh
git commit -m "feat(deploy): download-скрипт кампании — секция VMMRdb"
```

## Task B3: Прогон VehicleMakeNet на удалённом GPU

**Files:**
- Modify: `configs/remote_experiments.yaml` (experiments: [vehiclemakenet])

- [ ] **Step 1: Обновить конфиг сервера**

Сервер мог смениться с прогона A3. Обновить `host/port/identity` в `configs/remote_experiments.yaml`, `experiments: [vehiclemakenet]`, снять stale host key (`ssh-keygen -R "[<host>]:<old_port>"`).

- [ ] **Step 2: Dry-run**

Run: `python3 deploy/scripts/run_remote.py deploy --dry-run`
Expected: план с актуальным host, без ошибок валидации.

- [ ] **Step 3: Deploy + данные/веса**

```bash
python3 deploy/scripts/run_remote.py deploy
ssh -p <port> -i ~/.ssh/qudata root@<host> \
  'cd ~/cars-eval && source venv/bin/activate && \
   bash scripts/download_models.sh && \
   bash scripts/download_datasets_campaign.sh vmmrdb'
```
Expected: VehicleMakeNet ONNX в `models/vehiclemakenet/`, `data/vmmrdb/manifest.json` с записями только по 20 NGC-маркам.

- [ ] **Step 4: Run + collect**

```bash
python3 deploy/scripts/run_remote.py run
# дождаться строки "VehicleMakeNet: Top1=… Top3=…" в logs/vehiclemakenet.log
python3 deploy/scripts/run_remote.py collect
```
Expected: `results_collected/<host>/results/vehiclemakenet/metrics.json` + `per_class_metrics.csv`.

- [ ] **Step 5: Проверить результат**

Run: `python3 -c "import glob,json; d=json.load(open(glob.glob('results_collected/*/results/vehiclemakenet/metrics.json')[0])); print('Top1',d['metrics']['top1_accuracy'],'Top3',d['metrics']['top3_accuracy'],d['thresholds'])"`
Expected: числовые Top-1/Top-3 + статусы по порогам top1≥0.70, top3≥0.85. Любой исход валиден.

- [ ] **Step 6: Commit**

```bash
git add configs/remote_experiments.yaml results_collected/
git commit -m "feat(campaign): VehicleMakeNet на VMMRdb — прогон + метрики"
```

---

# Секция C — FaceDetect на WIDER FACE

Самая рискованная задача: эвалуатора нет, веса FaceNet надо найти на NGC, decode-параметры DetectNet_v2 для FaceNet требуют калибровки. FaceNet — DetectNet_v2, 1 класс «face», вход 736×416 → переиспользуем `detectnet_v2_decode`.

## Task C1: Конвертер WIDER FACE (TDD)

**Files:**
- Modify: `deploy/evaluation/dataset_prep.py` (добавить `widerface_faces_to_detections`)
- Test: `tests/test_dataset_prep.py`

- [ ] **Step 1: Написать падающий тест**

Добавить в `tests/test_dataset_prep.py`:
```python
from dataset_prep import widerface_faces_to_detections  # noqa: E402


def test_widerface_faces_to_detections():
    faces = {"bbox": [[10.0, 20.0, 30.0, 40.0],   # x,y,w,h → x1,y1,x2,y2
                      [5.0, 5.0, 0.0, 12.0]]}       # нулевая ширина → отброшен
    dets = widerface_faces_to_detections(faces)
    assert len(dets) == 1
    assert dets[0]["category"] == "face"
    assert dets[0]["box2d"] == {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0}


def test_widerface_empty():
    assert widerface_faces_to_detections({}) == []
    assert widerface_faces_to_detections({"bbox": []}) == []
```

- [ ] **Step 2: Запустить — убедиться, что падает**

Run: `uv run --with pytest python -m pytest tests/test_dataset_prep.py -v`
Expected: FAIL — `ImportError: cannot import name 'widerface_faces_to_detections'`.

- [ ] **Step 3: Реализовать**

Добавить в `deploy/evaluation/dataset_prep.py`:
```python
def widerface_faces_to_detections(faces: dict) -> list:
    """WIDER FACE faces-словарь HF → detections в формате пайплайна.

    HF-схема: faces = {"bbox": [[x, y, w, h], ...], ...}. Боксы с нулевой/
    отрицательной шириной или высотой отбрасываются (в WIDER FACE есть
    вырожденные аннотации «invalid»).
    """
    dets = []
    for bbox in faces.get("bbox", []):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        if w <= 0 or h <= 0:
            continue
        dets.append({
            "category": "face",
            "box2d": {"x1": float(x), "y1": float(y),
                      "x2": float(x + w), "y2": float(y + h)},
        })
    return dets
```

- [ ] **Step 4: Запустить — убедиться, что проходит**

Run: `uv run --with pytest python -m pytest tests/test_dataset_prep.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add deploy/evaluation/dataset_prep.py tests/test_dataset_prep.py
git commit -m "feat(eval): WIDER FACE faces→detections конвертер + тест"
```

## Task C2: Эвалуатор eval_facedetect + регистрация

**Files:**
- Modify: `deploy/evaluation/evaluate.py` (добавить `eval_facedetect`, запись в `EVAL_CONFIGS`)
- Modify: `deploy/scripts/run_remote.py:35-43` (добавить `facedetect` в `VALID_EXPERIMENTS`)
- Modify: `deploy/scripts/setup_server.sh` (каталоги)

- [ ] **Step 1: Добавить eval_facedetect в evaluate.py**

Вставить в `deploy/evaluation/evaluate.py` перед секцией `# ─── Main ───` (после `eval_nomeroff_ocr`):
```python
# ─── FaceDetect (TAO FaceNet / DetectNet_v2) ──────────────────────────────────

def eval_facedetect(cfg: dict) -> dict:
    """FaceNet (DetectNet_v2, 1 класс 'face', вход 736×416) на WIDER FACE.

    Переиспользует detectnet_v2_decode (как TrafficCamNet/LPDNet). Читает
    data/widerface/labels.json в том же формате, что TrafficCamNet:
    [{image_id, file_name, detections:[{category:'face', box2d:{x1,y1,x2,y2}}]}].
    """
    model_path = ROOT / cfg["model_path"]
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.error(f"FaceNet model not found: {model_path}")
        return {"error": "model not found"}

    meta_path = data_dir / "labels.json"
    if not meta_path.exists():
        log.error(f"WIDER FACE labels not found: {meta_path}")
        return {"error": "dataset not found"}

    with open(meta_path) as f:
        all_meta = json.load(f)

    sess = get_ort_session(str(model_path))
    input_name = sess.get_inputs()[0].name
    # FaceNet input: 736×416 (W×H). Подтвердить из sess.get_inputs()[0].shape;
    # если модель статической формы — взять оттуда.
    ishape = sess.get_inputs()[0].shape
    H = ishape[2] if isinstance(ishape[2], int) else 416
    W = ishape[3] if isinstance(ishape[3], int) else 736
    conf_thr = float(cfg.get("conf_thr", 0.4))
    bbox_norm = float(cfg.get("bbox_norm", 35.0))

    images_dir = data_dir / "images"
    predictions, ground_truths = [], []

    for item in tqdm(all_meta, desc="FaceDetect", unit="img"):
        img_path = images_dir / item["file_name"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        inp = cv2.resize(img, (W, H)).astype(np.float32)
        inp = inp[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0  # BGR→RGB NCHW

        outputs = sess.run(None, {input_name: inp})
        cov = outputs[0][0]   # [C, gh, gw]
        bbox = outputs[1][0]  # [4C, gh, gw]
        boxes_pred = detectnet_v2_decode(
            cov, bbox, target_cls=0, conf_thr=conf_thr,
            stride=16, bbox_norm=bbox_norm, img_w=W, img_h=H,
            scale_w=orig_w / W, scale_h=orig_h / H,
        )
        predictions.append({"image_id": item["image_id"], "boxes": boxes_pred})

        gt_boxes = []
        for det in item.get("detections", []):
            b = det.get("box2d", det.get("bbox2d", {}))
            if b:
                gt_boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
        ground_truths.append({"image_id": item["image_id"], "boxes": gt_boxes})

    m = compute_detection_metrics(predictions, ground_truths, conf_threshold=conf_thr)
    thresholds = {"precision": 0.70, "recall": 0.70, "f1": 0.70}
    status = check_thresholds(m.to_dict(), thresholds)

    result = {"metrics": m.to_dict(), "thresholds": status,
              "conf_thr": conf_thr,
              "note": "TAO FaceNet (DetectNet_v2) на WIDER FACE val; "
                      "WIDER FACE содержит крошечные/перекрытые лица (hard set) — "
                      "FAIL возможен и валиден"}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info(f"FaceDetect: P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")
    return result
```

- [ ] **Step 2: Зарегистрировать facedetect в EVAL_CONFIGS**

Добавить в `deploy/evaluation/evaluate.py`, в словарь `EVAL_CONFIGS` (например после `vehicletypenet`):
```python
    "facedetect": {
        "model_path": "models/facenet/facenet.onnx",
        "data_dir": "data/widerface",
        "results_dir": "results/facedetect",
        "eval_fn": eval_facedetect,
        "conf_thr": 0.4,
        "bbox_norm": 35.0,
    },
```

- [ ] **Step 3: Проверить, что модуль импортируется (синтаксис/имена)**

Run: `python3 -m py_compile deploy/evaluation/evaluate.py && echo OK`
Expected: `OK` (компиляция без синтаксических ошибок; импортов cv2/onnx это не требует).

- [ ] **Step 4: Добавить facedetect в VALID_EXPERIMENTS**

В `deploy/scripts/run_remote.py`, в множество `VALID_EXPERIMENTS`, добавить строку:
```python
    "facedetect",
```

- [ ] **Step 5: Прогнать тесты run_remote (регрессия валидации конфига)**

Run: `uv run --with pytest python -m pytest tests/test_run_remote.py -v`
Expected: PASS — добавление нового допустимого эксперимента не ломает существующие тесты.

- [ ] **Step 6: Добавить каталоги в setup_server.sh**

В `deploy/scripts/setup_server.sh` заменить три строки `mkdir -p` на:
```bash
mkdir -p models/{trafficcamnet,vehiclemakenet,vehicletypenet,lpdnet,lprnet,facenet,color}
mkdir -p data/{bdd100k,mad_cars,vmmrdb,widerface,nomeroff_lp,nomeroff_ocr_ru,bit_vehicle,stanford_cars,coco}
mkdir -p results/{trafficcamnet,vehiclemakenet,vehicletypenet,lpdnet,lprnet,facedetect,color}
```

- [ ] **Step 7: Commit**

```bash
git add deploy/evaluation/evaluate.py deploy/scripts/run_remote.py deploy/scripts/setup_server.sh
git commit -m "feat(eval): эвалуатор FaceDetect (FaceNet/DetectNet_v2) + регистрация в пайплайне"
```

## Task C3: Загрузка весов FaceNet + датасета WIDER FACE

**Files:**
- Modify: `deploy/scripts/download_models.sh` (FaceNet с NGC)
- Modify: `deploy/scripts/download_datasets_campaign.sh` (секция WIDER FACE)

- [ ] **Step 1: Добавить FaceNet в download_models.sh**

В `deploy/scripts/download_models.sh` перед финальным блоком `echo "=== Models downloaded ==="` добавить:
```bash
# 6. FaceNet (face detector, DetectNet_v2, 1 класс)
echo "[6/6] FaceNet — поиск ONNX-файла на NGC"
DIR="$MODELS_DIR/facenet"
mkdir -p "$DIR"
if [ -f "$DIR/facenet.onnx" ]; then
    echo "  [skip] facenet.onnx уже есть"
else
    # У FaceNet ONNX может лежать под разными версиями. Перебираем кандидатов,
    # для каждой запрашиваем список файлов и ищем первый .onnx.
    FOUND=""
    for V in deployable_v1.0 pruned_quantized_v2.0.1 pruned_v2.0.1 deployable_onnx_v1.0; do
        FILES_JSON="$(wget -qO- "$BASE_URL/facenet/versions/$V/files" || true)"
        ONNX_NAME="$(echo "$FILES_JSON" | grep -oE '"[^"]+\.onnx"' | head -1 | tr -d '"')"
        if [ -n "$ONNX_NAME" ]; then
            echo "  нашёл $ONNX_NAME в версии $V"
            download_file "$BASE_URL/facenet/versions/$V/files/$ONNX_NAME" \
                "$DIR/facenet.onnx" "facenet.onnx"
            FOUND="yes"
            break
        fi
    done
    if [ -z "$FOUND" ]; then
        echo "  ОШИБКА: ONNX FaceNet не найден ни в одной из проверенных версий NGC."
        echo "  Выполнить вручную: wget -qO- '$BASE_URL/facenet/versions/<V>/files'"
        echo "  чтобы найти версию с .onnx, и дописать её в список кандидатов."
        exit 1
    fi
fi
```
И обновить счётчики в эхо-строках `[1/5]…[5/5]` на `[1/6]…[5/6]` (косметика; не обязательно для работы).

- [ ] **Step 2: Добавить секцию WIDER FACE в download_datasets_campaign.sh**

Добавить перед финальной строкой скрипта:
```bash
# ── WIDER FACE ────────────────────────────────────────────────────────────────
if [ "$WHICH" = "widerface" ] || [ "$WHICH" = "all" ]; then
  echo "=== [WIDER FACE] FaceDetect ==="
  if [ -f data/widerface/labels.json ] && \
     [ "$(ls data/widerface/images 2>/dev/null | wc -l)" -ge 2000 ]; then
    echo "  [skip] data/widerface уже готов"
  else
    mkdir -p data/widerface/images
    python - <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "evaluation")
from dataset_prep import widerface_faces_to_detections

from datasets import load_dataset
ds = None
for repo in ("wider_face", "CUHK-CSE/wider_face"):
    try:
        print(f"пробую {repo} (split=validation)…")
        ds = load_dataset(repo, split="validation")
        print(f"OK из {repo}: {len(ds)} примеров; колонки {ds.column_names}")
        break
    except Exception as e:
        print(f"  не вышло: {type(e).__name__}: {e}")
assert ds is not None, "все HF-зеркала WIDER FACE недоступны"

img_dir = Path("data/widerface/images")
img_dir.mkdir(parents=True, exist_ok=True)
items = []
for i, ex in enumerate(ds):
    img = ex.get("image")
    if img is None or not hasattr(img, "save"):
        continue
    fname = f"wf_{i:06d}.jpg"
    dst = img_dir / fname
    if not dst.exists():
        img.convert("RGB").save(dst)
    dets = widerface_faces_to_detections(ex.get("faces", {}) or {})
    items.append({"image_id": f"wf_{i:06d}", "file_name": fname,
                  "detections": dets})

Path("data/widerface/labels.json").write_text(json.dumps(items))
print(f"WIDER FACE val: {len(items)} изображений, "
      f"{sum(len(it['detections']) for it in items)} лиц")
EOF
  fi
  echo "  готово: $(du -sh data/widerface 2>/dev/null | cut -f1)"
fi
```

- [ ] **Step 3: Проверить bash-синтаксис обоих скриптов**

Run: `bash -n deploy/scripts/download_models.sh && bash -n deploy/scripts/download_datasets_campaign.sh && echo OK`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add deploy/scripts/download_models.sh deploy/scripts/download_datasets_campaign.sh
git commit -m "feat(deploy): загрузка FaceNet (NGC) + WIDER FACE (HF)"
```

## Task C4: Прогон FaceDetect на удалённом GPU (+ калибровка decode)

**Files:**
- Modify: `configs/remote_experiments.yaml` (experiments: [facedetect])
- Возможно Modify: `deploy/evaluation/evaluate.py` (`bbox_norm`/`conf_thr` после калибровки)

- [ ] **Step 1: Обновить конфиг сервера**

Обновить host/port/identity, `experiments: [facedetect]`, снять stale host key.

- [ ] **Step 2: Deploy + веса + датасет**

```bash
python3 deploy/scripts/run_remote.py deploy
ssh -p <port> -i ~/.ssh/qudata root@<host> \
  'cd ~/cars-eval && source venv/bin/activate && \
   bash scripts/download_models.sh && \
   bash scripts/download_datasets_campaign.sh widerface'
```
Expected: `models/facenet/facenet.onnx` скачан (если NGC не отдаёт ONNX — скрипт упадёт с инструкцией; разобраться вручную через listing-API и дописать версию). `data/widerface/labels.json` создан (~3000 изображений val).

- [ ] **Step 3: Smoke-калибровка decode-параметров**

Перед полным прогоном проверить, что детектор вообще выдаёт боксы (FaceNet bbox_norm/stride могут отличаться). На сервере:
```bash
ssh -p <port> -i ~/.ssh/qudata root@<host> \
  'cd ~/cars-eval && source venv/bin/activate && \
   python -c "
import sys; sys.path.insert(0,\"evaluation\")
import onnxruntime as ort
s=ort.InferenceSession(\"models/facenet/facenet.onnx\",providers=[\"CUDAExecutionProvider\",\"CPUExecutionProvider\"])
print(\"input\", s.get_inputs()[0].name, s.get_inputs()[0].shape)
for o in s.get_outputs(): print(\"output\", o.name, o.shape)
"'
```
Expected: один вход формы `[1,3,416,736]`, два выхода (cov `[1,1,26,46]` и bbox `[1,4,26,46]`). Если форма входа иная — поправить W/H-фолбэк в `eval_facedetect`. Если выходов не два или каналы не совпадают (cov C=1, bbox 4C=4) — модель не DetectNet_v2-совместима; зафиксировать как блокер и сообщить.

- [ ] **Step 4: Run + collect**

```bash
python3 deploy/scripts/run_remote.py run
# дождаться "FaceDetect: P=… R=… F1=…" в logs/facedetect.log
python3 deploy/scripts/run_remote.py collect
```
Expected: `results_collected/<host>/results/facedetect/metrics.json`.

- [ ] **Step 5: Проверить результат и при необходимости откалибровать**

Run: `python3 -c "import glob,json; d=json.load(open(glob.glob('results_collected/*/results/facedetect/metrics.json')[0])); print('P',d['metrics']['precision'],'R',d['metrics']['recall'],'num_pred',d['metrics']['num_pred'],'num_gt',d['metrics']['num_gt'])"`
Expected: ненулевые `num_pred`/`num_gt`. Если `num_pred == 0` (детектор молчит) — `bbox_norm` или `conf_thr` неверны: снизить `conf_thr` до 0.1 и/или попробовать `bbox_norm` ∈ {35.0, 1.0} в `EVAL_CONFIGS["facedetect"]`, перезалить (`deploy`) и повторить `run`/`collect`. Когда боксы появятся — зафиксировать метрики как окончательные (PASS или FAIL — оба валидны).

- [ ] **Step 6: Commit**

```bash
git add configs/remote_experiments.yaml deploy/evaluation/evaluate.py results_collected/
git commit -m "feat(campaign): FaceDetect на WIDER FACE — прогон + метрики (+ калибровка decode)"
```

---

# Секция D — Финальная агрегация

После трёх прогонов собрать единый отчёт по всем шести парам (три новых + три уже собранных, которые `aggregate_summary.py` подхватит из `results_collected/`).

## Task D1: Агрегация, графики, notebook, merge в main

**Files:**
- Create: `results/SUMMARY.md`, `results/*.csv`, `plots/*.png` (генерируются)
- Create: `notebooks/validation_campaign_2026_06.ipynb`

- [ ] **Step 1: Сгенерировать агрегированный SUMMARY**

Run: `python3 deploy/evaluation/aggregate_summary.py`
Expected: `Wrote results/SUMMARY.md — N runs from results_collected`. N ≥ 6 (три новых прогона + ранее собранные nomeroff_lpd/nomeroff_ocr/vehicletypenet и baseline). Открыть `results/SUMMARY.md`, убедиться, что присутствуют строки `trafficcamnet`, `vehiclemakenet`, `facedetect` с актуальных хостов.

> Зависимость aggregate_summary.py — только stdlib, поэтому запускается локально: `uv run python3 deploy/evaluation/aggregate_summary.py` если системный python3 подходит — без `--with`.

- [ ] **Step 2: Выгрузить метрики в CSV (правило CLAUDE.md: results/ как JSON+CSV)**

Создать и запустить разовый экспорт; сохранить как часть notebook (Step 4) либо командой:
```bash
python3 - <<'EOF'
import glob, json, csv
from pathlib import Path
rows = []
for mj in sorted(glob.glob("results_collected/*/results/*/metrics.json")):
    d = json.load(open(mj))
    if "error" in d:
        continue
    parts = Path(mj).parts
    host, model = parts[1], parts[-2]
    m = d.get("metrics", {})
    th = d.get("thresholds", {})
    status = "PASS" if all(v.get("status") == "PASS" for v in th.values()) and th else "FAIL"
    rows.append({"host": host, "model": model, "status": status,
                 **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
Path("results").mkdir(exist_ok=True)
keys = sorted({k for r in rows for k in r})
with open("results/campaign_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"results/campaign_summary.csv — {len(rows)} строк")
EOF
```
Expected: `results/campaign_summary.csv` с одной строкой на (host, model) и общим pass/fail.

- [ ] **Step 3: Графики (правило CLAUDE.md: plots/ как PNG)**

Графики per-class/confusion для vehiclemakenet/vehicletypenet уже генерируются эвалуаторами и собираются в `results_collected/<host>/plots/`. Свести обзорный bar-chart pass/fail:
```bash
uv run --with matplotlib python3 - <<'EOF'
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
rows = list(csv.DictReader(open("results/campaign_summary.csv")))
labels = [f"{r['model']}" for r in rows]
colors = ["green" if r["status"] == "PASS" else "salmon" for r in rows]
vals = [float(r.get("f1") or r.get("top1_accuracy") or r.get("char_accuracy") or 0) for r in rows]
fig, ax = plt.subplots(figsize=(max(8, len(rows)), 5))
ax.bar(labels, vals, color=colors, edgecolor="black")
ax.set_ylim(0, 1.05); ax.set_ylabel("key metric (F1 / Top-1 / CharAcc)")
ax.set_title("Validation Campaign 2026-06 — обзор по моделям")
ax.tick_params(axis="x", rotation=45)
Path("plots").mkdir(exist_ok=True)
fig.tight_layout(); fig.savefig("plots/campaign_overview.png", dpi=150)
print("plots/campaign_overview.png")
EOF
```
Expected: `plots/campaign_overview.png`.

- [ ] **Step 4: Воспроизводимый notebook (правило CLAUDE.md: notebooks/)**

Создать `notebooks/validation_campaign_2026_06.ipynb` с ячейками, воспроизводящими Step 1–3 (вызов `aggregate_summary.py`, экспорт CSV, обзорный график) и markdown-разделом с итоговой таблицей шести пар и вердиктами. Сгенерировать через `jupyter nbconvert` или собрать программно из словаря ячеек; убедиться, что ноутбук выполняется сверху вниз без ошибок:
```bash
uv run --with jupyter --with matplotlib jupyter nbconvert --to notebook --execute \
  notebooks/validation_campaign_2026_06.ipynb --output validation_campaign_2026_06.ipynb
```
Expected: ноутбук исполняется без ошибок.

- [ ] **Step 5: Commit результатов кампании**

```bash
git add results/SUMMARY.md results/campaign_summary.csv plots/campaign_overview.png \
        notebooks/validation_campaign_2026_06.ipynb
git commit -m "docs(campaign): финальная агрегация 6 пар — SUMMARY + CSV + графики + notebook"
```

- [ ] **Step 6: Слить ветку кампании в main**

```bash
git checkout main
git merge --no-ff exp/validation-campaign-2026-06 -m "merge: валидационная кампания 2026-06 (3 модели + агрегация)"
git push origin main
```
Expected: fast-forward или чистый merge-commit (ветка отведена от актуального main, конфликтов нет). `origin/main` обновлён.

- [ ] **Step 7: Финальная верификация**

Run: `git checkout main && uv run --with pytest python -m pytest tests/test_dataset_prep.py tests/test_run_remote.py -v && python3 -m py_compile deploy/evaluation/evaluate.py && echo "CAMPAIGN OK"`
Expected: все тесты PASS, компиляция OK, печатает `CAMPAIGN OK`. SUMMARY.md содержит вердикты по всем шести парам.

---

## Self-Review (заполняется автором плана)

**Покрытие спека (дизайн §Охват, §Архитектура, §Порядок):**
- TrafficCamNet/BDD100K — Секция A (конвертер + загрузка + прогон). ✓
- VehicleMakeNet/VMMRdb — Секция B (make-extraction + meta_file + загрузка + прогон). ✓
- FaceDetect/WIDER FACE — Секция C (новый эвалуатор + веса NGC + WIDER FACE HF + калибровка + прогон). ✓
- Уже отвалидированные nomeroff_lpd/nomeroff_ocr/vehicletypenet — не перегоняются, зачитываются агрегацией (Секция D). ✓
- Финальная агрегация: SUMMARY (6 пар), графики, CSV, notebook, merge в main — Секция D. ✓
- Порядок (TrafficCamNet → VehicleMakeNet → FaceDetect, рискованный в конце) — соблюдён. ✓
- Одна общая ветка, последовательные прогоны на одном GPU — соблюдено. ✓
- Серверная ротация (host key, идемпотентный bootstrap) — учтена в Prerequisites и в каждом прогоне. ✓

**Сканирование плейсхолдеров:** код конвертеров, эвалуатора, скриптов — полный. Открытые внешние неизвестности оформлены как явные шаги с проверкой и инструкцией при провале (имя ONNX FaceNet на NGC → listing-перебор; раскладка внутри zip → unzip -l + правка glob; decode-параметры FaceNet → smoke-калибровка). Это не плейсхолдеры, а корректная обработка внешних неизвестностей.

**Согласованность типов/имён:**
- `dataset_prep`: `bdd100k_to_labels(list)->list`, `vmmrdb_make(str)->str`, `iter_vmmrdb_samples(Path,int)->list[(Path,str)]`, `widerface_faces_to_detections(dict)->list` — единообразны между тестами, скриптами и эвалуаторами.
- Формат `labels.json` (`image_id`/`file_name`/`detections`/`box2d`) одинаков для BDD100K и WIDER FACE и совпадает с тем, что читают `eval_trafficcamnet`/`eval_facedetect`.
- Манифест `[{file_name, brand}]` совпадает с тем, что читает `eval_vehiclemakenet` через `cfg["meta_file"]`.
- `facedetect` добавлен согласованно в `EVAL_CONFIGS`, `VALID_EXPERIMENTS`, `setup_server.sh`.
