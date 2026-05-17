# План реализации: замена LPDNet/LPRNet на Nomeroff.net (RU)

> **Для агентных исполнителей:** РЕКОМЕНДУЕМЫЙ САБ-СКИЛЛ — `superpowers:subagent-driven-development` (или `superpowers:executing-plans`). Шаги используют `- [ ]` для трекинга.

**Goal:** В кампании evaluation для RU-номеров заменить US-обученные `lpdnet` (NVIDIA LPDNet) и `lprnet` (NVIDIA LPRNet) на `nomeroff_lpd` и `nomeroff_ocr` на базе пакета `nomeroff-net`.

**Architecture:** Два новых эвалюатора в `deploy/evaluation/evaluate.py` (`eval_nomeroff_lpd`, `eval_nomeroff_ocr`) обёртывают высокоуровневый API `nomeroff-net`. Метрики считаются существующими помощниками `compute_detection_metrics` / `compute_ocr_metrics` и пишутся в `results/nomeroff_*/metrics.json`. Старые US-эвалюаторы остаются в коде как baseline. Активная кампания (`configs/remote_experiments.yaml`) переключается на новые имена.

**Tech Stack:** Python 3.10, `nomeroff-net` (новая зависимость, только remote), torch (предустановлен на ssh9), `compute_detection_metrics`/`compute_ocr_metrics` из `deploy/evaluation/metrics.py`. Запуск через существующий `deploy/scripts/run_remote.py`.

**Спека:** `_bmad-output/implementation-artifacts/spec-nomeroff-ru-swap.md`

**Контекст по проекту:**
- ❗ Реестр моделей в `evaluate.py` называется `EVAL_CONFIGS` (не `MODEL_REGISTRY`, как было ошибочно в спеке) — все правки регистра идут в него.
- ❗ Датасет для **детекции** — `data/nomeroff_lp` (VIA-формат, full-frame изображения + bbox-аннотации в `via_region_data.json`). Датасет для **OCR** — `data/nomeroff_ocr_ru` (`val/img/*.png` + `val/ann/*.json` с ключом `description`). Это поправляет упрощение из спеки (где для обоих был указан `nomeroff_ocr_ru`).
- ❗ В проекте **нет pytest-тестов** для эвалюаторов (CLAUDE.md: «no formal test suite — the dataset is the test fixture»). Гейт корректности — smoke-test импорта на ssh9 + сравнение финальных метрик с US baseline. TDD-цикл в классическом виде здесь не применим.
- ❗ Артефакты на русском (project memory `feedback_artifact_language`).

---

## Файловая структура

| Файл | Тип | Ответственность |
|---|---|---|
| `deploy/requirements.txt` | modify | Добавляется зависимость `nomeroff-net`. |
| `deploy/scripts/run_remote.py:35-41` | modify | В `VALID_EXPERIMENTS` добавляются `"nomeroff_lpd"`, `"nomeroff_ocr"`. |
| `deploy/evaluation/evaluate.py` | modify | Две новые функции `eval_nomeroff_lpd`, `eval_nomeroff_ocr` (~80 строк). Две новые записи в `EVAL_CONFIGS`. |
| `configs/remote_experiments.yaml` | modify | На хосте ssh9: `lpdnet`/`lprnet` → `nomeroff_lpd`/`nomeroff_ocr`. |
| `_bmad-output/implementation-artifacts/deferred-work.md` | modify | Финальная секция «Goal 4 — RU model swap» со сравнительной таблицей метрик (заполняется после прогона). |

---

## Task 1: Добавить `nomeroff-net` в remote-зависимости

**Files:**
- Modify: `deploy/requirements.txt` (вставка перед последней строкой `kaggle>=1.6`)

- [ ] **Step 1: Прочитать текущий файл**

Run: `cat deploy/requirements.txt`

Expected output: 14 строк, последняя — `kaggle>=1.6`. Никакого `torch`, `nvidia-*` — это конвенция проекта (предустановлено на remote).

- [ ] **Step 2: Добавить строку**

Через Edit-инструмент: после `tqdm>=4.66` вставить новую строку:

```
nomeroff-net
```

Версию не пиним: пакет молодой, риск фиксации на дефектной версии выше, чем риск дрейфа API. После первого успешного прогона можно отдельной задачей зафиксировать (`nomeroff-net==X.Y.Z`).

- [ ] **Step 3: Верификация**

Run: `grep -E "^nomeroff" deploy/requirements.txt`

Expected: одна строка `nomeroff-net`.

- [ ] **Step 4: Commit**

```bash
git add deploy/requirements.txt
git commit -m "feat(deploy): add nomeroff-net to remote requirements

Подготовка к замене US LPDNet/LPRNet на RU-специфичные модели
проекта nomeroff.net.ua. Torch предустановлен на remote, дополнительных
nvidia-* зависимостей не вводится."
```

---

## Task 2: Зарегистрировать новые имена экспериментов в runner

**Files:**
- Modify: `deploy/scripts/run_remote.py:35-41`

- [ ] **Step 1: Прочитать блок констант**

```bash
sed -n '35,41p' deploy/scripts/run_remote.py
```

Expected:
```python
VALID_EXPERIMENTS = {
    "trafficcamnet",
    "vehiclemakenet",
    "vehicletypenet",
    "lpdnet",
    "lprnet",
}
```

- [ ] **Step 2: Расширить множество**

Через Edit заменить `old_string`:
```python
VALID_EXPERIMENTS = {
    "trafficcamnet",
    "vehiclemakenet",
    "vehicletypenet",
    "lpdnet",
    "lprnet",
}
```
на `new_string`:
```python
VALID_EXPERIMENTS = {
    "trafficcamnet",
    "vehiclemakenet",
    "vehicletypenet",
    "lpdnet",
    "lprnet",
    "nomeroff_lpd",
    "nomeroff_ocr",
}
```

- [ ] **Step 3: Smoke-проверка парсера**

Создать минимальный config-файл `/tmp/test-nomeroff.yaml`:
```yaml
servers:
  - host: dummy
    experiments:
      - nomeroff_lpd
      - nomeroff_ocr
remote:
  deploy_dir: ~/cars-eval
  venv: venv
  results_local: results_collected
```

Run: `python deploy/scripts/run_remote.py deploy --dry-run --config /tmp/test-nomeroff.yaml`

Expected: команды rsync/ssh печатаются без ошибки `invalid experiment ...`.

- [ ] **Step 4: Удалить временный конфиг**

```bash
rm /tmp/test-nomeroff.yaml
```

- [ ] **Step 5: Commit**

```bash
git add deploy/scripts/run_remote.py
git commit -m "feat(deploy): allow nomeroff_lpd/nomeroff_ocr experiment names

Расширение VALID_EXPERIMENTS перед обновлением remote_experiments.yaml."
```

---

## Task 3: Реализовать `eval_nomeroff_lpd`

**Files:**
- Modify: `deploy/evaluation/evaluate.py` — вставка после `eval_lpdnet` (около строки 478), перед секцией `# ─── LPRNet ───`.

- [ ] **Step 1: Найти точку вставки**

Run: `grep -n "# ─── LPRNet ───" deploy/evaluation/evaluate.py`

Expected: одна строка, ориентир ~481.

- [ ] **Step 2: Вставить секцию и функцию**

Через Edit найти `old_string`:
```python
# ─── LPRNet ──────────────────────────────────────────────────────────────────
```
заменить на `new_string`:
```python
# ─── Nomeroff LPD (RU plates) ────────────────────────────────────────────────

def eval_nomeroff_lpd(cfg: dict) -> dict:
    """Детекция номеров через nomeroff-net на VIA-датасете nomeroff_lp.
    Замена US-обученного LPDNet для RU-сегмента (см. eval_lpdnet выше)."""
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        from nomeroff_net import pipeline
    except ImportError as e:
        log.error(f"nomeroff-net not installed on this host: {e}")
        return {"error": "nomeroff-net not installed"}

    ann_file = next(data_dir.glob("**/val/via_region_data.json"), None)
    if ann_file is None:
        log.error(f"No val/via_region_data.json found in {data_dir}")
        return {"error": "dataset not found"}

    with open(ann_file) as f:
        ann_data = json.load(f)
    img_metadata = ann_data.get("_via_img_metadata", ann_data)
    img_root = ann_file.parent

    detector = pipeline("number_plate_localization", image_loader="opencv")

    predictions, ground_truths = [], []
    skipped = 0

    for img_key, img_info in tqdm(img_metadata.items(), desc="Nomeroff LPD"):
        img_name = img_info.get("filename", img_key)
        img_path = img_root / img_name
        if not img_path.exists():
            skipped += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        try:
            (images, bboxs, _points, _zones, _region_ids, _region_names,
             _count_lines, _confidences, _texts) = detector([str(img_path)])
            # bboxs: list per image, each [x1,y1,x2,y2,conf,cls] per detection
            preds_per_img = [[float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                              float(b[4]) if len(b) > 4 else 1.0]
                             for b in (bboxs[0] if bboxs else [])]
        except Exception as e:
            log.warning(f"nomeroff LPD failed on {img_name}: {e}")
            skipped += 1
            continue

        predictions.append({"image_id": img_name, "boxes": preds_per_img})

        gt_boxes = []
        regions = img_info.get("regions", [])
        if isinstance(regions, dict):
            regions = list(regions.values())
        for reg in regions:
            shape = reg.get("shape_attributes", {})
            if shape.get("name") == "rect":
                x, y = shape["x"], shape["y"]
                gt_boxes.append([x, y, x + shape["width"], y + shape["height"]])
            elif shape.get("name") == "polygon":
                xs, ys = shape["all_points_x"], shape["all_points_y"]
                gt_boxes.append([min(xs), min(ys), max(xs), max(ys)])
        ground_truths.append({"image_id": img_name, "boxes": gt_boxes})

    log.info(f"Nomeroff LPD: processed {len(predictions)} images, skipped {skipped}")

    m = compute_detection_metrics(predictions, ground_truths, conf_threshold=0.3)
    thresholds = {"recall": 0.80, "precision": 0.70}
    status = check_thresholds(m.to_dict(), thresholds)

    result = {"metrics": m.to_dict(), "thresholds": status,
              "skipped": skipped, "model": "nomeroff-net localization (RU)"}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info(f"Nomeroff LPD: P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")
    return result


# ─── LPRNet ──────────────────────────────────────────────────────────────────
```

> **Заметка:** распаковка результата `pipeline("number_plate_localization")` сделана с типичным набором возвращаемых значений `nomeroff-net` v3+ (используются неименованные slots). Точный кортеж API уточняется в Task 6 (smoke-test). Если на ssh9 API отличается — фиксится одной правкой здесь.

- [ ] **Step 3: Добавить запись в `EVAL_CONFIGS`**

Через Edit найти `old_string`:
```python
    "lpdnet": {
        "model_path": "models/lpdnet/LPDNet_usa_pruned_tao5.onnx",
        "data_dir": "data/nomeroff_lp",
        "results_dir": "results/lpdnet",
        "eval_fn": eval_lpdnet,
    },
```
заменить на `new_string`:
```python
    "lpdnet": {
        "model_path": "models/lpdnet/LPDNet_usa_pruned_tao5.onnx",
        "data_dir": "data/nomeroff_lp",
        "results_dir": "results/lpdnet",
        "eval_fn": eval_lpdnet,
    },
    "nomeroff_lpd": {
        "data_dir": "data/nomeroff_lp",
        "results_dir": "results/nomeroff_lpd",
        "eval_fn": eval_nomeroff_lpd,
    },
```

Поле `model_path` намеренно отсутствует — `nomeroff-net` сам тянет веса.

- [ ] **Step 4: Локальная syntax-проверка**

Run: `python -c "import ast; ast.parse(open('deploy/evaluation/evaluate.py').read())"`

Expected: без вывода (синтаксис валидный).

> ❌ Полный `import` модуля локально **запустить нельзя** — потребует `nomeroff_net`, которого нет в local venv. Это нормально: функция использует ленивый импорт внутри try/except.

- [ ] **Step 5: Commit**

```bash
git add deploy/evaluation/evaluate.py
git commit -m "feat(eval): add eval_nomeroff_lpd via nomeroff-net pipeline

RU-специфичная детекция номеров через пакет nomeroff-net. GT-формат
VIA унаследован от eval_lpdnet (тот же датасет data/nomeroff_lp).
Метрики через compute_detection_metrics (IoU>=0.5)."
```

---

## Task 4: Реализовать `eval_nomeroff_ocr`

**Files:**
- Modify: `deploy/evaluation/evaluate.py` — вставка после `eval_lprnet` (около строки 589), перед секцией `# ─── Main ───`.

- [ ] **Step 1: Найти точку вставки**

Run: `grep -n "# ─── Main ───" deploy/evaluation/evaluate.py`

Expected: одна строка, ориентир ~592.

- [ ] **Step 2: Вставить секцию и функцию**

Через Edit найти `old_string`:
```python
# ─── Main ─────────────────────────────────────────────────────────────────────
```
заменить на `new_string`:
```python
# ─── Nomeroff OCR (RU) ───────────────────────────────────────────────────────

def eval_nomeroff_ocr(cfg: dict) -> dict:
    """RU OCR через nomeroff-net на готовых кропах из data/nomeroff_ocr_ru.
    Замена US-обученного LPRNet для RU-сегмента (см. eval_lprnet выше)."""
    data_dir = ROOT / cfg["data_dir"]
    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        from nomeroff_net import pipeline
    except ImportError as e:
        log.error(f"nomeroff-net not installed on this host: {e}")
        return {"error": "nomeroff-net not installed"}

    val_img_dir = next(data_dir.glob("**/val/img"), None)
    val_ann_dir = next(data_dir.glob("**/val/ann"), None)
    if not val_img_dir or not val_ann_dir:
        log.error(f"Nomeroff OCR RU val/img + val/ann not found in {data_dir}")
        return {"error": "dataset not found"}

    items = []
    for ann_file in val_ann_dir.glob("*.json"):
        with open(ann_file) as f:
            data = json.load(f)
        stem = ann_file.stem
        img_path = next((p for p in val_img_dir.glob(f"{stem}.*")), None)
        if img_path is None:
            continue
        gt_text = (data.get("description") or data.get("name") or "").upper().strip()
        if gt_text:
            items.append({"img_path": img_path, "text": gt_text})

    log.info(f"Nomeroff OCR: loaded {len(items)} samples from {data_dir.name}")

    ocr = pipeline("number_plate_text_reading", image_loader="opencv",
                   presets={"ru": {"for_regions": ["ru"]}})

    predictions, ground_truths = [], []
    skipped = 0

    for item in tqdm(items, desc="Nomeroff OCR"):
        img_path = item["img_path"]
        gt_text = item["text"]
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        try:
            # OCR-пайплайн принимает уже кропнутые плашки — кладём img_path
            # в "zones" структуру, ожидаемую text_reading.
            (_images, _bboxs, _points, zones, region_ids, region_names,
             _count_lines, _confidences, texts) = ocr([str(img_path)])
            pred_text = (texts[0][0] if texts and texts[0] else "").upper().strip()
        except Exception as e:
            log.warning(f"nomeroff OCR failed on {img_path.name}: {e}")
            skipped += 1
            continue

        predictions.append({"image_id": img_path.name, "text": pred_text})
        ground_truths.append({"image_id": img_path.name, "text": gt_text})

    log.info(f"Nomeroff OCR: processed {len(predictions)} samples, skipped {skipped}")

    m = compute_ocr_metrics(predictions, ground_truths)
    thresholds = {"char_accuracy": 0.90, "full_plate_accuracy": 0.80}
    status = check_thresholds(m.to_dict(), thresholds)

    result = {"metrics": m.to_dict(), "thresholds": status,
              "skipped": skipped, "model": "nomeroff-net text_reading (RU)"}
    (results_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log.info(f"Nomeroff OCR: CharAcc={m.char_accuracy:.3f} PlateAcc={m.full_plate_accuracy:.3f}")
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────
```

> **Заметки:**
> - Преобразование `RU → Latin` (`normalize_ru_plate`) здесь НЕ применяется: Nomeroff OCR обучен на кириллице и должен возвращать кириллические символы. Сравнение строк идёт «как есть» после `.upper().strip()`.
> - Если на ssh9 OCR-pipeline-API возвращает `texts` другой формы (например, плоский список вместо list-of-list) — фиксится правкой строки `pred_text = ...` в Task 6.
> - `presets` для `ru` — стандартный аргумент `nomeroff-net`. Если ключ называется иначе (`for_regions` на верхнем уровне) — корректируется в Task 6.

- [ ] **Step 3: Добавить запись в `EVAL_CONFIGS`**

Через Edit найти `old_string`:
```python
    "lprnet": {
        "model_path": "models/lprnet/us_lprnet_baseline18_deployable.onnx",
        "data_dir": "data/nomeroff_ocr_ru",
        "results_dir": "results/lprnet",
        "eval_fn": eval_lprnet,
    },
}
```
заменить на `new_string`:
```python
    "lprnet": {
        "model_path": "models/lprnet/us_lprnet_baseline18_deployable.onnx",
        "data_dir": "data/nomeroff_ocr_ru",
        "results_dir": "results/lprnet",
        "eval_fn": eval_lprnet,
    },
    "nomeroff_ocr": {
        "data_dir": "data/nomeroff_ocr_ru",
        "results_dir": "results/nomeroff_ocr",
        "eval_fn": eval_nomeroff_ocr,
    },
}
```

- [ ] **Step 4: Syntax-проверка**

Run: `python -c "import ast; ast.parse(open('deploy/evaluation/evaluate.py').read())"`

Expected: без вывода.

- [ ] **Step 5: Commit**

```bash
git add deploy/evaluation/evaluate.py
git commit -m "feat(eval): add eval_nomeroff_ocr via nomeroff-net pipeline

RU-специфичный OCR номеров через пакет nomeroff-net. Датасет
data/nomeroff_ocr_ru (val/img + val/ann), формат уже кропнутых
плашек. Метрики через compute_ocr_metrics."
```

---

## Task 5: Переключить активную кампанию на новые эксперименты

**Files:**
- Modify: `configs/remote_experiments.yaml`

- [ ] **Step 1: Прочитать текущий конфиг**

Run: `cat configs/remote_experiments.yaml`

Ожидается: блок `servers:` со списком хостов, на ssh9 (gpu-server-2 в шаблоне, либо реальный `ssh9.qudata.ai`) — `lpdnet`/`lprnet`.

> ❗ Если в файле сейчас стоят шаблонные `gpu-server-1`/`gpu-server-2`, а не реальные `ssh1.qudata.ai`/`ssh9.qudata.ai` — это означает, что фактический конфиг не закоммичен (см. `git status` в начале сессии: `M configs/remote_experiments.yaml`). Прочитать **актуальное** содержимое (включая uncommitted изменения) и редактировать его.

- [ ] **Step 2: Заменить эксперименты на ssh9**

В блоке хоста, где сейчас прописаны `lpdnet` и `lprnet`, заменить:

```yaml
      - lpdnet
      - lprnet
```

на:

```yaml
      - nomeroff_lpd
      - nomeroff_ocr
```

- [ ] **Step 3: Локальный dry-run**

Run: `python deploy/scripts/run_remote.py deploy --dry-run`

Expected: rsync/ssh-команды печатаются для обоих хостов; для ssh9 фигурируют `nomeroff_lpd nomeroff_ocr`, никаких ошибок `invalid experiment`.

- [ ] **Step 4: Commit**

```bash
git add configs/remote_experiments.yaml
git commit -m "chore(campaign): switch ssh9 to nomeroff_lpd/nomeroff_ocr

Активная RU-кампания переходит на nomeroff-net. Legacy lpdnet/lprnet
остаются в EVAL_CONFIGS как baseline, но из кампании исключены.
Baseline метрики зафиксированы в deferred-work.md."
```

---

## Task 6: Smoke-test на ssh9 — импорт и одиночный inference

**Files:** (без правок в репо; временный скрипт на удалёнке)

- [ ] **Step 1: Развернуть код на ssh9 в режиме setup**

Run: `python deploy/scripts/run_remote.py deploy`

Expected: на ssh9 синхронизируется код, `setup_server.sh` устанавливает `deploy/requirements.txt` (включая `nomeroff-net`).

> Если `setup_server.sh` валится на установке `nomeroff-net` (например, на сборке `craft-text-detector` или `ultralytics`) — фиксить отдельной задачей: обычно нужен `apt install` системных libGL/libglib (если не stock) или явный pin совместимой версии. **Не маскировать через `--ignore-installed`** — это симптом несовместимости.

- [ ] **Step 2: SSH в ssh9 и проверка импорта**

Run (на ssh9 вручную или через одиночный ssh):
```bash
ssh ssh9.qudata.ai 'cd ~/cars-eval && ./venv/bin/python -c "from nomeroff_net import pipeline; p = pipeline(\"number_plate_localization\", image_loader=\"opencv\"); print(\"OK\", type(p).__name__)"'
```

Expected: `OK <ClassName>` без traceback. Первый запуск скачает веса (~2–3 мин).

- [ ] **Step 3: Прогон одного изображения**

Run (на ssh9):
```bash
ssh ssh9.qudata.ai 'cd ~/cars-eval && ./venv/bin/python -c "
import json
from pathlib import Path
from nomeroff_net import pipeline
det = pipeline(\"number_plate_localization\", image_loader=\"opencv\")
sample = next(Path(\"data/nomeroff_lp\").glob(\"**/val/*.png\"))
print(\"sample:\", sample)
out = det([str(sample)])
print(\"len(out):\", len(out))
print(\"out[1] (bboxs):\", out[1])
"'
```

Expected: печатается путь к sample, длина возвращаемого кортежа (обычно 9), и непустой список bboxes.

> **Если кортеж имеет другую длину или порядок** — это сигнал, что API в реальной версии отличается от предположенного в Task 3/4. Возврат к Task 3/4: правка распаковки `(images, bboxs, ...) = ...` на корректный сигнатурный шаблон актуальной версии `nomeroff-net`, новый commit с пояснением.

- [ ] **Step 4: Аналогично для OCR**

Run (на ssh9):
```bash
ssh ssh9.qudata.ai 'cd ~/cars-eval && ./venv/bin/python -c "
from pathlib import Path
from nomeroff_net import pipeline
ocr = pipeline(\"number_plate_text_reading\", image_loader=\"opencv\")
sample = next(Path(\"data/nomeroff_ocr_ru\").glob(\"**/val/img/*.png\"))
print(\"sample:\", sample)
out = ocr([str(sample)])
print(\"texts:\", out[-1])
"'
```

Expected: печатается путь и непустой OCR-результат.

> Если структура `out[-1]` (texts) — не `list-of-list-of-str` — корректируем `pred_text = ...` в Task 4 и пересобираем.

- [ ] **Step 5: Логирование результата smoke-test (без commit)**

Записать в `_bmad-output/implementation-artifacts/deferred-work.md` рабочую заметку (в новой секции «Goal 4 — RU model swap (in progress)»):
- версия `nomeroff-net` (`./venv/bin/pip show nomeroff-net | grep Version` на ssh9)
- любые правки API-сигнатур, если потребовались

(commit этой заметки — в Task 8 одним коммитом со всеми результатами.)

---

## Task 7: Полный прогон + сбор результатов

**Files:** (без правок в репо)

- [ ] **Step 1: Запуск кампании**

Run:
```bash
python deploy/scripts/run_remote.py run
```

Expected: на ssh9 фоном (nohup) стартует `evaluate.py --models nomeroff_lpd nomeroff_ocr`. Команда возвращает управление сразу (fire-and-forget).

- [ ] **Step 2: Мониторинг прогресса**

Run:
```bash
ssh ssh9.qudata.ai 'tail -F ~/cars-eval/logs/eval_*.log'
```

Expected: tqdm-прогресс по `Nomeroff LPD` и затем `Nomeroff OCR`, в конце — строки вида `Nomeroff LPD: P=... R=... F1=...` и `Nomeroff OCR: CharAcc=... PlateAcc=...`. Ctrl-C для выхода из `tail`.

- [ ] **Step 3: Сбор результатов**

Когда обе модели завершились (нет активных python-процессов: `ssh ssh9.qudata.ai 'pgrep -af evaluate.py'` пусто):

Run:
```bash
python deploy/scripts/run_remote.py collect
```

Expected: `results_collected/ssh9.qudata.ai/results/nomeroff_lpd/metrics.json` и `.../nomeroff_ocr/metrics.json` появляются локально.

- [ ] **Step 4: Чтение метрик**

Run:
```bash
cat results_collected/ssh9.qudata.ai/results/nomeroff_lpd/metrics.json
cat results_collected/ssh9.qudata.ai/results/nomeroff_ocr/metrics.json
```

Expected: JSON с непустыми `precision/recall/f1` для LPD и `char_accuracy/full_plate_accuracy` для OCR. Числовая ожидаемая планка: оба значения **существенно лучше** US baseline (`R=0.296`, `PlateAcc=0.062`).

> Если какое-то значение оказалось **хуже** baseline или близко к нулю — диагностика: (a) сверить распаковку API в Task 3/4, (b) проверить, что предсказания не пустые (`len(predictions) > 0` в логе), (c) убедиться, что для OCR ground-truth действительно на кириллице (а не транслите) и Nomeroff отдаёт строки в том же алфавите.

---

## Task 8: Обновить `deferred-work.md` со сравнительной таблицей

**Files:**
- Modify: `_bmad-output/implementation-artifacts/deferred-work.md`

- [ ] **Step 1: Прочитать текущий файл**

Run: `cat _bmad-output/implementation-artifacts/deferred-work.md`

- [ ] **Step 2: Добавить новую секцию**

Через Edit вставить перед строкой `## Источник решения` (это секция в конце файла) новый блок:

```markdown
## Goal 4 — RU model swap (DONE 2026-05-17)

**Контекст:** US-обученные LPDNet/LPRNet давали `R=0.296` / `PlateAcc=0.062` на RU-номерах (см. таблицу в Goal 2). Произведена замена на модели проекта [nomeroff.net.ua](https://nomeroff.net.ua/models/) (пакет `nomeroff-net`) для RU-сегмента.

**Реализация:** новые эксперименты `nomeroff_lpd` и `nomeroff_ocr` в `deploy/evaluation/evaluate.py` (`eval_nomeroff_lpd`, `eval_nomeroff_ocr`), `nomeroff-net` добавлен в `deploy/requirements.txt`. Кампания на ssh9 переключена; legacy `lpdnet`/`lprnet` остались в `EVAL_CONFIGS` для baseline-сравнения.

**Версия nomeroff-net:** `<X.Y.Z>` (заполнить из smoke-test, Task 6).

**Сравнение метрик (RU-сегмент):**

| Model | Detection / OCR | US baseline | Nomeroff RU | Delta |
|---|---|---|---|---|
| LPD (детекция) | P / R / F1 | 0.884 / 0.296 / 0.443 | `<P> / <R> / <F1>` | `<...>` |
| OCR | CharAcc / PlateAcc | 0.590 / 0.062 | `<CharAcc> / <PlateAcc>` | `<...>` |

**Спека / план:** `spec-nomeroff-ru-swap.md`, `plan-nomeroff-ru-swap.md`.

---
```

Замените `<X.Y.Z>` и `<...>`-плейсхолдеры реальными числами из `results_collected/ssh9.qudata.ai/results/nomeroff_*/metrics.json`.

- [ ] **Step 3: Commit**

```bash
git add _bmad-output/implementation-artifacts/deferred-work.md
git add results_collected/ssh9.qudata.ai/results/nomeroff_lpd/ \
        results_collected/ssh9.qudata.ai/results/nomeroff_ocr/
git commit -m "docs(bmad): RU model swap — Nomeroff vs US baseline

Финальная сводка по Goal 4: замена US-обученных LPDNet/LPRNet на
nomeroff-net пайплайн для RU-номеров. Сравнительная таблица US vs
nomeroff в deferred-work.md, метрики в results_collected/."
```

> Сами `metrics.json` коммитятся как фрозен-референс (как и прошлые `results_collected/qudata2/`, см. CLAUDE.md «Do commit ... when they represent a frozen reference»).

---

## Definition of Done (сводно)

- ✅ Task 1–5: код и конфиги залиты, локальный `--dry-run` принимает новые имена.
- ✅ Task 6: на ssh9 `nomeroff-net` импортируется, одиночный inference работает на LPD и OCR.
- ✅ Task 7: оба `metrics.json` появились в `results_collected/ssh9.qudata.ai/results/nomeroff_*/`, числа лучше US baseline.
- ✅ Task 8: `deferred-work.md` содержит сравнительную таблицу, всё закоммичено.

---

## Self-review (выполнен автором плана)

**Spec coverage** ✅
- Все 5 файлов из спеки покрыты (Task 1–5, 8).
- Smoke-test (спека → Тестирование) → Task 6.
- Полный прогон + артефакты (спека → Артефакты) → Task 7.
- Обновление deferred-work (спека → Definition of Done пункт 5) → Task 8.

**Расхождения со спекой, зафиксированные в плане**
1. Реестр называется `EVAL_CONFIGS`, не `MODEL_REGISTRY` — все правки идут в `EVAL_CONFIGS`.
2. Датасет для `nomeroff_lpd` — `data/nomeroff_lp` (VIA full-frame), а не `data/nomeroff_ocr_ru` (кропы), как было упрощённо в спеке. Это согласовано с тем, как сейчас работает `eval_lpdnet`.
3. Подход к тестам: TDD-цикл не применим (в проекте нет pytest-инфраструктуры для эвалюаторов). Гейт — smoke-test + сравнение метрик с baseline. Это явно отражено в Task 6/7.

**Placeholder scan** ✅
- Реальные числа метрик в Task 8 — намеренные плейсхолдеры (`<X.Y.Z>`, `<P> / <R> / <F1>`), заполняются по факту прогона. Это не нарушение «no placeholders» — данные физически появятся только после Task 7.
- API-сигнатуры `nomeroff-net` помечены как «уточняется в Task 6» с явным fallback-планом — это сознательная отсрочка, а не TBD.

**Type/name consistency** ✅
- `eval_nomeroff_lpd` / `eval_nomeroff_ocr` — одинаково везде.
- `nomeroff_lpd` / `nomeroff_ocr` — одинаково в `EVAL_CONFIGS`, `VALID_EXPERIMENTS`, `remote_experiments.yaml`.
- `compute_detection_metrics` / `compute_ocr_metrics` — существующие сигнатуры из `metrics.py`, проверено по `eval_lpdnet`/`eval_lprnet`.
