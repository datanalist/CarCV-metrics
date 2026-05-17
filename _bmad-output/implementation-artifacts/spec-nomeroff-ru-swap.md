# Spec — Замена LPDNet/LPRNet на Nomeroff.net (RU-сегмент)

**Дата:** 2026-05-17
**Контекст:** Goal 4 из `deferred-work.md` — пункт «оценить целесообразность RU-моделей (NTechLab/nomeroff_lpd)».
**Триггер:** US-обученные NVIDIA TAO модели на RU-номерах дают `LPDNet R=0.296` и `LPRNet PlateAcc=0.062` (задокументированный domain gap).

## Цель

Заменить в кампании пайплайн «детекция номеров + OCR» с US-моделей (LPDNet/LPRNet) на модели проекта [Nomeroff.net](https://nomeroff.net.ua/models/), специализированные на RU/UA/EU-номерах, и получить метрики, сопоставимые с реальной production-применимостью для RU-сегмента.

## Из объёма исключено (YAGNI)

- UA/EU/KZ/BY и прочие регионы — только `ru`.
- Локальная установка `nomeroff-net` в `pyproject.toml` — пакет нужен только на remote-сервере.
- Удаление US-эвалюаторов (`eval_lpdnet`, `eval_lprnet`) из `evaluate.py` — остаются как baseline для будущих сравнений; исключаются только из `configs/remote_experiments.yaml`.
- Новые hydra-конфиги в `configs/experiment/` — у Nomeroff собственная предобработка внутри пакета, выносить наружу нечего.
- End-to-end метрика (плёночный pipeline detect+OCR одним числом) — детекцию и OCR меряем раздельно, как `eval_lpdnet`/`eval_lprnet` сейчас.

## Архитектурные решения

### Подход

**B (выбран):** интегрировать PyPI-пакет `nomeroff-net` как inference-движок. PyTorch уже предустановлен на remote-серверах (см. `project_remote_servers` memory), пакет встаёт штатным `pip install` через `deploy/requirements.txt`.

Альтернатива (A — только ONNX-веса из релизов Nomeroff + ручная реализация NMS / CTC-декодера) отклонена: высокий риск ошибок в постпроцессинге YOLO-выхода и расхождения с reference-реализацией.

### Именование

Новые эксперименты добавляются с префиксом `nomeroff_`:
- `nomeroff_lpd` — детекция номеров (соответствует `lpdnet`)
- `nomeroff_ocr` — RU OCR (соответствует `lprnet`)

Старые `lpdnet`/`lprnet` остаются валидными значениями в `VALID_EXPERIMENTS` и `MODEL_REGISTRY`, но **исключаются из активной кампании** (`configs/remote_experiments.yaml`).

### Целевой хост

`ssh9.qudata.ai:19478` (RTX 5090, 32 CPU, 30GB RAM, 46GB free) — тот же, где сейчас выполняются `lpdnet`/`lprnet`. Дисковый бюджет: ~500MB пакетных зависимостей + веса моделей ~200MB укладываются.

## Затронутые файлы

| Файл | Изменение |
|---|---|
| `deploy/evaluation/evaluate.py` | Добавить `eval_nomeroff_lpd(cfg)` и `eval_nomeroff_ocr(cfg)` с секционными баннерами в существующем стиле. Зарегистрировать в `MODEL_REGISTRY` под ключами `nomeroff_lpd`, `nomeroff_ocr`. |
| `deploy/requirements.txt` | Добавить `nomeroff-net` (без указания версии — пакет молодой, фиксируем актуальный мажор после первой сборки). |
| `configs/remote_experiments.yaml` | На хосте ssh9 заменить `lpdnet`, `lprnet` → `nomeroff_lpd`, `nomeroff_ocr`. |
| `deploy/scripts/run_remote.py` | В `VALID_EXPERIMENTS` добавить `"nomeroff_lpd"` и `"nomeroff_ocr"`. |
| `_bmad-output/implementation-artifacts/deferred-work.md` | После прогона дописать секцию «Goal 4 — RU model swap» с новыми метриками против US baseline. |

`pyproject.toml` **не трогаем** — пакет нужен только на remote.

## Контракты функций

```python
def eval_nomeroff_lpd(cfg: dict) -> dict:
    """Детекция номеров через nomeroff-net на data/nomeroff_ocr_ru.
    Возвращает {precision, recall, f1, num_images, num_gt, num_pred, ...}.
    Записывает results/nomeroff_lpd/metrics.json через те же помощники,
    что и eval_lpdnet (compute_detection_metrics, IoU >= 0.5)."""

def eval_nomeroff_ocr(cfg: dict) -> dict:
    """RU OCR через nomeroff-net на ground-truth-кропах из val/ann/*.json.
    Возвращает {char_accuracy, full_plate_accuracy, num_samples, ...}.
    Записывает results/nomeroff_ocr/metrics.json через compute_ocr_metrics."""
```

## MODEL_REGISTRY entries

```python
"nomeroff_lpd": {
    "data_dir": "data/nomeroff_ocr_ru",
    "results_dir": "results/nomeroff_lpd",
    "eval_fn": eval_nomeroff_lpd,
    # model_path не нужен — веса тянет сам nomeroff-net
},
"nomeroff_ocr": {
    "data_dir": "data/nomeroff_ocr_ru",
    "results_dir": "results/nomeroff_ocr",
    "eval_fn": eval_nomeroff_ocr,
    "region": "ru",
},
```

## Поток данных

**`nomeroff_lpd`:**
1. Итерируемся по `data/nomeroff_ocr_ru/val/img/*.png`.
2. Для каждого изображения: парсим парный `val/ann/<stem>.json`, извлекаем GT bbox-ы (`box`/`box2d` — читаем оба ключа, см. правило для BDD).
3. Прогон через `nomeroff_net.pipelines.number_plate_localization` (или эквивалент актуальной версии API).
4. Сопоставление pred ↔ GT по IoU≥0.5, агрегация через `compute_detection_metrics(...)` → `DetectionMetrics`.
5. Запись `metrics.json`.

**`nomeroff_ocr`:**
1. Итерируемся по тем же `val/ann/*.json`, для каждого аннотируемого номера: кропим из исходного изображения по GT bbox.
2. Прогон через `nomeroff_net.pipelines.number_plate_text_reading` с пресетом `ru`.
3. Нормализация строк: `.upper()`, оставляем `[A-Z0-9А-Я]`, убираем разделители.
4. `compute_ocr_metrics(predictions, ground_truths)` → `OCRMetrics`.
5. Запись `metrics.json`.

## Предобработка / постобработка

Полностью внутри пакета `nomeroff-net`. В отличие от ONNX-моделей (где предобработка прописана в hydra YAML), для `nomeroff-net` это **сознательный** отход от правила «preprocessing constants live in YAML»: пакет инкапсулирует свой пайплайн, и попытка вынести его параметры наружу нарушит интерфейс пакета. Документируется здесь, чтобы в `CLAUDE.md` потом отразить как исключение.

## Обработка ошибок

- Импорт `nomeroff_net` оборачивается в try/except внутри `eval_*`. При `ImportError` — `log.error("nomeroff-net not installed: ...")` и `return {}`. Соседние эксперименты не падают (соответствует текущему поведению `evaluate.py`).
- Per-image failures: `try/except` вокруг inference, лог + счётчик `skipped` (паттерн `skipped_oo_dist` из `eval_vehiclemakenet`), исключение из знаменателя.
- Отсутствующая аннотация / пустой `description`: `continue`, инкремент `skipped`.
- Битые изображения (`cv2.imread is None`): `continue`, инкремент `skipped`.

## Веса моделей

`nomeroff-net` по умолчанию тянет веса с GitHub releases при первом импорте/inference и кэширует их в `~/.cache/nomeroff-net/` (или аналогичный путь). Это значит:
- Первый прогон будет дольше обычного (~+1–3 мин на скачивание).
- На shared-хосте `/dev/shm/<bundle>` веса НЕ кэшируются — кэш живёт в `~/`, что соответствует CLAUDE.md (`SSH-keys и кэши на пользовательской стороне, не в bundle`).
- Если нужно ускорить — добавить предзагрузку в `setup_server.sh` отдельной задачей; в первоначальный объём не входит.

## Тестирование

- Smoke-test на ssh9 перед полным прогоном: маленький Python-snippet, импортирующий `nomeroff_net` и прогоняющий 1 изображение — подтверждает, что torch + GPU + модели стянулись.
- Unit-тестов в `tests/` — нет (правило: «no formal test suite, the dataset is the test fixture»).
- Validation gate: после прогона `nomeroff_lpd` должен выдать `R >> 0.296`, `nomeroff_ocr` — `PlateAcc >> 0.062`. Числовые пороги фиксируем после первого baseline-прогона, не до.

## Артефакты

После успешного прогона:
- `results_collected/ssh9.qudata.ai/results/nomeroff_lpd/metrics.json`
- `results_collected/ssh9.qudata.ai/results/nomeroff_ocr/metrics.json`
- Опционально: `plots/nomeroff_*.png` через `visualize.py` (отдельная задача после первичного baseline)
- Запись в `deferred-work.md` с таблицей «US baseline vs Nomeroff RU»

## Open questions для уточнения на этапе плана

- Точное имя API в актуальной версии `nomeroff-net` — выбор между `Detector`/`pipelines.number_plate_*`/`number_plate_detection_and_reading` фиксируется при первом импорте на ssh9.
- Формат `val/ann/*.json` — структура верифицируется чтением одного файла на ssh9 (там же `description` для GT-текста).

## Definition of Done

1. ✅ Все 5 файлов отредактированы по таблице выше.
2. ✅ `python deploy/scripts/run_remote.py deploy --dry-run` принимает `nomeroff_lpd`/`nomeroff_ocr` без ошибок.
3. ✅ На ssh9 импорт `nomeroff_net` проходит (smoke-test).
4. ✅ Полный прогон обеих моделей завершается с `metrics.json` в `results_collected/ssh9.qudata.ai/results/nomeroff_*/`.
5. ✅ `deferred-work.md` обновлён со сравнительной таблицей.
6. ✅ Commit + (по запросу) PR в `chore/bmad-deploy-artifacts`.
