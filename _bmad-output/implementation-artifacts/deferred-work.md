# Deferred Work

Откладывается из текущей итерации (split из QQ-цикла `validation-campaign-2026-05-17`).

---

## Goal 2 — Campaign config + execution (DONE 2026-05-17)

**Хосты (фактические):**
- ssh1.qudata.ai:11598 — RTX 4090, 64 CPU, 251GB RAM, 46GB free disk → `trafficcamnet`, `vehiclemakenet`
- ssh9.qudata.ai:19478 — RTX 5090, 32 CPU, 30GB RAM, 46GB free disk → `lpdnet`, `lprnet`

**Прогон `deploy → run → collect` выполнен.** Результаты в `results_collected/{ssh1,ssh9}.qudata.ai/results/*/metrics.json` + plot `vehiclemakenet_per_class.png`. Все 4 запущенные модели завершились без runtime-ошибок; низкие метрики — semantic (data/model mismatch), не execution.

**Сводка метрик (этап 1):**
| Model | Dataset | Key metric | Note |
|---|---|---|---|
| TrafficCamNet | COCO val2017 5K (как BDD-формат) | P=0.167, R=0.032, F1=0.053 (FAIL) | class-mapping mismatch: COCO id 1/2/3 vs TrafficCamNet 4-class — нужен анализ |
| VehicleMakeNet | mad-cars 4960 | Top1=0.002 (FAIL) | per_class сводится к 1 марке («acura») — label-mapping mad-cars↔NGC сломан |
| LPDNet (USA) | nomeroff LP ~6K | P=0.884 **PASS**, R=0.296 FAIL | USA-модель на RU plates: высокая precision, низкий recall |
| LPRNet (US) | nomeroff OCR RU 4893 | CharAcc=0.590, PlateAcc=0.062 (FAIL) | US Latin model on RU Cyrillic — domain gap отмечен в evaluate.py |

**Исключено из кампании:**
- **vehicletypenet** — `BIT-Vehicle` требует Kaggle auth, fallback URL (GitHub Curig7) → 404. Восстановление: либо `KAGGLE_USERNAME/KAGGLE_KEY` env, либо переход на Stanford Cars + Make→Type mapping per §6.5 (требует переписать `eval_vehicletypenet` в `evaluate.py`).

**Divergence от §6.5:** Текущие скрипты (`download_datasets_qudata2_v3.sh`, `download_datasets_qudata5.sh`) используют COCO+nomeroff+mad-cars/BIT-Vehicle вместо BDD100K+Stanford+Kaggle car-plates+AY000554, что и объясняет часть низких метрик. Полный переход на §6.5 — отдельный кусок работы (требует синхронной правки `evaluate.py` и data-loaders).

---

## Goal 3 — Aggregation: SUMMARY.md generator + post-processing notebook (partial DONE 2026-05-17)

**Зачем:** Автоматизировать сборку `results/SUMMARY.md` (правило из `CLAUDE.md`) и предоставить notebook для пост-обработки.

**DONE (MVP):**
- ✅ `deploy/evaluation/aggregate_summary.py` — CLI-агрегатор: обходит `results_collected/**/metrics.json` (поддерживает обе схемы — `qudata2/<model>/` и `<host>/results/<model>/`), классифицирует семейство по ключам метрик, пишет `results/SUMMARY.md` с тремя секциями (Overall table, US vs Nomeroff comparison, Detailed Results).
- ✅ `notebooks/post_processing.ipynb` — воспроизводимый notebook (`uv run python -m jupyter nbconvert --execute …` проходит без ошибок). Генерирует 5 PNG в `plots/`: `pass_fail_by_family`, `detection_metrics`, `ocr_metrics`, `classification_metrics`, `us_vs_nomeroff`.

**Deferred sub-items:**
- ⏳ **PR-curves / confusion matrices / per-image error breakdown.** Требует расширения `evaluate.py`: писать per-sample predictions (например, `results/<model>/predictions.parquet` с колонками `image_id, gt, pred, score`) под флагом `--save-predictions`. После этого notebook расширяется отдельной итерацией.
- ⏳ **Сегментация по `weather × timeofday × scene` (BDD100K).** Не применимо сейчас — TrafficCamNet прогонялся на COCO val2017, BDD-метаданных нет. Зависит от **Goal 4 §6.5 dataset migration** (переход на BDD100K val 10K).

**Зависит от:** Goal 2 ✅ (результаты в `results_collected/{ssh1,ssh9}.qudata.ai/results/`).

---

## Goal 4 — §6.5 dataset migration + class-mapping fixes (CODE READY 2026-05-17)

**Зачем:** Полученные в Goal 2 низкие метрики (TrafficCamNet F1=0.053, VehicleMakeNet Top1=0.002) — следствие data/label mismatch, не сломанной модели. Реальная валидация требует перехода на §6.5 шорт-лист.

**Перенос инфраструктуры (2026-05-17):** ssh1.qudata.ai стал недоступен → весь Goal 4 кампания перенесена на ssh9.qudata.ai (`configs/remote_experiments.yaml`).

**Status:**
- ✅ **VehicleMakeNet bug fix.** Корневая причина "acura collapse" найдена: `normalize_brand('')` возвращал "acura", потому что `"" in "acura"` всегда True → все 4960 семплов получали GT "acura". Фикс: явная проверка длины (≥3) + exact-match приоритет. Regression-tests в `tests/test_normalize_brand.py` (14/14 PASS).
- ✅ **TrafficCamNet multi-class evaluation.** Расширен `eval_trafficcamnet` до 4-классовой оценки (car/person/bicycle/road_sign), `conf_thr` параметризован через cfg (default 0.2 под cross-domain), добавлены `per_class` + `macro` метрики. BDD100K val 10K — опциональный path через `BDD100K_HF_REPO` env var; primary остаётся COCO val2017 (4-class mapping: COCO 1/2/3/13 → person/bicycle/car/traffic sign).
- ✅ **VehicleTypeNet миграция на Stanford Cars.** Переписан `eval_vehicletypenet`: вместо BIT-Vehicle (Kaggle-blocked) использует `data/stanford_cars/test.json` + автогенерируемый mapping body-type через `STANFORD_BODY_KEYWORDS` (sedan/coupe/suv/van/truck — `derive_typenet_label`). Audit-таблица записывается в `data/stanford_cars/type_mapping.csv` при каждом запуске (§7.5 reproducibility). Regression-tests в `tests/test_typenet_mapping.py` (17/17 PASS). BIT-Vehicle layout сохранён как fallback для обратной совместимости.
- ✅ **LPDNet/LPRNet RU swap** — закрыто Goal 5 (nomeroff-net).

**Доставка на ssh9 (deliverable артефакты):**
- `deploy/evaluation/evaluate.py` — фиксы + новые маппинги
- `deploy/scripts/download_datasets_ssh9_goal4.sh` — объединённый download (COCO/BDD100K + mad-cars + Stanford Cars через HF)
- `deploy/scripts/setup_server.sh` — добавил `stanford_cars`, `coco` каталоги
- `configs/remote_experiments.yaml` — ssh9-only, все 5 моделей
- `tests/test_normalize_brand.py`, `tests/test_typenet_mapping.py` — regression-tests

**Действия для прогона на ssh9** (выполняет пользователь — нет SSH-доступа из текущего окружения):
1. `python deploy/scripts/run_remote.py deploy` — синхронизирует код на ssh9 + setup_server.sh.
2. `ssh ssh9.qudata.ai 'cd cars-eval && bash deploy/scripts/download_datasets_ssh9_goal4.sh'` — скачать датасеты (опционально `BDD100K_HF_REPO=<hf-mirror>` для замены COCO на BDD100K).
3. `python deploy/scripts/run_remote.py run` — запустить параллельно все 5 экспериментов в фоне.
4. `python deploy/scripts/run_remote.py collect` — забрать results+plots в `results_collected/ssh9.qudata.ai/`.
5. Запустить `notebooks/post_processing.ipynb` для обновлённых графиков; `deploy/evaluation/aggregate_summary.py` для SUMMARY.md.

**Каveat (Honest Reporting):**
- TrafficCamNet всё ещё может FAIL на COCO val2017 даже с multi-class evaluation — domain gap (street-level vs traffic-cam top-down) остаётся. Для перехода на in-domain BDD100K val 10K требуется HF mirror с известным repo ID (placeholder в скрипте).
- VehicleTypeNet на Stanford Cars: catalog-photos vs road-scene — domain gap. Также распределение body-types в Stanford смещено к sedan/coupe.

**Зависит от:** Goal 2 ✅.

---

## Goal 5 — RU model swap: LPDNet/LPRNet → nomeroff-net (DONE 2026-05-17)

**Контекст:** US-обученные LPDNet/LPRNet давали `R=0.296` / `PlateAcc=0.062` на RU-номерах (Goal 2). Произведена замена на пакет `nomeroff-net` для RU-сегмента. Закрывает в Goal 4 пункт «LPDNet/LPRNet RU swap».

**Реализация:** новые эксперименты `nomeroff_lpd` и `nomeroff_ocr` в `deploy/evaluation/evaluate.py` (`eval_nomeroff_lpd`, `eval_nomeroff_ocr`), `nomeroff-net` в `deploy/requirements.txt`. Кампания на ssh9 (`configs/remote_experiments.yaml`) переключена; legacy `lpdnet`/`lprnet` остаются в `EVAL_CONFIGS` как baseline.

**Версия nomeroff-net (ssh9):** `4.0.1`. **GPU:** NVIDIA RTX 5090. **Время прогона:** LPD 21.2s, OCR 25.6s (параллельно).

**Сравнительная таблица — US baseline vs Nomeroff (RU, ssh9):**

| Model | Metric | US baseline | Nomeroff RU | Delta | Threshold |
|---|---|---:|---:|---:|---|
| LPD | Precision | 0.884 | **0.906** | +0.022 | ≥0.70 PASS |
| LPD | Recall    | 0.296 | **0.922** | **+0.626** | ≥0.80 PASS |
| LPD | F1        | 0.444 | **0.914** | **+0.470** | – |
| LPD | AP / mAP@0.5 | 0.262 | **0.881** | +0.619 | – |
| LPD | TP / GT / Pred | 114 / 385 / 129 | 355 / 385 / 392 | – | – |
| OCR | Char accuracy | 0.590 | **0.9995** | **+0.410** | ≥0.90 PASS |
| OCR | Full-plate accuracy | 0.062 | **0.9978** | **+0.936** | ≥0.80 PASS |
| OCR | Char error rate | 0.410 | **0.0005** | −0.410 | – |
| OCR | Samples processed | 4893 | 4893 | – | skipped 0 |

**Caveat по интерпретации:** датасеты `data/nomeroff_lp` (autoriaNumberplateDataset-2018-11-20) и `data/nomeroff_ocr_ru` (autoriaNumberplateOcrRu-2021-09-01) — собственный корпус проекта nomeroff.net.ua, использованный для **обучения** этих же моделей. Числа отражают «пакет работает на своём бенчмарке», а не независимую генерализацию. Для честной валидации требуется отдельный held-out RU-датасет (см. Goal 4 — §6.5 dataset migration). Тем не менее delta vs US baseline на одинаковых данных корректна.

**API-расхождения от спеки (зафиксированы при smoke-test, plan Task 6):**
- `pipeline("number_plate_localization")` возвращает 2 слота через `nomeroff_unzip(raw)`: `(images_bboxs, images)`. Каждый bbox — `[x1, y1, x2, y2, conf, cls_int, kps_normalized]` (НЕ 9-слотный кортеж, как в спеке).
- Для OCR на уже кропнутых плашках используется `NumberPlateTextReading` + `DumpyImageLoader` напрямую (вход: `(zone_array, region_label, count_lines, preprocessed_np)`). `pipeline("number_plate_detection_and_reading")` гонял бы YOLO повторно поверх кропов и портил результат.
- PyTorch ≥ 2.6 изменил дефолт `weights_only=True` в `torch.load`; при инициализации OCR pipeline применяется временный monkey-patch (НЕ thread-safe — `eval_nomeroff_ocr` нельзя запускать параллельно с другим кодом, вызывающим `torch.load`).
- Системные зависимости nomeroff-net **не в PyPI METADATA**: `modelhub-client` (git-only, ria-com) и `PyTurboJPEG`. Установлены отдельными строками в `setup_server.sh` (best-effort после code-review).
- `nomeroff-net` тянет `opencv-python` (с GUI) — на headless-серверах требуется `uv pip install opencv-python-headless --reinstall`.

**Артефакты:**
- `results_collected/ssh9.qudata.ai/results/nomeroff_lpd/metrics.json`
- `results_collected/ssh9.qudata.ai/results/nomeroff_ocr/metrics.json`
- Логи прогона: `~/cars-eval/logs/nomeroff_{lpd,ocr}.log` на ssh9

**Спека / план:** `_bmad-output/implementation-artifacts/spec-nomeroff-ru-swap.md`, `plan-nomeroff-ru-swap.md`.

**Definition of Done:** ✅ Task 1–8 плана выполнены, оба порога (`recall ≥ 0.80` для LPD, `full_plate_accuracy ≥ 0.80` для OCR) пройдены с большим запасом.

---

## Источник решения

QQ-цикл `validation-campaign-2026-05-17`, multi-goal check на step-01 → пользователь выбрал [S] Split. Goal 1 завершён в `4fee864`. Goal 2 завершён 2026-05-17 (этот файл).
