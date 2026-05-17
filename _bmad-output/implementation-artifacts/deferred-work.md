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

## Goal 3 — Aggregation: SUMMARY.md generator + post-processing notebook (deferred 2026-05-17)

**Зачем:** Автоматизировать сборку `results/SUMMARY.md` (правило из `CLAUDE.md`) и предоставить notebook для пост-обработки.

**Scope:**
- Скрипт-агрегатор `results_collected/{host}/results/*.json` → единый `results/SUMMARY.md` с per-модельными метриками (Precision/Recall/F1/mAP@0.5 для detection; Top-1/Top-3/Accuracy для classification; Full Plate Accuracy/Character Accuracy для LPR).
- Сегментация метрик по `weather × timeofday × scene` для BDD100K (метаданные доступны).
- `notebooks/post_processing.ipynb` — воспроизводимый ноутбук для plots в `plots/` (PNG): confusion matrices, PR-curves, error breakdown.

**Зависит от:** Goal 2 ✅ (результаты в `results_collected/{ssh1,ssh9}.qudata.ai/results/`).

---

## Goal 4 — §6.5 dataset migration + class-mapping fixes (new, surfaced by Goal 2)

**Зачем:** Полученные в Goal 2 низкие метрики (TrafficCamNet F1=0.053, VehicleMakeNet Top1=0.002) — следствие data/label mismatch, не сломанной модели. Реальная валидация требует перехода на §6.5 шорт-лист.

**Scope:**
- Восстановить vehicletypenet: Kaggle auth ИЛИ переход на Stanford Cars + Make→Type mapping (требует переписать `eval_vehicletypenet`).
- TrafficCamNet: проверить class-mapping COCO→TrafficCamNet labels (4 класса bicycle/car/person/road_sign), либо мигрировать на BDD100K val (10K) с native labels.
- VehicleMakeNet: чинить label-mapping mad-cars `brand` → NGC 20 makes (сейчас всё схлопывается в «acura»); параллельно завести Stanford Cars test для cross-check.
- LPDNet/LPRNet: оценить целесообразность RU-моделей (NTechLab/nomeroff_lpd) — текущие USA/US-веса дают ожидаемый low recall/PlateAcc.

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
