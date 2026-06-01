# Дизайн: валидационная кампания 5 моделей на datasets.md

Дата: 2026-06-02
Статус: согласован, готов к планированию реализации

## Цель

Провести честную валидацию пяти моделей из `models.md` на датасетах, указанных в
`datasets.md`. Каждая модель — в своей ветке (git worktree). Результат каждой —
зафиксированные метрики и вердикт pass/fail по заранее заданным порогам. FAIL по
порогам считается валидным результатом («модель не подходит под целевые данные»),
а не поводом для тюнинга или дообучения.

## Охват: модель × датасет

Из таблиц `models.md` × `datasets.md` валидируемы пять пар. Остальные строки
исключены объективно: VehicleMakeNet — нет датасета; Color — нет модели; Face
embedding — нет модели (Trainable).

| # | Модель | Задача | Датасет (datasets.md) | Источник датасета |
|---|--------|--------|------------------------|--------------------|
| 1 | TrafficCamNet | Detection | BDD100K | next.ogoun.name → bdd100k.zip |
| 2 | nomeroff_lpd | LP Detection | autoriaNumberplateOcrRu-2021-09-01 | nomeroff.net.ua |
| 3 | nomeroff_ocr | OCR | autoriaNumberplateOcrRu-2021-09-01 | nomeroff.net.ua |
| 4 | VehicleTypeNet | Type | VMMRdb | next.ogoun.name → VMMRdb.zip |
| 5 | FaceDetect | Face Detection | WIDER FACE | HuggingFace |

Примечание по FaceDetect: в `datasets.md` строка Face Detection пуста, а WIDER FACE
привязан к Face embedding. По согласованию валидируем FaceDetect именно на WIDER FACE.

## Исходное состояние (что уже есть)

Ветка `exp/validation-campaign-2026-05-17` содержит зрелую инфраструктуру и
частично выполненную кампанию:

- `deploy/evaluation/evaluate.py` — универсальный evaluator (detection / classification /
  OCR), уже содержит эвалуаторы `eval_trafficcamnet`, `eval_vehicletypenet`,
  `eval_nomeroff_lpd`, `eval_nomeroff_ocr` (а также US-baseline LPDNet/LPRNet/VehicleMakeNet).
- `deploy/scripts/run_remote.py` — SSH-оркестратор: tarball → `/dev/shm` → запуск → сбор.
- `deploy/evaluation/metrics.py`, `aggregate_summary.py` — метрики, пороги, агрегация.
- `configs/remote_experiments.yaml` — серверная привязка (host/port/identity).

Уже собранные результаты (на `ssh9.qudata.ai`) и их расхождение с `datasets.md`:

| Модель | Гонялось на | Результат | Соответствие datasets.md |
|--------|-------------|-----------|--------------------------|
| nomeroff_lpd | autoriaNumberplateOcrRu | PASS (P=0.91, R=0.92) | да |
| nomeroff_ocr | autoriaNumberplateOcrRu | PASS (CharAcc=0.9995) | да |
| TrafficCamNet | COCO val2017 (суррогат) | FAIL (P=0.08) | нет → нужен BDD100K |
| VehicleTypeNet | Stanford Cars (суррогат) | FAIL (Top1=0.36) | нет → нужен VMMRdb |
| FaceDetect | — | не реализован | нужен эвалуатор + WIDER FACE |

Вывод: nomeroff LPD/OCR уже валидны (ревалидируем для свежих чисел на текущем сервере);
TrafficCamNet и VehicleTypeNet надо перегнать на правильных датасетах; FaceDetect —
реализовать с нуля.

## Архитектура

Базис — существующий пайплайн `deploy/`, без переписывания. Новый код минимален (YAGNI):

1. Загрузчик BDD100K + class-mapping BDD → классы TrafficCamNet.
2. Загрузчик VMMRdb + mapping make/model → типы кузова VehicleTypeNet.
3. Эвалуатор FaceDetect (TAO FaceNet / DetectNet_v2, переиспользуя `detectnet_v2_decode`)
   + парсер аннотаций WIDER FACE + detection-метрики.

Остальное — конфиги экспериментов и переиспользование готовых эвалуаторов.

## Структура веток (worktree-на-модель)

Сначала привести `main` к зрелой базе: слить `exp/validation-campaign-2026-05-17` в
`main`. От обновлённого `main` отвести пять веток, каждая — отдельный git worktree:

| Ветка | Модель | Работа |
|-------|--------|--------|
| `exp/eval-nomeroff-lpd` | nomeroff_lpd | ревалидация на текущем сервере |
| `exp/eval-nomeroff-ocr` | nomeroff_ocr | ревалидация на текущем сервере |
| `exp/eval-trafficcamnet` | TrafficCamNet | конфиг на настоящем BDD100K + class-mapping, прогон |
| `exp/eval-vehicletypenet` | VehicleTypeNet | загрузчик VMMRdb + mapping в типы кузова, прогон |
| `exp/eval-facedetect` | FaceDetect | реализовать эвалуатор + WIDER FACE, прогон |

Каждая ветка изолирована: свой `configs/experiment/<model>_eval.yaml`, свой прогон,
свои `results/<model>/metrics.json` + графики.

Ветка `exp/eval-trafficcamnet` уже существует — переиспользуется (пересоздаётся от
обновлённого `main`).

## Серверная конфигурация

Один хост, эфемерный контейнер с одним GPU (на момент дизайна — `ssh2.qudata.ai:17108`,
RTX 3090 24 GB, ключ `~/.ssh/qudata`, user root). Особенности:

- Контейнер пересоздаётся между запусками: host/port и host key меняются,
  `~/datasets`, `~/models` каждый раз пустые, onnxruntime и uv отсутствуют
  (torch 2.8 + CUDA — есть).
- Host/port/identity вынесены в `configs/remote_experiments.yaml` — единственное место правки.
- Перед коннектом — очистка stale host key: `ssh-keygen -R '[host]:<old_port>'`,
  подключение с `StrictHostKeyChecking=accept-new`.

Следствие для пайплайна: bootstrap (`pip install onnxruntime-gpu` + `deploy/requirements.txt`,
без torch) и скачивание датасетов/весов — **идемпотентные скрипты, запускаемые в начале
каждого прогона** из bundle. Расчёта на сохранённое состояние между прогонами нет.
Один GPU → прогоны последовательны; worktree даёт изоляцию кода/конфигов/результатов.

## Поток данных (на каждый прогон)

1. `run_remote.py` упаковывает код+конфиг в tarball, копирует в `/dev/shm/<exp>` на сервере.
2. На сервере: bootstrap deps → скачивание датасета (в `/dev/shm`, 50 GB) и весов (NGC / nomeroff cache).
3. `evaluate.py` выполняет препроцессинг + инференс + метрики.
4. `metrics.json` + графики → rsync обратно в `results_collected/<host>/<model>/`.

## Веса моделей

- TAO (TrafficCamNet, VehicleTypeNet, FaceNet) — ONNX с NVIDIA NGC (без API-ключа),
  скрипт `download_models.sh`.
- nomeroff_lpd / nomeroff_ocr — авто-загрузка пакетом nomeroff-net при первом вызове `pipeline()`.

## Метрики и критерий «успеха»

Каждый прогон даёт метрики и pass/fail по порогам из конфига (`EVAL_CONFIGS` / YAML).
Пороги — те, что уже заданы в проекте. Семейства метрик:

- Detection (TrafficCamNet, nomeroff_lpd, FaceDetect): Precision, Recall, F1, AP, mAP@0.5.
- Classification (VehicleTypeNet): Top-1, Top-3 accuracy.
- OCR (nomeroff_ocr): char accuracy, full-plate accuracy, CER.

Цель — измерить и зафиксировать. FAIL — валидный, окончательный результат.

## Порядок выполнения

1. nomeroff_lpd — ревалидация (проверка всего пайплайна end-to-end на новом сервере).
2. nomeroff_ocr — ревалидация.
3. TrafficCamNet — смена датасета на BDD100K.
4. VehicleTypeNet — смена датасета на VMMRdb.
5. FaceDetect — новый эвалуатор (самый рискованный, в конце).

## Финальная агрегация

После прогона всех пяти — слить ветки в `main`, запустить `aggregate_summary.py`:
единый `results/SUMMARY.md` (таблица 5 моделей × метрики × pass/fail), графики в `plots/`,
JSON+CSV в `results/`, воспроизводимый notebook в `notebooks/`.

## Вне охвата (YAGNI)

- Тюнинг порогов, дообучение, замена моделей ради PASS.
- VehicleMakeNet, Color, Face embedding (нет пары модель+датасет).
- Мульти-серверная параллелизация (один GPU).
