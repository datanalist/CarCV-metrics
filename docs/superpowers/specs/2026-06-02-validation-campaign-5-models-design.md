# Дизайн: валидационная кампания на datasets.md

Дата: 2026-06-02
Статус: согласован, готов к планированию реализации
Поправка 2026-06-02: после обновления `datasets.md` пересмотрены пары модель×датасет
(VehicleMakeNet вернулся — у него появился датасет VMMRdb; VehicleTypeNet переведён
на Stanford Cars). Из шести валидируемых пар три уже отвалидированы на правильных
датасетах и в задачи кампании **не входят** (см. ниже); реальная работа — три модели.
Ветки — одна общая (не worktree-на-модель). Источник BDD100K и VMMRdb — прямой wget
с публичной Nextcloud-шары.

## Цель

Провести честную валидацию моделей из `models.md` на датасетах, указанных в
`datasets.md`. Шесть пар модель×датасет валидируемы; три из них уже измерены на
правильных датасетах (nomeroff_lpd, nomeroff_ocr, VehicleTypeNet) — их результаты
зачитываются как есть. **Эта кампания закрывает три оставшиеся модели**
(TrafficCamNet, VehicleMakeNet, FaceDetect). Весь код ведётся в одной feature-ветке;
задачи идут последовательно по моделям. Результат каждой — зафиксированные метрики
и вердикт pass/fail по заранее заданным порогам. FAIL по порогам считается валидным
результатом («модель не подходит под целевые данные»), а не поводом для тюнинга
или дообучения.

## Охват: модель × датасет

Из таблиц `models.md` × `datasets.md` валидируемы шесть пар. Остальные строки
исключены объективно: Color — нет модели (`UNDEF`); Face embedding — нет модели
(`Trainable`).

**Входят в задачи кампании (требуют работы):**

| # | Модель | Задача | Датасет (datasets.md) | Источник датасета |
|---|--------|--------|------------------------|--------------------|
| 1 | TrafficCamNet | Detection | BDD100K | next.ogoun.name → bdd100k.zip (wget) |
| 2 | VehicleMakeNet | Make | VMMRdb | next.ogoun.name → VMMRdb.zip (wget) |
| 3 | FaceDetect | Face Detection | WIDER FACE | HuggingFace |

**Уже отвалидированы на правильных датасетах — вне задач кампании** (результаты
зачитываются из `results_collected/`, повторно не гоняются):

| Модель | Задача | Датасет (datasets.md) |
|--------|--------|------------------------|
| nomeroff_lpd | LP Detection | autoria (детекционный, см. ниже) |
| nomeroff_ocr | OCR | autoriaNumberplateOcrRu-2021-09-01 |
| VehicleTypeNet | Type | Stanford Cars |

Примечание по FaceDetect: в `datasets.md` строка Face Detection пуста, а WIDER FACE
привязан к Face embedding. По согласованию валидируем FaceDetect именно на WIDER FACE.

**Датасет для LP Detection (решено).** Изначально `datasets.md` указывал для строки
LP Detection OCR-датасет кропов `autoriaNumberplateOcrRu-2021-09-01.zip` — без
детекционных bbox-аннотаций в сцене, на нём детекцию номера измерить нельзя.
`datasets.md` поправлен (2026-06-02) на **детекционный** датасет
`autoriaNumberplateDataset-2019-02-19.zip` (VIA-аннотации `via_region_data.json` со
сценовыми боксами) — именно на нём получен прошлый PASS nomeroff_lpd (P=0.91, R=0.92).

## Исходное состояние (что уже есть)

Ветка `exp/validation-campaign-2026-05-17` содержит зрелую инфраструктуру и
частично выполненную кампанию:

- `deploy/evaluation/evaluate.py` — универсальный evaluator (detection / classification /
  OCR), уже содержит эвалуаторы `eval_trafficcamnet`, `eval_vehicletypenet`,
  `eval_nomeroff_lpd`, `eval_nomeroff_ocr` (а также US-baseline LPDNet/LPRNet/VehicleMakeNet).
- `deploy/scripts/run_remote.py` — SSH-оркестратор с командами deploy / run / collect
  (rsync кода на сервер → фоновый запуск `evaluate.py` → сбор результатов).
- `deploy/evaluation/metrics.py`, `aggregate_summary.py` — метрики, пороги, агрегация.
- `configs/remote_experiments.yaml` — серверная привязка (host/port/identity).

Уже собранные результаты (на `ssh9.qudata.ai`) и их статус относительно обновлённого
`datasets.md`:

| Модель | Гонялось на | Результат | Статус по datasets.md |
|--------|-------------|-----------|------------------------|
| nomeroff_lpd | autoriaNumberplateDataset-2019 (детекц.) | PASS (P=0.91, R=0.92) | ✓ датасет верный → **вне задач** |
| nomeroff_ocr | autoriaNumberplateOcrRu-2021-09-01 | PASS (CharAcc=0.9995) | ✓ датасет верный → **вне задач** |
| VehicleTypeNet | Stanford Cars | FAIL (Top1=0.36) | ✓ датасет верный → **вне задач** |
| TrafficCamNet | COCO val2017 (суррогат) | FAIL (P=0.08) | ✗ нужен BDD100K |
| VehicleMakeNet | mad-cars (суррогат) | (Goal 4) | ✗ нужен VMMRdb |
| FaceDetect | — | не реализован | нужен эвалуатор + WIDER FACE |

Вывод по обновлённому `datasets.md`:
- nomeroff LPD/OCR и VehicleTypeNet (Stanford Cars) уже измерены на правильных
  датасетах — их результаты зачитываются как есть, повторно не гоняются и в задачи
  кампании не входят. FAIL VehicleTypeNet (Top1=0.36) — валидный окончательный результат.
- TrafficCamNet надо перегнать на настоящем BDD100K (вместо COCO-суррогата).
- VehicleMakeNet надо перегнать на VMMRdb (вместо mad-cars-суррогата) — добавить загрузчик.
- FaceDetect — реализовать эвалуатор с нуля.

## Архитектура

Базис — существующий пайплайн `deploy/`, без переписывания. Новый код минимален (YAGNI):

1. Загрузчик BDD100K (распаковка `bdd100k.zip` + конверсия родных val-аннотаций
   `bdd100k_labels_images_val.json` в формат `labels.json` пайплайна) +
   class-mapping BDD → 4 класса TrafficCamNet. Эвалуатор `eval_trafficcamnet` уже есть.
2. Загрузчик VMMRdb (распаковка `VMMRdb.zip`; марка = первый токен имени папки
   `<make>_<model>_<year>`) для VehicleMakeNet. Эвалуатор `eval_vehiclemakenet`
   уже есть — добавить ветку чтения VMMRdb-раскладки (каталоги-по-классам) рядом с
   существующей mad-cars JSON-веткой; марка нормализуется через `normalize_brand`,
   out-of-distribution (не в 20 NGC-марках) пропускается.
3. Эвалуатор FaceDetect (TAO FaceNet / DetectNet_v2, переиспользуя `detectnet_v2_decode`,
   1 класс «face», вход 736×416) + парсер аннотаций WIDER FACE (HF) + detection-метрики.

VehicleTypeNet (Stanford Cars) и nomeroff LPD/OCR — код не трогаем, они вне задач кампании.
Остальное — конфиги экспериментов и переиспользование готовых эвалуаторов.

## Структура веток (одна общая ветка)

Сначала привести `main` к зрелой базе: слить `exp/validation-campaign-2026-05-17` в
`main`. От обновлённого `main` отвести **одну** feature-ветку кампании
(`exp/validation-campaign-2026-06`), внутри которой работа идёт последовательно
задачами по моделям.

Причина отказа от worktree-на-модель (отступление от первоначального дизайна):
весь новый код добавляется в **общие** файлы (`deploy/evaluation/evaluate.py`,
`download_models.sh`, скрипты загрузки датасетов), а прогоны всё равно
последовательны (один GPU). Пять параллельных веток дали бы конфликты слияния этих
общих файлов в `main` без выигрыша — изоляция кода/конфигов/результатов на уровне
функций и каталогов `results/<model>/` достаточна.

| Задача (в одной ветке) | Модель | Работа |
|------------------------|--------|--------|
| eval-trafficcamnet | TrafficCamNet | загрузчик BDD100K + class-mapping, прогон |
| eval-vehiclemakenet | VehicleMakeNet | загрузчик VMMRdb + make-mapping, прогон |
| eval-facedetect | FaceDetect | реализовать эвалуатор + WIDER FACE, прогон |

Изоляция результатов: у каждой модели свой `results/<model>/metrics.json` + графики.

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
Один GPU → прогоны последовательны; изоляция кода/конфигов/результатов — на уровне
функций эвалуаторов и каталогов `results/<model>/`, без отдельных worktree.

## Поток данных (на каждый прогон)

1. `run_remote.py deploy` синхронизирует код+конфиг на сервер и поднимает окружение (setup).
2. На сервере: bootstrap deps (onnxruntime-gpu + `requirements.txt`, без torch) →
   скачивание датасета и весов. BDD100K и VMMRdb тянутся прямым `wget` с публичной
   Nextcloud-шары (download-API: `…/s/<token>/download?path=/DATASETS&files=<file>.zip`);
   веса TAO — с NGC.
3. `run_remote.py run` запускает `evaluate.py` (препроцессинг + инференс + метрики).
4. `metrics.json` + графики → `run_remote.py collect` → `results_collected/<host>/<model>/`.

## Веса моделей

- TAO для моделей кампании (TrafficCamNet, VehicleMakeNet, FaceNet) — ONNX с NVIDIA NGC
  (без API-ключа), скрипт `download_models.sh`. FaceNet добавляется в скрипт (новый).

## Метрики и критерий «успеха»

Каждый прогон даёт метрики и pass/fail по порогам из конфига (`EVAL_CONFIGS` / YAML).
Пороги — те, что уже заданы в проекте. Семейства метрик:

- Detection (TrafficCamNet, FaceDetect; в агрегации также nomeroff_lpd): Precision, Recall, F1, AP, mAP@0.5.
- Classification (VehicleMakeNet; в агрегации также VehicleTypeNet): Top-1, Top-3 accuracy.
- OCR (в агрегации nomeroff_ocr): char accuracy, full-plate accuracy, CER.

Цель — измерить и зафиксировать. FAIL — валидный, окончательный результат.

## Порядок выполнения

1. TrafficCamNet — загрузчик BDD100K + class-mapping, прогон (проверяет пайплайн
   end-to-end на новом сервере на «лёгком» по коду шаге — эвалуатор уже есть).
2. VehicleMakeNet — загрузчик VMMRdb + make-mapping, прогон.
3. FaceDetect — новый эвалуатор + WIDER FACE (самый рискованный, в конце).

## Финальная агрегация

После трёх прогонов — слить feature-ветку в `main`, запустить `aggregate_summary.py`:
единый `results/SUMMARY.md` агрегирует все шесть пар (три новых + три уже собранных из
`results_collected/`), графики в `plots/`, JSON+CSV в `results/`, воспроизводимый
notebook в `notebooks/`.

## Вне охвата (YAGNI)

- Повторный прогон уже отвалидированных nomeroff_lpd, nomeroff_ocr, VehicleTypeNet
  (результаты на правильных датасетах уже есть — только зачитываются в агрегации).
- Тюнинг порогов, дообучение, замена моделей ради PASS.
- Color, Face embedding (нет модели).
- Мульти-серверная параллелизация (один GPU).
