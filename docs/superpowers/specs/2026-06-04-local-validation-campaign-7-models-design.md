# Дизайн: локальная валидационная кампания, 7 моделей (ветка на эксперимент)

Дата: 2026-06-04
Статус: согласован, готов к планированию реализации
Предшественник: `2026-06-02-validation-campaign-5-models-design.md` (удалённая SSH-кампания qudata).
Этот дизайн **заменяет** его для нового запуска: всё гоняется **локально**, охват расширен
до 7 пар, каждая модель — отдельная ветка git.

## Цель

Честно измерить каждую валидируемую пару *модель × датасет* из `models.md` × `datasets.md`
**локально на RTX 3090**, по одному эксперименту на git-ветку. Перед каждым прогоном агент
**читает спеку модели и датасета** из `docs/about_models/` и `docs/about_datasets/` и
фиксирует из них препроцессинг, классы, пороги и риски в отчёте эксперимента
(`results/<model>/EXPERIMENT.md`). Результат каждого эксперимента — зафиксированные метрики
и вердикт **PASS / FAIL / UNDEF** по заранее заданным порогам.

**FAIL** («модель не проходит порог на целевых данных») и **UNDEF** («оценка невозможна —
нет весов / эвалуатор не запускается») — **валидные окончательные результаты**, а не повод
для тюнинга порогов, дообучения или замены модели.

## Чем этот дизайн отличается от предыдущего (2026-06-02)

1. **Место запуска — локально.** Раньше — эфемерные удалённые контейнеры qudata по SSH
   (`run_remote.py`). Теперь на машине есть NVIDIA RTX 3090 24 GB и локально лежат **все 7
   датасетов** и ключевые модели. SSH/qudata-оркестрация для этой кампании не используется.
2. **Охват — 7 пар, все заново.** Раньше 3 пары зачитывались из `results_collected/`. Теперь
   измеряем все 7 пар в одном локальном окружении (воспроизводимость; у LPD к тому же сменилась
   версия датасета: 2019 → 2021-05-12).
3. **Color стал валидируемым.** `bae_model_f3.onnx` есть локально, а `datasets.md` дал ему
   датасет (MAD-Cars). Раньше Color исключался как «нет модели».
4. **Ветка на эксперимент.** Раньше — одна общая feature-ветка. Теперь — базовая ветка обвязки
   + по ветке на модель.

## Окружение (сверено по диску 2026-06-04)

- GPU: NVIDIA GeForce RTX 3090, 24 GB, драйвер 580.126.09.
- `.venv` (uv): Python 3.13.11, onnxruntime 1.24.4, torch 2.11.0+cu130, opencv 4.13.0.
  **`nomeroff_net` не установлен.** `pip` в venv отсутствует — ставить через `uv pip install`.

## Охват и реальная выполнимость

`Face embedding` исключён объективно — модели нет (`None / Trainable`). Остальные 7 пар:

| # | Модель | Датасет (локальный путь) | Веса | Эвалуатор | Подготовка данных |
|---|--------|--------------------------|------|-----------|-------------------|
| 1 | TrafficCamNet | BDD100K val — `…/DATASETS/bdd100k` | ✅ локально `…/trafficcamnet_pruned_onnx_v1.0.4/resnet18_trafficcamnet_pruned.onnx` | ✅ есть `eval_trafficcamnet` | конвертер `bdd100k_labels_images_val.json` → `labels.json` + class-mapping |
| 2 | nomeroff_lpd | autoria-det-2021-05-12 (`val/via_region_data.json`) | — (через `nomeroff-net`) | ✅ есть `eval_nomeroff_lpd` | путь к `data_dir`; установка `nomeroff-net` |
| 3 | nomeroff_ocr | autoria-OCR-2021-09-01 (`val/img`+`val/ann`) | — (через `nomeroff-net`) | ✅ есть `eval_nomeroff_ocr` | путь к `data_dir`; установка `nomeroff-net` |
| 4 | VehicleMakeNet | VMMRdb — `…/DATASETS/VMMRdb` (9170 папок `make_model_year`) | ⬇️ NGC `pruned_onnx_v1.1.0` (без ключа) | ⚠️ `eval_vehiclemakenet` — добавить ветку чтения VMMRdb | NGC-загрузка + reader каталогов-по-классам |
| 5 | VehicleTypeNet | Stanford Cars — `…/DATASETS/Stanford Cars Dataset` (devkit `.mat`) | ⬇️ NGC `pruned_onnx_v1.1.0` | ✅ есть `eval_vehicletypenet` | сборка `test.json` (имя класса) из `cars_meta.mat` + annos |
| 6 | **Color** (bae_model_f3) | MAD-Cars sample — `…/CarCV/data/external/ymad_cars` (≈10k img: 9996 на диске, 10432 по `metadata.json`; `images_index.jsonl` с полем `color`) | ✅ локально `/home/mk/CarCV/models/bae_model_f3.onnx` | ❌ **новый** `eval_color` | hex→класс mapping, dedup по `car_id` |
| 7 | **FaceDetect** | WIDER FACE val — `…/DATASETS/Wider Face/WIDER_val` (3226 img, `wider_face_val_bbx_gt.txt`) | ❌ локально только зашифр. `model.etlt` → получить ONNX | ❌ **новый** `eval_facedetect` | парсер `wider_face_val_bbx_gt.txt` |

Условные обозначения: ✅ готово · ⚠️ доработать существующее · ❌ создать с нуля · ⬇️ скачать.

## Архитектура

Базис — существующий `deploy/evaluation/evaluate.py` (универсальный evaluator: detection /
classification / OCR; уже содержит `eval_trafficcamnet`, `eval_vehiclemakenet`,
`eval_vehicletypenet`, `eval_nomeroff_lpd`, `eval_nomeroff_ocr`). Пайплайн **не переписываем**.
Новый код минимален (YAGNI):

1. **Локальный раннер** `deploy/scripts/run_local.py` — заменяет SSH-оркестрацию `run_remote.py`
   для локального прогона. Берёт `EVAL_CONFIGS[name]` из `evaluate.py`, накладывает абсолютные
   локальные пути из `configs/local_paths.yaml` (`model_path`, `labels_path`, `data_dir`),
   вызывает `cfg["eval_fn"](cfg)`. `results_dir` остаётся репозиторным (`results/<model>/`).
2. **Конфиг путей** `configs/local_paths.yaml` — единственное место с абсолютными путями к
   локальным датасетам и весам. Большие данные/веса **в git не коммитятся** (см. `.gitignore`).
3. **Конвертеры/подготовка данных** (по эксперименту, идемпотентные, пишут небольшие
   производные файлы рядом с данными или в gitignore-каталог `data/prep/<model>/`):
   - BDD100K: `bdd100k_labels_images_val.json` → `labels.json` (формат пайплайна:
     `[{file_name,image_id,detections:[{category,bbox2d:{x1,y1,x2,y2}}]}]`) + class-mapping
     BDD → 4 класса TrafficCamNet (`TRAFFICCAMNET_GT_VOCAB` уже в коде).
   - Stanford Cars: `cars_meta.mat` (имена 196 классов) + annos → `test.json`
     (`[{file_name,label:"<Make> <Model> <BodyType> <Year>"}]`); body type выводится
     `derive_typenet_label` (уже в коде). Источник меток теста зафиксировать в `EXPERIMENT.md`
     (официальный `cars_test_annos.mat` без меток → использовать split с метками; см. Риски).
   - VMMRdb: чтение каталогов `<make>_<model>_<year>`; марка = первый токен; нормализация
     `normalize_brand` (уже в коде); out-of-distribution (не в 20 NGC-марках) — пропуск.
   - MAD-Cars: чтение `images_index.jsonl`; hex→класс через `HEX_TO_CARS_COLOR`
     (из `docs/about_datasets/mad_cars.md`); dedup по `car_id` (1 view/car) для headline-метрики.
4. **Новый эвалуатор `eval_color`** (`bae_model_f3`, EfficientNet-B3, вход `[N,3,384,384]`,
   выход `[N,15]` логиты): препроцессинг BGR→RGB, `/255`, `(x-mean)/std` с
   `mean=[0.43,0.40,0.39]`, `std=[0.27,0.26,0.26]` (System Design §6.5), resize 384×384, NCHW;
   **softmax в постобработке** (граф отдаёт логиты); 15 классов в алфавитном порядке
   (`beige…yellow`) — маппинг индекс→класс помечается как **непроверенный**; Top-1/per-class +
   confusion 15×15. GT: `HEX_TO_CARS_COLOR[row.color]`.
5. **Новый эвалуатор `eval_facedetect`** (TAO FaceNet / DetectNet_v2, вход 736×416, 1 класс
   `face`): препроцессинг как у TrafficCamNet (BGR→RGB, `/255`, NCHW); декодирование
   `detectnet_v2_decode(target_cls=0, stride=16, bbox_norm=35.0)` (переиспользуем) + NMS;
   парсер `wider_face_val_bbx_gt.txt` (порядок полей `x1 y1 w h blur expression illumination
   invalid occlusion pose` — из `docs/about_datasets/wider_face.md`); detection-метрики + AP@0.5.

Записи `color` и `facedetect` добавляются в `EVAL_CONFIGS`. Остальные эвалуаторы и метрики
(`deploy/evaluation/metrics.py`, `aggregate_summary.py`, `visualize.py`) не трогаем.

## Структура веток и интеграция

```
main
 └─ exp/local-base            # run_local.py + local_paths.yaml + .gitignore data/models → merge в main
     ├─ exp/eval-trafficcamnet   # от обновлённого main; merge обратно в main
     ├─ exp/eval-nomeroff-lpd
     ├─ exp/eval-nomeroff-ocr
     ├─ exp/eval-vehiclemakenet
     ├─ exp/eval-vehicletypenet
     ├─ exp/eval-color
     └─ exp/eval-facedetect
```

- Сначала `exp/local-base` → **merge в `main`** (общая обвязка, нужна всем).
- Затем по одной ветке на модель **от обновлённого `main`**; после прогона — **merge в `main`**.
  Новый код модели (новый эвалуатор / reader-ветка) живёт в её же ветке; общие файлы
  (`evaluate.py`) при последовательных merge дают тривиальные конфликты (дописывание функций +
  записи в `EVAL_CONFIGS`) — разрешаются вручную.
- Изоляция результатов: у каждой модели свой `results/<model>/`.
- Две устаревшие ветки эпохи qudata (`exp/eval-trafficcamnet`, `exp-eval-trafficcamnet-unprunned`)
  удаляются/переименовываются в base-шаге, чтобы не конфликтовать с новыми именами.
- Финальная агрегация — на `main` после всех merge.

## Протокол одного эксперимента (фиксированный, для каждой ветки)

1. **Read docs (обязательно).** Прочитать `docs/about_models/<model>.md` и
   `docs/about_datasets/<dataset>.md`; выписать в `results/<model>/EXPERIMENT.md`:
   препроцессинг (resize, цветовой порядок, mean/std, offsets), классы/маппинг, пороги,
   известные риски и расхождения с legacy-доками.
2. **Подготовка.** Веса (локальный путь или NGC-загрузка) + конвертер/sampling данных.
3. **Прогон.** `.venv/bin/python deploy/scripts/run_local.py --models <name>` (обёртка над
   `evaluate.py`).
4. **Фиксация.** `metrics.json` + графики (где есть) → дописать в `EXPERIMENT.md` измеренные
   числа и вердикт PASS/FAIL/UNDEF; коммит; merge в `main`.

## Метрики и пороги

Пороги — из кода (`EVAL_CONFIGS` / эвалуаторы) и спек; для двух новых эвалуаторов утверждены
в этом дизайне.

| Модель | Семейство | Пороги (PASS) |
|--------|-----------|----------------|
| TrafficCamNet | Detection | P≥0.90 · R≥0.85 · F1≥0.87 |
| nomeroff_lpd | Detection | P≥0.70 · R≥0.80 |
| nomeroff_ocr | OCR | char_acc≥0.90 · plate_acc≥0.80 |
| VehicleMakeNet | Classification | Top-1≥0.70 · Top-3≥0.85 |
| VehicleTypeNet | Classification | Top-1≥0.85 |
| **Color** (bae_model_f3) | Classification | Overall≥0.80 · best (black/white/red/blue)≥0.90 · challenging (beige/tan/gold/silver)≥0.70 (System Design §6.5) |
| **FaceDetect** | Detection | AP@0.5: Easy≥0.80 · Medium≥0.70 · Hard≥0.50 + отдельный automotive-срез (`14--Traffic`, `5--Car_Accident`, `59--people--driving--car`) |

## Риски и обработка блокеров

- **FaceDetect — главный блокер.** Локально только зашифрованный `model.etlt` (нужен
  `tlt-model-key`), ONNX нет. **Решение (утверждено):** (1) попытаться скачать deployable-ONNX
  FaceNet с NGC (как `download_models.sh`, без ключа); (2) если нет — экспорт `etlt→onnx`
  стандартными ключами (`tlt_encode` / `nvidia_tlt`); (3) если оба не выходят — зафиксировать
  **UNDEF/заблокировано** (валидный окончательный результат, без долгих раскопок TAO-тулинга).
- **WIDER Easy/Medium/Hard.** Официальное разбиение Easy/Med/Hard задаётся mat-файлами
  eval_tools MMLab, которых **нет** локально (есть только `wider_face_val.mat` с GT). Если их не
  удастся получить — Easy/Med/Hard аппроксимируется по масштабу лица + атрибутам (`blur`,
  `occlusion`) и **явно помечается как неофициальная** аппроксимация; иначе headline-метрика —
  общий AP@0.5 + automotive-срез. (Скорее всего неактуально: FaceDetect вероятно UNDEF.)
- **nomeroff-net на Python 3.13 / torch 2.11** может не установиться (исторически таргетит более
  старые версии). Если не встаёт — LPD/OCR помечаются **deferred** (не «выдуманный PASS»).
  В коде уже есть обход `torch.load(weights_only=False)` для torch≥2.6.
- **Stanford test-labels.** Официальный `cars_test_annos.mat` без меток классов. Берём split с
  метками (train annos) либо `*_withlabels`; конкретный источник фиксируется в `EXPERIMENT.md`.
  (VehicleTypeNet на Stanford ранее дал FAIL Top1≈0.36 удалённо — здесь меряем заново, FAIL
  остаётся валидным.)
- **Color — caveats (из спеки).** Выход — логиты (нужен softmax); mean/std нестандартные и
  непроверяемы без обучающего кода; маппинг индекс→класс — алфавитная гипотеза; `tan`
  отсутствует в данных (покрытие 14/15); `pink` ≈0.9% (широкий доверительный интервал). MAD-Cars
  — кадры с авто крупным планом → подаём resize 384×384 **без** TrafficCamNet-кропа. Эти факторы
  делают вероятным «осторожный»/заниженный вердикт — он валиден.
- **Лицензии данных.** MAD-Cars — CC BY-NC-SA 4.0; WIDER FACE — CC BY-NC-ND 4.0. Только
  **внутренний research-бенчмарк**; метрики не публиковать вовне без legal review.
- **Гигиена git.** Датасеты (десятки ГБ) и веса **не коммитятся** — `.gitignore` на `data/`,
  `models/` (кроме мелких labels), временный prep. Коммитятся только `results/` (JSON/CSV/MD)
  и `plots/`.

## Вне охвата (YAGNI)

- SSH/qudata-оркестрация (`run_remote.py`), мульти-сервер, мульти-GPU.
- Тюнинг порогов, дообучение, замена моделей ради PASS.
- `Face embedding` (нет модели); докачка train-сплитов WIDER/BDD; полноразмерная скачка MAD-Cars
  (используем готовый локальный sample ≈10k img).
- TensorRT/Jetson-замеры latency (кампания меряет качество, не перфоманс на устройстве).

## Артефакты на выходе

**На каждый эксперимент** (в его ветке, затем в `main`):
- `results/<model>/metrics.json` — метрики + статус порогов.
- `results/<model>/EXPERIMENT.md` — выписка из docs (шаг 1) + измеренные числа + вердикт.
- графики/CSV где применимо (`plots/`, `per_class_metrics.csv`, confusion).

**Финал на `main`** (после всех merge):
- `results/SUMMARY.md` — агрегат всех 7 пар (`aggregate_summary.py` / `evaluate.py --summary`).
- воспроизводимый notebook в `notebooks/`.
- обновление памяти кампании (PASS/FAIL/UNDEF по 7 моделям).

## Порядок выполнения

0. `exp/local-base`: `run_local.py` + `configs/local_paths.yaml` + `.gitignore` → merge в `main`.
1. **TrafficCamNet** — конвертер BDD100K + прогон (проверяет локальный пайплайн end-to-end на
   готовом эвалуаторе).
2. **VehicleTypeNet** — `test.json` из Stanford devkit + NGC-веса + прогон.
3. **VehicleMakeNet** — VMMRdb reader + NGC-веса + прогон.
4. **Color** — новый `eval_color` + MAD-Cars sample + прогон.
5. **nomeroff_lpd** — установка `nomeroff-net` + прогон (если установка успешна).
6. **nomeroff_ocr** — прогон на той же установке.
7. **FaceDetect** — получение ONNX (NGC/экспорт) + новый `eval_facedetect` + прогон, иначе UNDEF
   (самый рискованный — в конце).
8. Агрегация на `main`.
