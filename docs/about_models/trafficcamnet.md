# TrafficCamNet — детекция транспортных средств (PGIE)

## Общая информация

**TrafficCamNet** — purpose-built детектор объектов транспортной сцены от NVIDIA из каталога TAO / NGC. В семействе NVIDIA это специализированная модель класса DetectNet_v2, обученная на видеопотоке с дорожных камер (traffic-camera footage, преимущественно вид сверху/под углом сверху). В конвейере CARS она выступает **первичным детектором (PGIE)**: находит объекты в кадре, а её кропы передаются вторичным классификаторам (SGIE) — VehicleMakeNet (марка), VehicleTypeNet (тип кузова) — и далее по конвейеру (LP Detection, OCR, цвет).

| Поле | Значение |
|------|----------|
| Задача | Object Detection (4 класса) |
| Роль в CARS | PGIE — первичный детектор в DeepStream-конвейере |
| Вендор / источник | NVIDIA TAO, каталог NGC |
| Архитектура | DetectNet_v2, backbone ResNet-18 (pruned) |
| Версия (локально / `download_models.sh`) | `pruned_onnx_v1.0.4` |
| Версия (по `models.md`) | `pruned_quantized_v2.0.1` (в URL — `pruned_v1.0.3`) |
| Локальное расположение | `/home/mk/CarCV/models/trafficcamnet_pruned_onnx_v1.0.4` (вне репозитория CarCV-metrics) |
| URL карточки модели | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet |

> **Расхождение версий (зафиксировано честно).** `models.md` указывает версию `pruned_quantized_v2.0.1`, а в URL той же строки — `pruned_v1.0.3`. Однако и локальная копия, и скрипт `deploy/scripts/download_models.sh` фактически используют **`pruned_onnx_v1.0.4`** (NGC base `https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_onnx_v1.0.4/files/...`). Авторитетной для измеренных метрик и препроцессинга считается локальная версия `pruned_onnx_v1.0.4`; запись в `models.md` остаётся непрояснённой и подлежит сверке.

---

## Архитектура

- **Семейство:** DetectNet_v2 (NVIDIA) — grid-based детектор без anchor-боксов. Вместо якорей сеть выдаёт два покрытых сеткой тензора: coverage (вероятность присутствия объекта в ячейке) и bbox (нормированные смещения границ от центра ячейки).
- **Backbone:** ResNet-18, prune-обрезанный (pruned) для снижения числа параметров и латентности на edge-устройствах.
- **Квантизация:** в комплект локальной копии входит INT8 calibration cache (`resnet18_trafficcamnet_pruned_int8.txt`, заголовок `TRT-100300-EntropyCalibration2`) — кэш для построения INT8 TensorRT-engine. ONNX-файл при этом в FP32-весах; квантизация применяется на этапе сборки engine на целевом устройстве.
- **Якоря (anchors):** отсутствуют — координаты декодируются из grid-coverage + bbox-смещений (см. «Вход / выход и препроцессинг»).

---

## Вход / выход и препроцессинг

### Вход

Параметры взяты из `nvinfer_config.txt` локальной модели и подтверждены эвалуатором `eval_trafficcamnet` в `deploy/evaluation/evaluate.py` — значения совпадают.

| Параметр | Значение | Источник |
|----------|----------|----------|
| `infer-dims` | `3;544;960` → вход **960×544** (W×H), 3 канала | `nvinfer_config.txt` |
| Формат тензора | NCHW | `evaluate.py` (`transpose(2,0,1)[None]`) |
| `model-color-format` | `0` → RGB (на вход подаётся BGR→RGB) | `nvinfer_config.txt` |
| `net-scale-factor` | `0.00392156862745098` (= 1/255) | `nvinfer_config.txt` |
| `offsets` | `0.0;0.0;0.0` — **без вычитания среднего** | `nvinfer_config.txt` |
| `maintain-aspect-ratio` | `0` (resize без сохранения пропорций) | `nvinfer_config.txt` |
| `num-detected-classes` | `4` | `nvinfer_config.txt` |
| `network-type` | `0` (detector) | `nvinfer_config.txt` |

Семейство препроцессинга — **DetectNet_v2** (по диспетчеру `_bmad-output/project-context.md`): BGR→RGB, масштаб `/255`, форма NCHW, без mean/std-offsets.

Фрагмент из `eval_trafficcamnet` (`deploy/evaluation/evaluate.py`):

```python
inp = cv2.resize(img, (W, H)).astype(np.float32)        # W=960, H=544
inp = inp[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0  # BGR→RGB, NCHW, /255 (без offsets)
```

> **Расхождение в конфиге-черновике.** Файл `configs/experiment/trafficcamnet_eval.yaml` содержит ИНОЙ препроцессинг (`offsets`/`mean` 103.939;116.779;123.68, `class_names: [car, truck, bicycle, person]`, `confidence_threshold: 0.3`, `nms_iou_threshold: 0.45`). Это черновик/шаблон BDD100K-прогона, он **не** соответствует ни локальному `nvinfer_config.txt`, ни рабочему эвалуатору. Авторитетным считается путь `eval_trafficcamnet` + `nvinfer_config.txt`: масштаб `/255`, offsets `0`, NMS IoU `0.5`. Несовпадение YAML с фактическим препроцессингом помечено как требующее правки конфига.

### Выход и декодирование

Сеть выдаёт два тензора:

| Тензор | Форма | Имя | Содержание |
|--------|-------|-----|------------|
| coverage | `[C, gh, gw]` | `output_cov/Sigmoid` | sigmoid-вероятность объекта класса в ячейке |
| bbox | `[4C, gh, gw]` | `output_bbox/BiasAdd` | нормированные смещения границ (x1,y1,x2,y2 на класс) |

Декодер `detectnet_v2_decode` (`deploy/evaluation/evaluate.py`):

- `stride = 16`, `bbox_norm = 35.0`.
- Центр ячейки: `cx = (j+0.5)*16`, `cy = (i+0.5)*16`.
- Смещения границ: `dx1,dy1,dx2,dy2 = bbox[:, i, j] * 35.0`; `x1=cx-dx1`, `y1=cy-dy1`, `x2=cx+dx2`, `y2=cy+dy2`.
- Масштабирование к исходному размеру кадра (`scale_w = orig_w/960`, `scale_h = orig_h/544`).
- Финальный NMS с IoU = 0.5.

```python
cx = (j + 0.5) * stride                       # stride = 16
cy = (i + 0.5) * stride
dx1, dy1, dx2, dy2 = bbox_cls[:, i, j] * bbox_norm   # bbox_norm = 35.0
x1, y1, x2, y2 = cx - dx1, cy - dy1, cx + dx2, cy + dy2
# ... масштаб к orig, затем nms(boxes, iou_thr=0.5)
```

---

## Файлы и форматы

Локальный каталог `/home/mk/CarCV/models/trafficcamnet_pruned_onnx_v1.0.4` (вне репозитория CarCV-metrics):

```
/home/mk/CarCV/models/trafficcamnet_pruned_onnx_v1.0.4/
├── resnet18_trafficcamnet_pruned.onnx       # веса (ONNX), 5 365 751 байт ≈ 5.36 MB
├── labels.txt                               # 4 строки классов (29 байт)
├── nvinfer_config.txt                       # параметры nvinfer/препроцессинг (207 байт)
└── resnet18_trafficcamnet_pruned_int8.txt   # INT8 calibration cache (8 630 байт)
```

| Файл | Размер | Назначение |
|------|--------|------------|
| `resnet18_trafficcamnet_pruned.onnx` | 5 365 751 байт (≈5.36 MB) | веса модели в ONNX (используется для валидации через onnxruntime-gpu) |
| `labels.txt` | 29 байт | список классов, порядок строк = индексы классов |
| `nvinfer_config.txt` | 207 байт | параметры DeepStream nvinfer (вход, цвет, масштаб, число классов) |
| `resnet18_trafficcamnet_pruned_int8.txt` | 8 630 байт | INT8 calibration cache (`TRT-100300-EntropyCalibration2`) для INT8-engine |

- ONNX в комплекте **не зашифрован** (в отличие от `.etlt`, который требует ключа `tlt_encode`); `nvinfer_config.txt` содержит `tlt-model-key=tlt_encode` как наследие TAO-конфига, но локально используется именно ONNX.
- В репозитории `download_models.sh` тянет ONNX в `deploy/models/trafficcamnet/resnet18_trafficcamnet_pruned.onnx` + `labels.txt`. INT8-кэш и `nvinfer_config.txt` скриптом не загружаются (они присутствуют только в локальной копии `/home/mk/CarCV/models/...`).
- FP16/INT8 TensorRT-engine в локальной копии **отсутствует** — engine собирается на целевом Jetson из ONNX (+ INT8-кэш при необходимости).

---

## Классы / выходной словарь

В `labels.txt` ровно 4 класса (порядок строк = индекс класса, подтверждено чтением файла):

| Индекс | Класс | Роль в CARS |
|--------|-------|-------------|
| 0 | `car` | **основной** — кропы идут в SGIE-классификаторы (марка/тип) |
| 1 | `bicycle` | вспомогательный |
| 2 | `person` | вспомогательный (детекция людей) |
| 3 | `road_sign` | вспомогательный |

> Внимание: порядок в файле — `car, bicycle, person, road_sign`. Эвалуатор использует индекс из `labels.txt` (`labels_lower.index(cls_name)`), поэтому декодирование классов привязано именно к этому порядку.

Маппинг меток датасета на классы TrafficCamNet (для подсчёта TP) задан `TRAFFICCAMNET_GT_VOCAB` в `evaluate.py`:

| Класс модели | Категории GT, засчитываемые как TP |
|--------------|------------------------------------|
| `car` | `{car}` |
| `person` | `{person, pedestrian}` |
| `bicycle` | `{bike, bicycle, rider}` |
| `road_sign` | `{traffic sign, sign, traffic_sign, trafficsign}` |

---

## Использование для CarCV

### Применимость

- ✅ **Первичный детектор (PGIE)** в DeepStream-конвейере CARS: ResNet-18 pruned + ONNX 5.36 MB — лёгкая модель под edge-устройство (Jetson Orin Nano 8GB).
- ✅ Класс `car` напрямую обслуживает основной сценарий CARS — детекция ТС и подача кропов в SGIE (VehicleMakeNet, VehicleTypeNet).
- ✅ Совместима с INT8-квантизацией (в комплекте calibration cache) — потенциально ниже латентность/память на Jetson.
- ✅ Дополнительные классы `person`, `bicycle`, `road_sign` дают контекст сцены без отдельных моделей.

### Ограничения

- ❌ **Доменный разрыв (viewpoint).** Модель обучена на видах с дорожных камер (вид сверху/под углом сверху), тогда как automotive POV CARS — бортовая съёмка с уровня дороги (§5.1 целевого сценария: дистанция 5–50 м, угол 0–30°). На уличной фотосъёмке с уровня земли (COCO-суррогат) детектор показал тяжёлый провал (см. «Валидация»).
- ⚠️ Только **4 класса**; нет различения легковой/грузовой (нет `truck`/`bus`) — тип кузова определяется отдельной моделью VehicleTypeNet.
- ⚠️ `maintain-aspect-ratio=0`: resize 960×544 без сохранения пропорций искажает геометрию кадров с иным соотношением сторон (например 1920×1080) — это часть штатного препроцессинга, но влияет на качество боксов.
- ⚠️ Расхождение версий между `models.md` (`pruned_quantized_v2.0.1` / `pruned_v1.0.3`) и фактической локальной/скриптовой (`pruned_onnx_v1.0.4`) — источник истины для измерений: локальная версия.
- ⚠️ FP32-ONNX используется только для валидации на x86; продакшн-engine (FP16/INT8) собирается отдельно на Jetson — измеренные на ONNX числа считаются верхней границей точности для TensorRT-engine.

### Рекомендации

1. **Валидировать на BDD100K, а не на COCO.** COCO val2017 — уличная фотосъёмка с уровня земли (доменный разрыв); целевой бенчмарк PGIE — BDD100K (дорожная съёмка). См. [bdd100k.md](../about_datasets/bdd100k.md).
2. **conf_thr под домен.** Для cross-domain (COCO) — `conf_thr=0.2`; для in-domain (BDD100K, ближе к training-условиям) — `conf_thr=0.4` (заложено в `eval_trafficcamnet`).
3. **Сверить версию модели.** Привести `models.md` к фактической `pruned_onnx_v1.0.4` или явно зафиксировать, какая версия идёт в продакшн.
4. **Поправить `trafficcamnet_eval.yaml`.** Привести препроцессинг/классы черновика в соответствие с `nvinfer_config.txt` и `eval_trafficcamnet` (масштаб `/255`, offsets `0`, NMS IoU `0.5`, 4 класса `car/bicycle/person/road_sign`).
5. **Вердикт pass/fail фиксировать как есть.** FAIL из-за доменного разрыва — валидный окончательный результат, а не повод для тюнинга в рамках валидационной кампании.

---

## Развёртывание на Jetson

- **Целевое устройство:** NVIDIA Jetson Orin Nano 8GB.
- **Роль:** PGIE (первичный детектор) в DeepStream SDK + TensorRT-конвейере.
- **Engine:** TensorRT, продакшн — FP16 (по legacy-докам файл вида `models/baseline/resnet18_trafficcamnet_fp16.engine` — это legacy-путь из `docs/architecture.md`, а не локальная копия модели `/home/mk/CarCV/models/...`). INT8-сборка возможна из ONNX + поставляемого calibration cache (`resnet18_trafficcamnet_pruned_int8.txt`).
- **Целевая latency:** ~8–10 ms (по legacy-докам). ⚠️ **Не измерено** в данном репозитории — это аспирационная цифра из legacy-документации (README / docs/architecture / system-design), не подтверждённый замер. Приводится с явной пометкой «не измерено».
- **Точность TensorRT-engine:** измеряется не здесь; числа из валидации на ONNX (x86) считаются верхней границей точности для engine на Jetson.

---

## Валидация

**Целевой датасет:** BDD100K — спецификация [docs/about_datasets/bdd100k.md](../about_datasets/bdd100k.md). Источник — `datasets.md` (пара Detection × BDD100K).

**Пороги pass/fail** (из `eval_trafficcamnet`, метрика по классу `car` как primary):

| Метрика | Порог |
|---------|-------|
| precision | ≥ 0.90 |
| recall | ≥ 0.85 |
| f1 | ≥ 0.87 |

`conf_thr`: 0.2 для cross-domain (COCO street-level), 0.4 для in-domain (BDD100K).

### Статус кампании

По `docs/superpowers/specs/2026-06-02-validation-campaign-5-models-design.md`: эвалуатор `eval_trafficcamnet` уже реализован; **требуется перегнать модель на настоящем BDD100K val** (добавляется загрузчик `bdd100k.zip` — источник `next.ogoun.name → bdd100k.zip (wget)` по campaign-спеке — + конверсия `bdd100k_labels_images_val.json` в `labels.json` + class-mapping BDD → 4 класса). Прогон на BDD100K — **TBD** (на момент написания не выполнен).

### Предыдущий прогон (суррогат COCO val2017) — FAIL

Измеренный результат (`results_collected/ssh9.qudata.ai/results/trafficcamnet/metrics.json`, `conf_thr=0.2`, мультиклассовый прогон):

| Класс | precision | recall | f1 | ap | num_gt | num_pred | num_tp |
|-------|-----------|--------|-----|-----|--------|----------|--------|
| **car** (primary) | 0.0819 | 0.0528 | 0.0642 | 0.0192 | 1932 | 1245 | 102 |
| person | 0.36 | 0.2435 | 0.2905 | 0.1947 | 11004 | 7441 | 2679 |
| bicycle | 0.0429 | 0.0095 | 0.0155 | 0.0152 | 316 | 70 | 3 |
| road_sign | 0.0233 | 0.0267 | 0.0248 | 0.0048 | 75 | 86 | 2 |
| **macro (4 класса)** | 0.1270 | 0.0831 | 0.0987 | 0.0585 | — | — | — |

**Вердикт по порогам (класс `car`):** precision 0.0819 < 0.90 — FAIL; recall 0.0528 < 0.85 — FAIL; f1 0.0642 < 0.87 — FAIL. → **FAIL**.

**Причина FAIL:** доменный разрыв. TrafficCamNet обучен на видах с дорожных камер (вид сверху), COCO val2017 — уличная фотосъёмка с уровня земли: иной ракурс, масштаб и освещение. FAIL — валидный окончательный результат, не повод для тюнинга.

> **Расхождение в legacy-сводке.** `results_collected/FINAL_REPORT.md` (§1) приводит для TrafficCamNet иные car-числа (`P=0.167, R=0.032, F1=0.053, num_pred=365, num_tp=61`, «car only»). Это, по-видимому, отдельный car-only прогон с другим `conf_thr` (по сводной таблице FINAL_REPORT — `P=0.17`). Авторитетным считается актуальный мультиклассовый `metrics.json` (`P=0.0819`); расхождение со сводкой FINAL_REPORT помечено явно.

> **Legacy-метрики не использовать.** README / docs/architecture / system-design описывают старый стек и аспирационные/непроверенные числа (например TrafficCamNet «P=0.92–0.95»). Эти значения **не** являются измеренными и здесь не воспроизводятся.

---

## Лицензия

- **Модель/веса:** NVIDIA TAO / NGC (purpose-built модель TrafficCamNet). Конкретный текст лицензии — см. карточку модели на NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet. ⚠️ Точные условия (тип лицензии, ограничения на коммерческое использование/дистрибуцию весов и engine) в данной спецификации **не проверены пофайлово** и помечены как требующие сверки с карточкой NGC.

**Вывод для CARS (КОММЕРЧЕСКИЙ продукт):**

- ⚠️ Условия лицензии NVIDIA на purpose-built модель (включая право на встраивание весов/engine в коммерческий продукт и его дистрибуцию заказчику) необходимо **подтвердить legal review** до поставки. NGC-модели NVIDIA обычно поставляются под отдельным лицензионным соглашением — его применимость к сценарию CARS нужно проверить отдельно.
- ⚠️ До legal review рассматривать использование как **внутреннюю валидацию/прототип**; внешнюю поставку (engine в составе продукта, публикация) согласовать с юридическим контролем проекта.

---

## Ссылки

- [Карточка модели TrafficCamNet на NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet)
- [NGC base (versions/pruned_onnx_v1.0.4)](https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_onnx_v1.0.4)
- Локальная модель: `/home/mk/CarCV/models/trafficcamnet_pruned_onnx_v1.0.4`
- Эвалуатор: `deploy/evaluation/evaluate.py` (`eval_trafficcamnet`, `detectnet_v2_decode`, `EVAL_CONFIGS["trafficcamnet"]`)
- Загрузчик весов: `deploy/scripts/download_models.sh`
- Черновик конфига прогона: `configs/experiment/trafficcamnet_eval.yaml`
- Измеренные метрики: `results_collected/ssh9.qudata.ai/results/trafficcamnet/metrics.json`
- Сводный отчёт (legacy/COCO): `results_collected/FINAL_REPORT.md` (§1)
- Спецификация датасета: [docs/about_datasets/bdd100k.md](../about_datasets/bdd100k.md)
- Статус кампании: `docs/superpowers/specs/2026-06-02-validation-campaign-5-models-design.md`

---

## История изменений

- **2026-06-04** — Создана спецификация модели в рамках документирования стека models.md.
