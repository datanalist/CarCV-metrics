# nomeroff_lpd — детекция номерных знаков (LP Detection)

## Общая информация

**nomeroff_lpd** — детектор номерных пластин на базе **Nomeroff Net** (проект `ria-com/nomeroff-net`, сайт nomeroff.net.ua). В конвейере CARS отвечает за стадию **LP Detection**: локализует номерную пластину в кадре и одновременно выдаёт **4 угловые точки (keypoints)** пластины, по которым выполняется перспективное выравнивание кропа перед OCR (`nomeroff_ocr`).

Модель введена как **замена US-обученного LPDNet (NVIDIA TAO)** для RU/UA-сегмента. Legacy-LPDNet консервативен на пластинах не-US формата (см. раздел «Валидация» — контраст), поэтому для российских/украинских номеров используется keypoint-детектор Nomeroff Net.

| Поле | Значение | Источник |
|------|----------|----------|
| Имя в стеке | `nomeroff_lpd` | `models.md` |
| Задача в конвейере CARS | LP Detection (bbox пластины + 4 угла) | `models.md`, `_bmad-output/project-context.md` |
| Семейство / архитектура | YOLOv11x (Ultralytics YOLO11, размер `x`) с keypoint/pose-головой | URL весов, карточка Nomeroff Net |
| Вендор / источник | Nomeroff Net (`ria-com/nomeroff-net`) | `models.md` |
| URL весов | https://nomeroff.net.ua/models/object_detection/yolov11x-keypoints-2026-01-21.pt | `models.md` |
| Версия (по `models.md`) | `None` | `models.md` (поле `version` = `None`) |
| Локальный путь весов | не задан (столбец `Local path` в строке отсутствует) | `models.md` (5-й столбец опущен, см. ниже) |
| Запуск | Python / nomeroff-net (`pipeline("number_plate_localization")`) | `deploy/evaluation/evaluate.py` |

> **О версии и локальном пути.** Заголовок таблицы `models.md` содержит 5 столбцов (`Task | model name | URL | version | Local path`), но строка `nomeroff_lpd` заполнена лишь 4 столбцами: поле `version` = `None`, а 5-й столбец `Local path` в строке **отсутствует вовсе** (не задан, а не равен литералу `None` — в отличие, например, от строки TrafficCamNet, где заполнены все 5 столбцов). По имени файла весов фактическая версия — **`yolov11x-keypoints-2026-01-21`** (датирована 2026-01-21). Локальной копии `.pt` в репозитории всё равно нет: веса подтягиваются пайплайном `nomeroff-net` по требованию (lazy download при первом вызове `pipeline("number_plate_localization")`), что подтверждается отсутствием `model_path` в `EVAL_CONFIGS["nomeroff_lpd"]` (`deploy/evaluation/evaluate.py`).

---

## Архитектура

- **Семейство:** YOLO11 (Ultralytics) — однопроходный детектор; вариант **`x`** (extra-large, самый тяжёлый размер семейства).
- **Голова:** keypoint/pose-голова — помимо bounding box каждой пластины модель регрессирует **4 угловые точки** (углы четырёхугольника пластины), что напрямую соответствует перспективно-искажённой пластине в реальной сцене.
- **Веса:** PyTorch `.pt` (`yolov11x-keypoints-2026-01-21.pt`). Часть pipeline `number_plate_localization` в `nomeroff-net`.
- **Pruning / quantization:** не применяется в исходных весах — это полноразмерная модель `x` без прунинга/квантизации (в отличие от pruned-моделей TAO в остальном стеке CARS). Квантизация/оптимизация — только на этапе экспорта под Jetson (см. «Развёртывание на Jetson»).

> Точные гиперпараметры архитектуры (число слоёв, размер входа по умолчанию, число параметров) в локальной копии **отсутствуют** — модель не лежит в репозитории и не была вскрыта; описание опирается на имя файла весов (`yolov11x-keypoints`) и pipeline `nomeroff-net`. Помечено как **непроверяемо локально**.

---

## Вход / выход и препроцессинг

Модель вызывается **не напрямую**, а через высокоуровневый pipeline `nomeroff-net`. В эвалуаторе CARS (`deploy/evaluation/evaluate.py`, `eval_nomeroff_lpd`) подаётся **путь к файлу полного кадра**, а весь препроцессинг выполняется **внутри YOLO** (Ultralytics): letterbox-resize, `/255`, перевод в RGB, формат NCHW. Параметры net-scale-factor/offsets, как у TAO-моделей CARS, здесь **не задаются вручную** — препроцессинг инкапсулирован в pipeline.

```python
# deploy/evaluation/evaluate.py — eval_nomeroff_lpd (фрагмент)
from nomeroff_net import pipeline
from nomeroff_net.tools import unzip as nomeroff_unzip

detector = pipeline("number_plate_localization", image_loader="opencv")

# подаётся ПОЛНЫЙ кадр (путь к файлу), препроцессинг — внутренний для YOLO
raw = detector([str(img_path)])
images_bboxs, _images = nomeroff_unzip(raw)
bboxs = images_bboxs  # список, индексируется номером изображения; bboxs[0] — для первого
preds_per_img = [[float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                  float(b[4]) if len(b) > 4 else 1.0]
                 for b in ((bboxs[0] or []) if bboxs else [])
                 if b is not None]
```

| Параметр | Значение | Примечание |
|----------|----------|------------|
| Вход эвалуатора | путь к файлу полного кадра | `detector([str(img_path)])` |
| Загрузчик изображений | `image_loader="opencv"` | задаётся при создании pipeline |
| Цветовой формат | RGB (внутри YOLO) | препроцессинг инкапсулирован |
| Масштаб | `/255` (внутри YOLO) | препроцессинг инкапсулирован |
| Resize | letterbox (внутри YOLO) | препроцессинг инкапсулирован |
| Формат тензора | NCHW | семейство YOLO |
| Разрешение входа (W×H) | внутреннее значение YOLO11 (по умолчанию обычно 640) — **в локальной копии не зафиксировано (TBD)** | не задаётся в `evaluate.py` |

> **Семейство препроцессинга.** В диспетчере препроцессинга проекта (`_bmad-output/project-context.md`) перечислены три семейства: Generic ImageNet classifier, TAO classifier и DetectNet_v2. Nomeroff `nomeroff_lpd` **не относится ни к одному из них** — это YOLO11 с собственным (внутренним) препроцессингом Ultralytics. Не путать с DetectNet_v2 (TrafficCamNet/FaceNet: BGR→RGB, `/255`, ручной декодер) — здесь весь препроцессинг и декодирование боксов скрыты в pipeline.

### Выход и его декодирование

`pipeline("number_plate_localization")` возвращает структуру, которая через `nomeroff_unzip(raw)` разбивается на **2 слота**: `(images_bboxs, images)`. `images_bboxs` — список, индексированный по изображению; для одного входа берётся `bboxs[0]`. Каждый элемент детекции:

```
[x1, y1, x2, y2, conf, cls_int, kps_normalized]
```

| Индекс | Поле | Семантика |
|--------|------|-----------|
| `b[0..3]` | `x1, y1, x2, y2` | axis-aligned bbox пластины (абс. пиксели) |
| `b[4]` | `conf` | уверенность детекции |
| `b[5]` | `cls_int` | целочисленный класс (единственный — пластина) |
| `b[6]` | `kps_normalized` | **4 нормированных keypoints** (углы пластины) |

В CARS-эвалуаторе для метрики **используется только bbox** (`b[0..4]`); keypoints (`b[6]`) в `eval_nomeroff_lpd` не извлекаются — они доступны в выходе модели, но текущая метрика их не задействует (4 угла GT-полигона напрямую сопоставимы с 4 keypoints, например через OKS, — это **потенциальная**, ещё не реализованная метрика; см. спеку датасета).

---

## Файлы и форматы

| Файл | Формат | Размер | Статус |
|------|--------|--------|--------|
| `yolov11x-keypoints-2026-01-21.pt` | PyTorch checkpoint (`.pt`) | в локальной копии отсутствует (не измерен) | тянется пайплайном nomeroff-net по требованию |
| labels / алфавит | — | — | один класс (пластина), отдельного файла меток нет |
| calibration | — | — | отсутствует (квантизация под Jetson не выполнялась локально) |

```
# Локально в репозитории CarCV-metrics весов нет:
#   models.md → строка nomeroff_lpd: столбец Local path не задан (5-й столбец опущен; version = None)
#   EVAL_CONFIGS["nomeroff_lpd"] не содержит "model_path"
# Веса скачиваются nomeroff-net при первом pipeline("number_plate_localization").
# В deploy/scripts/download_models.sh nomeroff_lpd НЕ перечислен
#   (скрипт качает только US-LPDNet: lpdnet/LPDNet_usa_pruned_tao5.onnx).
```

> Файл весов **не зашифрован** (обычный PyTorch `.pt`, в отличие от `.etlt` TAO-моделей). Однако для Jetson требуется отдельный экспорт `.pt` → ONNX → TensorRT (см. ниже) — это нетривиальный шаг, а не прямая загрузка готового engine.

---

## Классы / выходной словарь

- **Один класс:** номерная пластина (`numberplate`).
- Дополнительно к боксу — **4 keypoints** (угловые точки пластины) на каждую детекцию.
- Алфавита/словаря символов у этой модели **нет**: распознавание текста номера выполняет отдельная модель **`nomeroff_ocr`** (CRNN/ResNet18 + CTC), вне зоны ответственности `nomeroff_lpd`.

---

## Использование для CarCV

### Применимость

- ✅ **Стадия LP Detection RU/UA-сегмента.** Заменяет US-обученный LPDNet, который пропускает не-US пластины (низкий recall — см. «Валидация»).
- ✅ **Keypoint-выход (4 угла) для перспективного выравнивания.** Углы пластины позволяют выпрямить перспективно искажённый кроп перед подачей в OCR — это прямое улучшение точности `nomeroff_ocr`.
- ✅ **Один объектный класс, простой контракт выхода** (`[x1,y1,x2,y2,conf,cls,kps]`) — удобно сводить к bbox-метрикам (IoU) и keypoint-метрикам (OKS).
- ✅ **Высокая измеренная точность** на целевом RU/UA-распределении (PASS, P=0.91 / R=0.92 — см. «Валидация»).

### Ограничения

- ⚠️ **Самая тяжёлая модель стека CARS** (YOLO11 размер `x`, без прунинга). Несёт основные риски по latency и памяти на Jetson Orin Nano 8GB.
- ⚠️ **Запуск через Python/nomeroff-net, а не через DeepStream SGIE.** В отличие от legacy-LPDNet (DeepStream SGIE3), `nomeroff_lpd` работает как Python-сервис — интеграция в DeepStream-конвейер требует отдельного экспорта и обвязки.
- ⚠️ **Веса не лежат локально** — тянутся пайплайном по требованию; нет калибровочного набора и готового engine в репозитории.
- ⚠️ **GPL-3.0 у кода nomeroff-net** — для коммерческого продукта это copyleft-обязательство, требующее legal review (см. «Лицензия»).
- ❌ **Не распознаёт текст номера.** Только локализация + углы; OCR — отдельная модель `nomeroff_ocr`.
- ⚠️ **Точное разрешение входа и параметры архитектуры локально не зафиксированы** (TBD) — препроцессинг скрыт в pipeline.

### Рекомендации

1. **Использовать как основной LP-детектор для RU/UA.** US-LPDNet оставить только как контрольный baseline (на нём задокументирован FAIL по recall — см. «Валидация»).
2. **Задействовать keypoints для выравнивания.** В CARS-эвалуаторе keypoints (`b[6]`) сейчас не извлекаются; при интеграции с OCR использовать 4 угла для перспективной коррекции кропа.
3. **Жёстко контролировать latency/память на Jetson.** Перед продакшеном измерить фактическую задержку engine FP16/INT8 на Orin Nano 8GB; при превышении бюджета рассмотреть более лёгкий размер YOLO11 (`m`/`l`) или прунинг.
4. **Зафиксировать версию весов.** В `models.md` версия = `None`; закрепить фактический тег `yolov11x-keypoints-2026-01-21` и кэшировать `.pt` локально для воспроизводимости (сейчас зависит от доступности nomeroff.net.ua).
5. **Провести legal review GPL-3.0** до включения кода nomeroff-net в дистрибутив коммерческого продукта.

---

## Развёртывание на Jetson

| Параметр | Значение | Примечание |
|----------|----------|------------|
| Целевая платформа | NVIDIA Jetson Orin Nano 8GB | продакшн CARS |
| Формат продакшн-движка | TensorRT engine (FP16 / INT8) | требует экспорта |
| Путь экспорта | `.pt` → ONNX → TensorRT | нетривиальный шаг |
| Роль в конвейере | Python-сервис (nomeroff-net), **не** DeepStream SGIE | в отличие от legacy-LPDNet (SGIE3) |
| Целевая latency | **не определена (TBD)** — требует замера | модель — самая тяжёлая в стеке |

> **Риск.** YOLO11x — наиболее ресурсоёмкая модель стека. На Jetson Orin Nano 8GB экспорт `.pt` → ONNX → TensorRT обязателен (готового engine нет), а итоговые latency и потребление памяти **должны быть измерены отдельно** — они могут стать узким местом конвейера реального времени. Это явный риск, а не подтверждённый результат.

---

## Валидация

**Датасет:** AUTO.RIA Numberplate Dataset (Detection), версия `autoriaNumberplateDataset-2021-05-12` — детекционный корпус полных кадров с VIA-полигонами пластин. Спека: [autoria_numberplate_detection_ru.md](../about_datasets/autoria_numberplate_detection_ru.md).

**Метрика.** GT-полигоны сводятся к axis-aligned bbox (`min/max` по `all_points_x`/`all_points_y`), предсказания — bbox из выхода pipeline; сопоставление по **IoU** с `conf_threshold=0.3` (`compute_detection_metrics`, `eval_nomeroff_lpd`). 4 keypoints выхода напрямую сопоставимы с 4 углами GT-полигона (метрика по углам / OKS — потенциальная, в текущем эвалуаторе не реализована).

**Пороги pass/fail** (`deploy/evaluation/evaluate.py`, `thresholds = {"recall": 0.80, "precision": 0.70}`):

| Метрика | Порог | Измерено | Статус |
|---------|-------|----------|--------|
| Recall | ≥ 0.80 | **0.9221** | PASS |
| Precision | ≥ 0.70 | **0.9056** | PASS |

**Полный измеренный результат** (`results_collected/ssh9.qudata.ai/results/nomeroff_lpd/metrics.json`, модель `"nomeroff-net localization (RU)"`):

| Показатель | Значение |
|------------|----------|
| precision | 0.9056 |
| recall | 0.9221 |
| f1 | 0.9138 |
| ap | 0.8806 |
| map50 | 0.8806 |
| num_gt | 385 |
| num_pred | 392 |
| num_tp | 355 |
| skipped | 0 |

**Вердикт: PASS** (оба порога пройдены). В дизайне кампании округлено до **P=0.91 / R=0.92**.

> **Статус в кампании.** `nomeroff_lpd` уже отвалидирован на правильном (детекционном) датасете и потому **вне активных задач** валидационной кампании 5 моделей (`docs/superpowers/specs/2026-06-02-validation-campaign-5-models-design.md`): результат зачитывается как есть, повторный прогон/тюнинг не требуется. **FAIL был бы валидным окончательным результатом** — в данном случае получен PASS.

> **Контраст: US-LPDNet (TAO) — FAIL по recall.** На том же локальном детекционном датасете (`data/nomeroff_lp`, `num_gt=385`) baseline-LPDNet даёт **P=0.8837, R=0.2961, F1=0.4436, ap=0.2616** (`results_collected/ssh9.qudata.ai/results/lpdnet/metrics.json`) — высокая точность, но катастрофически низкий recall: детектор консервативен и пропускает RU/UA-пластины. Это и есть обоснование замены LPDNet на `nomeroff_lpd`.

> **Расхождение с legacy-доками.** В legacy `results_collected/FINAL_REPORT.md` (§3) задокументирован только US-LPDNet на датасете `autoriaNumberplateDataset-2018-11-20` (val=375), а в дизайн-доке кампании упоминается версия 2019 — там же фигурирует прошлый PASS `nomeroff_lpd` (P=0.91/R=0.92). Актуальные `metrics.json` для обоих детекторов получены на одном локальном срезе с `num_gt=385`. Числа FINAL_REPORT по версиям датасета (375 vs 385) трактуйте как **legacy/неточные**; за измеренный результат принимаются значения из `metrics.json`. FINAL_REPORT прямой строки по `nomeroff_lpd` не содержит — он описывает только legacy-стек US-LPDNet/LPRNet.

---

## Лицензия

- **Код Nomeroff Net** (`ria-com/nomeroff-net`): **GPL-3.0** (copyleft). Использование кода pipeline `number_plate_localization` в составе продукта влечёт обязательства GPL-3.0 (раскрытие исходников производных работ и т. п.).
- **Веса** (`yolov11x-keypoints-2026-01-21.pt`, nomeroff.net.ua): распространяются с сайта проекта; отдельная лицензия именно на файл весов **в локальной копии отсутствует и не подтверждена** (TBD — уточнить у правообладателя/в карточке модели).
- **Данные AUTO.RIA** (датасет для валидации) — отдельно под **CC BY 4.0** (см. спеку датасета). Это лицензия **на данные**, к лицензии на код/веса отношения не имеет.

**Вывод для CARS (КОММЕРЧЕСКИЙ продукт):**
- ⚠️ **GPL-3.0 у кода nomeroff-net — copyleft.** Включение этого кода в дистрибутив или в процессы коммерческого продукта требует **обязательного legal review**: GPL-3.0 может вынуждать раскрывать исходный код производных компонентов. Это ключевой юридический риск стадии LP Detection.
- ⚠️ **Лицензия на сами веса не подтверждена** — до коммерческого использования необходимо явно прояснить условия распространения `.pt` с nomeroff.net.ua.
- ✅ Валидационные **метрики** на датасете AUTO.RIA можно публиковать (датасет под CC BY 4.0 при корректной атрибуции) — но это касается данных, а не кода/весов модели.

---

## Ссылки

- [Веса модели `yolov11x-keypoints-2026-01-21.pt` (Nomeroff Net)](https://nomeroff.net.ua/models/object_detection/yolov11x-keypoints-2026-01-21.pt)
- [Сайт проекта Nomeroff Net](https://nomeroff.net.ua/)
- [GitHub `ria-com/nomeroff-net`](https://github.com/ria-com/nomeroff-net)
- [Спека датасета валидации: AUTO.RIA Numberplate Dataset (Detection)](../about_datasets/autoria_numberplate_detection_ru.md)
- [Ultralytics YOLO11 (семейство архитектуры)](https://docs.ultralytics.com/models/yolo11/)
- Исходники CARS: `deploy/evaluation/evaluate.py` (`eval_nomeroff_lpd`, `eval_lpdnet`), `results_collected/ssh9.qudata.ai/results/{nomeroff_lpd,lpdnet}/metrics.json`, `models.md`, `docs/superpowers/specs/2026-06-02-validation-campaign-5-models-design.md`

---

## История изменений

- **2026-06-04** — Создана спецификация модели в рамках документирования стека `models.md`.
