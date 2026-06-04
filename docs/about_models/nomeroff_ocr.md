# nomeroff_ocr — распознавание текста номера (OCR)

## Общая информация

**nomeroff_ocr** — OCR-модель для чтения текста российского регистрационного знака (ГОСТ-Р 50577) с уже вырезанного кропа пластины. Это PyTorch-чекпойнт CRNN из проекта **Nomeroff Net** (`ria-com/nomeroff-net`, сайт `nomeroff.net.ua`), RU-вариант `model_v3.3` на backbone ResNet-18.

В стеке CARS `nomeroff_ocr` — финальная ступень распознавания номера: после детекции/выравнивания пластины (LP Detection, `nomeroff_lpd`) кроп подаётся в OCR, который возвращает строку символов. Реализован как **Python-сервис** (PyTorch / ONNX Runtime), а не как DeepStream-узел PGIE/SGIE. По `models.md` `nomeroff_ocr` назначен официальной OCR-моделью CARS и **заменяет US-обученный LPRNet (NVIDIA TAO)** для российского сегмента: US-модель на RU-номерах проваливает валидацию (см. § Валидация).

| Поле | Значение |
|------|----------|
| Имя в `models.md` | `nomeroff_ocr` |
| Задача | OCR — чтение текста однострочного RU-номера с кропа |
| Роль в конвейере CARS | Python-сервис (после LP Detection) |
| Семейство/архитектура | CRNN (CNN ResNet-18 + рекуррентная часть + CTC-декодер) |
| Вендор/источник | Nomeroff Net (`ria-com/nomeroff-net`, `nomeroff.net.ua`) |
| Версия (по `models.md`) | `None` (поле version пустое в `models.md`); веса — `model_v3.3`, `resnet18`, RU |
| URL весов | https://nomeroff.net.ua/models/ocr/ru/torch/model_v3.3/resnet18/anpr_ocr_ru_2023_02_01_resnet18.ckpt |
| Локальный путь (по `models.md`) | `None` — фреймворк `nomeroff-net` тянет чекпойнт по требованию (`model_path="latest"`) |
| Локальный путь (фактически найден на диске) | `/home/mk/CarCV/models/nomeroff_net/ocr/anpr_ocr_ru_2023_02_01_resnet18.ckpt` (~27 MB, не в этом репозитории) |

> **Пометка о расположении.** В `models.md` локальный путь указан как `None` (чекпойнт скачивается `nomeroff-net` на лету при `model_path="latest"`). При этом на машине разработчика обнаружена ручная копия чекпойнта `anpr_ocr_ru_2023_02_01_resnet18.ckpt` (~27 MB) по пути `/home/mk/CarCV/models/nomeroff_net/ocr/` — это вне репозитория `CarCV-metrics` и не является источником истины; эвалуатор её не использует напрямую (он полагается на загрузчик весов nomeroff-net).

---

## Архитектура

- **Семейство:** CRNN (Convolutional Recurrent Neural Network) с CTC-декодером — классическая связка для распознавания однострочного текста переменной длины.
- **Backbone (feature extractor):** ResNet-18 (по имени файла весов `..._resnet18.ckpt` и каталогу URL `.../resnet18/`).
- **Рекуррентная часть + декодер:** последовательное кодирование признаков и **CTC** (Connectionist Temporal Classification) для выравнивания последовательности символов без посимвольной разметки. Точная конфигурация рекуррентного блока задаётся внутри фреймворка Nomeroff Net (в локальной копии репозитория CARS-metrics исходники nomeroff-net отсутствуют — деталь не верифицируется здесь).
- **Формат весов:** PyTorch-чекпойнт `.ckpt` старого образца. Чекпойнт содержит сериализованный объект `StrLabelConverter` (конвертер индексов↔символов), из-за чего его нельзя загрузить в безопасном режиме `weights_only=True` (см. § Вход/выход и препроцессинг, нюанс окружения).
- **Pruning/quantization:** в публичной карточке весов не заявлены; модель и так лёгкая (ResNet-18 CRNN). FP16/INT8 — вопрос экспорта под Jetson (см. § Развёртывание).

> **Семейство препроцессинга.** `nomeroff_ocr` — это CRNN/CTC, и он **не относится** ни к одному из трёх семейств из таблицы препроцессинга `_bmad-output/project-context.md` (Generic ImageNet classifier / TAO classifier / DetectNet_v2). Препроцессинг для RU-OCR выполняется **внутри пайплайна Nomeroff Net**, а не диспетчером CARS (см. ниже).

---

## Вход / выход и препроцессинг

В эвалуаторе CARS (`deploy/evaluation/evaluate.py`, `eval_nomeroff_ocr`) модель используется **напрямую через `NumberPlateTextReading`** (task `"number_plate_text_reading"`), минуя YOLO-детекцию. Препроцессинг (resize/нормализация/CTC-декод) полностью инкапсулирован в пайплайне Nomeroff Net — в коде CARS он не дублируется. Поэтому точные значения resize-разрешения / mean / std не задаются в репозитории и являются внутренней деталью nomeroff-net (**не верифицируются здесь**; для справки legacy-линия LPR_STN использовала вход 188×48×3 RGB — это другая модель, см. § Валидация).

**Параметры пайплайна** (как в `eval_nomeroff_ocr`):

| Параметр | Значение | Назначение |
|----------|----------|------------|
| `task` | `"number_plate_text_reading"` | только чтение текста, без детекции |
| `image_loader` | в конструктор передаётся `image_loader="opencv"`, затем `text_reader.image_loader` переопределяется на `DumpyImageLoader()` (`evaluate.py:904`, `910`) | `DumpyImageLoader` передаёт numpy-кроп «как есть», без повторной YOLO-детекции |
| `presets["ru"]["for_regions"]` | `["ru"]` | регион — Россия |
| `presets["ru"]["for_count_lines"]` | `[1]` | однострочные номера |
| `presets["ru"]["model_path"]` | `"latest"` | подтянуть актуальные RU-веса по требованию |
| `default_label` | `"ru"` | дефолтная разметка региона |
| `off_number_plate_classification` | `True` | отключить классификацию типа/региона пластины |

**Вход:** numpy-кроп пластины (как из `cv2.imread`, BGR uint8). На вход пайплайну подаётся список кортежей `(zone_array, region_label, count_lines, preprocessed_np)`:

```python
raw = text_reader([(img, "ru", 1, None)])
texts, _images = nomeroff_unzip(raw)
pred_text = (texts[0] if texts else "").upper().strip()
```

**Выход:** строка текста номера (нативный RU-алфавит, см. § Классы). `NumberPlateTextReading` возвращает список `(text, zone_array)`; после `nomeroff_net.tools.unzip` получаются 2 слота: `(texts, images)`.

**Нюанс окружения (важно).** PyTorch ≥ 2.6 сменил дефолт `torch.load` на `weights_only=True`, что ломает загрузку старого чекпойнта nomeroff (он содержит `StrLabelConverter`). В `eval_nomeroff_ocr` это решается **временным monkey-patch** `torch.load(..., weights_only=False)`, применяемым **только на время инициализации** пайплайна и затем откатываемым:

```python
def _torch_load_unsafe(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(f, *args, **kwargs)
```

> ⚠️ Патч **не потокобезопасен** — `eval_nomeroff_ocr` нельзя запускать параллельно с другим кодом, вызывающим `torch.load`.

---

## Файлы и форматы

| Файл | Расположение | Размер | Примечание |
|------|--------------|--------|------------|
| `anpr_ocr_ru_2023_02_01_resnet18.ckpt` | по URL nomeroff.net.ua (тянется на лету); ручная копия — `/home/mk/CarCV/models/nomeroff_net/ocr/` | ~27 MB (27 798 077 байт у локальной копии) | PyTorch-чекпойнт, содержит `StrLabelConverter`; требует `weights_only=False` |
| Веса в репозитории `CarCV-metrics` | — | — | **в репозитории отсутствуют** (модель скачивается nomeroff-net) |
| ONNX/TensorRT engine | — | — | **в локальной копии отсутствует**; экспорт под Jetson — TBD (см. § Развёртывание) |
| labels/алфавит как отдельный файл | — | — | алфавит зашит внутри `StrLabelConverter` в чекпойнте, отдельного `labels.txt` нет |

> В отличие от US-LPRNet (ONNX `models/lprnet/us_lprnet_baseline18_deployable.onnx`, ~57.7 MB), для `nomeroff_ocr` в репозитории CARS **нет** локального onnx/engine — инференс идёт через PyTorch-пайплайн nomeroff-net. Шифрования весов нет (открытый `.ckpt`); закрыт только режим `weights_only`.

---

## Классы / выходной словарь

RU-номера (ГОСТ-Р 50577) используют **только 12 букв, визуально совпадающих с латиницей**, плюс 10 цифр. Готовой строки-литерала с этим RU-алфавитом в `evaluate.py` **нет**: он выводится из значений словаря `RU_TO_LATIN` (12 букв) плюс цифры `0–9`. Производная нотация (составлено из `RU_TO_LATIN` + цифры; не фрагмент кода `evaluate.py`):

```
0 1 2 3 4 5 6 7 8 9   A B E K M H O P C T Y X
```

| Группа | Символы | Кол-во |
|--------|---------|--------|
| Цифры | `0 1 2 3 4 5 6 7 8 9` | 10 |
| Буквы (кириллица, визуально = латиница) | А В Е К М Н О Р С Т У Х | 12 |
| **Итого содержательных** | | **22** |

Транслитерация RU→Latin (из `evaluate.py`, `RU_TO_LATIN`), используется **только для сравнения** с US-моделью; сам `nomeroff_ocr` выдаёт нативный RU-текст:

```
А→A  В→B  Е→E  К→K  М→M  Н→H  О→O  Р→P  С→C  Т→T  У→Y  Х→X
```

> **Отличие от US LPRNet.** US LPRNet работает с полным латинским алфавитом `A–Z` + `0–9` (36 символов, `US_CHARS` в `evaluate.py`) и использует символ `Z` (индекс 35) как CTC-blank. У `nomeroff_ocr` алфавит — нативный RU (22 содержательных символа); это и есть причина несовместимости US-модели с RU-номерами.

---

## Использование для CarCV

### Применимость (✅)

- ✅ **Целевая задача OCR §6.4** — чтение однострочного RU-номера. Подтверждён **PASS** на `autoriaNumberplateOcrRu-2021-09-01/val` (char_accuracy=0.9995, full_plate_accuracy=0.9978).
- ✅ **Нативный RU-алфавит** (ГОСТ-Р 50577, 22 символа) — правильный домен, в отличие от US LPRNet.
- ✅ **Лёгкая модель** (ResNet-18 CRNN, ~27 MB чекпойнт) — реалистична для бортового Python-сервиса; контрастирует с тяжёлым YOLO11x в `nomeroff_lpd`.
- ✅ Работает на **готовых кропах** (numpy через `DumpyImageLoader`), без повторной детекции — корректная стыковка после ступени LP Detection.

### Ограничения (❌/⚠️)

- ⚠️ **Только однострочные номера** (`for_count_lines=[1]`). Двухстрочные/спецсерии этим пресетом не покрываются; валидационный датасет их фактически не содержит (`count_lines=0`).
- ⚠️ **Нюанс загрузки весов** (PyTorch ≥ 2.6, `weights_only`): требует monkey-patch `torch.load(weights_only=False)`, **не потокобезопасный** — нельзя параллелить с другими вызовами `torch.load`.
- ⚠️ **Риск train-leakage** при оценке: `autoriaNumberplateOcrRu` — первоисточник обучения Nomeroff Net OCR; сверхвысокий результат (0.9995) может быть оптимистично смещён (см. спеку датасета).
- ⚠️ **Препроцессинг непрозрачен** для CARS: resize/нормализация/CTC-декод инкапсулированы в nomeroff-net; точные параметры входа (разрешение, mean/std) **не зафиксированы в репозитории** и не верифицированы здесь.
- ❌ **Не покрывает детекцию пластины** — это OCR-only; recall детектора обеспечивает `nomeroff_lpd`.
- ❌ **Нет локального ONNX/TensorRT-engine** в репозитории — продакшн-экспорт под Jetson ещё не оформлен (TBD).

### Рекомендации

1. **Источник истины — `description`.** При валидации эталон брать только из поля `description` JSON-аннотации (не из имени файла, `name` или `predicted`) — как и реализовано в `eval_nomeroff_ocr` (`data.get("description") or data.get("name")`).
2. **Не использовать train для оценки.** Считать метрики на `val` (4893 кропа) / `val+test`; train (49 382) исключить из-за риска train-leakage.
3. **Изолировать загрузку весов.** Из-за monkey-patch `torch.load` запускать `eval_nomeroff_ocr` отдельно, без конкурентных вызовов `torch.load`.
4. **Не путать с legacy LPR_STN.** Метрики из `docs/architecture.md` (LPR_STN_PRE_POST, char 99.44%) относятся к **другой** модели и не должны выдаваться за результаты `nomeroff_ocr` (см. § Валидация).
5. **Перед продакшеном — оформить экспорт.** Зафиксировать ONNX/TensorRT-engine и измерить latency на Jetson Orin Nano (сейчас TBD).
6. **Учитывать однострочное ограничение.** Для двухстрочных/нестандартных номеров пресет `for_count_lines=[1]` недостаточен; при необходимости расширять пресеты и переоценивать.

---

## Развёртывание на Jetson

- **Роль:** Python-сервис (PyTorch / ONNX Runtime), запускается после ступени LP Detection (детекция/выравнивание пластины). Это **не** DeepStream PGIE/SGIE-узел.
- **Модель лёгкая:** ResNet-18 CRNN, чекпойнт ~27 MB — реалистично для бортового исполнения на Jetson Orin Nano 8GB. Контрастирует с тяжёлым YOLO11x в `nomeroff_lpd`.
- **TensorRT engine (FP16/INT8):** **в локальной копии отсутствует**; готового onnx/engine для `nomeroff_ocr` в репозитории CARS-metrics нет. Экспорт `.ckpt` → ONNX → TensorRT под Jetson — **TBD** (не оформлен).
- **Целевая latency:** **не измерена** (для справки: legacy LPR_STN давал ~5.04 мс на CPU — это другая модель, не `nomeroff_ocr`).

> Измеренные числа точности (см. § Валидация) получены на x86 GPU-сервере через PyTorch-пайплайн nomeroff-net и считаются **верхней границей** точности для будущего TensorRT-engine на Jetson.

---

## Валидация

- **Датасет:** `autoriaNumberplateOcrRu-2021-09-01`, сплит **val = 4893 кропа**. Спека: [`docs/about_datasets/autoria_numberplate_ocr_ru.md`](../about_datasets/autoria_numberplate_ocr_ru.md).
- **Пороги pass/fail** (`evaluate.py`, `eval_nomeroff_ocr`): `char_accuracy ≥ 0.90`, `full_plate_accuracy ≥ 0.80`.
- **Метрики** (`metrics.py`, `compute_ocr_metrics`): `char_accuracy` — посимвольное совпадение `zip(pred, gt)` / `max(len)`; `full_plate_accuracy` — доля точных совпадений строки; `char_error_rate = 1 − char_accuracy`.

**Измеренный результат** (`results_collected/ssh9.qudata.ai/results/nomeroff_ocr/metrics.json`, модель `"nomeroff-net NumberPlateTextReading direct (RU)"`):

| Метрика | Значение | Порог | Статус |
|---------|----------|-------|--------|
| `char_accuracy` | **0.9995** | ≥ 0.90 | ✅ PASS |
| `full_plate_accuracy` | **0.9978** | ≥ 0.80 | ✅ PASS |
| `char_error_rate` | 0.0005 | — | — |
| `num_samples` | 4893 | — | — |
| `skipped` | 0 | — | — |

**Вердикт: PASS.** Модель уже отвалидирована и находится **вне активных задач** валидационной кампании (см. `docs/superpowers/specs/2026-06-02-validation-campaign-5-models-design.md`).

**Контраст: US LPRNet (NVIDIA TAO) на тех же данных — FAIL** (`results_collected/ssh9.qudata.ai/results/lprnet/metrics.json`):

| Метрика | nomeroff_ocr (RU) | US LPRNet | Порог |
|---------|-------------------|-----------|-------|
| `char_accuracy` | 0.9995 ✅ | 0.5904 ❌ | ≥ 0.90 |
| `full_plate_accuracy` | 0.9978 ✅ | 0.0621 ❌ | ≥ 0.80 |

Причина провала US-модели — неверный алфавит/формат: US-знак 6–7 символов (латиница `A–Z`), RU-знак 8–9 символов (кириллица-как-латиница); плюс критический фикс декодера — NVIDIA US LPRNet использует символ `Z` (индекс 35) как CTC-blank. Это и обосновало замену LPRNet на `nomeroff_ocr` для RU-сегмента.

> **Legacy-замечание (не путать метрики).** `docs/architecture.md` §6 описывает **другую** RU-OCR-модель — **LPR_STN_PRE_POST** (STN + CNN + Bi-LSTM + CTC, вход 188×48×3, алфавит 23 символа с `-`, char 99.44% / plate 98.75%). Это **отдельная legacy-линия**; в актуальном `models.md` OCR-моделью назначен именно `nomeroff_ocr`. Метрики LPR_STN — **legacy/непроверенные в рамках этой кампании** и не относятся к `nomeroff_ocr`.

---

## Лицензия

- **Код Nomeroff Net** (`ria-com/nomeroff-net`): **GPL-3.0** (copyleft).
- **Веса** (`anpr_ocr_ru_2023_02_01_resnet18.ckpt`): распространяются с `nomeroff.net.ua` в составе проекта Nomeroff Net.
- **Обучающие данные** (`autoriaNumberplateOcrRu`): **CC BY 4.0** (см. спеку датасета) — коммерчески совместимы при указании авторства.

**Вывод для CARS (КОММЕРЧЕСКИЙ продукт):**
- ⚠️ **GPL-3.0 — copyleft.** Распространение продукта, линкующего/включающего GPL-код Nomeroff Net, может потребовать раскрытия исходников производной работы. Для коммерческой поставки CARS — **обязателен legal review** на предмет того, как именно `nomeroff_ocr` интегрируется (отдельный процесс/сервис vs. линковка в один бинарь) и распространяется.
- ⚠️ Использование как **внутренний benchmark** (валидация OCR, отчёты команды) — допустимо в research-режиме; продакшн-дистрибуция требует юридической проработки лицензии GPL-3.0.
- ✅ Сами данные валидации (CC BY 4.0) для метрик/публикации безопасны при корректном указании авторства.

---

## Ссылки

- [Веса nomeroff_ocr (resnet18, RU, model_v3.3)](https://nomeroff.net.ua/models/ocr/ru/torch/model_v3.3/resnet18/anpr_ocr_ru_2023_02_01_resnet18.ckpt)
- [Nomeroff Net (официальный сайт)](https://nomeroff.net.ua/)
- [ria-com/nomeroff-net (GitHub)](https://github.com/ria-com/nomeroff-net)
- [Спека валидационного датасета — autoriaNumberplateOcrRu](../about_datasets/autoria_numberplate_ocr_ru.md)
- [ГОСТ Р 50577-2018 (docs.cntd.ru)](https://docs.cntd.ru/document/1200160380)
- Источники в репозитории: `models.md`, `datasets.md`, `deploy/evaluation/evaluate.py` (`eval_nomeroff_ocr`, `eval_lprnet`, `normalize_ru_plate`, `ctc_decode`), `deploy/evaluation/metrics.py` (`compute_ocr_metrics`), `results_collected/ssh9.qudata.ai/results/{nomeroff_ocr,lprnet}/metrics.json`, `results_collected/FINAL_REPORT.md` (§4), `docs/architecture.md` (§6, legacy LPR_STN)

---

## История изменений

- **2026-06-04** — Создана спецификация модели в рамках документирования стека `models.md`.
