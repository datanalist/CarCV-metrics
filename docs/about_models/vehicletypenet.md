# VehicleTypeNet — классификация типа кузова (Type)

## Общая информация

**VehicleTypeNet** — предобученный классификатор типа кузова транспортного средства от NVIDIA TAO (Train, Adapt, Optimize), распространяемый через каталог NGC. Модель принимает на вход кроп автомобиля и относит его к одному из **6 типов кузова**. В конвейере CARS это вторичный классификатор (**SGIE2**), работающий на кропе ТС, выданном детектором TrafficCamNet (PGIE).

| Поле | Значение |
|------|----------|
| Имя модели | VehicleTypeNet |
| Задача в CARS | Type — классификация типа кузова (6 классов) |
| Роль в конвейере | SGIE2 (secondary GPU inference engine, на кропе ТС) |
| Вендор / источник | NVIDIA TAO, каталог NGC |
| Семейство | TAO classification (TF1) |
| Backbone | ResNet-18 (pruned) |
| Версия (по `models.md`) | `pruned_v1.0.2` |
| Версия (фактически скачивается `download_models.sh`) | `pruned_onnx_v1.1.0` |
| Локальное расположение (целевое, по `download_models.sh`) | `models/vehicletypenet/resnet18_pruned.onnx` + `labels.txt` |
| Локальный архив | `/home/mk/Загрузки/CarCVModels/vehicletypenet_pruned_onnx_v1.1.0.zip` |
| URL модели | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehicletypenet?version=pruned_v1.0.2 |

> **⚠️ Расхождение версий (требует сверки).** В источнике истины `models.md` (строка 10) для VehicleTypeNet указана версия **`pruned_v1.0.2`**, и URL в карточке ведёт на `?version=pruned_v1.0.2`. Однако скрипт `deploy/scripts/download_models.sh` (блок VehicleTypeNet — строки 43–50) выкачивает версию **`pruned_onnx_v1.1.0`** (присваивание `V="pruned_onnx_v1.1.0"` на строке 46), и именно из неё собран локальный архив `vehicletypenet_pruned_onnx_v1.1.0.zip`. То есть **фактически валидировалась версия `pruned_onnx_v1.1.0`**, а в реестре моделей записана `pruned_v1.0.2`. Расхождение оставлено как есть — синхронизация записей `models.md` и скрипта загрузки **TBD**.

> **Примечание о локальной копии.** Каталог `models/vehicletypenet/` в репозитории на момент написания спецификации **отсутствует** (модель распаковывается из архива при запуске эвалуатора). Все факты по файлам ниже сверены непосредственно по содержимому архива `vehicletypenet_pruned_onnx_v1.1.0.zip`.

---

## Архитектура

- **Семейство:** TAO classification (NVIDIA TAO Toolkit, наследие TF1/Keras-конвейера классификаторов TAO).
- **Backbone:** **ResNet-18**, **pruned** (структурно прорежённый для снижения числа параметров и ускорения inference на Jetson).
- **Голова:** полносвязный классификатор на **6 выходных классов** (типы кузова).
- **Квантизация:** в составе архива поставляется калибровочный кэш `resnet18_pruned_int8.txt` (6 918 байт) — таблица калибровки для **INT8**-движка TensorRT. Сама ONNX-модель — в FP32; INT8/FP16-движок собирается на целевом устройстве.

> Точные гиперпараметры обучения (датасет обучения, число эпох, степень прунинга) в локальной копии **отсутствуют** — берутся по карточке NGC. Здесь подтверждены только backbone (ResNet-18 pruned), число классов (6) и наличие INT8-калибровки.

---

## Вход / выход и препроцессинг

### Вход

| Параметр | Значение |
|----------|----------|
| Разрешение | **224 × 224** (W × H) |
| Каналы | 3 |
| Цветовой формат | **BGR без свопа** (как читает OpenCV) |
| Масштабирование | **нет** (`net-scale-factor = 1.0`) |
| Вычитание среднего (offsets, порядок B;G;R) | **(104, 117, 124)** |
| Формат тензора | **NCHW** |

VehicleTypeNet относится к семейству **TAO classifier** (как и VehicleMakeNet) — препроцессинг диспетчеризуется функцией `preprocess_tao_bgr` из `deploy/evaluation/evaluate.py`: BGR **без** перевода в RGB, без деления на 255, только поканальное вычитание offsets. Это отличает её от детекторов DetectNet_v2 (TrafficCamNet/FaceNet: BGR→RGB, /255) и от generic ImageNet-классификаторов (bae_model_f3: BGR→RGB, /255 + (x−mean)/std).

Фрагмент препроцессинга (`evaluate.py`, `preprocess_tao_bgr`):

```python
def preprocess_tao_bgr(img_bgr: np.ndarray, size: int = 224,
                      offsets=(103.939, 116.779, 123.68)) -> np.ndarray:
    """TAO TF1 classification preprocessing: BGR, no scaling, mean subtract.
    VehicleMakeNet/TypeNet use offsets=(104, 117, 124) (B;G;R)."""
    img = cv2.resize(img_bgr, (size, size)).astype(np.float32)
    img[..., 0] -= offsets[0]   # B
    img[..., 1] -= offsets[1]   # G
    img[..., 2] -= offsets[2]   # R
    return img.transpose(2, 0, 1)[None]  # NCHW
```

Вызов для VehicleTypeNet (`eval_vehicletypenet`):

```python
# VehicleTypeNet: BGR + offsets=(104, 117, 124)
inp = preprocess_tao_bgr(img, size=224, offsets=(104.0, 117.0, 124.0))
```

### Выход и декодирование

- Выходной тензор: вектор логитов размерности **6** (по числу классов).
- Декодирование: **softmax** → top-k. В эвалуаторе берётся **top-3** (`probs.argsort()[::-1][:3]`); top-1 — первый элемент.

```python
out = sess.run(None, {input_name: inp})[0][0]
probs = softmax(out)
top_k = [labels_norm[i] for i in probs.argsort()[::-1][:3]]
```

---

## Файлы и форматы

Состав архива `vehicletypenet_pruned_onnx_v1.1.0.zip` (18 446 791 байт ≈ **18.4 MB** на диске):

| Файл | Размер | Назначение |
|------|--------|------------|
| `resnet18_pruned.onnx` | 19 888 237 байт (≈ **19.9 MB** в десятичных MB; ≈ 18.97 MiB) | веса модели в формате ONNX (FP32) |
| `labels.txt` | 44 байта | список из 6 меток классов (по одной в строке) |
| `resnet18_pruned_int8.txt` | 6 918 байт | калибровочный кэш для сборки INT8-движка TensorRT |

Целевые локальные пути после загрузки (`download_models.sh`):

```
models/vehicletypenet/
├── resnet18_pruned.onnx
└── labels.txt
```

> Эвалуатор (`EVAL_CONFIGS["vehicletypenet"]`) ожидает файлы по путям `models/vehicletypenet/resnet18_pruned.onnx` и `models/vehicletypenet/labels.txt`. Ничего зашифрованного (`.etlt`) или требующего экспортного ключа в этой версии нет — ONNX поставляется в открытом виде.

> **Единицы измерения (без конфликта).** Значение «19.9 MB» из `FINAL_REPORT.md` — это те же 19 888 237 байт в десятичных MB (10⁶), а не другой файл и не противоречие; в MiB (2²⁰) те же байты дают ≈ 18.97 MiB. Размеры в таблицах выше приведены к десятичным MB.

---

## Классы / выходной словарь

Полный список из 6 классов (`labels.txt` архива; порядок = индексы выхода):

| Индекс | Класс | Тип кузова |
|--------|-------|------------|
| 0 | `coupe` | купе |
| 1 | `largevehicle` | крупное ТС (автобус / крупногабарит) |
| 2 | `sedan` | седан |
| 3 | `suv` | внедорожник / кроссовер |
| 4 | `truck` | грузовик / пикап |
| 5 | `van` | фургон / минивэн |

> В эвалуаторе метки нормализуются (`strip().lower()`). Класс `largevehicle` **не встречается среди ground-truth** при валидации на Stanford Cars (ни одно ключевое слово в именах классов Stanford на него не отображается), поэтому в `per_class_accuracy` метрик он отсутствует — см. раздел «Валидация».

---

## Использование для CarCV

В CARS VehicleTypeNet — вторичный классификатор **SGIE2**: на каждом кропе ТС, выданном PGIE-детектором TrafficCamNet, определяет тип кузова из 6 категорий. Результат используется как атрибут трека ТС.

### Применимость

- ✅ Готовый предобученный классификатор типа кузова от NVIDIA TAO, **из коробки совместимый с DeepStream/TensorRT** (тот же TAO-конвейер и препроцессинг, что у VehicleMakeNet).
- ✅ Лёгкий backbone **ResNet-18 (pruned)** + поставляемый **INT8-калибровочный кэш** → пригоден для целевого устройства Jetson Orin Nano как SGIE2.
- ✅ Препроцессинг детерминирован и совпадает с конвейером VehicleMakeNet (BGR, offsets=(104,117,124), NCHW) — единая SGIE-обработка кропа ТС.

### Ограничения

- ❌ **Провалена валидация на доступном датасете** (Stanford Cars): измеренный **top-1 = 0.3575** против порога **0.85** — см. «Валидация». Это окончательный валидный результат, а не повод для тюнинга.
- ⚠️ **Доменный разрыв и шумные метки.** Stanford Cars размечен по *make/model*, а не по типу кузова; тип кузова в валидации **выводится приблизительно** по ключевым словам из имени класса (`derive_typenet_label`). Метрики на Stanford Cars не отражают истинную точность по типу кузова.
- ⚠️ Класс **`largevehicle`** на Stanford Cars вообще не покрыт (нет соответствующих ключевых слов) — оценка по нему отсутствует.
- ⚠️ Все измерения сделаны на x86 GPU через ONNX Runtime — это **верхняя граница** точности относительно TensorRT FP16/INT8-движка на Jetson (не измерено на устройстве).
- ⚠️ Расхождение версий `pruned_v1.0.2` (реестр) ↔ `pruned_onnx_v1.1.0` (фактическая загрузка) — см. «Общая информация».

### Рекомендации

1. **Не интерпретировать FAIL на Stanford Cars как качество по типу кузова.** Stanford размечен по make/model; вывод типа по ключевым словам приблизителен и шумен. Для достоверной оценки нужен датасет, размеченный **именно по типу кузова** (исходно планировался BIT-Vehicle — заблокирован Kaggle-аутентификацией).
2. **Считать задачу Type закрытой в рамках текущей кампании.** Модель уже отвалидирована на доступном датасете (с известными оговорками), результат зафиксирован → вне активных задач кампании.
3. **При появлении корректного датасета** (тип кузова как нативная метка) перезапустить `eval_vehicletypenet`, исключив суррогатный keyword-mapping, и пересмотреть вердикт.
4. **Синхронизировать версии:** привести в соответствие запись `models.md` (`pruned_v1.0.2`) и фактически загружаемую версию `pruned_onnx_v1.1.0` (`download_models.sh`).
5. **Сборку движка на Jetson** вести с использованием поставляемого INT8-кэша (`resnet18_pruned_int8.txt`) и сверять точность INT8-движка с ONNX-baseline отдельно.

---

## Развёртывание на Jetson

| Параметр | Значение |
|----------|----------|
| Целевое устройство | NVIDIA Jetson Orin Nano 8GB |
| Рантайм | DeepStream SDK + TensorRT (engine) |
| Точность движка | **FP16** (есть также INT8-калибровка `resnet18_pruned_int8.txt`) |
| Роль | **SGIE2** (вторичный классификатор на кропе ТС) |
| Источник кропа | PGIE TrafficCamNet (детекция ТС) |
| Целевая latency | **3–4 ms** (*legacy-оценка, на устройстве не измерено*) |

> Целевая latency 3–4 ms взята из legacy-документации стека и **не подтверждена измерением** на Jetson. Все точностные числа из этого репозитория получены на x86 GPU через ONNX/onnxruntime-gpu и считаются верхней границей для TensorRT-движка на Jetson.

---

## Валидация

- **Датасет:** Stanford Cars Dataset (см. `datasets.md`, строка для Type → VehicleTypeNet). Спецификация датасета: [stanford_cars.md](../about_datasets/stanford_cars.md).
- **Порог pass/fail** (`evaluate.py`, `eval_vehicletypenet`): **`top1_accuracy ≥ 0.85`**.
- **Вердикт:** **FAIL** (окончательный, валидный результат).

### Измеренный результат

Источник: `results_collected/ssh9.qudata.ai/results/vehicletypenet/metrics.json` (число образцов **7 483**).

| Метрика | Значение | Порог | Статус |
|---------|----------|-------|--------|
| top-1 accuracy | **0.3575** | ≥ 0.85 | **FAIL** |
| top-3 accuracy | 0.7009 | — | — |

Per-class top-1 (по выведенным меткам типа кузова):

| Класс | top-1 |
|-------|-------|
| `truck` | 0.6361 (лучший) |
| `suv` | 0.3928 |
| `van` | 0.3815 |
| `coupe` | 0.3160 |
| `sedan` | 0.2894 (худший) |
| `largevehicle` | — (нет ground-truth на Stanford) |

### Методика вывода меток (суррогатный mapping)

Stanford-классы имеют формат `"<Make> <Model> <BodyType> <Year>"`; тип кузова извлекается по ключевым словам функцией `derive_typenet_label` (таблица `STANFORD_BODY_KEYWORDS`, приоритет — от более специфичных слов к общим; порядок строк ниже точно повторяет исходник `evaluate.py`, строки 405–421, и задаёт реальный приоритет: первое совпавшее ключевое слово выигрывает):

| # | Ключевое слово в имени Stanford | → метка VehicleTypeNet |
|---|---------------------------------|------------------------|
| 1 | `convertible` | `coupe` |
| 2 | `hatchback` | `sedan` |
| 3 | `minivan` | `van` |
| 4 | `crew cab` | `truck` |
| 5 | `extended cab` | `truck` |
| 6 | `regular cab` | `truck` |
| 7 | `cargo van` | `van` |
| 8 | `supercab` | `truck` |
| 9 | `wagon` | `sedan` |
| 10 | `sedan` | `sedan` |
| 11 | `coupe` | `coupe` |
| 12 | `suv` | `suv` |
| 13 | `hummer` | `suv` |
| 14 | `van` | `van` |
| 15 | `cab` | `truck` |

> **Порядок важен.** `cargo van → van` (поз. 7) стоит **выше** `supercab → truck` (поз. 8). Plain `cab → truck` — **последняя, самая низкоприоритетная пара** (поз. 15, ниже `wagon`/`sedan`/`coupe`/`suv`/`hummer`/`van`): она ловит остаточные «*Cab*» только после того, как все более специфичные слова не совпали.

Если ни одно ключевое слово не совпало, образец **пропускается** (`skipped_no_mapping`). Аудит соответствий пишется в `data/stanford_cars/type_mapping.csv` для воспроизводимости.

### Причина FAIL

**Доменный разрыв + шумные метки.** Stanford Cars размечен по make/model, а не по типу кузова; вывод типа по ключевым словам — приблизителен (например, любой `convertible` → `coupe`, `wagon` → `sedan`). При этом класс `largevehicle` на этом датасете не представлен. FAIL фиксируется как окончательный валидный результат — задача Type уже отвалидирована на доступном (пусть и суррогатном) датасете и **выведена из активных задач кампании**, тюнинг не предполагается.

> **Историческая заметка (legacy).** Изначально для Type планировался **BIT-Vehicle** (заблокирован Kaggle-аутентификацией). В `evaluate.py` сохранена legacy-ветка чтения BIT-раскладки (словарь `BIT_TO_TYPENET`: `bus/microbus → largevehicle`, `minivan/van → van`, `sedan → sedan`, `suv → suv`, `truck → truck`) на случай появления датасета с каталогами-по-классу. В `results_collected/FINAL_REPORT.md` VehicleTypeNet на одном из срезов фигурирует как «not evaluated / Pending dataset (no BIT-Vehicle access)» — это **более ранний статус**; актуальный измеренный результат — FAIL на Stanford Cars (см. `metrics.json` выше).

---

## Лицензия

- **Модель/веса:** NVIDIA TAO / NGC. Условия использования определяются лицензией модели на карточке NGC (NVIDIA Model EULA / соответствующая лицензия NGC) — **подтвердить по карточке** https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehicletypenet. Полный текст лицензии в локальной копии **отсутствует**.

**Вывод для CARS (КОММЕРЧЕСКИЙ продукт):**
- ⚠️ Перед включением VehicleTypeNet в состав коммерческого продукта необходимо **подтвердить условия лицензии NVIDIA на карточке NGC** (право на коммерческое использование/дистрибуцию предобученных весов и собранного TensorRT-движка) — требуется **legal review**.
- ⚠️ Точные условия (коммерческое использование, редистрибуция движка на устройстве) **не верифицированы** в рамках этой спецификации и помечены как TBD до юридической сверки.

---

## Ссылки

- [VehicleTypeNet на NGC (карточка модели)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehicletypenet?version=pruned_v1.0.2)
- Источник истины по моделям: `models.md` (строка 10)
- Пары модель × датасет: `datasets.md`
- Эвалуатор: `deploy/evaluation/evaluate.py` (`eval_vehicletypenet`, `derive_typenet_label`, `STANFORD_BODY_KEYWORDS`, `preprocess_tao_bgr`, `BIT_TO_TYPENET`)
- Скрипт загрузки: `deploy/scripts/download_models.sh` (строки 43–50)
- Измеренные метрики: `results_collected/ssh9.qudata.ai/results/vehicletypenet/metrics.json`
- Сводный отчёт: `results_collected/FINAL_REPORT.md`
- Спецификация датасета валидации: [stanford_cars.md](../about_datasets/stanford_cars.md)

---

## История изменений

- **2026-06-04** — Создана спецификация модели в рамках документирования стека models.md.
